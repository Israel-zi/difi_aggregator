"""
input_capture.py
----------------
DIFI Input Capture module.

Listens on multiple UDP ports simultaneously (one per Generator/Modem),
parses incoming DIFI packets (Context + Data), tags them with their
source stream ID, and places them into a shared queue for the Aggregator.

Each port corresponds to one DIFI stream:
  Port 50001 -> Stream ID 0x00000001 (Modem 1)
  Port 50002 -> Stream ID 0x00000002 (Modem 2)
"""

import heapq
import os
import sys
import socket
import struct
import threading
import queue
import time
from dataclasses import dataclass

if not getattr(sys, 'frozen', False):
    _src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _src not in sys.path:
        sys.path.insert(0, _src)

from core.difi_packet import (
    DifiDataPacket,
    DifiContextPacket,
    PACKET_TYPE_DATA,
    PACKET_TYPE_CONTEXT,
    PROLOGUE_WORDS,
    peek_header,
    peek_first_iq,
)
from pipeline_logger import wall_clock_str, sample_fingerprint
from thread_priority import boost_current_thread, THREAD_PRIORITY_TIME_CRITICAL, THREAD_PRIORITY_HIGHEST
from socket_warmup import warm_up_socket


# ─────────────────────────────────────────────
# Tagged packet container
# ─────────────────────────────────────────────

@dataclass
class CapturedPacket:
    """A received DIFI packet tagged with its source port."""
    source_port:    int
    received_at:    float                           # time.monotonic() timestamp
    packet:         DifiDataPacket | DifiContextPacket


@dataclass
class RawCapturedPacket:
    """LAN relay mode's lightweight counterpart to CapturedPacket (2026-09-03).

    At hold_ms=0 the Combiner never touches a packet's payload -- it only
    filters by stream_id and retransmits unchanged. The full-parse path
    (CapturedPacket wrapping a DifiDataPacket) was paying for a numpy IQ
    unpack on receipt (from_bytes), pickling that numpy array across the
    packet_q process boundary, and a numpy IQ re-pack before sendto()
    (to_bytes) -- on every single packet, for zero benefit, since nothing
    about the payload is ever inspected or modified in relay mode. Measured
    on real-VM logs as a major, previously-uncounted contributor to
    packet_q hand-off loss (separate from the queue-capacity mechanism
    already documented in capture_worker.py's docstring).

    This carries the RAW wire bytes straight through -- combiner_worker.py's
    _relay_loop() does a plain sendto(raw_bytes, dest), no re-encode -- plus
    only the header fields peek_header()/peek_first_iq() can read without
    touching the payload, which is everything the relay path's filtering
    and CSV logging actually need."""
    source_port:    int
    received_at:    float
    raw_bytes:      bytes
    pkt_type:       int
    stream_id:      int
    seq_num:        int
    timestamp_int:  int
    timestamp_frac: int
    n_samples:      int
    first_i:        object   # float, or "" for CONTEXT packets
    first_q:        object


# ─────────────────────────────────────────────
# Per-port listener thread
# ─────────────────────────────────────────────

class PortListener(threading.Thread):
    """
    Listens on a single UDP port, parses DIFI packets, and puts them
    into the shared output queue.
    """

    MAX_UDP_SIZE = 65535

    def __init__(
        self,
        port: int,
        out_queue: queue.Queue,
        host: str = "0.0.0.0",
        timeout: float = 1.0,
        packet_logger = None,   # pipeline_logger.PacketLogger, or None
        rcvbuf_bytes: int = 512 * 1024,
        relay_mode: bool = False,   # True -> cheap header-peek + raw-bytes passthrough, see RawCapturedPacket
    ):
        super().__init__(daemon=True, name=f"listener-{port}")
        self.port      = port
        self.out_queue = out_queue
        self.host      = host
        self.timeout   = timeout
        self._packet_logger = packet_logger
        self._rcvbuf_bytes = rcvbuf_bytes
        self._relay_mode = relay_mode
        self._stop_evt = threading.Event()

        # Bind synchronously in the caller's thread so a failure (port already
        # in use by another process, permission denied, etc.) raises immediately
        # here instead of silently killing a background thread with no one
        # ever finding out why no packets are arriving.
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # Windows' default UDP receive buffer is small (measured 64KB on this
        # host) -- some headroom above that default is worth having so a
        # brief burst doesn't get dropped outright. The 512KB default below
        # was sized against an old, DIFFERENT failure mode: a SUSTAINED low
        # drain rate (~275 pkt/s) under GIL contention, where a big buffer
        # just hid the deficit as multi-second creeping latency instead of
        # dropping (measured: an 8MB buffer produced a steady 7-9s hold).
        # 2026-09-01 field evidence (real VM, real network) points at a
        # DIFFERENT mechanism instead: a hard, RATE-INDEPENDENT receive
        # ceiling (~620-660 pkt/s regardless of offered load, CPU/RAM both
        # idle) with a receive-side inter-arrival pattern dominated by a
        # rhythmic ~2.2ms gap not present in the sender's own send pattern in
        # the same proportion -- consistent with vmxnet3 interrupt-coalescing
        # batching delivery rather than a sustained Python-side drain
        # bottleneck. Against THAT failure mode the old reasoning doesn't
        # directly apply -- there's no sustained deficit to hide as creeping
        # latency, only (locally, reproduced via the real multiprocessing
        # worker topology) occasional 100-400ms recvfrom() stalls that a
        # bigger buffer could plausibly absorb without loss. Configurable
        # via rcvbuf_bytes specifically to A/B test this rather than assume
        # either the old warning or "bigger is just better".
        try:
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self._rcvbuf_bytes)
        except OSError:
            pass   # best-effort -- some platforms/permissions cap this; not fatal
        self._sock.settimeout(self.timeout)
        try:
            self._sock.bind((self.host, self.port))
        except OSError:
            self._sock.close()
            raise

        # per-stream statistics
        self.stats = {
            "data_received":    0,
            "context_received": 0,
            "parse_errors":     0,
            "bytes_received":   0,   # raw wire bytes of every successfully-parsed packet
        }

    def stop(self):
        self._stop_evt.set()
        # Close socket immediately to unblock recvfrom — no 1-second timeout wait
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass

    def run(self):
        # See thread_priority.py: this is the single most latency-critical
        # thread in the whole pipeline -- every millisecond it isn't
        # scheduled to call recvfrom() is a millisecond the OS's UDP
        # receive buffer isn't being drained, and on a CPU-constrained
        # machine that's exactly what causes real datagrams to be dropped
        # before this process ever sees them. Best-effort; a failed boost
        # (non-Windows, or the call itself failing) never blocks capture.
        boost_current_thread(THREAD_PRIORITY_TIME_CRITICAL)

        # See socket_warmup.py: a just-bound socket does not actually
        # deliver inbound traffic for several seconds on this host, even
        # though bind() itself already returned. Any real packets sent to
        # this port before this finishes are silently dropped -- absorb
        # that cost here, before declaring this port ready, rather than
        # losing real acquisition data to it.
        warm_ms, leaked = warm_up_socket(self._sock, self.port)
        print(f"[Capture] Port {self.port} socket warm-up took {warm_ms:.0f} ms")
        if leaked is not None and not self._stop_evt.is_set():
            self._parse_and_enqueue(leaked)

        print(f"[Capture] Listening on {self.host}:{self.port}")

        while not self._stop_evt.is_set():
            try:
                data, _ = self._sock.recvfrom(self.MAX_UDP_SIZE)
                self._parse_and_enqueue(data)
            except socket.timeout:
                continue
            except OSError:
                break   # socket closed (by stop() or network error)

        try:
            self._sock.close()
        except OSError:
            pass  # already closed by stop()
        print(f"[Capture] Port {self.port} listener stopped")

    def _parse_and_enqueue(self, data: bytes):
        """Detect packet type from header and parse accordingly."""
        if len(data) < 4:
            return

        if self._relay_mode:
            self._parse_and_enqueue_relay(data)
            return

        # peek at bits 31-28 of Word 1 to detect packet type
        word1    = int.from_bytes(data[:4], "big")
        pkt_type = (word1 >> 28) & 0xF

        try:
            if pkt_type == PACKET_TYPE_DATA:
                pkt = DifiDataPacket.from_bytes(data)
                self.stats["data_received"] += 1
                n_samples = len(pkt.payload)
                first_i, first_q = sample_fingerprint(pkt.payload)

            elif pkt_type == PACKET_TYPE_CONTEXT:
                pkt = DifiContextPacket.from_bytes(data)
                self.stats["context_received"] += 1
                n_samples = 0
                first_i, first_q = "", ""

            else:
                # unknown packet type — skip silently
                return

            self.stats["bytes_received"] += len(data)

            if self._packet_logger is not None:
                pkt_kind = "DATA" if pkt_type == PACKET_TYPE_DATA else "CONTEXT"
                self._packet_logger.log(
                    wall_clock_str(), self.port, f"0x{pkt.stream_id:08X}", pkt_kind,
                    pkt.seq_num, pkt.timestamp_int, pkt.timestamp_frac, n_samples,
                    first_i, first_q,
                )

            # Non-blocking put: if the queue is full, drop this packet rather
            # than blocking the receive thread (which would let the OS UDP
            # buffer fill and silently lose newer packets from the TX).
            try:
                self.out_queue.put_nowait(CapturedPacket(
                    source_port = self.port,
                    received_at = time.monotonic(),
                    packet      = pkt,
                ))
            except queue.Full:
                self.stats["parse_errors"] += 1   # reuse counter; counts drops

        except (ValueError, struct.error) as exc:
            self.stats["parse_errors"] += 1
            print(f"[Capture] Parse error on port {self.port}: {exc}")

    def _parse_and_enqueue_relay(self, data: bytes):
        """LAN relay mode's cheap path -- see RawCapturedPacket. Header-only
        peek (no numpy IQ unpack); the raw wire bytes go into out_queue
        unchanged, ready for combiner_worker.py's _relay_loop() to
        sendto() directly with no re-encode."""
        try:
            pkt_type, stream_id, seq_num, ts_int, ts_frac = peek_header(data)
        except (ValueError, struct.error) as exc:
            self.stats["parse_errors"] += 1
            print(f"[Capture] Parse error on port {self.port}: {exc}")
            return

        if pkt_type == PACKET_TYPE_DATA:
            self.stats["data_received"] += 1
            n_samples = (len(data) - PROLOGUE_WORDS * 4) // 4
            first_i, first_q = peek_first_iq(data)
        elif pkt_type == PACKET_TYPE_CONTEXT:
            self.stats["context_received"] += 1
            n_samples = 0
            first_i, first_q = "", ""
        else:
            return   # unknown packet type -- skip silently, same as the full-parse path

        self.stats["bytes_received"] += len(data)

        if self._packet_logger is not None:
            pkt_kind = "DATA" if pkt_type == PACKET_TYPE_DATA else "CONTEXT"
            self._packet_logger.log(
                wall_clock_str(), self.port, f"0x{stream_id:08X}", pkt_kind,
                seq_num, ts_int, ts_frac, n_samples, first_i, first_q,
            )

        try:
            self.out_queue.put_nowait(RawCapturedPacket(
                source_port=self.port, received_at=time.monotonic(), raw_bytes=data,
                pkt_type=pkt_type, stream_id=stream_id, seq_num=seq_num,
                timestamp_int=ts_int, timestamp_frac=ts_frac, n_samples=n_samples,
                first_i=first_i, first_q=first_q,
            ))
        except queue.Full:
            self.stats["parse_errors"] += 1


# ─────────────────────────────────────────────
# Multi-port capture manager
# ─────────────────────────────────────────────

class InputCapture:
    """
    Manages multiple PortListener threads and exposes a single queue
    containing tagged packets from all sources.

    Usage
    -----
        capture = InputCapture(ports=[50001, 50002])
        capture.start()

        while True:
            captured = capture.get(timeout=1.0)
            if captured:
                process(captured.packet)

        capture.stop()
    """

    def __init__(
        self,
        ports: list,
        host: str        = "0.0.0.0",
        queue_maxsize: int = 30,
        packet_logger    = None,   # pipeline_logger.PacketLogger, or None -- shared across all ports
        packet_logger_factory = None,   # optional Callable[[port:int], logger] -- ONE PER PORT instead
        rcvbuf_bytes: int = 512 * 1024,
        relay_mode: bool = False,   # see PortListener/RawCapturedPacket
    ):
        # 2026-09-01: tried passing a multiprocessing.Queue in here directly
        # (PortListener enqueuing straight into it, no in-process relay
        # thread/hop) to cut capture_worker.py down to fewer threads --
        # measured WORSE, not better: loss at 8000 pkt/s went from ~37% (with
        # the relay thread) to ~53% (without it), collapsing further at
        # higher rates. Reverted. The relay thread isn't just overhead: it
        # shields THIS thread (which must call recvfrom() as promptly and
        # consistently as possible) from multiprocessing.Queue.put_nowait()'s
        # own latency (pickling + internal lock/pipe handoff), which an
        # isolated single-purpose-thread benchmark showed is fast on
        # average (clean past 15000 pkt/s) but apparently variable enough
        # that folding it into the recvfrom-critical thread itself costs
        # real drops. Keep out_queue as a plain in-process queue.Queue;
        # capture_worker.py's own relay thread is what should be tuned
        # further, not this.
        self._out_queue = queue.Queue(maxsize=queue_maxsize)
        self._listeners = []
        self._packet_logger = packet_logger
        self._packet_logger_factory = packet_logger_factory
        self._rcvbuf_bytes = rcvbuf_bytes
        self._relay_mode = relay_mode
        # Ports that failed to bind (e.g. already in use by another program) —
        # collected instead of raised so the other ports still start.
        self.bind_errors: dict = {}
        for p in ports:
            try:
                self._listeners.append(
                    PortListener(port=p, out_queue=self._out_queue, host=host,
                                 packet_logger=self._logger_for(p), rcvbuf_bytes=rcvbuf_bytes,
                                 relay_mode=relay_mode)
                )
            except OSError as exc:
                self.bind_errors[p] = str(exc)
                print(f"[Capture] Failed to bind port {p}: {exc}")

    def _logger_for(self, port: int):
        """Per-port logger if a factory was given (2026-09-03: multiple
        PortListener threads sharing ONE AsyncPacketLogger means multiple
        producer threads contending on that logger's internal queue.Queue
        lock on every single packet -- measured to matter in exactly the
        same way already documented for Aggregator/Packetizer sharing a
        hold_log: with 8 simultaneous streams at ~681 pkt/s each (~5450
        pkt/s combined, well within already-proven-clean aggregate rates
        for 2-3 streams), real loss appeared (31-43%) that vanished at
        lower stream counts carrying the SAME or higher aggregate rate --
        pointing at stream COUNT, not byte rate, and this shared-logger
        contention was the leading suspect. One dedicated logger per port
        removes that contention entirely), else the single shared logger
        (unchanged behavior for every other caller)."""
        if self._packet_logger_factory is not None:
            return self._packet_logger_factory(port)
        return self._packet_logger

    def start(self):
        """Start all listener threads."""
        for listener in self._listeners:
            listener.start()
        print(f"[Capture] Started {len(self._listeners)} listener(s)")

    def stop(self):
        """Stop all listener threads."""
        for listener in self._listeners:
            listener.stop()
        for listener in self._listeners:
            listener.join(timeout=2.0)
            if listener.is_alive():
                # A listener stuck here keeps its socket bound and keeps
                # running in the background, invisible, while the next
                # Listen cycle creates a brand new set on top of it -- a
                # plausible explanation for performance degrading across
                # repeated Stop/Start cycles within one long-running session.
                print(f"[Capture] WARNING: listener on port {listener.port} "
                      f"did not stop within 2s -- still running")
        print("[Capture] All listeners stopped")

    def add_port(self, port: int, host: str = "0.0.0.0"):
        """Start a new listener on the given port while already running.

        Raises OSError (unchanged) if the port can't be bound — callers
        should catch this and surface it rather than let it vanish.
        """
        listener = PortListener(port=port, out_queue=self._out_queue, host=host,
                                 packet_logger=self._logger_for(port), rcvbuf_bytes=self._rcvbuf_bytes,
                                 relay_mode=self._relay_mode)
        self._listeners.append(listener)
        listener.start()
        print(f"[Capture] Added listener on port {port}")

    def remove_port(self, port: int):
        """Stop and remove the listener for the given port."""
        for listener in list(self._listeners):
            if listener.port == port:
                listener.stop()
                listener.join(timeout=2.0)
                self._listeners.remove(listener)
                print(f"[Capture] Removed listener on port {port}")
                return

    def get(self, timeout: float = 0.1):
        """
        Retrieve the next captured packet.
        Returns None if nothing arrived within `timeout` seconds.
        """
        try:
            return self._out_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stats(self) -> dict:
        """Return combined statistics from all listeners."""
        combined = {"data_received": 0, "context_received": 0, "parse_errors": 0, "bytes_received": 0}
        for listener in self._listeners:
            for key in combined:
                combined[key] += listener.stats[key]
        return combined

    def port_stats(self) -> dict:
        """Return {port: data_received} for every currently active listener."""
        return {listener.port: listener.stats["data_received"] for listener in self._listeners}

    @property
    def queue_size(self) -> int:
        return self._out_queue.qsize()


# ─────────────────────────────────────────────
# Jitter / reorder buffer (WAN deployments)
# ─────────────────────────────────────────────

class JitterBuffer:
    """
    Per-stream timestamp-ordered reorder buffer for WAN deployments.

    Sits between InputCapture and Aggregator.  DIFI Data packets are held
    per stream in a min-heap keyed by (timestamp_int, timestamp_frac) and
    released in chronological order once they have been in the buffer for at
    least ``hold_ms`` milliseconds.  This absorbs network jitter up to
    ``hold_ms`` and corrects out-of-order packet arrival within each stream.

    Context packets are forwarded immediately (stateless, no IQ data).

    With ``hold_ms=0`` (the default, appropriate for LAN) this is a
    zero-overhead pass-through equivalent to using InputCapture directly.

    Parameters
    ----------
    capture  : InputCapture to read raw packets from.
    hold_ms  : Jitter budget in milliseconds.
               0  → LAN pass-through (zero added latency).
               100-300 → typical WAN setting.
    """

    def __init__(self, capture: InputCapture, hold_ms: float = 0.0):
        self._capture = capture
        self._hold_s  = hold_ms / 1000.0
        self._enabled = hold_ms > 0

        # per-stream min-heap of (ts_int, ts_frac, seq, CapturedPacket)
        # seq is a monotonic push counter used as a tiebreaker so that Python
        # never falls through to comparing CapturedPacket objects (no __lt__).
        self._heaps: dict  = {}
        self._push_seq: int = 0
        self._lock         = threading.Lock()

        # 2026-09-01: raised from 128 -- combiner_worker.py now always
        # engages this class (even at hold_ms=0, pure pass-through) because
        # its own intake thread turned out to matter for throughput, not
        # just WAN jitter: it shields the Aggregator's own thread from
        # multiprocessing.Queue.get()'s latency the same way
        # capture_worker.py's forward thread shields PortListener from
        # put()'s latency (see combiner_worker.py's do_listen() comment).
        # At real target rates (several thousand pkt/s per stream) 128 is
        # far too small a burst buffer for that role; sized to match
        # packet_q's own 4096 instead.
        self._out_queue = queue.Queue(maxsize=4096)
        self._stop_evt  = threading.Event()
        self._thread    = threading.Thread(
            target=self._run, daemon=True, name="jitter-buffer"
        )

        self.gaps_detected = 0

    # ── lifecycle ──────────────────────────────────────────────────────────

    def start(self):
        self._thread.start()
        mode = f"hold={self._hold_s * 1000:.0f} ms (WAN)" if self._enabled else "pass-through (LAN)"
        print(f"[JitterBuffer] Started — {mode}")

    def stop(self):
        self._stop_evt.set()
        self._thread.join(timeout=3.0)
        if self._thread.is_alive():
            print("[JitterBuffer] WARNING: worker thread did not stop within 3s -- still running")
        # _run()'s loop exits as soon as _stop_evt is set, without draining
        # whatever's still sitting in the per-stream heaps waiting out its
        # hold window -- every one of those packets was real, already-
        # captured data that would otherwise be silently discarded here,
        # invisible to every counter (confirmed directly: at 1000 pkt/s with
        # a 200ms hold, capture-stage receive count ran ~900 packets ahead
        # of what the Aggregator ever emitted, entirely explained by this).
        # Flush them out now, still in timestamp order per stream, so a
        # stop mid-flight loses nothing that was actually captured.
        flushed = 0
        with self._lock:
            for sid, heap in self._heaps.items():
                while heap:
                    _, _, _, captured = heapq.heappop(heap)
                    try:
                        self._out_queue.put_nowait(captured)
                        flushed += 1
                    except queue.Full:
                        self.gaps_detected += 1
            self._heaps.clear()
        print(f"[JitterBuffer] Stopped | gaps detected: {self.gaps_detected} | flushed on stop: {flushed}")

    def get(self, timeout: float = 0.1):
        """Drop-in replacement for InputCapture.get()."""
        try:
            return self._out_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def set_hold_ms(self, hold_ms: float):
        """Change the jitter budget at runtime (no restart needed)."""
        self._hold_s  = hold_ms / 1000.0
        self._enabled = hold_ms > 0

    # ── internal ──────────────────────────────────────────────────────────

    def _run(self):
        # High, not time-critical: still above normal (so it doesn't fall
        # behind Packetizer/GUI work under CPU contention), but the port
        # listeners themselves come first -- see thread_priority.py.
        boost_current_thread(THREAD_PRIORITY_HIGHEST)
        while not self._stop_evt.is_set():
            try:
                captured = self._capture.get(timeout=0.02)
                if captured is not None:
                    if not self._enabled:
                        try:
                            self._out_queue.put_nowait(captured)
                        except queue.Full:
                            # Was previously silently discarded with no
                            # counter at all -- pass-through mode's own
                            # gaps_detected stat only ever incremented on
                            # the hold-mode _drain() path, so this specific
                            # drop point was completely invisible even in
                            # memory, let alone in any log.
                            self.gaps_detected += 1
                    else:
                        self._push(captured)
                if self._enabled:
                    self._drain(time.monotonic())
            except Exception as exc:
                print(f"[JitterBuffer] Internal error (thread continues): {exc}")

    def _push(self, captured: CapturedPacket):
        pkt = captured.packet
        sid = pkt.stream_id

        if not isinstance(pkt, DifiDataPacket):
            # Context packets carry no IQ samples — forward immediately.
            try:
                self._out_queue.put_nowait(captured)
            except queue.Full:
                pass
            return

        with self._lock:
            if sid not in self._heaps:
                self._heaps[sid] = []
            heapq.heappush(
                self._heaps[sid],
                (pkt.timestamp_int, pkt.timestamp_frac, self._push_seq, captured),
            )
            self._push_seq += 1

    def _drain(self, now: float):
        """
        Release packets whose hold window has expired.

        Each packet is held for at least hold_s seconds after it arrived.
        By that time, any packet with a smaller DIFI timestamp that was still
        in transit across the WAN should have arrived — or is declared lost.
        Packets are always emitted in ascending (ts_int, ts_frac) order.
        """
        with self._lock:
            for sid, heap in self._heaps.items():
                while heap:
                    ts_int, ts_frac, _seq, captured = heap[0]
                    if now - captured.received_at < self._hold_s:
                        break   # oldest packet hasn't waited long enough yet
                    heapq.heappop(heap)
                    try:
                        self._out_queue.put_nowait(captured)
                    except queue.Full:
                        self.gaps_detected += 1


# ─────────────────────────────────────────────
# Quick self-test (standalone)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    TEST_PORTS = [50001, 50002]
    capture    = InputCapture(ports=TEST_PORTS)
    capture.start()

    print(f"\n[Test] Waiting for DIFI packets on ports {TEST_PORTS} ...")
    print("       Run generator.py in another terminal to send packets.")
    print("       Press Ctrl+C to stop.\n")

    received = 0
    try:
        while True:
            captured = capture.get(timeout=1.0)
            if captured:
                received += 1
                pkt = captured.packet
                print(
                    f"  Port {captured.source_port} | "
                    f"type={'DATA' if isinstance(pkt, DifiDataPacket) else 'CTX'} | "
                    f"stream=0x{pkt.stream_id:08X} | "
                    f"seq={pkt.seq_num}"
                )
    except KeyboardInterrupt:
        pass
    finally:
        capture.stop()
        print(f"\n[Test] Received {received} packets total")
        print(f"[Test] Stats: {capture.stats()}")