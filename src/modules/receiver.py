"""
receiver.py
-----------
DIFI Receiver module.

Listens on a single UDP port for the multiplexed DIFI stream emitted by
the Sender.  Incoming packets carry their original Stream IDs (as defined
in IEEE-ISTO Std 4900-2021 Figure 7), so each stream is tracked
independently.

Per-stream rolling IQ buffers are maintained; callers use
get_stream_snapshots() to retrieve all active streams at once for display.
"""

import os
import sys
import socket
import threading
import time
import queue

import numpy as np

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
from socket_warmup import warm_up_socket


class DifiReceiver:
    """
    Receives a multiplexed DIFI stream and maintains per-stream IQ buffers.

    Parameters
    ----------
    host        : local address to bind
    port        : UDP port to listen on
    buffer_size : number of IQ samples per stream in the rolling display buffer
    """

    MAX_UDP_SIZE = 65535

    # 2026-09-05: full parse cost (DifiDataPacket.from_bytes(), ~44x a
    # header-only peek per this project's own earlier measurement) was
    # confirmed as the dominant cost keeping this thread from sustaining a
    # real 2-stream ~4500 pkt/s combined load (measured directly: ~1470
    # packet/s drops, ~30% of arrival, even after the circular-buffer fix
    # removed the OTHER per-packet cost this same hot path had). The
    # rolling display buffer only actually needs to be refreshed often
    # enough to look smooth at the GUI's own ~10Hz redraw rate, not on
    # every single packet -- decoding 1 in DECODE_EVERY_N still delivers
    # far more fresh samples per second than that requires, while cutting
    # the expensive-parse call rate (and this thread's CPU cost) by
    # roughly (DECODE_EVERY_N-1)/DECODE_EVERY_N. Every packet still gets
    # counted, sequence-checked and logged either way -- see _handle()'s
    # peek_header()/peek_first_iq() path for the skipped ones, the same
    # header-only functions ring_capture_main and generator.py already
    # use for exactly this reason.
    DECODE_EVERY_N = 4

    # 2026-09-05: same DIAGNOSTIC MODE convention as ring_pipeline.py --
    # env-gated (DIFI_DEBUG_REORDER=1), zero cost otherwise, never set by
    # the real EXEs' own startup path. Added after confirming this
    # process's memory grew unbounded (1.36GB -> 2.47GB in ~2 minutes)
    # with no timestamped signal anywhere that would have shown WHEN the
    # backlog started forming or how large _raw_q had gotten at that
    # moment -- the only evidence was periodic manual `tasklist` memory
    # samples, days too coarse to actually catch the mechanism. This
    # heartbeat gives the same kind of "walk backward from the last known
    # good moment" timeline this project's other DIFI_DEBUG_REORDER
    # instrumentation already relies on.
    _DEBUG_REORDER = os.environ.get("DIFI_DEBUG_REORDER") == "1"

    def __init__(
        self,
        host: str        = "0.0.0.0",
        port: int        = 50010,
        buffer_size: int = 8192,
        packet_logger    = None,   # pipeline_logger.PacketLogger, or None
    ):
        self._host        = host
        self._port        = port
        self._buffer_size = buffer_size
        self._packet_logger = packet_logger
        self._sock        = None
        self._stop_evt    = threading.Event()
        # 2026-09-05: unbounded until now -- confirmed directly: over a
        # real multi-minute run, this process's memory climbed from
        # 1.36GB to 2.47GB in about 2 minutes with nothing else wrong in
        # the pipeline (Combiner ingress/egress counts stayed healthy the
        # whole time). Unlike ring_capture_main's own raw_q, _handle()
        # here can't switch to the cheap peek_header()/peek_first_iq()
        # path that fixed the equivalent spot there -- this class actually
        # needs the fully decoded IQ payload for the rolling display
        # buffer, not just header fields -- so DifiDataPacket.from_bytes()
        # (the ~44x-more-expensive full parse, per this project's own
        # earlier measurement) runs on every packet here by necessity.
        # Under a real burst (the Combiner releases a whole hold-window's
        # worth of both streams at once), that parse cost can fall behind
        # the multiplexed arrival rate, and an unbounded queue.Queue just
        # keeps the backlog in memory forever instead of ever catching up
        # or giving up. Bounding it converts silent unbounded growth into
        # a visible, counted drop -- same principle already applied to
        # AsyncPacketLogger's own queue for the same reason.
        self._raw_q       = queue.Queue(maxsize=20_000)
        self.raw_q_dropped = 0
        # 2026-09-04: split recvfrom() into its own dedicated thread (raw
        # bytes only, handed off to _raw_q) with this thread left doing
        # only the CPU-bound work (DIFI parse, ring-buffer update,
        # synchronous CSV log write) -- previously both lived in the same
        # loop (_run() called recvfrom() then _handle() inline). A real
        # multi-stream GUI test (Base/LOGS, 2026-09-04 ~19:04-19:07)
        # showed near-lossless delivery TX->Combiner and Combiner-ingress
        # -> Combiner-egress (off by ~1 packet, just an in-flight-at-stop
        # artifact) but 13-41% loss AND real timestamp-order inversions
        # specifically on the Combiner->Receiver leg -- exactly the
        # signature of this same anti-pattern already found and fixed in
        # ring_capture_main this session (see ring_pipeline.py's own
        # docstring): CPU-bound work sharing recvfrom()'s thread starves
        # how often the OS socket is actually drained, so a burst
        # (unavoidable here -- egress catches up on every stream whose
        # 200ms hold has elapsed, all at once) overflows the kernel
        # receive buffer and the OS silently drops/reorders what arrives
        # while this thread is busy elsewhere.
        self._recv_thread = threading.Thread(
            target=self._recv_loop, daemon=True, name="receiver-recv"
        )
        self._thread      = threading.Thread(
            target=self._run, daemon=True, name="receiver-process"
        )

        # per-stream state — keyed by stream_id (int)
        self._iq_buffers:  dict = {}   # stream_id -> np.ndarray[complex64], CIRCULAR storage (see _write_idx)
        self._write_idx:   dict = {}   # stream_id -> next write position in _iq_buffers[sid]
        self._decode_counter: dict = {}   # stream_id -> packets seen, for the every-Nth-packet full decode below
        self._contexts:    dict = {}   # stream_id -> DifiContextPacket
        self._last_seqs:   dict = {}   # stream_id -> last seen seq_num (0-15)
        self._last_update: dict = {}   # stream_id -> time.monotonic() of last data packet
        self._lock         = threading.Lock()

        # stats
        self.data_received    = 0
        self.context_received = 0
        self.parse_errors     = 0
        self.seq_errors       = 0
        self.bytes_received   = 0

    # ── lifecycle ──────────────────────────────────────────────────────────

    def start(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # See PortListener's identical setsockopt in input_capture.py for why:
        # the OS default UDP receive buffer is small enough that a real burst
        # can overflow it and get silently dropped before Python ever calls
        # recvfrom() -- invisible to every application-level counter here.
        # 2026-09-04: bumped 512KB -> 16MB to match ring_capture_main's own
        # rcvbuf_bytes default -- the Combiner's egress deliberately releases
        # every stream whose target-delay hold has elapsed in one tight
        # catch-up burst each tick, and 512KB left no real headroom for
        # that burst while this socket's OWN thread was also busy parsing/
        # logging the previous batch (see the recv/process split below).
        try:
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 * 1024 * 1024)
        except OSError:
            pass
        self._sock.settimeout(1.0)
        self._sock.bind((self._host, self._port))
        self._thread.start()
        self._recv_thread.start()
        print(f"[Receiver] Listening on {self._host}:{self._port}")

    def stop(self):
        self._stop_evt.set()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        self._recv_thread.join(timeout=2.0)
        self._thread.join(timeout=2.0)
        print(
            f"[Receiver] Stopped | "
            f"data={self.data_received} ctx={self.context_received} "
            f"streams={list(f'0x{s:08X}' for s in self._contexts)}"
        )

    def rebind(self, port: int):
        """Stop listening on the current port and rebind to a new one at runtime."""
        self._stop_evt.set()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        self._recv_thread.join(timeout=2.0)
        self._thread.join(timeout=2.0)

        self._port     = port
        self._stop_evt = threading.Event()
        self._raw_q    = queue.Queue(maxsize=self._raw_q.maxsize)
        self.raw_q_dropped = 0
        self._sock     = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 * 1024 * 1024)
        except OSError:
            pass
        self._sock.settimeout(1.0)
        self._sock.bind((self._host, self._port))
        self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True, name="receiver-recv")
        self._thread = threading.Thread(target=self._run, daemon=True, name="receiver-process")
        self._thread.start()
        self._recv_thread.start()
        print(f"[Receiver] Rebound to {self._host}:{self._port}")

    # ── data access ────────────────────────────────────────────────────────

    def get_stream_snapshots(self, tail_samples: int | None = None) -> dict:
        """
        Return a dict of {stream_id: (iq_array, context)} for all active streams.
        Both arrays are copies (thread-safe).  Streams without a context packet
        yet are included with context=None.

        tail_samples: if given, copy only the last N samples per stream
        instead of the full rolling buffer (default buffer_size=8192).
        Pass 0 to skip the IQ copy entirely (an empty array is returned) --
        for callers that only need the context objects.

        2026-09-04: added after finding the GUI's own display timer
        (receiver_app.py's _tick(), every 100ms) was copying the FULL
        buffer for every stream just to feed a 1024-sample FFT window
        (_stream_fft's own seg_len) -- an 8x larger copy than needed, held
        under the SAME lock the background receive thread needs for every
        incoming packet. On a loaded system this widened the window for
        the GUI thread to visibly stall (spectrum frozen, looking like a
        disconnect) even though the underlying receive thread itself never
        actually lost data (confirmed separately via the packet logs)."""
        with self._lock:
            if tail_samples == 0:
                return {
                    sid: (np.empty(0, dtype=np.complex64), self._contexts.get(sid))
                    for sid in self._iq_buffers
                }
            return {
                sid: (self._chronological(sid, tail_samples), self._contexts.get(sid))
                for sid in self._iq_buffers
            }

    def get_iq_snapshot(self) -> np.ndarray:
        """
        Return combined IQ snapshot across all streams (concatenated, not interleaved).
        Preserves backwards compatibility with single-stream callers.
        """
        with self._lock:
            if not self._iq_buffers:
                return np.zeros(self._buffer_size, dtype=np.complex64)
            return np.concatenate([self._chronological(sid) for sid in self._iq_buffers])

    @property
    def context(self):
        """Return context for the first available stream (backwards compatibility)."""
        with self._lock:
            if self._contexts:
                return next(iter(self._contexts.values()))
            return None

    def flush(self):
        """Zero all IQ buffers and reset sequence tracking after parameter changes."""
        with self._lock:
            for sid in self._iq_buffers:
                self._iq_buffers[sid][:] = 0
                self._write_idx[sid] = 0
            self._last_seqs.clear()

    def get_sample_rate(self) -> float:
        """Return sample rate from the first available context (or default)."""
        ctx = self.context
        return ctx.sample_rate_hz if ctx else 48_000.0

    # ── main loop ──────────────────────────────────────────────────────────

    def _recv_loop(self):
        """recvfrom() only -- nothing else -- so this thread is always
        available to drain the socket, no matter how busy _run()'s parse/
        log/ring-buffer work is on any given packet."""
        # See socket_warmup.py: a just-bound socket does not actually
        # deliver inbound traffic for several seconds on this host, even
        # though bind() itself already returned -- absorb that cost here
        # before treating any loss as real.
        warm_ms, leaked = warm_up_socket(self._sock, self._port)
        print(f"[Receiver] Socket warm-up took {warm_ms:.0f} ms")
        if leaked is not None:
            self._put_raw(leaked)

        while not self._stop_evt.is_set():
            try:
                data, _ = self._sock.recvfrom(self.MAX_UDP_SIZE)
                self._put_raw(data)
            except socket.timeout:
                continue
            except OSError:
                break

    def _put_raw(self, data: bytes):
        try:
            self._raw_q.put_nowait(data)
        except queue.Full:
            # The processing side (see _handle()'s own note on why it
            # can't avoid the expensive full parse) has fallen behind --
            # drop this packet's bytes rather than let the backlog grow
            # without limit. Same "never propagate backpressure, but never
            # go silent about it either" shape as AsyncPacketLogger.
            self.raw_q_dropped += 1
            if self.raw_q_dropped == 1 or self.raw_q_dropped % 10_000 == 0:
                print(f"[Receiver] *** WARNING *** processing has fallen behind "
                      f"arrival rate -- {self.raw_q_dropped} raw packet(s) dropped "
                      f"so far (queue full at {self._raw_q.maxsize})")

    def _run(self):
        """Consumer side: parse + ring-buffer update + CSV log, decoupled
        from recvfrom() by _raw_q (see _recv_loop)."""
        _diag_proc = None
        _last_heartbeat = 0.0
        if self._DEBUG_REORDER:
            try:
                import psutil
                _diag_proc = psutil.Process()
            except Exception as _exc:
                print(f"[Receiver-DEBUG] psutil unavailable, memory tracking disabled: {_exc}")

        while not self._stop_evt.is_set():
            try:
                data = self._raw_q.get(timeout=0.2)
            except queue.Empty:
                data = None
            if self._DEBUG_REORDER:
                now = time.time()
                if now - _last_heartbeat >= 2.0:
                    _last_heartbeat = now
                    mem_str = ""
                    if _diag_proc is not None:
                        try:
                            mem_str = f" working_set={_diag_proc.memory_info().rss / 1e6:.1f}MB"
                        except Exception:
                            pass
                    print(f"[Receiver-DEBUG] {wall_clock_str()} raw_q_qsize={self._raw_q.qsize()} "
                          f"raw_q_dropped={self.raw_q_dropped} data_received={self.data_received} "
                          f"seq_errors={self.seq_errors}{mem_str}")
            if data is None:
                continue
            self._handle(data)
            # Drain whatever else already queued before checking stop_evt
            # again -- same "fully drain before yielding" shape used
            # elsewhere in this project (aggregator.py, ring_capture_main).
            while True:
                try:
                    self._handle(self._raw_q.get_nowait())
                except queue.Empty:
                    break

    def _handle(self, data: bytes):
        if len(data) < 8:
            return

        word1    = int.from_bytes(data[:4], "big")
        pkt_type = (word1 >> 28) & 0xF
        sid      = int.from_bytes(data[4:8], "big")   # Stream ID is always word 2

        try:
            if pkt_type == PACKET_TYPE_DATA:
                self.bytes_received += len(data)
                count = self._decode_counter.get(sid, 0)
                self._decode_counter[sid] = count + 1
                full_decode = (count % self.DECODE_EVERY_N) == 0

                if full_decode:
                    ctx = self._contexts.get(sid)
                    bit_depth = ctx.sample_bit_depth if ctx else 16
                    pkt = DifiDataPacket.from_bytes(data, sample_bit_depth=bit_depth)
                    seq_num, timestamp_int, timestamp_frac = pkt.seq_num, pkt.timestamp_int, pkt.timestamp_frac
                    samples = len(pkt.payload)
                    first_i, first_q = sample_fingerprint(pkt.payload)
                    self._update_stream_buffer(sid, pkt.payload)
                else:
                    _pt, _sid2, seq_num, timestamp_int, timestamp_frac = peek_header(data)
                    samples = max(0, (len(data) - PROLOGUE_WORDS * 4) // 4)
                    first_i, first_q = peek_first_iq(data)
                    self._last_update[sid] = time.monotonic()   # _update_stream_buffer's own bookkeeping, skipped here

                # Detect sequence-number gaps (DIFI seq wraps 0-15) --
                # works identically whichever path above supplied seq_num.
                last_seq = self._last_seqs.get(sid)
                seq_gap = last_seq is not None and seq_num != (last_seq + 1) & 0xF
                if seq_gap:
                    self.seq_errors += 1
                    # 2026-09-05: this used to print unconditionally, every
                    # single gap -- confirmed directly: once _raw_q starts
                    # dropping under real overload, EVERY surviving packet
                    # after a drop shows a gap, so this printed on nearly
                    # every packet, adding real per-packet overhead (stdout
                    # is a real file in the frozen EXE, not a no-op) right
                    # in the same hot path that was already falling behind
                    # -- a genuine negative feedback loop, not just log
                    # noise. Throttled the same way _put_raw's own overload
                    # warning already is: still counted exactly (seq_errors
                    # not affected), just not printed for every occurrence.
                    if self.seq_errors == 1 or self.seq_errors % 10_000 == 0:
                        print(
                            f"[Receiver] Seq gap stream 0x{sid:08X}: "
                            f"expected {(last_seq + 1) & 0xF}, got {seq_num} "
                            f"({self.seq_errors} total so far)"
                        )
                self._last_seqs[sid] = seq_num
                self.data_received += 1
                if self._packet_logger is not None:
                    self._packet_logger.log(
                        wall_clock_str(), f"0x{sid:08X}", "DATA", seq_num,
                        timestamp_int, timestamp_frac, samples, seq_gap,
                        first_i, first_q,
                        "", "", "",   # rf_ref_hz/sample_rate_hz/bandwidth_hz -- DATA packets don't carry these
                        "",           # active -- only STATUS rows (see receiver_app.py's _tick()) carry this
                    )

            elif pkt_type == PACKET_TYPE_CONTEXT:
                self.bytes_received += len(data)
                pkt = DifiContextPacket.from_bytes(data)
                with self._lock:
                    self._contexts[pkt.stream_id] = pkt
                    if pkt.stream_id not in self._iq_buffers:
                        self._iq_buffers[pkt.stream_id] = np.zeros(
                            self._buffer_size, dtype=np.complex64
                        )
                        self._write_idx[pkt.stream_id] = 0
                        print(f"[Receiver] New stream: 0x{pkt.stream_id:08X}")
                self.context_received += 1
                if self._packet_logger is not None:
                    self._packet_logger.log(
                        wall_clock_str(), f"0x{sid:08X}", "CONTEXT", pkt.seq_num,
                        pkt.timestamp_int, pkt.timestamp_frac, 0, False,
                        "", "",
                        pkt.rf_ref_freq_hz, pkt.sample_rate_hz, pkt.bandwidth_hz,
                        "",
                    )

        except Exception as exc:
            self.parse_errors += 1
            print(f"[Receiver] Parse error (sid=0x{sid:08X}): {exc}")

    def stream_last_seen(self) -> dict:
        """Return {stream_id: monotonic timestamp} of the last data packet per stream."""
        with self._lock:
            return dict(self._last_update)

    def _update_stream_buffer(self, sid: int, new_samples: np.ndarray):
        """Write new_samples into this stream's CIRCULAR buffer at O(n) cost.

        2026-09-05: this used to be np.roll(buf, -n) on the WHOLE buffer
        every single packet -- an O(buffer_size) copy (8192 elements,
        every call) to insert n=2200 new samples, on top of the already-
        expensive full DIFI parse this same hot path needs (see _handle()'s
        own note on why that parse can't be skipped here). Confirmed
        directly: under a real 2-stream, ~4500 pkt/s combined load, the
        processing thread fell far enough behind that its (now-bounded,
        see _raw_q) queue was dropping ~30-65% of arriving packets. A
        circular buffer with an explicit write index writes only the n new
        samples per call, with reconstruction into chronological order
        deferred to get_stream_snapshots() -- called at the GUI's own
        display-refresh rate (10-30Hz), not per packet."""
        n = len(new_samples)
        with self._lock:
            if sid not in self._iq_buffers:
                self._iq_buffers[sid] = np.zeros(self._buffer_size, dtype=np.complex64)
                self._write_idx[sid] = 0
            buf = self._iq_buffers[sid]
            if n >= self._buffer_size:
                buf[:] = new_samples[-self._buffer_size:]
                self._write_idx[sid] = 0
            else:
                idx = self._write_idx[sid]
                end = idx + n
                if end <= self._buffer_size:
                    buf[idx:end] = new_samples
                else:
                    first = self._buffer_size - idx
                    buf[idx:] = new_samples[:first]
                    buf[:end - self._buffer_size] = new_samples[first:]
                self._write_idx[sid] = end % self._buffer_size
            self._last_update[sid] = time.monotonic()

    def _chronological(self, sid: int, tail_samples: int | None = None) -> np.ndarray:
        """Reconstruct the last `tail_samples` (or the full buffer) in
        oldest-to-newest order from the circular storage. Caller must hold
        self._lock. Cost is proportional to what's actually requested, not
        the full buffer_size, matching get_stream_snapshots()'s own
        existing note on why that mattered for tail reads."""
        buf = self._iq_buffers[sid]
        size = len(buf)
        idx = self._write_idx.get(sid, 0)
        n = size if tail_samples is None else min(tail_samples, size)
        if n == size and idx == 0:
            return buf.copy()
        start = (idx - n) % size
        if start + n <= size:
            return buf[start:start + n].copy()
        first = size - start
        return np.concatenate([buf[start:], buf[:n - first]])


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import time

    receiver = DifiReceiver(port=50010)
    receiver.start()

    print("\n[Receiver] Waiting for multiplexed DIFI stream on port 50010 ...")
    print("           Run main.py or the Packetizer GUI to start the pipeline.")
    print("           Press Ctrl+C to stop.\n")

    try:
        while True:
            time.sleep(2.0)
            snaps = receiver.get_stream_snapshots()
            if snaps:
                for sid, (iq, ctx) in snaps.items():
                    fs_str = f"{ctx.sample_rate_hz/1e6:.3f} MHz" if ctx else "?"
                    rf_str = f"{ctx.rf_ref_freq_hz/1e6:.3f} MHz" if ctx else "?"
                    print(
                        f"  stream=0x{sid:08X}  samples={len(iq):,} "
                        f"fs={fs_str}  RF={rf_str}"
                    )
            else:
                print("  (no streams yet)")
    except KeyboardInterrupt:
        pass
    finally:
        receiver.stop()
