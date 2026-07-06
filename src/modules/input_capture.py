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
)


# ─────────────────────────────────────────────
# Tagged packet container
# ─────────────────────────────────────────────

@dataclass
class CapturedPacket:
    """A received DIFI packet tagged with its source port."""
    source_port:    int
    received_at:    float                           # time.monotonic() timestamp
    packet:         DifiDataPacket | DifiContextPacket


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
    ):
        super().__init__(daemon=True, name=f"listener-{port}")
        self.port      = port
        self.out_queue = out_queue
        self.host      = host
        self.timeout   = timeout
        self._stop_evt = threading.Event()

        # Bind synchronously in the caller's thread so a failure (port already
        # in use by another process, permission denied, etc.) raises immediately
        # here instead of silently killing a background thread with no one
        # ever finding out why no packets are arriving.
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
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

        # peek at bits 31-28 of Word 1 to detect packet type
        word1    = int.from_bytes(data[:4], "big")
        pkt_type = (word1 >> 28) & 0xF

        try:
            if pkt_type == PACKET_TYPE_DATA:
                pkt = DifiDataPacket.from_bytes(data)
                self.stats["data_received"] += 1

            elif pkt_type == PACKET_TYPE_CONTEXT:
                pkt = DifiContextPacket.from_bytes(data)
                self.stats["context_received"] += 1

            else:
                # unknown packet type — skip silently
                return

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
    ):
        self._out_queue = queue.Queue(maxsize=queue_maxsize)
        self._listeners = []
        # Ports that failed to bind (e.g. already in use by another program) —
        # collected instead of raised so the other ports still start.
        self.bind_errors: dict = {}
        for p in ports:
            try:
                self._listeners.append(
                    PortListener(port=p, out_queue=self._out_queue, host=host)
                )
            except OSError as exc:
                self.bind_errors[p] = str(exc)
                print(f"[Capture] Failed to bind port {p}: {exc}")

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
        print("[Capture] All listeners stopped")

    def add_port(self, port: int, host: str = "0.0.0.0"):
        """Start a new listener on the given port while already running.

        Raises OSError (unchanged) if the port can't be bound — callers
        should catch this and surface it rather than let it vanish.
        """
        listener = PortListener(port=port, out_queue=self._out_queue, host=host)
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
        combined = {"data_received": 0, "context_received": 0, "parse_errors": 0}
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

        # Small queue — keeps pipeline latency low.  At ~47 pps per stream a
        # depth of 128 gives ~2.7 s of burst tolerance without building a backlog.
        self._out_queue = queue.Queue(maxsize=128)
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
        print(f"[JitterBuffer] Stopped | gaps detected: {self.gaps_detected}")

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
        while not self._stop_evt.is_set():
            try:
                captured = self._capture.get(timeout=0.02)
                if captured is not None:
                    if not self._enabled:
                        try:
                            self._out_queue.put_nowait(captured)
                        except queue.Full:
                            pass
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