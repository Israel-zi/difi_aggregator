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
        self._raw_q       = queue.Queue()
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
        self._iq_buffers:  dict = {}   # stream_id -> np.ndarray[complex64]
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
        self._raw_q    = queue.Queue()
        self._sock     = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 512 * 1024)
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
            if tail_samples is None:
                return {
                    sid: (buf.copy(), self._contexts.get(sid))
                    for sid, buf in self._iq_buffers.items()
                }
            return {
                sid: (buf[-tail_samples:].copy(), self._contexts.get(sid))
                for sid, buf in self._iq_buffers.items()
            }

    def get_iq_snapshot(self) -> np.ndarray:
        """
        Return combined IQ snapshot across all streams (concatenated, not interleaved).
        Preserves backwards compatibility with single-stream callers.
        """
        with self._lock:
            if not self._iq_buffers:
                return np.zeros(self._buffer_size, dtype=np.complex64)
            return np.concatenate(list(self._iq_buffers.values()))

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
            self._raw_q.put(leaked)

        while not self._stop_evt.is_set():
            try:
                data, _ = self._sock.recvfrom(self.MAX_UDP_SIZE)
                self._raw_q.put(data)
            except socket.timeout:
                continue
            except OSError:
                break

    def _run(self):
        """Consumer side: parse + ring-buffer update + CSV log, decoupled
        from recvfrom() by _raw_q (see _recv_loop)."""
        while not self._stop_evt.is_set():
            try:
                data = self._raw_q.get(timeout=0.2)
            except queue.Empty:
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
                ctx = self._contexts.get(sid)
                bit_depth = ctx.sample_bit_depth if ctx else 16
                pkt = DifiDataPacket.from_bytes(data, sample_bit_depth=bit_depth)
                # Detect sequence-number gaps (DIFI seq wraps 0-15)
                last_seq = self._last_seqs.get(sid)
                seq_gap = last_seq is not None and pkt.seq_num != (last_seq + 1) & 0xF
                if seq_gap:
                    self.seq_errors += 1
                    print(
                        f"[Receiver] Seq gap stream 0x{sid:08X}: "
                        f"expected {(last_seq + 1) & 0xF}, got {pkt.seq_num}"
                    )
                self._last_seqs[sid] = pkt.seq_num
                self._update_stream_buffer(pkt.stream_id, pkt.payload)
                self.data_received += 1
                if self._packet_logger is not None:
                    first_i, first_q = sample_fingerprint(pkt.payload)
                    self._packet_logger.log(
                        wall_clock_str(), f"0x{sid:08X}", "DATA", pkt.seq_num,
                        pkt.timestamp_int, pkt.timestamp_frac, len(pkt.payload), seq_gap,
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
        n = len(new_samples)
        with self._lock:
            if sid not in self._iq_buffers:
                self._iq_buffers[sid] = np.zeros(self._buffer_size, dtype=np.complex64)
            buf = self._iq_buffers[sid]
            if n >= self._buffer_size:
                buf[:] = new_samples[-self._buffer_size:]
            else:
                self._iq_buffers[sid] = np.roll(buf, -n)
                self._iq_buffers[sid][-n:] = new_samples
            self._last_update[sid] = time.monotonic()


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
