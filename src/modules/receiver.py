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

import socket
import threading
import numpy as np

from core.difi_packet import (
    DifiDataPacket,
    DifiContextPacket,
    PACKET_TYPE_DATA,
    PACKET_TYPE_CONTEXT,
)


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
    ):
        self._host        = host
        self._port        = port
        self._buffer_size = buffer_size
        self._sock        = None
        self._stop_evt    = threading.Event()
        self._thread      = threading.Thread(
            target=self._run, daemon=True, name="receiver"
        )

        # per-stream state — keyed by stream_id (int)
        self._iq_buffers: dict = {}   # stream_id -> np.ndarray[complex64]
        self._contexts:   dict = {}   # stream_id -> DifiContextPacket
        self._last_seqs:  dict = {}   # stream_id -> last seen seq_num (0-15)
        self._lock        = threading.Lock()

        # stats
        self.data_received    = 0
        self.context_received = 0
        self.parse_errors     = 0
        self.seq_errors       = 0

    # ── lifecycle ──────────────────────────────────────────────────────────

    def start(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.settimeout(1.0)
        self._sock.bind((self._host, self._port))
        self._thread.start()
        print(f"[Receiver] Listening on {self._host}:{self._port}")

    def stop(self):
        self._stop_evt.set()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        self._thread.join(timeout=2.0)
        print(
            f"[Receiver] Stopped | "
            f"data={self.data_received} ctx={self.context_received} "
            f"streams={list(f'0x{s:08X}' for s in self._contexts)}"
        )

    # ── data access ────────────────────────────────────────────────────────

    def get_stream_snapshots(self) -> dict:
        """
        Return a dict of {stream_id: (iq_array, context)} for all active streams.
        Both arrays are copies (thread-safe).  Streams without a context packet
        yet are included with context=None.
        """
        with self._lock:
            return {
                sid: (buf.copy(), self._contexts.get(sid))
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

    def _run(self):
        while not self._stop_evt.is_set():
            try:
                data, _ = self._sock.recvfrom(self.MAX_UDP_SIZE)
                self._handle(data)
            except socket.timeout:
                continue
            except OSError:
                break

    def _handle(self, data: bytes):
        if len(data) < 8:
            return

        word1    = int.from_bytes(data[:4], "big")
        pkt_type = (word1 >> 28) & 0xF
        sid      = int.from_bytes(data[4:8], "big")   # Stream ID is always word 2

        try:
            if pkt_type == PACKET_TYPE_DATA:
                ctx = self._contexts.get(sid)
                bit_depth = ctx.sample_bit_depth if ctx else 16
                pkt = DifiDataPacket.from_bytes(data, sample_bit_depth=bit_depth)
                # Detect sequence-number gaps (DIFI seq wraps 0-15)
                last_seq = self._last_seqs.get(sid)
                if last_seq is not None and pkt.seq_num != (last_seq + 1) & 0xF:
                    self.seq_errors += 1
                    print(
                        f"[Receiver] Seq gap stream 0x{sid:08X}: "
                        f"expected {(last_seq + 1) & 0xF}, got {pkt.seq_num}"
                    )
                self._last_seqs[sid] = pkt.seq_num
                self._update_stream_buffer(pkt.stream_id, pkt.payload)
                self.data_received += 1

            elif pkt_type == PACKET_TYPE_CONTEXT:
                pkt = DifiContextPacket.from_bytes(data)
                with self._lock:
                    self._contexts[pkt.stream_id] = pkt
                    if pkt.stream_id not in self._iq_buffers:
                        self._iq_buffers[pkt.stream_id] = np.zeros(
                            self._buffer_size, dtype=np.complex64
                        )
                        print(f"[Receiver] New stream: 0x{pkt.stream_id:08X}")
                self.context_received += 1

        except Exception as exc:
            self.parse_errors += 1
            print(f"[Receiver] Parse error (sid=0x{sid:08X}): {exc}")

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
