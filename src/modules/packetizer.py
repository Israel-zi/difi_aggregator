"""
packetizer.py
-------------
DIFI Packetizer module.

Receives AggregatedChunk objects from the Aggregator and re-packetizes each
constituent stream into a proper DIFI Data + Context packet pair, preserving
each stream's original Stream ID.

All per-stream packets are placed in a single output queue so the Sender
multiplexes them onto one UDP destination port.  The downstream Receiver
distinguishes streams by their original Stream IDs, as defined in
IEEE-ISTO Std 4900-2021 Figure 7:
  "Data Link With Multiple Simultaneous Information Streams with Distinct
   Stream IDs."
"""

import os
import sys
import queue
import time
import threading

if not getattr(sys, 'frozen', False):
    _src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _src not in sys.path:
        sys.path.insert(0, _src)

from core.difi_packet import (
    DifiDataPacket,
    DifiContextPacket,
    TSI_UTC,
    TSF_REAL_TIME,
)
from modules.aggregator import Aggregator, StreamBlock


CONTEXT_MIN_INTERVAL_S = 0.05   # max 20 context packets/s per DIFI standard §4.3.1


class Packetizer:
    """
    Re-packetizes each stream from an AggregatedChunk into individual DIFI
    Data + Context packet pairs, preserving the original Stream IDs.

    One (context_bytes_or_None, data_bytes) tuple is emitted per stream per
    chunk.  The Sender forwards all tuples to a single destination port,
    multiplexing the streams.

    Parameters
    ----------
    aggregator     : Aggregator instance to read chunks from
    out_queue_size : max depth of the output queue (should be >= expected
                     streams × burst depth)
    """

    def __init__(
        self,
        aggregator: Aggregator,
        out_queue_size: int = 32,
    ):
        self._aggregator      = aggregator
        self._out_queue       = queue.Queue(maxsize=out_queue_size)
        self._stop_evt        = threading.Event()
        self._thread          = threading.Thread(
            target=self._run, daemon=True, name="packetizer"
        )

        # Per-stream counters — keyed by stream_id
        self._data_seqs:      dict = {}   # stream_id -> int (0-15)
        self._ctx_seqs:       dict = {}   # stream_id -> int (0-15)
        self._last_ctx_times: dict = {}   # stream_id -> monotonic float

        # None = forward all streams; frozenset = whitelist (empty = forward none)
        self._forward_filter: frozenset | None = None

        self.packets_produced = 0
        self.packets_dropped  = 0

    # ── lifecycle ──────────────────────────────────────────────────────────

    def start(self):
        self._thread.start()
        print("[Packetizer] Started — forwarding original stream IDs")

    def stop(self):
        self._stop_evt.set()
        self._thread.join(timeout=3.0)
        print(f"[Packetizer] Stopped | packets produced: {self.packets_produced}")

    def get(self, timeout: float = 0.1):
        """Retrieve next (context_bytes_or_None, data_bytes). Returns None on timeout."""
        try:
            return self._out_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def set_forward_filter(self, sids):
        """Restrict which stream IDs are forwarded. None = all; frozenset() = none."""
        self._forward_filter = frozenset(sids) if sids is not None else None

    def flush_queue(self):
        """Drain the output queue and reset context timers so the next chunk
        sends fresh context packets for all streams to the receiver."""
        while True:
            try:
                self._out_queue.get_nowait()
            except queue.Empty:
                break
        self._last_ctx_times.clear()

    # ── per-stream sequence helpers ────────────────────────────────────────

    def _next_data_seq(self, sid: int) -> int:
        seq = self._data_seqs.get(sid, 0)
        self._data_seqs[sid] = (seq + 1) & 0xF
        return seq

    def _next_ctx_seq(self, sid: int) -> int:
        seq = self._ctx_seqs.get(sid, 0)
        self._ctx_seqs[sid] = (seq + 1) & 0xF
        return seq

    # ── packet builders ────────────────────────────────────────────────────

    def _build_context(self, block: StreamBlock) -> bytes:
        sid = block.stream_id
        src = block.context
        return DifiContextPacket(
            stream_id           = sid,
            seq_num             = self._next_ctx_seq(sid),
            timestamp_int       = block.data_ts_int,
            timestamp_frac      = block.data_ts_frac,
            sample_rate_hz      = src.sample_rate_hz,
            rf_ref_freq_hz      = src.rf_ref_freq_hz,
            if_ref_freq_hz      = src.if_ref_freq_hz,
            bandwidth_hz        = src.bandwidth_hz,
            reference_level_dbm = src.reference_level_dbm,
            sample_bit_depth    = src.sample_bit_depth,
            context_changed     = True,
            tsi                 = TSI_UTC,
            tsf                 = TSF_REAL_TIME,
        ).to_bytes()

    def _build_data(self, block: StreamBlock) -> bytes:
        sid = block.stream_id
        src = block.context
        return DifiDataPacket(
            stream_id        = sid,
            seq_num          = self._next_data_seq(sid),
            timestamp_int    = block.data_ts_int,
            timestamp_frac   = block.data_ts_frac,
            payload          = block.samples,
            sample_bit_depth = src.sample_bit_depth,
            tsi              = TSI_UTC,
            tsf              = TSF_REAL_TIME,
        ).to_bytes()

    # ── main loop ──────────────────────────────────────────────────────────

    def _run(self):
        while not self._stop_evt.is_set():
            chunk = self._aggregator.get(timeout=0.05)
            if chunk is None:
                continue

            now = time.monotonic()

            if self.packets_produced == 0:
                sids = [f"0x{s.stream_id:08X}" for s in chunk.streams]
                print(f"[Packetizer] First chunk — streams: {sids}")

            # Emit streams in ascending timestamp order so the receiver sees
            # packets in true chronological sequence across all streams.
            ordered = sorted(
                chunk.streams,
                key=lambda b: (b.data_ts_int, b.data_ts_frac),
            )

            for block in ordered:
                sid = block.stream_id

                if self._forward_filter is not None and sid not in self._forward_filter:
                    continue

                ctx_bytes = None
                if (now - self._last_ctx_times.get(sid, 0.0)) >= CONTEXT_MIN_INTERVAL_S:
                    ctx_bytes = self._build_context(block)
                    self._last_ctx_times[sid] = now

                data_bytes = self._build_data(block)

                try:
                    self._out_queue.put_nowait((ctx_bytes, data_bytes))
                    self.packets_produced += 1
                except queue.Full:
                    self.packets_dropped += 1


# ─────────────────────────────────────────────
# Quick self-test (standalone)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import numpy as np
    from core.difi_packet import now_timestamp
    from modules.aggregator import AggregatedChunk

    print("=== Packetizer Self-Test (stream-ID preservation) ===\n")

    _ts_int, _ts_frac = now_timestamp()

    def _fake_ctx(sid, rf_hz):
        return DifiContextPacket(
            stream_id      = sid,
            seq_num        = 0,
            timestamp_int  = _ts_int,
            timestamp_frac = _ts_frac,
            sample_rate_hz = 10_000_000.0,
            rf_ref_freq_hz = rf_hz,
            bandwidth_hz   = 1_000_000.0,
        )

    def _make_chunk():
        s1 = np.exp(1j * 2 * np.pi * 1e6 * np.arange(1024) / 10e6).astype(np.complex64)
        s2 = np.exp(1j * 2 * np.pi * 2e6 * np.arange(1024) / 10e6).astype(np.complex64)
        return AggregatedChunk(streams=[
            StreamBlock(stream_id=0x00000001, samples=s1,
                        context=_fake_ctx(0x00000001, 1e6), received_at=time.monotonic(),
                        data_ts_int=_ts_int, data_ts_frac=_ts_frac),
            StreamBlock(stream_id=0x00000002, samples=s2,
                        context=_fake_ctx(0x00000002, 2e6), received_at=time.monotonic(),
                        data_ts_int=_ts_int, data_ts_frac=_ts_frac),
        ])

    class _FakeAgg:
        def __init__(self):
            self._n = 0
        def get(self, **_):
            if self._n < 3:
                self._n += 1
                return _make_chunk()
            return None

    pktzr = Packetizer(aggregator=_FakeAgg())
    pktzr.start()

    received = []
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and len(received) < 6:
        item = pktzr.get(timeout=0.2)
        if item:
            received.append(item)

    pktzr.stop()

    print(f"Received {len(received)} packet pair(s) (expected 6 = 3 chunks × 2 streams)")
    for i, (ctx_b, data_b) in enumerate(received):
        dpkt = DifiDataPacket.from_bytes(data_b)
        print(f"  [{i}] stream=0x{dpkt.stream_id:08X}  samples={len(dpkt.payload)}"
              f"  ctx={'yes' if ctx_b else 'no'}")
        if ctx_b:
            cpkt = DifiContextPacket.from_bytes(ctx_b)
            assert cpkt.stream_id == dpkt.stream_id, "stream ID mismatch!"
            print(f"       ctx stream=0x{cpkt.stream_id:08X}  rf={cpkt.rf_ref_freq_hz/1e6:.1f}MHz")

    assert len(received) == 6
    print("\n✅ Packetizer self-test passed — original stream IDs preserved!")
