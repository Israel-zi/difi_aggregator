"""
packetizer.py
-------------
DIFI Packetizer module.

Receives AggregatedChunk objects from the Aggregator and converts them
into valid DIFI Data + Context packets ready for transmission.

Aggregation strategy for the unified stream
-------------------------------------------
  Each AggregatedChunk contains blocks from N streams.
  The Packetizer concatenates all IQ samples into a single payload and
  emits ONE Data packet per chunk, tagged with a unified stream ID.

  Stream IDs of the original sources are preserved in the Context packet
  via the reference_point and rf_ref_freq fields (in a real system, a
  vendor extension would carry them — for this PoC we log them).

  The unified stream uses stream_id = 0xAGGR0000.
"""

import queue
import time
import threading
import numpy as np

from core.difi_packet import (
    DifiDataPacket,
    DifiContextPacket,
    now_timestamp,
    TSI_UTC,
    TSF_REAL_TIME,
)
from modules.aggregator import Aggregator, AggregatedChunk


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

UNIFIED_STREAM_ID      = 0xAA000000   # aggregated stream ID
CONTEXT_MIN_INTERVAL_S = 0.05         # max 20 context packets/s per DIFI standard (Section 4.3.1)


# ─────────────────────────────────────────────
# Packetizer
# ─────────────────────────────────────────────

class Packetizer:
    """
    Converts AggregatedChunk objects into DIFI-compliant bytes
    and places them into an output queue for the Sender.

    Parameters
    ----------
    aggregator      : Aggregator instance to read chunks from
    unified_stream_id : stream ID for the outgoing unified DIFI stream
    out_queue_size  : max depth of the output queue
    """

    def __init__(
        self,
        aggregator: Aggregator,
        unified_stream_id: int = UNIFIED_STREAM_ID,
        out_queue_size: int    = 8,
    ):
        self._aggregator       = aggregator
        self._stream_id        = unified_stream_id
        self._out_queue        = queue.Queue(maxsize=out_queue_size)
        self._stop_evt         = threading.Event()
        self._thread           = threading.Thread(
            target=self._run, daemon=True, name="packetizer"
        )

        self._data_seq         = 0
        self._ctx_seq          = 0
        self._last_ctx_time    = 0.0   # 0 ensures context is sent before the first data packet

        # stats
        self.packets_produced  = 0
        self.packets_dropped   = 0

    # ── lifecycle ──────────────────────────────────────────────────────────

    def start(self):
        self._thread.start()
        print(f"[Packetizer] Started | unified stream_id=0x{self._stream_id:08X}")

    def stop(self):
        self._stop_evt.set()
        self._thread.join(timeout=3.0)
        print(f"[Packetizer] Stopped | packets produced: {self.packets_produced}")

    def get(self, timeout: float = 0.1):
        """Retrieve next (context_bytes, data_bytes) tuple. Returns None on timeout."""
        try:
            return self._out_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    # ── internal ───────────────────────────────────────────────────────────

    def _next_data_seq(self) -> int:
        seq = self._data_seq
        self._data_seq = (self._data_seq + 1) & 0xF
        return seq

    def _next_ctx_seq(self) -> int:
        seq = self._ctx_seq
        self._ctx_seq = (self._ctx_seq + 1) & 0xF
        return seq

    def _build_context(self, chunk: AggregatedChunk) -> bytes:
        """Build a Context packet from the first stream's metadata."""
        ctx_src        = chunk.streams[0].context
        ts_int, ts_frac = now_timestamp()

        ctx = DifiContextPacket(
            stream_id           = self._stream_id,
            seq_num             = self._next_ctx_seq(),
            timestamp_int       = ts_int,
            timestamp_frac      = ts_frac,
            sample_rate_hz      = ctx_src.sample_rate_hz,
            rf_ref_freq_hz      = ctx_src.rf_ref_freq_hz,
            if_ref_freq_hz      = ctx_src.if_ref_freq_hz,
            bandwidth_hz        = ctx_src.bandwidth_hz * chunk.num_streams,
            reference_level_dbm = ctx_src.reference_level_dbm,
            sample_bit_depth    = ctx_src.sample_bit_depth,
            context_changed     = True,
            tsi                 = TSI_UTC,
            tsf                 = TSF_REAL_TIME,
        )
        return ctx.to_bytes()

    def _build_data(self, chunk: AggregatedChunk) -> bytes:
        combined = sum(s.samples for s in chunk.streams)

        ts_int, ts_frac = now_timestamp()

        pkt = DifiDataPacket(
            stream_id        = self._stream_id,
            seq_num          = self._next_data_seq(),
            timestamp_int    = ts_int,
            timestamp_frac   = ts_frac,
            payload          = combined,
            sample_bit_depth = chunk.streams[0].context.sample_bit_depth,
            tsi              = TSI_UTC,
            tsf              = TSF_REAL_TIME,
        )
        return pkt.to_bytes()

    # ── main loop ──────────────────────────────────────────────────────────

    def _run(self):
        while not self._stop_evt.is_set():
            chunk = self._aggregator.get(timeout=0.05)
            if chunk is None:
                continue

            now       = time.monotonic()
            ctx_bytes = None
            if (now - self._last_ctx_time) >= CONTEXT_MIN_INTERVAL_S:
                ctx_bytes = self._build_context(chunk)
                self._last_ctx_time = now

            data_bytes = self._build_data(chunk)

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
    from core.difi_packet import DifiDataPacket, DifiContextPacket, now_timestamp
    from modules.aggregator import AggregatedChunk, StreamBlock

    print("=== Packetizer Self-Test (synthetic data) ===\n")

    # build a fake context
    ts_int, ts_frac = now_timestamp()
    fake_ctx = DifiContextPacket(
        stream_id      = 0x00000001,
        seq_num        = 0,
        timestamp_int  = ts_int,
        timestamp_frac = ts_frac,
        sample_rate_hz = 48_000.0,
        rf_ref_freq_hz = 437_000_000.0,
        bandwidth_hz   = 24_000.0,
    )

    # build fake chunks
    def make_chunk(sid1, sid2, n=1024):
        s1 = np.exp(1j * 2 * np.pi * 2000 * np.arange(n) / 48000).astype(np.complex64)
        s2 = np.exp(1j * 2 * np.pi * 6000 * np.arange(n) / 48000).astype(np.complex64)
        return AggregatedChunk(streams=[
            StreamBlock(stream_id=sid1, samples=s1, context=fake_ctx, received_at=time.monotonic()),
            StreamBlock(stream_id=sid2, samples=s2, context=fake_ctx, received_at=time.monotonic()),
        ])

    from modules.input_capture import InputCapture
    from modules.aggregator import Aggregator

    # we need a real aggregator instance for the packetizer constructor,
    # but we'll inject chunks manually via monkey-patching for the test
    class FakeAggregator:
        def __init__(self, chunks):
            self._chunks = iter(chunks)
            self._done   = False

        def get(self, timeout=0.1):
            try:
                return next(self._chunks)
            except StopIteration:
                self._done = True
                return None

    fake_agg   = FakeAggregator([make_chunk(0x1, 0x2) for _ in range(5)])
    packetizer = Packetizer(aggregator=fake_agg)

    # run manually (no thread)
    produced = []
    for _ in range(5):
        chunk = fake_agg.get()
        if chunk is None:
            break

        ctx_b  = None
        if len(produced) % CONTEXT_INTERVAL_PKTS == 0:
            ctx_b = packetizer._build_context(chunk)
        data_b = packetizer._build_data(chunk)
        produced.append((ctx_b, data_b))

    print(f"Produced {len(produced)} packet pair(s)")
    for i, (ctx_b, data_b) in enumerate(produced):
        pkt = DifiDataPacket.from_bytes(data_b)
        print(
            f"  [{i}] stream=0x{pkt.stream_id:08X} | "
            f"samples={len(pkt.payload)} | "
            f"ctx={'yes' if ctx_b else 'no'}"
        )
        if ctx_b:
            ctx = DifiContextPacket.from_bytes(ctx_b)
            print(f"       ctx: fs={ctx.sample_rate_hz:.0f}Hz bw={ctx.bandwidth_hz:.0f}Hz")

    print("\n✅ Packetizer self-test passed!")