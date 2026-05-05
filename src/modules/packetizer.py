"""
packetizer.py
-------------
DIFI Packetizer module.

Receives AggregatedChunk objects from the Aggregator and converts them
into valid DIFI Data + Context packets ready for transmission.

Frequency-stitching combiner
-----------------------------
  Each input stream carries IQ data sampled relative to its own LO
  (rf_ref_freq_hz).  Simply adding the raw samples is wrong when the LOs
  differ — you would be adding signals referenced to different carrier
  frequencies.

  Instead the Packetizer performs proper frequency stitching:

    1. Compute the combined wideband parameters from all stream contexts:
         new_center = midpoint of all (LO ± fs/2) edges
         new_fs     = orig_fs × ceil(total_span / orig_fs)   [integer multiple]

    2. For each stream, upsample to new_fs via FFT zero-padding and then
       frequency-shift by (stream_LO − new_center) so every stream is
       expressed relative to the same carrier.

    3. Sum the shifted streams.  The result is a single wideband IQ signal
       at new_center / new_fs that contains all input signals at their
       correct spectral positions.

  When all streams share the same LO the shift is zero and new_fs == orig_fs,
  so the fast path reduces to a plain element-wise sum — no extra cost.
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
MAX_UPSAMPLE           = 32           # cap for freq-stitching upsample factor.
                                      # Beyond this the LO separation exceeds ~16× fs, and
                                      # creating a dense time-domain wideband signal is
                                      # impractical.  Falls back to plain sum (same-LO path).


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

    # ── wideband combining ─────────────────────────────────────────────────

    @staticmethod
    def _compute_wideband_params(chunk: AggregatedChunk) -> tuple:
        """
        Derive the combined stream's center frequency, sample rate, total
        bandwidth, upsample factor, and output FFT length from the incoming
        stream contexts.

        Each stream occupies  LO ± orig_fs/2  (its full Nyquist band).
        n_new is rounded up to the next power of 2 so that IFFT is fast.
        Returns (new_center_hz, new_fs_hz, total_bw_hz, upsample_factor, n_new).
        """
        orig_fs  = chunk.streams[0].context.sample_rate_hz
        n_orig   = len(chunk.streams[0].samples)

        # Only active (non-zero) streams define the combined center frequency.
        # A SIGNAL_OFF stream sends exactly-zero samples; including its LO would
        # pull the combined center toward an irrelevant midpoint.
        active = [s for s in chunk.streams if np.any(s.samples != 0)]
        if not active:
            active = chunk.streams   # all streams OFF — fallback to prevent empty list

        edges    = []
        for s in active:
            lo = s.context.rf_ref_freq_hz
            edges.append(lo - orig_fs / 2.0)
            edges.append(lo + orig_fs / 2.0)

        min_edge   = min(edges)
        max_edge   = max(edges)
        total_bw   = max_edge - min_edge
        new_center = (min_edge + max_edge) / 2.0
        upsample   = max(1, int(np.ceil(total_bw / orig_fs)))

        if upsample > MAX_UPSAMPLE:
            # LO separation is too large to bridge at the current sample rate.
            # Fall back to the same-LO fast path: sum samples as-is, use average LO.
            upsample = 1

        if upsample > 1:
            # round up to next power of 2 for fast FFT/IFFT
            n_new  = 1 << int(np.ceil(np.log2(n_orig * upsample)))
            new_fs = orig_fs * n_new / n_orig
        else:
            n_new  = n_orig
            new_fs = orig_fs

        return new_center, new_fs, total_bw, upsample, n_new

    @staticmethod
    def _freq_stitch(chunk: AggregatedChunk,
                     new_center: float,
                     new_fs: float,
                     upsample: int,
                     n_new: int) -> np.ndarray:
        """
        Combine all streams into one wideband IQ signal.

        Fast path (upsample == 1, same LO): plain element-wise sum, no FFT.

        Stitching path: all work stays in the frequency domain —
          1. FFT each stream's n_orig samples.
          2. Zero-pad to n_new bins (power-of-2 for fast IFFT).
          3. Circular-shift the spectrum by delta_k bins to place the stream's
             carrier at the correct position relative to new_center.
          4. Accumulate into X_combined.
          5. Single IFFT at the end → one fast transform total.
        """
        n_orig = len(chunk.streams[0].samples)

        if upsample == 1:
            combined = np.zeros(n_orig, dtype=np.complex128)
            for s in chunk.streams:
                combined += s.samples.astype(np.complex128)
            return combined.astype(np.complex64)

        half    = n_orig // 2
        neg_len = n_orig - half - 1
        X_combined = np.zeros(n_new, dtype=np.complex128)

        for s in chunk.streams:
            X    = np.fft.fft(s.samples)
            X_up = np.zeros(n_new, dtype=np.complex128)
            X_up[:half + 1] = X[:half + 1]          # DC + positive freqs
            if neg_len:
                X_up[n_new - neg_len:] = X[half + 1:]   # negative freqs

            delta_k = round((s.context.rf_ref_freq_hz - new_center) / new_fs * n_new)
            if delta_k != 0:
                X_up = np.roll(X_up, delta_k)

            X_combined += X_up

        # single IFFT; scale so amplitude matches orig (zero-padding alone scales by n_new/n_orig)
        return (np.fft.ifft(X_combined) * (n_new / n_orig)).astype(np.complex64)

    # ── packet builders ────────────────────────────────────────────────────

    def _build_context(self, chunk: AggregatedChunk, params: tuple) -> bytes:
        new_center, new_fs, total_bw, _, _n = params
        ctx_src         = chunk.streams[0].context
        ts_int, ts_frac = now_timestamp()

        ctx = DifiContextPacket(
            stream_id           = self._stream_id,
            seq_num             = self._next_ctx_seq(),
            timestamp_int       = ts_int,
            timestamp_frac      = ts_frac,
            sample_rate_hz      = new_fs,
            rf_ref_freq_hz      = new_center,
            if_ref_freq_hz      = ctx_src.if_ref_freq_hz,
            bandwidth_hz        = total_bw,
            reference_level_dbm = ctx_src.reference_level_dbm,
            sample_bit_depth    = ctx_src.sample_bit_depth,
            context_changed     = True,
            tsi                 = TSI_UTC,
            tsf                 = TSF_REAL_TIME,
        )
        return ctx.to_bytes()

    def _build_data(self, chunk: AggregatedChunk, params: tuple) -> bytes:
        new_center, new_fs, _, upsample, n_new = params
        combined        = self._freq_stitch(chunk, new_center, new_fs, upsample, n_new)
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

            # Compute wideband params once; reuse for both context and data.
            params = self._compute_wideband_params(chunk)
            _, new_fs, _, upsample, n_new = params
            if self.packets_produced == 0:
                if upsample > 1:
                    print(
                        f"[Packetizer] Freq-stitch: upsample x{upsample} "
                        f"n_new={n_new} "
                        f"-> combined fs={new_fs/1e6:.1f} MHz "
                        f"center={params[0]/1e6:.3f} MHz"
                    )
                else:
                    print(
                        f"[Packetizer] Same-LO fast path "
                        f"(or LO sep > {MAX_UPSAMPLE}x fs, capped) "
                        f"center={params[0]/1e6:.3f} MHz"
                    )

            now       = time.monotonic()
            ctx_bytes = None
            if (now - self._last_ctx_time) >= CONTEXT_MIN_INTERVAL_S:
                ctx_bytes = self._build_context(chunk, params)
                self._last_ctx_time = now

            data_bytes = self._build_data(chunk, params)

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