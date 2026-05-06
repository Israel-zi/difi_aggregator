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
UDP_MAX_PAYLOAD        = 65507        # bytes: IPv4 max UDP payload (65535 − 20 IP − 8 UDP)
DIFI_DATA_HEADER_B     = 28           # bytes: 7 DIFI prologue words × 4 bytes each


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
            upsample = 1

        if upsample > 1:
            n_new_candidate  = 1 << int(np.ceil(np.log2(n_orig * upsample)))
            # Reject if the combined DIFI data packet would exceed the UDP MTU.
            # bit_depth bits per I and Q → 2 × ceil(bit_depth/8) bytes per complex sample.
            bit_depth        = chunk.streams[0].context.sample_bit_depth
            bytes_per_sample = 2 * max(1, (bit_depth + 7) // 8)
            if n_new_candidate * bytes_per_sample + DIFI_DATA_HEADER_B <= UDP_MAX_PAYLOAD:
                n_new  = n_new_candidate
                new_fs = orig_fs * n_new / n_orig
            else:
                upsample = 1
                n_new    = n_orig
                new_fs   = orig_fs
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

        half       = n_orig // 2
        neg_len    = n_orig - half - 1
        x_combined = np.zeros(n_new, dtype=np.complex128)

        for s in chunk.streams:
            x_fft = np.fft.fft(s.samples)
            x_up  = np.zeros(n_new, dtype=np.complex128)
            x_up[:half + 1] = x_fft[:half + 1]          # DC + positive freqs
            if neg_len:
                x_up[n_new - neg_len:] = x_fft[half + 1:]   # negative freqs

            delta_k = round((s.context.rf_ref_freq_hz - new_center) / new_fs * n_new)
            if delta_k != 0:
                x_up = np.roll(x_up, delta_k)

            x_combined += x_up

        # single IFFT; scale so amplitude matches orig (zero-padding alone scales by n_new/n_orig)
        return (np.fft.ifft(x_combined) * (n_new / n_orig)).astype(np.complex64)

    # ── packet builders ────────────────────────────────────────────────────

    def _build_context(self, chunk: AggregatedChunk, params: tuple) -> bytes:
        new_center, new_fs, total_bw, _, _n = params
        ctx_src = chunk.streams[0].context
        # Use the source stream's data timestamp so the context packet is
        # anchored to sample capture time, not the combiner's wall-clock.
        ts_int  = chunk.streams[0].data_ts_int
        ts_frac = chunk.streams[0].data_ts_frac

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
        combined = self._freq_stitch(chunk, new_center, new_fs, upsample, n_new)
        # Preserve the source timestamp: the combined packet's sample epoch
        # is the same as the input streams' sample epoch.
        ts_int  = chunk.streams[0].data_ts_int
        ts_frac = chunk.streams[0].data_ts_frac

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
            new_center, new_fs, _, upsample, n_new = params
            if self.packets_produced == 0:
                if upsample > 1:
                    print(
                        f"[Packetizer] Freq-stitch: upsample x{upsample} "
                        f"n_new={n_new} "
                        f"-> combined fs={new_fs/1e6:.1f} MHz "
                        f"center={new_center/1e6:.3f} MHz"
                    )
                else:
                    print(
                        f"[Packetizer] Same-LO fast path "
                        f"(or LO sep > {MAX_UPSAMPLE}x fs, capped) "
                        f"center={new_center/1e6:.3f} MHz"
                    )
                # Show where each stream's signal appears in the combined spectrum.
                # Combined rf_ref = midpoint; signals stay at their original LO positions.
                for s in chunk.streams:
                    lo  = s.context.rf_ref_freq_hz
                    off = (lo - new_center) / 1e6
                    active_flag = "(active)" if np.any(s.samples != 0) else "(OFF/silent)"
                    print(
                        f"[Packetizer]   stream=0x{s.stream_id:08X} "
                        f"LO={lo/1e6:.3f} MHz  offset={off:+.3f} MHz  "
                        f"-> signal at {lo/1e6:.3f} MHz in display {active_flag}"
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
    from core.difi_packet import DifiDataPacket, DifiContextPacket, now_timestamp
    from modules.aggregator import StreamBlock

    print("=== Packetizer Self-Test (synthetic data) ===\n")

    _ts_int, _ts_frac = now_timestamp()
    _fake_ctx = DifiContextPacket(
        stream_id      = 0x00000001,
        seq_num        = 0,
        timestamp_int  = _ts_int,
        timestamp_frac = _ts_frac,
        sample_rate_hz = 48_000.0,
        rf_ref_freq_hz = 437_000_000.0,
        bandwidth_hz   = 24_000.0,
    )

    def _make_chunk(sid1, sid2, n=1024):
        s1 = np.exp(1j * 2 * np.pi * 2000 * np.arange(n) / 48000).astype(np.complex64)
        s2 = np.exp(1j * 2 * np.pi * 6000 * np.arange(n) / 48000).astype(np.complex64)
        return AggregatedChunk(streams=[
            StreamBlock(stream_id=sid1, samples=s1, context=_fake_ctx, received_at=time.monotonic()),
            StreamBlock(stream_id=sid2, samples=s2, context=_fake_ctx, received_at=time.monotonic()),
        ])

    class _FakeAggregator:
        def __init__(self, chunks):
            self._chunks = iter(chunks)

        def get(self, **_):
            try:
                return next(self._chunks)
            except StopIteration:
                return None

    _fake_agg = _FakeAggregator([_make_chunk(0x1, 0x2) for _ in range(5)])
    _pktzr    = Packetizer(aggregator=_fake_agg)  # type: ignore[arg-type]
    _pktzr.start()

    _produced = []
    for _ in range(5):
        _item = _pktzr.get(timeout=1.0)
        if _item is None:
            break
        _produced.append(_item)

    _pktzr.stop()

    print(f"Produced {len(_produced)} packet pair(s)")
    for _i, (_ctx_b, _data_b) in enumerate(_produced):
        _pkt = DifiDataPacket.from_bytes(_data_b)
        print(
            f"  [{_i}] stream=0x{_pkt.stream_id:08X} | "
            f"samples={len(_pkt.payload)} | "
            f"ctx={'yes' if _ctx_b else 'no'}"
        )
        if _ctx_b:
            _ctx = DifiContextPacket.from_bytes(_ctx_b)
            print(f"       ctx: fs={_ctx.sample_rate_hz:.0f}Hz bw={_ctx.bandwidth_hz:.0f}Hz")

    print("\n✅ Packetizer self-test passed!")