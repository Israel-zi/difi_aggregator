"""
generator.py
------------
DIFI Modem Generator + Sender.

Signal types:
  CW  — pure complex sinusoid (single tone, for testing)
  BW  — bandpass-filtered AWGN (simulates a real waveform occupying BW Hz)
"""

import socket
import time
import numpy as np
from scipy import signal as scipy_signal

from core.difi_packet import (
    DifiDataPacket,
    DifiContextPacket,
    now_timestamp,
    TSI_UTC,
    TSF_REAL_TIME,
)

CONTEXT_MIN_INTERVAL_S = 0.05   # max 20 context packets/s per DIFI standard (Section 4.3.1)
SIGNAL_CW  = "CW"
SIGNAL_BW  = "BW"
SIGNAL_OFF = "OFF"


class DifiGenerator:
    """
    Generates and transmits a DIFI stream over UDP.

    Parameters
    ----------
    stream_id       : unique 32-bit stream identifier
    tone_hz         : center/carrier frequency (Hz) — used for both CW and BW
    signal_type     : "CW" or "BW"
    dest_host       : destination IP
    dest_port       : destination UDP port
    sample_rate_hz  : IQ sample rate (Hz)
    samples_per_pkt : IQ samples per Data packet
    bit_depth       : sample bit depth (4-16)
    rf_ref_freq_hz  : RF reference frequency for Context packet
    bandwidth_hz    : signal bandwidth (used in BW mode and Context packet)
    ref_level_dbm   : reference level (dBm) — scales signal amplitude
    """

    def __init__(
        self,
        stream_id: int,
        tone_hz: float,
        signal_type: str        = SIGNAL_CW,
        dest_host: str          = "127.0.0.1",
        dest_port: int          = 50001,
        sample_rate_hz: float   = 10_000_000,
        samples_per_pkt: int    = 1024,
        bit_depth: int          = 16,
        rf_ref_freq_hz: float   = 1_000_000_000,
        bandwidth_hz: float     = 1_000_000,
        ref_level_dbm: float    = -20.0,
    ):
        self.stream_id       = stream_id
        self.tone_hz         = tone_hz
        self.signal_type     = signal_type
        self.dest            = (dest_host, dest_port)
        self.sample_rate_hz  = sample_rate_hz
        self.samples_per_pkt = samples_per_pkt
        self.bit_depth       = bit_depth
        self.rf_ref_freq_hz  = rf_ref_freq_hz
        self.bandwidth_hz    = bandwidth_hz
        self.ref_level_dbm   = ref_level_dbm

        self._sock          = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._data_seq      = 0
        self._ctx_seq       = 0
        self._pkt_count     = 0
        self._phase         = 0.0
        self._running       = True
        self._bw_filter     = None
        self._bw_zi         = None
        self._last_ctx_time = 0.0   # 0 ensures context is sent before the first data packet

        self._build_bw_filter()

    # ── signal generation ──────────────────────────────────────────────────

    def _build_bw_filter(self):
        """Build a bandpass FIR filter for BW mode."""
        fs    = self.sample_rate_hz
        nyq   = fs / 2.0
        bw    = self.bandwidth_hz
        fc    = self.tone_hz

        low  = max((fc - bw / 2) / nyq, 1e-6)
        high = min((fc + bw / 2) / nyq, 1 - 1e-6)

        if low >= high or low <= 0 or high >= 1:
            # fallback: lowpass at BW/2, clamped so firwin doesn't raise
            self._bw_filter = scipy_signal.firwin(101, min(bw / nyq, 0.999))
        else:
            self._bw_filter = scipy_signal.firwin(
                101, [low, high], pass_zero=False
            )

        # init filter state
        zi = scipy_signal.lfilter_zi(self._bw_filter, [1.0])
        self._bw_zi_i = zi * 0
        self._bw_zi_q = zi * 0

    def _generate_cw(self) -> np.ndarray:
        """Generate one packet of pure CW (complex sinusoid)."""
        n   = self.samples_per_pkt
        t   = np.arange(n) / self.sample_rate_hz
        amp = 10 ** (self.ref_level_dbm / 20.0)   # dBm → linear
        sig = amp * np.exp(
            1j * (2.0 * np.pi * self.tone_hz * t + self._phase)
        ).astype(np.complex64)

        self._phase = (
            self._phase + 2.0 * np.pi * self.tone_hz * n / self.sample_rate_hz
        ) % (2.0 * np.pi)
        return sig

    def _generate_bw(self) -> np.ndarray:
        """Generate one packet of bandpass-filtered AWGN centered at tone_hz."""
        n   = self.samples_per_pkt
        amp = 10 ** (self.ref_level_dbm / 20.0)

        # white noise
        noise_i = np.random.randn(n).astype(np.float32)
        noise_q = np.random.randn(n).astype(np.float32)

        # apply bandpass filter with state continuity
        filt_i, self._bw_zi_i = scipy_signal.lfilter(
            self._bw_filter, [1.0], noise_i, zi=self._bw_zi_i
        )
        filt_q, self._bw_zi_q = scipy_signal.lfilter(
            self._bw_filter, [1.0], noise_q, zi=self._bw_zi_q
        )

        # shift to center frequency
        t   = np.arange(n) / self.sample_rate_hz
        mix = np.exp(1j * (2.0 * np.pi * self.tone_hz * t + self._phase))
        self._phase = (
            self._phase + 2.0 * np.pi * self.tone_hz * n / self.sample_rate_hz
        ) % (2.0 * np.pi)

        sig = amp * (filt_i + 1j * filt_q) * mix
        return sig.astype(np.complex64)

    def _generate_samples(self) -> np.ndarray:
        if self.signal_type == SIGNAL_OFF:
            return np.zeros(self.samples_per_pkt, dtype=np.complex64)
        if self.signal_type == SIGNAL_CW:
            return self._generate_cw()
        return self._generate_bw()

    # ── packet building ────────────────────────────────────────────────────

    def _next_seq(self, counter_name: str) -> int:
        val = getattr(self, counter_name)
        setattr(self, counter_name, (val + 1) & 0xF)
        return val

    def _make_context(self) -> bytes:
        ts_int, ts_frac = now_timestamp()
        return DifiContextPacket(
            stream_id           = self.stream_id,
            seq_num             = self._next_seq("_ctx_seq"),
            timestamp_int       = ts_int,
            timestamp_frac      = ts_frac,
            sample_rate_hz      = self.sample_rate_hz,
            rf_ref_freq_hz      = self.rf_ref_freq_hz,
            bandwidth_hz        = self.bandwidth_hz,
            reference_level_dbm = self.ref_level_dbm,
            sample_bit_depth    = self.bit_depth,
            tsi                 = TSI_UTC,
            tsf                 = TSF_REAL_TIME,
        ).to_bytes()

    def _make_data(self, samples: np.ndarray) -> bytes:
        ts_int, ts_frac = now_timestamp()
        return DifiDataPacket(
            stream_id        = self.stream_id,
            seq_num          = self._next_seq("_data_seq"),
            timestamp_int    = ts_int,
            timestamp_frac   = ts_frac,
            payload          = samples,
            sample_bit_depth    = self.bit_depth,
            tsi              = TSI_UTC,
            tsf              = TSF_REAL_TIME,
        ).to_bytes()

    # ── public API ─────────────────────────────────────────────────────────

    def update_params(
        self,
        tone_hz: float       = None,
        signal_type: str     = None,
        bandwidth_hz: float  = None,
        rf_ref_freq_hz: float = None,
        ref_level_dbm: float = None,
        sample_rate_hz: float = None,
    ):
        """Update generator parameters at runtime (thread-safe for simple types)."""
        rebuild_filter = False
        if tone_hz is not None:
            self.tone_hz = tone_hz
            rebuild_filter = True
        if signal_type is not None:
            self.signal_type = signal_type
        if bandwidth_hz is not None:
            self.bandwidth_hz = bandwidth_hz
            rebuild_filter = True
        if rf_ref_freq_hz is not None:
            self.rf_ref_freq_hz = rf_ref_freq_hz
        if ref_level_dbm is not None:
            self.ref_level_dbm = ref_level_dbm
        if sample_rate_hz is not None:
            self.sample_rate_hz = sample_rate_hz
            rebuild_filter = True
        if rebuild_filter:
            self._build_bw_filter()

    def send_one_packet(self):
        now = time.monotonic()
        if (now - self._last_ctx_time) >= CONTEXT_MIN_INTERVAL_S:
            self._sock.sendto(self._make_context(), self.dest)
            self._last_ctx_time = now

        samples = self._generate_samples()
        self._sock.sendto(self._make_data(samples), self.dest)
        self._pkt_count += 1

    def run(self, num_packets: int = 0, packet_rate_hz: float = 0.0):
        interval = 1.0 / packet_rate_hz if packet_rate_hz > 0 else 0.0
        count    = 0
        print(
            f"[Generator] stream=0x{self.stream_id:08X} | "
            f"type={self.signal_type} | tone={self.tone_hz:.0f}Hz | "
            f"fs={self.sample_rate_hz:.0f}Hz | dest={self.dest}"
        )
        try:
            while self._running and (num_packets == 0 or count < num_packets):
                t0 = time.monotonic()
                self.send_one_packet()
                count += 1
                if interval > 0:
                    sleep = interval - (time.monotonic() - t0)
                    if sleep > 0:
                        time.sleep(sleep)
        except (KeyboardInterrupt, OSError):
            pass

    def close(self):
        self._running = False
        try:
            self._sock.close()
        except OSError:
            pass