"""
difi_packet.py
--------------
VITA-49.2 / DIFI packet builder and parser.

Supports:
  - Data Packet Class 0x0000    (Standard Flow Signal Data)
  - Context Packet Class 0x0001 (Standard Flow Signal Context)

Reference: IEEE-ISTO Std 4900-2021 v1.3.0
"""

import struct
import time
import numpy as np


# ─────────────────────────────────────────────
# DIFI Constants (from the standard)
# ─────────────────────────────────────────────

DIFI_OUI = 0x6A621E          # DIFI Organizationally Unique Identifier

# Packet Types (bits 31-28 of Word 1)
PACKET_TYPE_DATA    = 0x1    # Signal Data packet with stream ID
PACKET_TYPE_CONTEXT = 0x4    # Context packet with stream ID

# Packet Classes
PACKET_CLASS_DATA    = 0x0000   # Standard Flow Signal Data
PACKET_CLASS_CONTEXT = 0x0001   # Standard Flow Signal Context

# Information Class (Basic Data Plane)
INFO_CLASS_BASIC = 0x0000

# Timestamp Integer (TSI) - bits 23-22
TSI_UTC   = 0b01   # UTC time (default)
TSI_GPS   = 0b10
TSI_POSIX = 0b11

# Timestamp Fractional (TSF) - bits 21-20
TSF_SAMPLE_COUNT = 0b01   # Sample count
TSF_REAL_TIME    = 0b10   # Real time picoseconds (used in class 0x0000)

# Context Indicator Field values (Word 8 of Context Packet)
CIF0_CHANGE    = 0xFBB98000   # context changed
CIF0_NO_CHANGE = 0x7BB98000   # no change

PROLOGUE_WORDS = 7   # Words 1-7 (header + stream ID + class ID + timestamps)


# ─────────────────────────────────────────────
# Data Packet
# ─────────────────────────────────────────────

class DifiDataPacket:
    """
    DIFI Standard Flow Signal Data Packet (Class 0x0000).

    Wire format (32-bit words, big-endian):
      Word 1 : Packet Header
      Word 2 : Stream Identifier
      Word 3 : Class ID high  (Pad | 0 | OUI)
      Word 4 : Class ID low   (Info Class | Packet Class)
      Word 5 : Integer-seconds Timestamp
      Word 6 : Fractional-seconds Timestamp (high 32 bits)
      Word 7 : Fractional-seconds Timestamp (low  32 bits)
      Word 8+: IQ Signal Data Payload
    """

    def __init__(
        self,
        stream_id: int,
        seq_num: int,               # 0-15, modulo 16
        timestamp_int: int,         # integer seconds (UTC)
        timestamp_frac: int,        # picoseconds since last integer second
        payload: np.ndarray,        # complex64 IQ samples
        sample_bit_depth: int = 16, # bits per I or Q sample (4-16)
        info_class: int = INFO_CLASS_BASIC,
        tsi: int = TSI_UTC,
        tsf: int = TSF_REAL_TIME,
    ):
        self.stream_id        = stream_id
        self.seq_num          = seq_num & 0xF   # only 4 bits
        self.timestamp_int    = timestamp_int
        self.timestamp_frac   = timestamp_frac
        self.payload          = payload.astype(np.complex64)
        self.sample_bit_depth = sample_bit_depth
        self.info_class       = info_class
        self.tsi              = tsi
        self.tsf              = tsf

    # ── build ──────────────────────────────────────────────────────────────

    def to_bytes(self) -> bytes:
        """Serialise packet to bytes ready for UDP transmission."""
        payload_bytes = self._pack_iq_samples()
        payload_words = (len(payload_bytes) + 3) // 4
        packet_size   = PROLOGUE_WORDS + payload_words

        word1 = self._build_header(packet_size)
        word2 = self.stream_id & 0xFFFFFFFF
        word3 = DIFI_OUI & 0xFFFFFF
        word4 = ((self.info_class & 0xFFFF) << 16) | (PACKET_CLASS_DATA & 0xFFFF)
        word5 = self.timestamp_int & 0xFFFFFFFF
        word6 = (self.timestamp_frac >> 32) & 0xFFFFFFFF
        word7 = self.timestamp_frac & 0xFFFFFFFF

        header = struct.pack(">7I", word1, word2, word3, word4, word5, word6, word7)
        pad    = (4 - len(payload_bytes) % 4) % 4
        return header + payload_bytes + bytes(pad)

    def _build_header(self, packet_size: int) -> int:
        """Build Word 1 of the DIFI packet header."""
        return (
            (PACKET_TYPE_DATA & 0xF) << 28 |
            1                        << 27 |   # Class ID present
            (self.tsi  & 0x3)        << 22 |
            (self.tsf  & 0x3)        << 20 |
            (self.seq_num & 0xF)     << 16 |
            (packet_size & 0xFFFF)
        )

    def _pack_iq_samples(self) -> bytes:
        """Pack complex64 samples into interleaved signed int16 bytes."""
        scale            = (2 ** (self.sample_bit_depth - 1)) - 1
        i_samples        = np.clip(self.payload.real * scale, -scale, scale).astype(np.int16)
        q_samples        = np.clip(self.payload.imag * scale, -scale, scale).astype(np.int16)
        interleaved      = np.empty(len(i_samples) * 2, dtype=np.int16)
        interleaved[0::2] = i_samples
        interleaved[1::2] = q_samples
        return interleaved.tobytes()

    # ── parse ──────────────────────────────────────────────────────────────

    @classmethod
    def from_bytes(cls, data: bytes) -> "DifiDataPacket":
        """Parse a received UDP payload into a DifiDataPacket."""
        if len(data) < PROLOGUE_WORDS * 4:
            raise ValueError(f"Packet too short: {len(data)} bytes")

        word1, word2, word3, word4, word5, word6, word7 = struct.unpack_from(">7I", data)

        pkt_type = (word1 >> 28) & 0xF
        if pkt_type != PACKET_TYPE_DATA:
            raise ValueError(f"Expected Data packet (type 0x1), got 0x{pkt_type:X}")

        tsi            = (word1 >> 22) & 0x3
        tsf            = (word1 >> 20) & 0x3
        seq_num        = (word1 >> 16) & 0xF
        stream_id      = word2
        info_class     = (word4 >> 16) & 0xFFFF
        timestamp_int  = word5
        timestamp_frac = (word6 << 32) | word7
        payload        = cls._unpack_iq_samples(data[PROLOGUE_WORDS * 4:])

        return cls(
            stream_id      = stream_id,
            seq_num        = seq_num,
            timestamp_int  = timestamp_int,
            timestamp_frac = timestamp_frac,
            payload        = payload,
            info_class     = info_class,
            tsi            = tsi,
            tsf            = tsf,
        )

    @staticmethod
    def _unpack_iq_samples(payload_bytes: bytes) -> np.ndarray:
        """Unpack interleaved int16 IQ bytes into complex64 array."""
        n = len(payload_bytes) // 2
        if n == 0:
            return np.array([], dtype=np.complex64)
        raw       = np.frombuffer(payload_bytes[:n * 2], dtype=np.int16)
        scale     = 32767.0
        i_samples = raw[0::2].astype(np.float32) / scale
        q_samples = raw[1::2].astype(np.float32) / scale
        return (i_samples + 1j * q_samples).astype(np.complex64)

    def __repr__(self) -> str:
        return (
            f"DifiDataPacket(stream_id=0x{self.stream_id:08X}, "
            f"seq={self.seq_num}, "
            f"ts={self.timestamp_int}.{self.timestamp_frac}, "
            f"samples={len(self.payload)})"
        )


# ─────────────────────────────────────────────
# Context Packet
# ─────────────────────────────────────────────

class DifiContextPacket:
    """
    DIFI Standard Flow Signal Context Packet (Class 0x0001).
    Fixed size: 27 words (108 bytes).

    Contains metadata required to interpret the Data Packet stream.
    """

    PACKET_SIZE_WORDS = 27

    def __init__(
        self,
        stream_id: int,
        seq_num: int,
        timestamp_int: int,
        timestamp_frac: int,
        sample_rate_hz: float,
        rf_ref_freq_hz: float      = 0.0,
        if_ref_freq_hz: float      = 0.0,
        bandwidth_hz: float        = 0.0,
        reference_level_dbm: float = 0.0,
        sample_bit_depth: int      = 16,
        context_changed: bool      = True,
        info_class: int            = INFO_CLASS_BASIC,
        tsi: int                   = TSI_UTC,
        tsf: int                   = TSF_REAL_TIME,
        reference_point: int       = 100,   # 0x64 = IF converter input
    ):
        self.stream_id           = stream_id
        self.seq_num             = seq_num & 0xF
        self.timestamp_int       = timestamp_int
        self.timestamp_frac      = timestamp_frac
        self.sample_rate_hz      = sample_rate_hz
        self.rf_ref_freq_hz      = rf_ref_freq_hz
        self.if_ref_freq_hz      = if_ref_freq_hz
        self.bandwidth_hz        = bandwidth_hz
        self.reference_level_dbm = reference_level_dbm
        self.sample_bit_depth    = sample_bit_depth
        self.context_changed     = context_changed
        self.info_class          = info_class
        self.tsi                 = tsi
        self.tsf                 = tsf
        self.reference_point     = reference_point

    # ── helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _encode_freq(hz: float) -> tuple:
        """Encode frequency in VITA-49.2 64-bit fixed-point format."""
        value64 = int(round(hz)) << 20
        return (value64 >> 32) & 0xFFFFFFFF, value64 & 0xFFFFFFFF

    @staticmethod
    def _decode_freq(high: int, low: int) -> float:
        """Decode VITA-49.2 frequency field back to Hz."""
        return float(((high << 32) | low) >> 20)

    @staticmethod
    def _encode_ref_level(dbm: float) -> int:
        """Encode Reference Level: 16-bit fixed-point (9 integer + 7 fractional bits)."""
        return int(round(dbm * 128)) & 0xFFFF

    def _build_data_payload_format(self) -> tuple:
        """Build the Data Packet Payload Format field (Words 26-27) per Figure 11."""
        bd = self.sample_bit_depth - 1
        word26 = (
            1      << 31 |   # Packing Method = Link Efficient
            0b01   << 29 |   # Real-Complex = Complex Cartesian
            bd     <<  6 |   # Item Packing Field Size
            bd     <<  0     # Data Item Size
        )
        return word26, 0

    # ── build ──────────────────────────────────────────────────────────────

    def to_bytes(self) -> bytes:
        """Serialise context packet to 108 bytes (27 words)."""
        word1  = (
            (PACKET_TYPE_CONTEXT & 0xF) << 28 |
            1                           << 27 |   # Class ID present
            1                           << 24 |   # TSM = coarse (Info Class 0x0000)
            (self.tsi  & 0x3)           << 22 |
            (self.tsf  & 0x3)           << 20 |
            (self.seq_num & 0xF)        << 16 |
            (self.PACKET_SIZE_WORDS & 0xFFFF)
        )
        word2  = self.stream_id & 0xFFFFFFFF
        word3  = DIFI_OUI & 0xFFFFFF
        word4  = ((self.info_class & 0xFFFF) << 16) | (PACKET_CLASS_CONTEXT & 0xFFFF)
        word5  = self.timestamp_int & 0xFFFFFFFF
        word6  = (self.timestamp_frac >> 32) & 0xFFFFFFFF
        word7  = self.timestamp_frac & 0xFFFFFFFF
        word8  = CIF0_CHANGE if self.context_changed else CIF0_NO_CHANGE
        word9  = self.reference_point & 0xFFFFFFFF
        word10, word11 = self._encode_freq(self.bandwidth_hz)
        word12, word13 = self._encode_freq(self.if_ref_freq_hz)
        word14, word15 = self._encode_freq(self.rf_ref_freq_hz)
        word16 = 0   # IF Band Offset high (zero-IF)
        word17 = 0   # IF Band Offset low
        word18 = self._encode_ref_level(self.reference_level_dbm) & 0xFFFF
        word19 = 0   # Gain (reserved)
        word20, word21 = self._encode_freq(self.sample_rate_hz)
        word22 = 0   # Timestamp Adjustment high (PoC: 0)
        word23 = 0   # Timestamp Adjustment low
        word24 = 0   # Timestamp Calibration Time
        word25 = 0   # State and Event Indicators
        word26, word27 = self._build_data_payload_format()

        return struct.pack(
            ">27I",
            word1,  word2,  word3,  word4,  word5,  word6,  word7,
            word8,  word9,  word10, word11, word12, word13, word14,
            word15, word16, word17, word18, word19, word20, word21,
            word22, word23, word24, word25, word26, word27,
        )

    # ── parse ──────────────────────────────────────────────────────────────

    @classmethod
    def from_bytes(cls, data: bytes) -> "DifiContextPacket":
        """Parse received bytes into a DifiContextPacket."""
        expected = cls.PACKET_SIZE_WORDS * 4
        if len(data) < expected:
            raise ValueError(f"Context packet too short: {len(data)} < {expected} bytes")

        (word1,  word2,  word3,  word4,  word5,  word6,  word7,
         word8,  word9,  word10, word11, word12, word13, word14,
         word15, _,      _,      word18, _,      word20, word21,
         _,      _,      _,      _,      word26, _) = struct.unpack_from(">27I", data)

        pkt_type = (word1 >> 28) & 0xF
        if pkt_type != PACKET_TYPE_CONTEXT:
            raise ValueError(f"Expected Context packet (type 0x4), got 0x{pkt_type:X}")

        tsi             = (word1 >> 22) & 0x3
        tsf             = (word1 >> 20) & 0x3
        seq_num         = (word1 >> 16) & 0xF
        stream_id       = word2
        info_class      = (word4 >> 16) & 0xFFFF
        timestamp_int   = word5
        timestamp_frac  = (word6 << 32) | word7
        context_changed = (word8 == CIF0_CHANGE)
        reference_point = word9
        bandwidth_hz    = cls._decode_freq(word10, word11)
        if_ref_freq_hz  = cls._decode_freq(word12, word13)
        rf_ref_freq_hz  = cls._decode_freq(word14, word15)
        sample_rate_hz  = cls._decode_freq(word20, word21)

        ref_raw = word18 & 0xFFFF
        if ref_raw & 0x8000:
            ref_raw -= 0x10000
        reference_level_dbm = ref_raw / 128.0
        sample_bit_depth    = (word26 & 0x3F) + 1

        return cls(
            stream_id           = stream_id,
            seq_num             = seq_num,
            timestamp_int       = timestamp_int,
            timestamp_frac      = timestamp_frac,
            sample_rate_hz      = sample_rate_hz,
            rf_ref_freq_hz      = rf_ref_freq_hz,
            if_ref_freq_hz      = if_ref_freq_hz,
            bandwidth_hz        = bandwidth_hz,
            reference_level_dbm = reference_level_dbm,
            sample_bit_depth    = sample_bit_depth,
            context_changed     = context_changed,
            info_class          = info_class,
            tsi                 = tsi,
            tsf                 = tsf,
            reference_point     = reference_point,
        )

    def __repr__(self) -> str:
        return (
            f"DifiContextPacket(stream_id=0x{self.stream_id:08X}, "
            f"seq={self.seq_num}, "
            f"fs={self.sample_rate_hz:.0f}Hz, "
            f"rf={self.rf_ref_freq_hz:.0f}Hz)"
        )


# ─────────────────────────────────────────────
# Timestamp helpers
# ─────────────────────────────────────────────

def now_timestamp() -> tuple:
    """Return current UTC time as (integer_seconds, picoseconds)."""
    t       = time.time()
    ts_int  = int(t)
    ts_frac = int((t - ts_int) * 1e12)
    return ts_int, ts_frac


def sample_count_timestamp(sample_index: int, sample_rate_hz: float, epoch_int: int = 0) -> tuple:
    """Compute (integer_seconds, sample_count) timestamp for a given sample index."""
    ts_int  = epoch_int + int(sample_index // sample_rate_hz)
    ts_frac = sample_index % int(sample_rate_hz)
    return ts_int, ts_frac


# ─────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=== DIFI Packet Self-Test ===\n")

    ts_int, ts_frac = now_timestamp()

    # Context Packet
    ctx       = DifiContextPacket(
        stream_id           = 0x00000001,
        seq_num             = 0,
        timestamp_int       = ts_int,
        timestamp_frac      = ts_frac,
        sample_rate_hz      = 48_000.0,
        rf_ref_freq_hz      = 437_000_000.0,
        bandwidth_hz        = 24_000.0,
        reference_level_dbm = -20.0,
        sample_bit_depth    = 16,
    )
    ctx_bytes = ctx.to_bytes()
    ctx2      = DifiContextPacket.from_bytes(ctx_bytes)
    print(f"Context packet : {len(ctx_bytes)} bytes (expected 108)")
    print(f"  {ctx2}")
    print(f"  Sample rate  : {ctx2.sample_rate_hz:.0f} Hz")
    print(f"  RF freq      : {ctx2.rf_ref_freq_hz:.0f} Hz")
    print(f"  Ref level    : {ctx2.reference_level_dbm:.1f} dBm")
    print()

    # Data Packet
    samples   = np.exp(1j * 2 * np.pi * 1000 * np.arange(1024) / 48000).astype(np.complex64)
    pkt       = DifiDataPacket(
        stream_id      = 0x00000001,
        seq_num        = 0,
        timestamp_int  = ts_int,
        timestamp_frac = ts_frac,
        payload        = samples,
    )
    pkt_bytes = pkt.to_bytes()
    pkt2      = DifiDataPacket.from_bytes(pkt_bytes)
    print(f"Data packet    : {len(pkt_bytes)} bytes")
    print(f"  {pkt2}")
    print(f"  Samples      : {len(pkt2.payload)}")

    max_err = float(np.max(np.abs(pkt2.payload - samples)))
    print(f"  Round-trip error: {max_err:.6f}")
    assert max_err < 0.0001
    print("\n✅ All tests passed!")