from dataclasses import dataclass
import numpy as np

from dataclasses import dataclass
import time
import numpy as np

from core.difi_packet import DifiPacket


@dataclass
class CwConfig:
    fs_hz: float = 48000.0              # sample rate
    tone_hz: float = 1000.0             # CW frequency (baseband)
    amplitude: float = 0.8              # 0..1
    samples_per_packet: int = 1024      # payload size
    num_packets: int = 10               # how many packets to generate


class CwDifiGenerator:
    def __init__(self, config: CwConfig):
        self.cfg = config

    def generate_stream(self, stream_id: int) -> list[DifiPacket]:
        cfg = self.cfg
        packets: list[DifiPacket] = []

        total_samples = cfg.samples_per_packet * cfg.num_packets
        n = np.arange(total_samples, dtype=np.float64)

        # Complex CW (baseband): A * exp(j*2*pi*f*n/Fs)
        x = cfg.amplitude * np.exp(1j * 2.0 * np.pi * cfg.tone_hz * n / cfg.fs_hz)
        x = x.astype(np.complex64)

        t0 = time.time()
        for k in range(cfg.num_packets):
            start = k * cfg.samples_per_packet
            end = start + cfg.samples_per_packet
            payload = x[start:end]

            # "timestamp" here is a simple simulation timestamp (not a DIFI timing implementation)
            ts = t0 + start / cfg.fs_hz

            packets.append(
                DifiPacket(
                    stream_id=stream_id,
                    sequence=k,
                    timestamp=ts,
                    payload=payload,
                )
            )

        return packets
@dataclass
class DifiPacket:
    stream_id: int
    sequence: int
    timestamp: float  # seconds
    payload: np.ndarray  # complex64 samples