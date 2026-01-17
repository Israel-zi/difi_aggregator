from dataclasses import dataclass
import numpy as np


@dataclass
class DifiPacket:
    stream_id: int
    sequence: int
    timestamp: float  # seconds
    payload: np.ndarray  # complex64 samples