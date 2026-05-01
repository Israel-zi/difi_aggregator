"""
aggregator.py
-------------
DIFI Aggregator module.

Receives tagged CapturedPacket objects from the InputCapture module,
separates Data packets from Context packets per stream, and produces
AggregatedChunk objects — each containing IQ samples from all active
streams plus their associated context metadata.

The Aggregator does NOT re-pack into DIFI format — that is the
responsibility of the Packetizer module.

Aggregation strategy (PoC)
--------------------------
  - Collect samples from each stream independently into per-stream buffers.
  - When ALL streams have accumulated at least `chunk_size` samples,
    emit one AggregatedChunk containing one block per stream.
  - Preserve original stream IDs so the Packetizer can encode them
    as separate sub-streams inside the unified DIFI packet stream.
"""

import queue
import time
import threading
import numpy as np
from dataclasses import dataclass, field

from core.difi_packet import DifiDataPacket, DifiContextPacket
from modules.input_capture import CapturedPacket, InputCapture


# ─────────────────────────────────────────────
# Output data structures
# ─────────────────────────────────────────────

@dataclass
class StreamBlock:
    """One stream's worth of IQ data inside an aggregated chunk."""
    stream_id:   int
    samples:     np.ndarray          # complex64
    context:     DifiContextPacket   # most recent context for this stream
    received_at: float               # time.monotonic() of last sample


@dataclass
class AggregatedChunk:
    """
    Output of the Aggregator: one chunk of samples from all active streams,
    ready to be handed to the Packetizer.
    """
    streams:     list                # list[StreamBlock]
    created_at:  float = field(default_factory=time.monotonic)

    @property
    def stream_ids(self) -> list:
        return [s.stream_id for s in self.streams]

    @property
    def num_streams(self) -> int:
        return len(self.streams)


# ─────────────────────────────────────────────
# Per-stream buffer
# ─────────────────────────────────────────────

class StreamBuffer:
    """Accumulates IQ samples and tracks the latest Context for one stream."""

    def __init__(self, stream_id: int):
        self.stream_id   = stream_id
        self._samples    = []          # list of np.ndarray chunks
        self._total      = 0           # total samples buffered
        self.context     = None        # latest DifiContextPacket
        self.last_update = time.monotonic()

    def add_data(self, pkt: DifiDataPacket):
        self._samples.append(pkt.payload.copy())
        self._total      += len(pkt.payload)
        self.last_update  = time.monotonic()

    def add_context(self, pkt: DifiContextPacket):
        self.context     = pkt
        self.last_update = time.monotonic()

    def ready(self, chunk_size: int) -> bool:
        return self._total >= chunk_size and self.context is not None

    def consume(self, chunk_size: int) -> np.ndarray:
        """Return exactly `chunk_size` samples, keeping the remainder."""
        all_samples = np.concatenate(self._samples).astype(np.complex64)
        out         = all_samples[:chunk_size]
        remainder   = all_samples[chunk_size:]

        if len(remainder) > 0:
            self._samples = [remainder]
            self._total   = len(remainder)
        else:
            self._samples = []
            self._total   = 0

        return out

    @property
    def buffered_samples(self) -> int:
        return self._total


# ─────────────────────────────────────────────
# Aggregator
# ─────────────────────────────────────────────

class Aggregator:
    """
    Reads CapturedPackets from InputCapture, buffers per-stream IQ data,
    and emits AggregatedChunk objects when all streams are ready.

    Parameters
    ----------
    capture          : InputCapture instance to read from
    expected_streams : list of stream IDs we expect to aggregate
    chunk_size       : samples per stream per aggregated chunk
    out_queue_size   : max depth of the output queue
    stale_timeout    : seconds before a stream is considered stale/missing
    """

    def __init__(
        self,
        capture: InputCapture,
        expected_streams: list,
        chunk_size: int       = 1024,
        out_queue_size: int   = 100,
        stale_timeout: float  = 5.0,
    ):
        self._capture          = capture
        self._expected         = set(expected_streams)
        self._chunk_size       = chunk_size
        self._stale_timeout    = stale_timeout
        self._out_queue        = queue.Queue(maxsize=out_queue_size)
        self._buffers          = {}        # stream_id -> StreamBuffer
        self._stop_evt         = threading.Event()
        self._thread           = threading.Thread(
            target=self._run, daemon=True, name="aggregator"
        )

        # stats
        self.chunks_emitted  = 0
        self.packets_dropped = 0

    # ── lifecycle ──────────────────────────────────────────────────────────

    def start(self):
        self._thread.start()
        print(
            f"[Aggregator] Started | "
            f"streams={[hex(s) for s in self._expected]} | "
            f"chunk_size={self._chunk_size}"
        )

    def stop(self):
        self._stop_evt.set()
        self._thread.join(timeout=3.0)
        print(f"[Aggregator] Stopped | chunks emitted: {self.chunks_emitted}")

    def get(self, timeout: float = 0.1):
        """Retrieve the next AggregatedChunk. Returns None on timeout."""
        try:
            return self._out_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    # ── main loop ──────────────────────────────────────────────────────────

    def _run(self):
        while not self._stop_evt.is_set():
            captured = self._capture.get(timeout=0.05)
            if captured is None:
                continue

            self._handle_packet(captured)
            self._try_emit_chunk()

    def _handle_packet(self, captured: CapturedPacket):
        pkt       = captured.packet
        stream_id = pkt.stream_id

        # only process streams we expect
        if stream_id not in self._expected:
            return

        # lazy-create buffer
        if stream_id not in self._buffers:
            self._buffers[stream_id] = StreamBuffer(stream_id)

        buf = self._buffers[stream_id]

        if isinstance(pkt, DifiDataPacket):
            buf.add_data(pkt)
        elif isinstance(pkt, DifiContextPacket):
            buf.add_context(pkt)

    def _try_emit_chunk(self):
        """Emit one AggregatedChunk if ALL expected streams are ready."""

        # check all expected streams have a buffer and are ready
        if not all(
            sid in self._buffers and self._buffers[sid].ready(self._chunk_size)
            for sid in self._expected
        ):
            return

        # build one StreamBlock per stream
        blocks = []
        for sid in sorted(self._expected):
            buf = self._buffers[sid]
            blocks.append(StreamBlock(
                stream_id   = sid,
                samples     = buf.consume(self._chunk_size),
                context     = buf.context,
                received_at = buf.last_update,
            ))

        chunk = AggregatedChunk(streams=blocks)

        try:
            self._out_queue.put_nowait(chunk)
            self.chunks_emitted += 1
        except queue.Full:
            self.packets_dropped += 1
            print("[Aggregator] Output queue full — chunk dropped")

    # ── diagnostics ────────────────────────────────────────────────────────

    def buffer_status(self) -> dict:
        return {
            hex(sid): buf.buffered_samples
            for sid, buf in self._buffers.items()
        }


# ─────────────────────────────────────────────
# Quick self-test (standalone)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    TEST_PORTS      = [50001, 50002]
    EXPECTED_STREAM = [0x00000001, 0x00000002]

    capture    = InputCapture(ports=TEST_PORTS)
    aggregator = Aggregator(
        capture          = capture,
        expected_streams = EXPECTED_STREAM,
        chunk_size       = 1024,
    )

    capture.start()
    aggregator.start()

    print("\n[Test] Waiting for aggregated chunks ...")
    print("       Run two generator.py instances in separate terminals.")
    print("       Press Ctrl+C to stop.\n")

    chunks = 0
    try:
        while True:
            chunk = aggregator.get(timeout=1.0)
            if chunk:
                chunks += 1
                print(
                    f"  Chunk #{chunks} | "
                    f"streams={[hex(s) for s in chunk.stream_ids]} | "
                    f"samples/stream={chunk.streams[0].samples.shape}"
                )
                if chunks >= 10:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        aggregator.stop()
        capture.stop()
        print(f"\n[Test] Total chunks aggregated: {chunks}")