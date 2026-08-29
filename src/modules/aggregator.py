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

import os
import sys
import queue
import time
import threading
from collections import deque

import numpy as np
from dataclasses import dataclass, field

if not getattr(sys, 'frozen', False):
    _src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _src not in sys.path:
        sys.path.insert(0, _src)

from core.difi_packet import DifiDataPacket, DifiContextPacket
from modules.input_capture import CapturedPacket, InputCapture
from pipeline_logger import wall_clock_str


def _advance_ts(ts_int: int, ts_frac: int, n_samples: int, sample_rate_hz: float) -> tuple:
    """Advance a DIFI (integer_sec, picosecond) timestamp by n_samples at sample_rate_hz."""
    if sample_rate_hz <= 0:
        return ts_int, ts_frac
    ps_advance = int(n_samples * 1_000_000_000_000 / sample_rate_hz)
    new_frac   = ts_frac + ps_advance
    return ts_int + new_frac // 1_000_000_000_000, new_frac % 1_000_000_000_000


# ─────────────────────────────────────────────
# Output data structures
# ─────────────────────────────────────────────

@dataclass
class StreamBlock:
    """One stream's worth of IQ data inside an aggregated chunk."""
    stream_id:    int
    samples:      np.ndarray          # complex64
    context:      DifiContextPacket   # most recent context for this stream
    received_at:  float               # time.monotonic() of last sample
    data_ts_int:  int  = 0            # DIFI integer timestamp of the last data packet
    data_ts_frac: int  = 0            # DIFI fractional timestamp of the last data packet


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
    """Accumulates IQ samples and tracks the latest Context for one stream.

    Samples are kept as a queue of (ts_int, ts_frac, array) per received
    packet -- not one flat array with a single timestamp latched only when
    the buffer was last empty. That older design meant that once a backlog
    built up (production briefly outrunning consumption, e.g. under thread
    scheduling contention with several other streams), every consume() took
    a "remainder" path that dead-reckoned the timestamp forward from a
    stale origin by chunk_size each time -- divorced from the packets'
    own real stamps, and able to drift for as long as the backlog persisted.
    Tracking per-packet timestamps means consume() always reports a real
    packet's own true stamp (exactly, in the common case where chunk_size
    equals samples_per_pkt -- one packet in, one packet out).
    """

    def __init__(self, stream_id: int, max_samples: int = 0):
        self.stream_id      = stream_id
        self._packets       = deque()     # (ts_int, ts_frac, np.ndarray) per packet
        self._total         = 0           # total samples buffered
        self.context        = None        # latest DifiContextPacket
        self.last_update    = time.monotonic()
        # Hard cap: when > 0, oldest packets are dropped once _total exceeds this.
        # Keeps pipeline latency bounded at max_samples / sample_rate seconds.
        self._max_samples   = max_samples
        # DIFI timestamp of the most recently emitted chunk (set by consume()).
        self.data_ts_int    = 0
        self.data_ts_frac   = 0

    def add_data(self, pkt: DifiDataPacket):
        self._packets.append((pkt.timestamp_int, pkt.timestamp_frac, pkt.payload.copy()))
        self._total      += len(pkt.payload)
        self.last_update  = time.monotonic()

        # If the buffer has grown beyond the cap, drop the oldest whole packets so
        # the pipeline always shows near-real-time data rather than a stale backlog.
        while self._max_samples and self._total > self._max_samples and len(self._packets) > 1:
            _, _, arr = self._packets.popleft()
            self._total -= len(arr)

    def add_context(self, pkt: DifiContextPacket):
        self.context     = pkt
        self.last_update = time.monotonic()

    def ready(self, chunk_size: int) -> bool:
        return self._total >= chunk_size and self.context is not None

    def consume(self, chunk_size: int, sample_rate_hz: float = 0.0) -> np.ndarray:
        """Return exactly `chunk_size` samples, keeping any remainder.

        Reports data_ts_int/frac as the *first consumed packet's own DIFI
        timestamp* -- exact, not extrapolated, whenever chunk_size doesn't
        split that packet. Only a genuine split (chunk_size smaller than
        one packet, or spanning a partial final packet) falls back to
        sample-rate-based advancement for the leftover portion.
        """
        if not self._packets:
            self.data_ts_int, self.data_ts_frac = 0, 0
            return np.zeros(0, dtype=np.complex64)

        self.data_ts_int, self.data_ts_frac = self._packets[0][0], self._packets[0][1]

        collected = []
        remaining = chunk_size
        while remaining > 0 and self._packets:
            p_ts_int, p_ts_frac, arr = self._packets[0]
            if len(arr) <= remaining:
                collected.append(arr)
                remaining -= len(arr)
                self._total -= len(arr)
                self._packets.popleft()
            else:
                collected.append(arr[:remaining])
                self._total -= remaining
                if sample_rate_hz > 0:
                    p_ts_int, p_ts_frac = _advance_ts(p_ts_int, p_ts_frac, remaining, sample_rate_hz)
                self._packets[0] = (p_ts_int, p_ts_frac, arr[remaining:])
                remaining = 0

        return np.concatenate(collected).astype(np.complex64) if collected else np.zeros(0, dtype=np.complex64)

    @property
    def buffered_samples(self) -> int:
        return self._total


# ─────────────────────────────────────────────
# Aggregator
# ─────────────────────────────────────────────

class Aggregator:
    """
    Reads CapturedPackets from InputCapture, buffers per-stream IQ data, and
    emits one AggregatedChunk as soon as every currently-expected stream has
    real data ready — or, if some are still missing once target_latency_ms
    has elapsed since the first one became ready, emits anyway with the
    stragglers zero-filled, instead of stalling forever.

    This answers the Tx-aggregation problem — N independently-delayed
    sources (different fixed network delay + random jitter per stream) must
    combine into ONE stream with bounded, deterministic end-to-end latency
    — WITHOUT losing real IQ data and WITHOUT emitting faster than data can
    actually arrive. Emission is driven by real packet readiness, not by an
    idealized wall-clock cadence derived from the declared sample rate: an
    earlier version of this class scheduled cycles against
    chunk_size/sample_rate_hz directly, and at a high declared rate (e.g.
    1024-sample chunks at 10 MHz demand ~9765 cycles/sec) that cadence
    outran what a Python thread can sustain end-to-end — it manufactured a
    flood of meaningless zero-filled cycles and, worse, skipping the
    resulting backlog discarded real, already-buffered IQ data outright.
    Tying emission to real readiness self-throttles to whatever the
    pipeline can actually sustain (exactly like a plain "wait for everyone"
    design when every stream keeps up), and the deadline only ever forces a
    decision when SOME streams are ready and one specific one is genuinely
    late — exactly the WAN-delay scenario this project targets, not a
    CPU/throughput ceiling.

    Parameters
    ----------
    capture           : InputCapture (or JitterBuffer) instance to read from
    sample_rate_hz    : nominal shared sample rate — used only to size each
                        stream's backlog cap relative to target_latency_ms.
                        Per-stream DIFI timestamps use that stream's own
                        Context sample_rate_hz.
    expected_streams  : explicit list of stream IDs to wait for, OR None to
                        auto-discover stream IDs from incoming packets
    expected_count    : when expected_streams=None, how many unique streams to
                        wait for before emitting (typically = number of listen
                        ports).  If also None, emit as soon as any stream is ready.
    chunk_size        : samples per stream per aggregated chunk
    out_queue_size    : max depth of the output queue
    stale_timeout     : seconds a stream may stay silent — either never having
                        sent a first packet at all, or having gone quiet after
                        being active — before it's dropped from the cycle
                        entirely instead of being waited on forever. Should
                        comfortably exceed target_latency_ms (the cold-start
                        grace period reuses this value: a stream due any
                        moment within its configured delay must not be
                        excluded before it's even had a chance to appear).
    target_latency_ms : maximum time a partially-ready cycle waits for the
                        remaining streams before zero-filling them and
                        emitting anyway — must comfortably exceed the
                        largest configured sim delay + jitter across streams
    """

    def __init__(
        self,
        capture: InputCapture,
        sample_rate_hz: float   = 48_000.0,
        expected_streams: list  = None,
        expected_count: int     = None,
        chunk_size: int         = 1024,
        out_queue_size: int     = 32,
        put_timeout_s: float    = 0.2,
        stale_timeout: float    = 5.0,
        target_latency_ms: float = 200.0,
        hold_log                = None,   # pipeline_logger.PacketLogger, or None
    ):
        self._capture          = capture
        self._hold_log          = hold_log
        self._sample_rate_hz   = sample_rate_hz
        self._expected         = set(expected_streams) if expected_streams else None
        self._expected_count   = (
            len(expected_streams) if expected_streams else expected_count
        )
        self._chunk_size       = chunk_size
        self._stale_timeout    = stale_timeout
        self._target_latency_s = target_latency_ms / 1000.0
        self._out_queue        = queue.Queue(maxsize=out_queue_size)
        # See Packetizer's identical put_timeout_s: absorbs a transient
        # downstream stall instead of permanently discarding a whole
        # already-aggregated chunk (both streams' real IQ data at once).
        self._put_timeout_s    = put_timeout_s
        self._buffers          = {}        # stream_id -> StreamBuffer
        self._stop_evt         = threading.Event()
        self._thread           = threading.Thread(
            target=self._run, daemon=True, name="aggregator"
        )

        # display tap — latest chunk written by aggregator thread, read by GUI thread.
        # Assignment is atomic in CPython; chunks are immutable after creation.
        self.last_chunk        = None

        # port each stream_id was first received on (populated on first packet)
        self._stream_ports: dict = {}   # stream_id -> source_port

        # wall-clock time we started expecting each never-yet-seen stream —
        # its cold-start grace period (see _cycle_stream_ids) runs from here,
        # not from aggregator startup, so a stream added later via
        # update_stream_filter() gets its own full grace window.
        self._expected_since: dict = {}   # stream_id -> monotonic time
        if self._expected is not None:
            now = time.monotonic()
            self._expected_since = {sid: now for sid in self._expected}

        # per-stream "next expected chunk" DIFI timestamp cursor, used to stamp
        # zero-filled cycles so downstream timestamps stay continuous through
        # gaps. Seeded from real data on that stream's first successful consume.
        self._next_expected_ts: dict = {}   # stream_id -> (ts_int, ts_frac)

        # wall-clock time the current cycle first had ANY stream ready, or
        # None if nothing is pending. Starts the target_latency_ms clock.
        self._pending_since: float | None = None

        # stats
        self.chunks_emitted   = 0
        self.packets_dropped  = 0
        self.deadline_misses  = 0
        self.deadline_misses_by_stream: dict = {}
        self._drop_warn_count = 0

    # ── lifecycle ──────────────────────────────────────────────────────────

    def start(self):
        self._pending_since = None
        self._thread.start()
        if self._expected:
            print(f"[Aggregator] Started | streams={[hex(s) for s in self._expected]} | "
                  f"chunk_size={self._chunk_size} | target_latency={self._target_latency_s*1000:.0f}ms")
        else:
            print(f"[Aggregator] Started | auto-detect mode | expecting {self._expected_count} stream(s) | "
                  f"chunk_size={self._chunk_size} | target_latency={self._target_latency_s*1000:.0f}ms")

    def stop(self):
        self._stop_evt.set()
        self._thread.join(timeout=3.0)
        print(f"[Aggregator] Stopped | chunks emitted: {self.chunks_emitted} | "
              f"deadline misses: {self.deadline_misses}")

    def get(self, timeout: float = 0.1):
        """Retrieve the next AggregatedChunk. Returns None on timeout."""
        try:
            return self._out_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    # ── main loop ──────────────────────────────────────────────────────────

    def _run(self):
        while not self._stop_evt.is_set():
            # Fully drain whatever has already arrived before judging
            # readiness — draining only one packet per iteration would let
            # a burst of emissions race ahead of real data that's already
            # sitting in the queue but not yet handled.
            captured = self._capture.get(timeout=0.005)
            drained = 0
            while captured is not None:
                self._handle_packet(captured)
                drained += 1
                if drained >= 512:   # generous safety cap, not a normal limit
                    break
                captured = self._capture.get(timeout=0.0)

            # Keep emitting while a full cycle (or a deadline-expired
            # partial one) is available — lets a backlog drain at whatever
            # rate the pipeline can sustain, rather than one emission per
            # outer-loop tick.
            while self._maybe_emit():
                pass

    def _maybe_emit(self) -> bool:
        """Emit one AggregatedChunk if a full cycle is ready, or a partial
        one has waited past target_latency_ms. Returns True iff it emitted."""
        stream_ids = self._cycle_stream_ids()
        if not stream_ids:
            self._pending_since = None
            return False

        ready = {
            sid for sid in stream_ids
            if (buf := self._buffers.get(sid)) is not None and buf.ready(self._chunk_size)
        }
        if not ready:
            return False   # nothing to emit yet — don't start the deadline clock on nothing

        now = time.monotonic()
        if self._pending_since is None:
            self._pending_since = now

        if ready < stream_ids and (now - self._pending_since) < self._target_latency_s:
            return False   # still within budget — give the stragglers a chance

        self._emit_cycle(stream_ids, ready)
        self._pending_since = None
        return True

    def _handle_packet(self, captured: CapturedPacket):
        pkt       = captured.packet
        stream_id = pkt.stream_id

        # fixed mode: ignore streams not in the expected set
        if self._expected is not None and stream_id not in self._expected:
            return

        if stream_id not in self._buffers:
            # Backlog cap sized off the latency budget (with headroom for the
            # configured JitterBuffer hold + scheduling slack), not a fixed
            # small multiple of chunk_size — a 150ms simulated delay needs
            # ~150ms of buffering room to survive without discarding samples.
            max_samples = max(
                self._chunk_size * 4,
                int(self._sample_rate_hz * self._target_latency_s * 3),
            )
            self._buffers[stream_id] = StreamBuffer(stream_id, max_samples=max_samples)
            self._stream_ports[stream_id] = captured.source_port
            if self._expected is None:
                print(f"[Aggregator] Discovered stream 0x{stream_id:08X} on port {captured.source_port} ({len(self._buffers)} of {self._expected_count or '?'})")

        buf = self._buffers[stream_id]

        if isinstance(pkt, DifiDataPacket):
            buf.add_data(pkt)
        elif isinstance(pkt, DifiContextPacket):
            buf.add_context(pkt)

    def _cycle_stream_ids(self) -> set:
        """Stream IDs to consider for the current cycle.

        Two distinct kinds of "not ready" get different treatment, and
        conflating them is what caused a real bug: a fast (little/no delay)
        stream would race ahead and drain its entire startup backlog alone
        the moment a slower expected stream simply hadn't sent its first
        packet YET — that stream wasn't "excluded", it was invisible, so
        "all ready" was trivially true among whoever happened to exist so
        far. Once the slow stream finally did appear, it was permanently
        paired with the fast stream's much-later backlog instead of its
        true original moment — a lasting cross-stream misalignment, not a
        transient glitch.

        - Never-yet-seen (expected, no buffer at all): stays a candidate —
          i.e. still counts toward "not everyone is ready" — for up to
          stale_timeout seconds since it became expected. This is what
          correctly makes a fast stream WAIT for a slower one that's still
          within its configured delay, instead of running ahead.
        - Previously active but gone silent (has a buffer, but stale):
          excluded — a stream whose listener/generator was turned off must
          eventually stop being waited on, or every other stream pays the
          full target_latency_ms wait-then-zero-fill penalty forever.
        """
        now = time.monotonic()
        if self._expected is not None:
            result = set()
            for sid in self._expected:
                buf = self._buffers.get(sid)
                if buf is not None:
                    if now - buf.last_update < self._stale_timeout:
                        result.add(sid)
                elif now - self._expected_since.get(sid, now) < self._stale_timeout:
                    result.add(sid)
            return result
        # Auto-detect mode: only ever considers streams actually seen at
        # least once, so there's no "never-yet-seen" case to distinguish.
        return {
            sid for sid, buf in self._buffers.items()
            if now - buf.last_update < self._stale_timeout
        }

    def _emit_cycle(self, stream_ids: set, ready: set):
        """Emit exactly one AggregatedChunk for the current cycle.

        `ready` (precomputed by _maybe_emit, which already checked each
        buffer) says which streams get real samples; every other expected
        stream gets a zero-filled block stamped with its own expected next
        timestamp — so a late or missing stream never blocks the others,
        and downstream timestamps stay continuous. A stream that has never
        sent a first packet at all (true cold start) is skipped entirely.
        """
        blocks = []
        for sid in sorted(stream_ids):
            buf = self._buffers.get(sid)

            if sid in ready:
                fs = buf.context.sample_rate_hz if buf.context else self._sample_rate_hz
                samples = buf.consume(self._chunk_size, sample_rate_hz=fs)
                ts_int, ts_frac = buf.data_ts_int, buf.data_ts_frac   # set by consume() itself
                self._next_expected_ts[sid] = _advance_ts(ts_int, ts_frac, self._chunk_size, fs)
                blocks.append(StreamBlock(
                    stream_id=sid, samples=samples, context=buf.context,
                    received_at=buf.last_update, data_ts_int=ts_int, data_ts_frac=ts_frac,
                ))
                self._log_hold(sid, "READY", ts_int, ts_frac, len(samples))
            elif buf is not None and buf.context is not None and sid in self._next_expected_ts:
                # Missed this cycle's deadline — zero-fill to preserve phase
                # alignment for the other streams instead of stalling everyone.
                ts_int, ts_frac = self._next_expected_ts[sid]
                fs = buf.context.sample_rate_hz
                blocks.append(StreamBlock(
                    stream_id=sid, samples=np.zeros(self._chunk_size, dtype=np.complex64),
                    context=buf.context, received_at=buf.last_update,
                    data_ts_int=ts_int, data_ts_frac=ts_frac,
                ))
                self._next_expected_ts[sid] = _advance_ts(ts_int, ts_frac, self._chunk_size, fs)
                self.deadline_misses += 1
                self.deadline_misses_by_stream[sid] = self.deadline_misses_by_stream.get(sid, 0) + 1
                self._log_hold(sid, "TIMEOUT_ZEROFILL", ts_int, ts_frac, self._chunk_size)
            # else: cold start — this stream has never sent a first packet yet.

        if not blocks:
            return

        chunk = AggregatedChunk(streams=blocks)
        self.last_chunk = chunk   # display tap — no queue consumption required

        try:
            self._out_queue.put(chunk, timeout=self._put_timeout_s)
            self.chunks_emitted += 1
        except queue.Full:
            self.packets_dropped += 1
            self._drop_warn_count += 1
            if self._drop_warn_count <= 3 or self._drop_warn_count % 1000 == 0:
                print(f"[Aggregator] Output queue full — chunk dropped (total: {self.packets_dropped})")
            for block in blocks:
                self._log_hold(block.stream_id, "LOST_QUEUE_FULL",
                                block.data_ts_int, block.data_ts_frac, len(block.samples))

    def _log_hold(self, sid: int, outcome: str, ts_int: int, ts_frac: int, samples: int):
        """Record one stream's per-cycle outcome to the hold/loss evidence log."""
        if self._hold_log is None:
            return
        hold_ms = (time.time() - (ts_int + ts_frac / 1e12)) * 1000.0
        self._hold_log.log(
            wall_clock_str(), "AGGREGATOR", f"0x{sid:08X}", outcome,
            f"{hold_ms:.2f}", ts_int, ts_frac, samples,
        )

    # ── diagnostics ────────────────────────────────────────────────────────

    def remove_stream_by_port(self, port: int):
        """Remove buffers and tracking for the stream that was first seen on this port."""
        for sid, p in list(self._stream_ports.items()):
            if p == port:
                self._buffers.pop(sid, None)
                self._stream_ports.pop(sid, None)
                self._next_expected_ts.pop(sid, None)
                self.deadline_misses_by_stream.pop(sid, None)
                # Removing the buffer alone would make this look like a fresh
                # cold start (no buffer -> full new stale_timeout grace period)
                # if it's still in self._expected — force it stale immediately
                # instead of restarting its clock.
                if sid in self._expected_since:
                    self._expected_since[sid] = time.monotonic() - self._stale_timeout - 1.0
                print(f"[Aggregator] Removed stream 0x{sid:08X} (was on port {port})")

    def update_stream_filter(self, allowed_ids: set):
        """
        Dynamically restrict which stream IDs are included in aggregated chunks.
        Takes effect on the next _emit_cycle call (no pipeline restart needed).
        Pass None to revert to accepting all discovered streams.
        """
        if allowed_ids is None:
            self._expected       = None
            self._expected_count = None
        else:
            self._expected       = frozenset(allowed_ids)
            self._expected_count = len(allowed_ids)
            now = time.monotonic()
            for sid in self._expected:
                self._expected_since.setdefault(sid, now)   # own grace window if newly added
            for sid in list(self._next_expected_ts):
                if sid not in self._expected:
                    self._next_expected_ts.pop(sid, None)
                    self.deadline_misses_by_stream.pop(sid, None)

    def flush_queue(self):
        """Drain the output queue so stale chunks don't reach the receiver."""
        drained = 0
        while True:
            try:
                self._out_queue.get_nowait()
                drained += 1
            except queue.Empty:
                break
        return drained

    def buffer_status(self) -> dict:
        return {
            hex(sid): buf.buffered_samples
            for sid, buf in self._buffers.items()
        }

    def stream_last_seen(self) -> dict:
        """Return {stream_id: last_update monotonic time} for all known streams."""
        return {sid: buf.last_update for sid, buf in self._buffers.items()}

    def stream_source_ports(self) -> dict:
        """Return {stream_id: source_port} for all discovered streams."""
        return dict(self._stream_ports)

    def get_stream_previews(self) -> list:
        """Return [(stream_id, samples, context)] for any stream that has data.
        Does NOT consume buffered samples — read-only snapshot for display."""
        result = []
        for sid, buf in self._buffers.items():
            if buf.context is not None and buf._total > 0:
                samples = np.concatenate([arr for _, _, arr in buf._packets]).astype(np.complex64)
                result.append((sid, samples, buf.context))
        return result


# ─────────────────────────────────────────────
# Quick self-test (standalone)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    TEST_PORTS      = [50001, 50002]
    EXPECTED_STREAM = [0x00000001, 0x00000002]

    capture    = InputCapture(ports=TEST_PORTS)
    aggregator = Aggregator(
        capture          = capture,
        sample_rate_hz   = 48_000.0,   # matches generator.py's self-test default
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
            if chunk is None:
                continue
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