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
import heapq
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
from pipeline_logger import wall_clock_str, sample_fingerprint
from thread_priority import boost_current_thread, THREAD_PRIORITY_HIGHEST


def _advance_ts(ts_int: int, ts_frac: int, n_samples: int, sample_rate_hz: float) -> tuple:
    """Advance a DIFI (integer_sec, picosecond) timestamp by n_samples at sample_rate_hz."""
    if sample_rate_hz <= 0:
        return ts_int, ts_frac
    ps_advance = int(n_samples * 1_000_000_000_000 / sample_rate_hz)
    new_frac   = ts_frac + ps_advance
    return ts_int + new_frac // 1_000_000_000_000, new_frac % 1_000_000_000_000


def _seq_run_complete(seqs: list) -> bool:
    """True iff seq_num values (4-bit, mod-16) form one unbroken run with no
    gap -- i.e. no packet belonging to this (stream_id, timestamp) group was
    lost in transit between InputCapture and the Aggregator."""
    if len(seqs) <= 1:
        return True
    expected = seqs[0]
    for s in seqs:
        if s != expected:
            return False
        expected = (expected + 1) & 0xF
    return True


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

    Reassembly, not just buffering
    -------------------------------
    A single logical capture instant can arrive as several distinct UDP DATA
    packets that all carry the *same* Stream ID and the *same* DIFI
    timestamp -- either because the source genuinely split one block across
    multiple packets, or (observed in practice on this project) because the
    wall-clock timestamp source ticks slower than the packet rate, so a
    burst of real, independent packets collide onto one timestamp. Either
    way, the combiner must not silently treat these as one packet's worth of
    data cut off wherever chunk_size happens to land -- it must gather every
    packet sharing that (stream_id, timestamp) key, verify none of them was
    lost in transit (via unbroken seq_num continuity), and only then treat
    the reassembled result as ready to forward.

    add_data() accumulates packets into an "open" group keyed by the DIFI
    timestamp. The group closes -- gets concatenated into one array and
    pushed onto the queue that ready()/consume() read from -- the moment a
    packet with a *different* timestamp arrives (the normal case: the
    source has moved on, so nothing more will arrive for the old one), or
    after poll_idle() sees the group has gone quiet for group_timeout_s (the
    tail case: this was the last group before the stream paused/stopped, so
    nothing will ever arrive to trigger the normal close). Each close is
    logged (packet count, seq range, completeness) via group_log.

    Once closed, groups are just (ts_int, ts_frac, array) entries in a queue
    -- consume() slices exactly chunk_size samples off the front, splitting
    a group only when chunk_size doesn't land on a group boundary (rare;
    falls back to sample-rate-based timestamp advancement for the leftover,
    same as before reassembly was added).
    """

    def __init__(self, stream_id: int, max_samples: int = 0,
                 group_timeout_s: float = 0.05, group_log=None,
                 target_latency_s: float = 0.2, max_staleness_s: float = None):
        self.stream_id      = stream_id
        self._packets       = deque()     # (ts_int, ts_frac, np.ndarray, first_received_at) per CLOSED group
        self._total         = 0           # total samples buffered across closed groups
        self.context        = None        # latest DifiContextPacket
        self.last_update    = time.monotonic()
        # Hard cap: when > 0, oldest closed groups are dropped once _total exceeds
        # this. Keeps pipeline latency bounded at max_samples / sample_rate seconds.
        #
        # This is sized off the caller's ASSUMED sample_rate_hz at construction
        # time, before any real context packet has arrived -- if the actual
        # stream turns out to run much faster (a real, measured case: 1.024
        # MSps vs. an assumed 48 kHz default, a 21x difference), that cap
        # represents only a few tens of milliseconds of true buffering
        # headroom instead of the intended few hundred, and any jitter/hold
        # near that scale silently evicts whole groups of real, already-
        # captured data with zero counter or log anywhere. add_context()
        # below recomputes this once the real rate is known.
        self._max_samples      = max_samples
        self._target_latency_s = target_latency_s
        # Groups evicted here were real, successfully-captured data that
        # arrived too late relative to this cap -- previously not counted
        # anywhere at all.
        self.groups_dropped_capacity = 0
        self.samples_dropped_capacity = 0

        # 2026-09-03: staleness bound, distinct from the capacity cap above.
        # The capacity cap only fires once _total (buffered SAMPLE COUNT)
        # exceeds a size threshold -- it says nothing about how long any
        # individual group has actually been waiting. Real-VM evidence
        # (5MHz+20MHz, hold_ms=200) showed hold_and_loss.csv reporting a
        # MEDIAN of 23 SECONDS and a p95 of 53 seconds between a packet's
        # own DIFI timestamp and when it was finally emitted -- while the
        # jitter-hold heap upstream (_HoldReorderBuffer) was correctly
        # releasing packets after ~200ms as configured. The gap was this
        # buffer: once a group closes, it just sits in _packets until
        # consume() gets around to it, with NO age check at all -- if the
        # Aggregator thread falls behind the offered rate for any reason,
        # already-captured data keeps accumulating and eventually gets
        # emitted anyway, however stale, rather than ever being dropped.
        # For a live system that's backwards: better to bound worst-case
        # latency and drop what can't be delivered within it than to keep
        # emitting data so old it's no longer useful. max_staleness_s=None
        # disables this (matches old behavior, e.g. config_gui.py's
        # standalone use where no caller has opted in).
        self._max_staleness_s = max_staleness_s
        self.groups_dropped_stale  = 0
        self.samples_dropped_stale = 0
        # DIFI timestamp of the most recently emitted chunk (set by consume()).
        self.data_ts_int    = 0
        self.data_ts_frac   = 0
        # monotonic time (this process's own clock only) that the first packet
        # of the most recently emitted chunk was captured off the wire by
        # InputCapture -- set by consume(). Unlike hold_ms (which compares
        # this machine's wall clock against the DIFI timestamp stamped on a
        # DIFFERENT machine, the Transmitter), a delta against this is immune
        # to any clock skew between VMs -- it measures only how long this
        # process itself took, nothing about whether the two machines agree
        # on what time it is.
        self.data_received_at = 0.0

        # -- open (not yet closed) group state --
        self._group_log        = group_log   # pipeline_logger.PacketLogger, or None
        self._group_timeout_s  = group_timeout_s
        self._open_ts          = None        # (ts_int, ts_frac) or None if nothing open
        self._open_parts       = []          # list[np.ndarray] pieces seen so far
        self._open_seqs        = []          # list[int] seq_num per piece, arrival order
        self._open_last_seen   = 0.0         # monotonic time of the open group's last packet
        self._open_gap         = False       # a seq_num gap landed while this group was open
        self._open_first_recv  = 0.0         # monotonic captured_at of this group's first packet
        self._last_seq         = None        # last seq_num seen on this stream (any group)

        # stats
        self.groups_closed     = 0
        self.groups_incomplete = 0

    def add_data(self, pkt: DifiDataPacket, received_at: float, capacity_multiplier: float = 1.0):
        now = time.monotonic()
        ts  = (pkt.timestamp_int, pkt.timestamp_frac)

        if self._open_ts is not None and ts != self._open_ts:
            self._close_open_group("TS_CHANGED", capacity_multiplier)
        if self._open_ts is None:
            self._open_ts         = ts
            self._open_first_recv = received_at

        gap = self._last_seq is not None and pkt.seq_num != (self._last_seq + 1) & 0xF
        if gap:
            self._open_gap = True
        self._last_seq = pkt.seq_num

        self._open_parts.append(pkt.payload.copy())
        self._open_seqs.append(pkt.seq_num)
        self._open_last_seen = now
        self.last_update     = now

    def poll_idle(self):
        """Force-close the open group if no new packet has arrived for it in
        group_timeout_s. Without this, the last group of a burst -- the one
        with no later, differently-stamped packet to trigger a normal close
        -- would sit "open" (invisible to ready()/consume()) forever."""
        if self._open_ts is not None and (time.monotonic() - self._open_last_seen) >= self._group_timeout_s:
            self._close_open_group("TIMEOUT")

    def force_close(self):
        """Unconditionally close the open group regardless of idle time --
        used only when the pipeline is shutting down, so whatever real data
        arrived most recently isn't left invisible to ready()/consume()
        just because it hasn't been quiet long enough yet."""
        if self._open_ts is not None:
            self._close_open_group("SHUTDOWN")

    def _close_open_group(self, reason: str, capacity_multiplier: float = 1.0):
        ts_int, ts_frac = self._open_ts
        seqs    = self._open_seqs
        samples = np.concatenate(self._open_parts).astype(np.complex64)
        complete = not self._open_gap and _seq_run_complete(seqs)

        self._packets.append((ts_int, ts_frac, samples, self._open_first_recv))
        self._total += len(samples)
        self.groups_closed += 1
        if not complete:
            self.groups_incomplete += 1

        if self._group_log is not None:
            first_i, first_q = sample_fingerprint(samples)
            local_latency_ms = (time.monotonic() - self._open_first_recv) * 1000.0
            self._group_log.log(
                wall_clock_str(), f"0x{self.stream_id:08X}", ts_int, ts_frac,
                len(seqs), len(samples), seqs[0], seqs[-1], complete, reason,
                first_i, first_q, f"{local_latency_ms:.2f}",
            )

        self._open_ts         = None
        self._open_parts      = []
        self._open_seqs       = []
        self._open_gap        = False
        self._open_first_recv = 0.0

        # If the buffer has grown beyond the cap, drop the oldest whole groups so
        # the pipeline always shows near-real-time data rather than a stale backlog.
        #
        # 2026-09-04: capacity_multiplier temporarily widens this cap right
        # after a stream joins (see Aggregator._topology_grace_s) -- a
        # progressive-join reproduction found THIS mechanism, not the
        # staleness bound, was the dominant one during a join transient:
        # this cap is checked on every single group close (far more often
        # than evict_stale's once-per-loop-iteration check), so it catches
        # and evicts a sudden backlog spike much faster. Widening the
        # staleness bound alone measurably did NOT fix the join-triggered
        # collapse; this cap is where the real fix has to apply too.
        effective_cap = self._max_samples * capacity_multiplier if self._max_samples else 0
        while effective_cap and self._total > effective_cap and len(self._packets) > 1:
            _, _, arr, _ = self._packets.popleft()
            self._total -= len(arr)
            self.groups_dropped_capacity += 1
            self.samples_dropped_capacity += len(arr)

    def evict_stale(self, now: float, max_staleness_s: float = None):
        """Drop closed groups from the FRONT of _packets (oldest first,
        since consume() always takes from the front too) that have aged
        past the staleness bound just sitting here waiting for consume() --
        see max_staleness_s's own comment for the real-VM evidence this
        responds to. Call this often (Aggregator._run() does, alongside
        poll_idle()) so "aged past the bound" and "actually evicted" stay
        close together in wall-clock time.

        max_staleness_s overrides self._max_staleness_s for this call when
        given (Aggregator._run() passes its own _effective_staleness_s(),
        which can be temporarily widened right after a stream join -- see
        that method's docstring); falls back to the constructor value
        otherwise, e.g. from callers that don't know about the topology
        grace period."""
        bound = max_staleness_s if max_staleness_s is not None else self._max_staleness_s
        if not bound:
            return
        while self._packets:
            ts_int, ts_frac, arr, first_recv = self._packets[0]
            age_s = now - first_recv
            if age_s <= bound:
                break
            self._packets.popleft()
            self._total -= len(arr)
            self.groups_dropped_stale  += 1
            self.samples_dropped_stale += len(arr)
            if self.groups_dropped_stale == 1 or self.groups_dropped_stale % 5000 == 0:
                print(f"[Aggregator] Stream 0x{self.stream_id:08X}: dropping already-reassembled "
                      f"groups that aged past {bound*1000:.0f}ms waiting to be "
                      f"consumed (total: {self.groups_dropped_stale})")
            if self._group_log is not None:
                first_i, first_q = sample_fingerprint(arr)
                self._group_log.log(
                    wall_clock_str(), f"0x{self.stream_id:08X}", ts_int, ts_frac,
                    0, len(arr), 0, 0, False, "STALE_DROPPED",
                    first_i, first_q, f"{age_s * 1000.0:.2f}",
                )

    def add_context(self, pkt: DifiContextPacket):
        self.context     = pkt
        self.last_update = time.monotonic()
        # The backlog cap was sized off an assumed rate at buffer-creation
        # time, before any context packet had arrived. Now that the real
        # rate is known, widen the cap if it turns out the stream runs
        # faster than assumed -- never shrink it, only grow to cover
        # whatever the true rate demands for the configured latency budget.
        if pkt.sample_rate_hz > 0:
            real_cap = int(pkt.sample_rate_hz * self._target_latency_s * 3)
            if real_cap > self._max_samples:
                self._max_samples = real_cap

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
            self.data_received_at = 0.0
            return np.zeros(0, dtype=np.complex64)

        self.data_ts_int, self.data_ts_frac = self._packets[0][0], self._packets[0][1]
        self.data_received_at = self._packets[0][3]

        collected = []
        remaining = chunk_size
        while remaining > 0 and self._packets:
            p_ts_int, p_ts_frac, arr, p_recv_at = self._packets[0]
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
                self._packets[0] = (p_ts_int, p_ts_frac, arr[remaining:], p_recv_at)
                remaining = 0

        return np.concatenate(collected).astype(np.complex64) if collected else np.zeros(0, dtype=np.complex64)

    @property
    def buffered_samples(self) -> int:
        return self._total


# ─────────────────────────────────────────────
# Inline hold/reorder buffer (WAN mode)
# ─────────────────────────────────────────────

class _HoldReorderBuffer:
    """Per-stream timestamp-ordered hold buffer -- the same min-heap logic
    JitterBuffer (input_capture.py) uses, but driven synchronously from
    inside Aggregator's OWN thread instead of a separate background
    thread + its own intermediate queue.Queue.

    Why (2026-09-03): a single-stream WAN-mode benchmark against the real
    production stack found the full JitterBuffer+Aggregator pipeline
    degrading (packet_q hand-off loss) starting around 4000 pkt/s, while
    an isolated benchmark of JUST the packet_q IPC/pickle cost (no
    JitterBuffer or Aggregator logic at all) stayed clean past 10,000
    pkt/s at the same 2200-sample/pkt jumbo size -- the IPC itself isn't
    the bottleneck. The one thing standing between those two numbers is
    JitterBuffer and Aggregator running as two separate Python threads
    SHARING ONE GIL inside combiner_worker.py's own process -- exactly
    the failure mode thread_priority.py's docstring already documented
    once (CPU-bound work sharing a thread's process: loss jumps sharply)
    and that capture_worker.py/packetizer_worker.py were each split into
    their own OS process specifically to avoid. JitterBuffer/Aggregator
    were never given the same treatment. A full third process is
    possible but adds ANOTHER pickle/IPC hop for the already-parsed IQ
    payload (measured expensive at this packet size); merging the two
    THREADS into one removes the GIL back-and-forth entirely while
    keeping exactly the same amount of real IPC work, at zero added
    serialization cost. This class is what makes that merge possible
    without reimplementing JitterBuffer's (already-validated: 0% loss, 0
    sequence gaps, 0% cross-stream ordering violations) hold/reorder
    logic from scratch.

    Not thread-safe -- by design, this is meant to be pushed to and
    drained from a single thread only. JitterBuffer's own copy of this
    logic (input_capture.py) is left untouched since it still spins its
    own thread for config_gui.py's legacy usage.
    """

    def __init__(self, hold_s: float):
        self._hold_s   = hold_s
        self._heaps: dict = {}
        self._push_seq = 0

    def push(self, captured) -> list:
        """Queue a DATA packet for later release, or return a CONTEXT
        packet immediately (stateless, no IQ data, no need to hold it)."""
        pkt = captured.packet
        if not isinstance(pkt, DifiDataPacket):
            return [captured]
        sid = pkt.stream_id
        self._heaps.setdefault(sid, [])
        heapq.heappush(self._heaps[sid], (pkt.timestamp_int, pkt.timestamp_frac, self._push_seq, captured))
        self._push_seq += 1
        return []

    def drain_ready(self, now: float) -> list:
        """Release every packet, across all streams, whose hold window has
        elapsed -- always in ascending (ts_int, ts_frac) order per stream."""
        released = []
        for heap in self._heaps.values():
            while heap:
                _ts_int, _ts_frac, _seq, captured = heap[0]
                if now - captured.received_at < self._hold_s:
                    break
                heapq.heappop(heap)
                released.append(captured)
        return released

    def flush_all(self) -> list:
        """Release everything still held, regardless of hold window --
        used only on shutdown so already-captured data isn't silently
        discarded just because it hadn't waited out its hold time yet."""
        released = []
        for heap in self._heaps.values():
            while heap:
                _, _, _, captured = heapq.heappop(heap)
                released.append(captured)
        self._heaps.clear()
        return released


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
    group_timeout_ms  : how long a stream's in-progress (stream_id, timestamp)
                        reassembly group may sit with no new packet before it's
                        force-closed and treated as ready — see StreamBuffer.
                        Only matters for the last group before a stream goes
                        quiet; every other group closes immediately once a
                        differently-stamped packet arrives.
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
        group_timeout_ms: float = 50.0,
        hold_log                = None,   # pipeline_logger.PacketLogger, or None
        group_log               = None,   # pipeline_logger.PacketLogger, or None
        hold_ms: float          = 0.0,    # WAN jitter/reorder hold -- see _HoldReorderBuffer
        chunk_sink              = None,   # optional Callable[[AggregatedChunk], bool] -- see set_chunk_sink()
        max_staleness_ms: float = None,   # None -> 3x target_latency_ms; 0/negative disables -- see StreamBuffer.evict_stale()
    ):
        self._capture          = capture
        # See _HoldReorderBuffer's own docstring for why this replaces a
        # separate JitterBuffer thread: same hold/reorder guarantee, done
        # inline on this thread instead of a second one sharing this
        # process's GIL. None when hold_ms<=0 -- capture packets flow
        # straight to _handle_packet() with no hold at all, same as
        # capture being an InputCapture/JitterBuffer(hold_ms=0) before.
        self._hold_buf = _HoldReorderBuffer(hold_ms / 1000.0) if hold_ms > 0 else None
        self._hold_log          = hold_log
        self._group_log        = group_log
        self._group_timeout_s  = group_timeout_ms / 1000.0
        self._sample_rate_hz   = sample_rate_hz
        self._expected         = set(expected_streams) if expected_streams else None
        self._expected_count   = (
            len(expected_streams) if expected_streams else expected_count
        )
        self._chunk_size       = chunk_size
        self._stale_timeout    = stale_timeout
        self._target_latency_s = target_latency_ms / 1000.0
        # See StreamBuffer.evict_stale()'s comment for why this exists at
        # all -- default is 3x the reorder/hold budget, the same multiple
        # already used elsewhere in this file for backlog-capacity sizing,
        # so "how long is too long to wait" stays consistent across both
        # mechanisms without introducing a second magic number.
        self._max_staleness_s = (
            (max_staleness_ms / 1000.0) if max_staleness_ms is not None
            else self._target_latency_s * 3.0
        )
        # 2026-09-04: a new stream joining mid-run steps the OFFERED rate up
        # in one instant -- reproduced directly (backend-only, no GUI, ruled
        # out as a logging/GUI artifact): capture stays perfectly smooth
        # through every join, but processing (READY events) collapses to
        # near-zero for 2-6 seconds right after the 3rd/4th stream joins,
        # then recovers on its own once the pipeline catches up to the new
        # rate. That's a textbook queueing-theory transient (a step
        # increase in arrival rate temporarily grows queue depth even when
        # the new steady-state rate is sustainable) -- the fixed
        # max_staleness_s bound has no slack for it, so packets that WOULD
        # have been fine once the backlog drained get shed anyway. Widen
        # the bound temporarily right after any new stream is discovered,
        # narrowing back down once the transient has had time to pass.
        self._topology_grace_s          = 5.0   # how long the widened bound applies after a join
        self._topology_grace_multiplier = 3.0   # widen max_staleness_s by this factor during grace
        self._last_topology_change      = time.monotonic()

        # 2026-09-04: a SECOND, DIFFERENT failure mode found via a real-GUI
        # A/B test (1 vs 2 vs 3 real Transmitter windows running together,
        # same combined offered rate throughout) -- not a topology change at
        # all. 3 concurrent real GUI processes measurably reduce this
        # machine's actual achievable pipeline throughput, but only to
        # roughly the offered rate, not below it: a live instrumented run
        # (outer-loop iterations/sec) proved this thread keeps running fast
        # (1600-2700+ iters/sec, i.e. NOT starved by Windows scheduling --
        # ruled out directly, including with the worker processes bumped to
        # HIGH_PRIORITY_CLASS, which measured no improvement) while still
        # discarding nearly every packet as already-stale, continuously, for
        # the rest of the run. The mechanism: once an early hiccup leaves the
        # front of packet_q older than max_staleness_s, and drain rate only
        # barely keeps pace with arrival rate (not comfortably ahead of it),
        # that backlog age never shrinks back under the bound on its own --
        # it just persists, so EVERY subsequent packet keeps getting judged
        # "already too late" forever, even though nothing is really staler
        # than it was a second ago. _topology_grace_* above doesn't cover
        # this (it's keyed to stream-join events, not to "am I currently
        # stuck"), and it starts from Aggregator construction, not from
        # whenever a real deficit actually begins.
        #
        # Fix: track a ROLLING window of recent intake outcomes (drop vs
        # success) and, once the drop RATIO within that window crosses
        # _overload_drop_ratio_trigger, treat it as proof the pipeline is
        # stuck at its ceiling rather than seeing one-off loss -- widen the
        # bound drastically for _overload_recovery_s so the very same
        # backlog that would otherwise be dropped forever gets a real
        # window to actually flush through (accepting extra latency for
        # that stretch) instead of being discarded outright. Sticky for the
        # full recovery window (not re-evaluated packet-by-packet) to avoid
        # flapping between wide/narrow every other packet.
        #
        # 2026-09-04, revised TWICE after real-GUI validation:
        #
        # v1 required an UNBROKEN streak of intake-level drops (reset to
        # zero by any single success). Fine at the "marginal, mostly-drops"
        # rates from the first pass (78-81% -> 27-41% loss), but a
        # genuinely-over-capacity heavy-rate run (5MHz+20MHz, ~11,364 pkt/s
        # vs the documented ~10,000 pkt/s packet_q ceiling -- see the
        # earlier packet_q entry above) let just enough isolated packets
        # through to keep resetting the streak, so it NEVER triggered
        # despite 86.9% real loss (confirmed: zero trigger-log lines).
        #
        # v2 replaced the streak with a drop-RATIO over a rolling window --
        # but scoped ONLY to _ingest()'s own intake-level check. Re-tested
        # against the SAME heavy run: still 86.7% loss, still zero
        # triggers. Root cause: in that run, most of the real loss wasn't
        # even happening at intake -- it was StreamBuffer.evict_stale()
        # dropping ALREADY-REASSEMBLED groups that had passed the intake
        # check just fine (10,049 groups dropped there vs "only" 14,136 at
        # intake, out of ~284,000 offered). A ratio computed from intake
        # outcomes alone never saw the dominant loss mechanism.
        #
        # v3 (this one): the ratio is computed from BOTH loss mechanisms
        # combined (packets_dropped_stale_intake delta + summed
        # groups_dropped_stale delta across all StreamBuffers) against
        # packets_ingested_ok delta, sampled once per _run() outer-loop
        # iteration rather than only inside _ingest() -- see _run()'s own
        # comment for where this is evaluated.
        self._overload_window_s           = 0.5   # evaluate the drop ratio on this cadence
        self._overload_drop_ratio_trigger = 0.5   # >= this fraction lost in a window means "stuck"
        self._overload_recovery_s         = 8.0    # how long the drastically-widened bound stays in effect once triggered
        self._overload_widen_multiplier   = 10.0   # e.g. 600ms -> 6000ms during recovery
        self._overload_window_start       = time.monotonic()
        self._overload_prev_intake_drops  = 0   # snapshot of packets_dropped_stale_intake at last window
        self._overload_prev_group_drops   = 0   # snapshot of summed groups_dropped_stale at last window
        self._overload_prev_ok            = 0   # snapshot of packets_ingested_ok at last window
        self._overload_recovery_until     = 0.0
        self._out_queue        = queue.Queue(maxsize=out_queue_size)
        # See Packetizer's identical put_timeout_s: absorbs a transient
        # downstream stall instead of permanently discarding a whole
        # already-aggregated chunk (both streams' real IQ data at once).
        self._put_timeout_s    = put_timeout_s
        # Optional direct hand-off, bypassing _out_queue entirely -- see
        # set_chunk_sink()'s docstring for why combiner_worker.py uses this
        # instead of a separate relay thread reading .get().
        self._chunk_sink       = chunk_sink
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
        # See _ingest()'s staleness check -- packets dropped before ever
        # reaching a StreamBuffer, distinct from StreamBuffer.evict_stale()'s
        # own groups_dropped_stale (which only catches staleness accrued
        # AFTER a group has already closed).
        self.packets_dropped_stale_intake = 0
        # 2026-09-04: counts successful (non-stale) _ingest() calls -- the
        # "good" half of the sustained-overload drop-ratio signal, see
        # _overload_* fields' own comment.
        self.packets_ingested_ok = 0

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
        if self._thread.is_alive():
            # join() timing out silently here would leave this thread running
            # in the background, invisible, competing for the GIL with
            # whatever the NEXT Listen cycle creates -- exactly the kind of
            # leak that would explain performance degrading across repeated
            # Stop/Start cycles within one long-running Combiner session.
            print("[Aggregator] WARNING: worker thread did not stop within 3s -- still running")
        cap_drops     = sum(b.groups_dropped_capacity for b in self._buffers.values())
        cap_samples   = sum(b.samples_dropped_capacity for b in self._buffers.values())
        stale_drops   = sum(b.groups_dropped_stale for b in self._buffers.values())
        stale_samples = sum(b.samples_dropped_stale for b in self._buffers.values())
        print(f"[Aggregator] Stopped | chunks emitted: {self.chunks_emitted} | "
              f"deadline misses: {self.deadline_misses} | "
              f"output-queue-full drops: {self.packets_dropped} | "
              f"capacity-cap drops: {cap_drops} groups ({cap_samples} samples) | "
              f"staleness drops: {stale_drops} groups ({stale_samples} samples) + "
              f"{self.packets_dropped_stale_intake} packets at intake "
              f"(bound={self._max_staleness_s*1000:.0f}ms)")

    def get(self, timeout: float = 0.1):
        """Retrieve the next AggregatedChunk. Returns None on timeout.
        Unused (returns nothing new) whenever a chunk_sink is set -- see
        set_chunk_sink()."""
        try:
            return self._out_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _current_widen_multiplier(self, now: float) -> float:
        """The larger of the two independent reasons to temporarily widen
        both the staleness bound and the capacity-cap multiplier: a recent
        topology change (_topology_grace_*), or a currently-active
        sustained-overload recovery window (_overload_*) -- see that
        field's own comment for why these are genuinely different failure
        modes, not duplicates of each other. Takes the max rather than
        stacking them (no evidence either needs to compound with the
        other, and stacking would make the widened bound's actual value
        harder to reason about from the logs)."""
        m = 1.0
        if now - self._last_topology_change < self._topology_grace_s:
            m = max(m, self._topology_grace_multiplier)
        if now < self._overload_recovery_until:
            m = max(m, self._overload_widen_multiplier)
        return m

    def _effective_staleness_s(self, now: float):
        """max_staleness_s, temporarily widened per _current_widen_multiplier().
        Returns the same falsy value max_staleness_s itself would (None/0)
        when staleness dropping is disabled entirely, so callers can keep
        using a single truthiness check."""
        if not self._max_staleness_s:
            return self._max_staleness_s
        return self._max_staleness_s * self._current_widen_multiplier(now)

    def _effective_capacity_multiplier(self, now: float) -> float:
        """Companion to _effective_staleness_s() for StreamBuffer's
        SEPARATE size-based backlog cap (max_samples) -- see
        StreamBuffer._close_open_group()'s comment for why a progressive
        stream-join reproduction found THIS mechanism, not the staleness
        bound, dominant during the join transient (checked far more often:
        every group close, vs. once per outer-loop iteration)."""
        return self._current_widen_multiplier(now)

    def set_chunk_sink(self, sink):
        """Install (or clear, with None) a direct hand-off callback,
        bypassing _out_queue/.get() entirely.

        Why (2026-09-03): combiner_worker.py's WAN pipeline used to read
        chunks via .get() on a separate "chunk relay" thread that then
        pickled each one across a multiprocessing.Queue to
        packetizer_worker.py's process -- a second thread sharing this
        process's one GIL with Aggregator's own thread, the exact same
        class of problem the JitterBuffer/Aggregator merge (see
        _HoldReorderBuffer) just fixed for the capture-side hand-off. A
        real-VM run (5MHz+20MHz, hold_ms=200) after that merge still
        showed "chunk_q full" drops -- this hand-off was the next
        GIL-contention bottleneck in line. sink is
        Callable[[AggregatedChunk], bool] -- called synchronously from
        _emit_cycle() on Aggregator's own thread, returning True if the
        chunk was actually handed off (counted in chunks_emitted) or
        False if it couldn't be (counted in packets_dropped, logged
        LOST_QUEUE_FULL, exactly like an _out_queue.Full used to be).
        Exceptions from sink are not caught here -- a broken sink should
        surface loudly, not silently degrade to _out_queue."""
        self._chunk_sink = sink

    # ── main loop ──────────────────────────────────────────────────────────

    def _ingest(self, captured):
        """Route one captured packet to _handle_packet(), through the
        inline hold/reorder buffer first if WAN mode (hold_ms>0) is
        active -- see _HoldReorderBuffer's docstring for why this replaces
        a separate JitterBuffer thread.

        2026-09-03: checks staleness FIRST, before any further processing
        at all. StreamBuffer.evict_stale() alone (applied only to the
        POST-hold-buffer backlog) turned out not to be enough: a real-VM
        overload test (5MHz+20MHz) showed heavy staleness drops there yet
        packets that DID survive still reported a 10+ second median
        hold_ms -- meaning most of the real delay was happening BEFORE a
        packet ever reached that stage (sitting in packet_q, or in
        _HoldReorderBuffer's own heap, while this thread's overall
        throughput fell behind the offered rate). A packet already older
        than the staleness bound at the moment this thread finally gets
        to it has no chance of being delivered within budget regardless
        of what happens next (hold + group-close + emit only add more
        delay) -- so it's dropped here, immediately, rather than paying
        for hold-buffer/reassembly work on data that's already too late
        to be useful."""
        now = time.monotonic()
        effective_bound = self._effective_staleness_s(now)
        if effective_bound and (now - captured.received_at) > effective_bound:
            self.packets_dropped_stale_intake += 1
            # 2026-09-03 real-VM finding: without this, a sustained overload
            # makes EVERY packet stale by the time this thread gets to it --
            # packet_q keeps draining fine (never prints "full"), but zero
            # output reaches the Receiver for as long as it lasts, with
            # nothing in any log to say why. Same periodic-print convention
            # as packet_q/chunk_q's own full-drop warnings.
            if self.packets_dropped_stale_intake == 1 or self.packets_dropped_stale_intake % 5000 == 0:
                print(f"[Aggregator] Dropping packets at intake -- already stale on arrival "
                      f"(total: {self.packets_dropped_stale_intake}, bound={self._max_staleness_s*1000:.0f}ms). "
                      f"This thread cannot keep up with the offered rate.")
            return
        self.packets_ingested_ok += 1
        if self._hold_buf is None:
            self._handle_packet(captured)
            return
        for released in self._hold_buf.push(captured):
            self._handle_packet(released)

    def _run(self):
        # See thread_priority.py -- this thread drains InputCapture's own
        # queue; keeping it ahead of Packetizer/GUI under CPU contention
        # prevents that queue backing up and, upstream, InputCapture
        # dropping packets it never gets asked to enqueue in time for.
        boost_current_thread(THREAD_PRIORITY_HIGHEST)
        # TEMPORARY diagnostic (2026-09-04), gated behind an env var so it
        # never runs unless explicitly asked for -- see memory note "an
        # outer-loop-iterations/sec counter is a fast, decisive way to rule
        # OS/GIL scheduling starvation in or out". Answers: when throughput
        # collapses with 3+ real GUI-hosted streams, is THIS thread still
        # being scheduled normally (many fast iterations/sec, pointing at an
        # algorithmic/queueing cause) or is it itself being starved of CPU
        # (few iterations/sec, pointing at real OS scheduling contention)?
        import os as _os_dbg
        _debug_rate = _os_dbg.environ.get("DIFI_DEBUG_AGGREGATOR_RATE") == "1"
        _dbg_iters = 0
        _dbg_last_print = time.monotonic()
        while not self._stop_evt.is_set():
            if _debug_rate:
                _dbg_iters += 1
                _now_dbg = time.monotonic()
                if _now_dbg - _dbg_last_print >= 1.0:
                    print(f"[AggregatorDebug] {_dbg_iters} outer-loop iters/"
                          f"{_now_dbg - _dbg_last_print:.2f}s "
                          f"(stale_intake_total={self.packets_dropped_stale_intake})")
                    _dbg_iters = 0
                    _dbg_last_print = _now_dbg
            # Fully drain whatever has already arrived before judging
            # readiness — draining only one packet per iteration would let
            # a burst of emissions race ahead of real data that's already
            # sitting in the queue but not yet handled.
            captured = self._capture.get(timeout=0.005)
            drained = 0
            while captured is not None:
                self._ingest(captured)
                drained += 1
                if drained >= 512:   # generous safety cap, not a normal limit
                    break
                captured = self._capture.get(timeout=0.0)

            # Release anything whose hold window has now elapsed -- same
            # per-loop-iteration cadence JitterBuffer's own thread used.
            if self._hold_buf is not None:
                for released in self._hold_buf.drain_ready(time.monotonic()):
                    self._handle_packet(released)

            # Drop any ALREADY-CLOSED group that's aged past the staleness
            # bound while just waiting its turn for consume() -- see
            # StreamBuffer.evict_stale()'s docstring. This is a DIFFERENT
            # queue than _HoldReorderBuffer's hold-and-reorder heap above
            # (which already released this packet on schedule); it's the
            # per-stream backlog between "released, reassembled, closed"
            # and "actually consumed into an output chunk."
            now = time.monotonic()
            effective_bound = self._effective_staleness_s(now)
            for buf in self._buffers.values():
                buf.evict_stale(now, effective_bound)

            # Sustained-overload detection -- see _overload_* fields' own
            # comment (v3: combines BOTH loss mechanisms -- intake drops
            # AND already-reassembled-group eviction drops -- against
            # successful intakes, sampled here once per outer-loop
            # iteration rather than only inside _ingest(), since a heavy
            # real-overload run showed the group-eviction path can
            # dominate the real loss while intake alone still looks fine).
            if now - self._overload_window_start >= self._overload_window_s:
                total_group_drops = sum(b.groups_dropped_stale for b in self._buffers.values())
                d_intake = self.packets_dropped_stale_intake - self._overload_prev_intake_drops
                d_group  = total_group_drops - self._overload_prev_group_drops
                d_ok     = self.packets_ingested_ok - self._overload_prev_ok
                d_bad    = d_intake + d_group
                d_total  = d_bad + d_ok
                if (d_total > 0
                        and d_bad / d_total >= self._overload_drop_ratio_trigger
                        and now >= self._overload_recovery_until):
                    self._overload_recovery_until = now + self._overload_recovery_s
                    print(f"[Aggregator] Sustained overload detected "
                          f"({d_bad}/{d_total} lost in the last "
                          f"{self._overload_window_s:.1f}s: {d_intake} at intake, "
                          f"{d_group} at group-eviction) -- widening staleness bound "
                          f"{self._overload_widen_multiplier:.0f}x for "
                          f"{self._overload_recovery_s:.1f}s to let the backlog drain "
                          f"instead of discarding it.")
                self._overload_window_start      = now
                self._overload_prev_intake_drops = self.packets_dropped_stale_intake
                self._overload_prev_group_drops  = total_group_drops
                self._overload_prev_ok           = self.packets_ingested_ok

            # Force-close any stream's in-progress reassembly group that's
            # gone quiet — otherwise the tail group of a burst (nothing later
            # ever arrives to trigger its normal TS_CHANGED close) would sit
            # open forever, invisible to ready()/consume().
            for buf in self._buffers.values():
                buf.poll_idle()

            # Keep emitting while a full cycle (or a deadline-expired
            # partial one) is available — lets a backlog drain at whatever
            # rate the pipeline can sustain, rather than one emission per
            # outer-loop tick.
            while self._maybe_emit():
                pass

        # Shutting down: drain whatever arrived in the final moments, flush
        # anything still sitting in the hold buffer regardless of its hold
        # window (real, already-captured data that must not be silently
        # discarded just because it hadn't waited out hold_ms yet -- the
        # same reasoning JitterBuffer.stop() used to apply before this
        # class replaced it inline), force-close any still-open reassembly
        # group instead of waiting for its normal idle timeout, and
        # force-emit anything that's ready without waiting out
        # target_latency_ms for stragglers. Measured directly (pre-merge):
        # this was a real, repeatable loss source in multi-stream runs,
        # invisible to every other counter.
        captured = self._capture.get(timeout=0.02)
        while captured is not None:
            self._ingest(captured)
            captured = self._capture.get(timeout=0.0)
        if self._hold_buf is not None:
            for released in self._hold_buf.flush_all():
                self._handle_packet(released)
        for buf in self._buffers.values():
            buf.force_close()
        while self._maybe_emit(force=True):
            pass

    def _maybe_emit(self, force: bool = False) -> bool:
        """Emit one AggregatedChunk if a full cycle is ready, or a partial
        one has waited past target_latency_ms. Returns True iff it emitted.

        force=True skips the "wait for stragglers" grace period entirely --
        used only when shutting down, so a stream that was already ready
        and simply hadn't waited out its full target_latency_ms yet doesn't
        have its real, already-buffered data abandoned just because the
        pipeline stopped a few milliseconds before that clock ran out."""
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

        if not force and ready < stream_ids and (now - self._pending_since) < self._target_latency_s:
            return False   # still within budget — give the stragglers a chance

        self._emit_cycle(stream_ids, ready)
        # 2026-09-04: only clear the deadline clock once EVERY stream made
        # it into this cycle -- a real, reproduced bug (progressive-join
        # test) had this reset unconditionally, which re-armed a fresh
        # target_latency_s wait before the NEXT cycle could emit for as
        # long as any one straggler (e.g. a stream still cold-starting)
        # kept missing ready(). Since a cycle only ever drains one
        # chunk_size's worth per call, that capped the WHOLE aggregator's
        # throughput at one cycle per target_latency_s (~5/s at 200ms)
        # regardless of how much real backlog the already-ready streams
        # had queued -- which then aged past the staleness bound and got
        # evicted instead of consumed. Once we've already paid the "give
        # the straggler a chance" wait once, keep draining the backlog
        # immediately every subsequent call until that straggler either
        # catches up (ready >= stream_ids) or ages out on its own.
        if ready >= stream_ids:
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
            self._buffers[stream_id] = StreamBuffer(
                stream_id, max_samples=max_samples,
                group_timeout_s=self._group_timeout_s, group_log=self._group_log,
                target_latency_s=self._target_latency_s,
                max_staleness_s=self._max_staleness_s,
            )
            self._stream_ports[stream_id] = captured.source_port
            # See _topology_grace_s's comment -- a new stream steps the
            # offered rate up in one instant; widen the staleness bound for
            # a few seconds so the transient queueing spike while this
            # thread catches up doesn't get shed as if it were permanent
            # overload.
            self._last_topology_change = time.monotonic()
            if self._expected is None:
                print(f"[Aggregator] Discovered stream 0x{stream_id:08X} on port {captured.source_port} ({len(self._buffers)} of {self._expected_count or '?'})")

        buf = self._buffers[stream_id]

        if isinstance(pkt, DifiDataPacket):
            buf.add_data(pkt, captured.received_at,
                         capacity_multiplier=self._effective_capacity_multiplier(time.monotonic()))
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
                self._log_hold(sid, "READY", ts_int, ts_frac, len(samples), buf.data_received_at)
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
                self._log_hold(sid, "TIMEOUT_ZEROFILL", ts_int, ts_frac, self._chunk_size, None)
            # else: cold start — this stream has never sent a first packet yet.

        if not blocks:
            return

        chunk = AggregatedChunk(streams=blocks)
        self.last_chunk = chunk   # display tap — no queue consumption required

        if self._chunk_sink is not None:
            # See set_chunk_sink()'s docstring -- same accounting as the
            # _out_queue path below, just handed off directly instead of
            # through a second thread.
            if self._chunk_sink(chunk):
                self.chunks_emitted += 1
            else:
                self.packets_dropped += 1
                self._drop_warn_count += 1
                if self._drop_warn_count <= 3 or self._drop_warn_count % 1000 == 0:
                    print(f"[Aggregator] chunk_sink full — chunk dropped (total: {self.packets_dropped})")
                for block in blocks:
                    self._log_hold(block.stream_id, "LOST_QUEUE_FULL",
                                    block.data_ts_int, block.data_ts_frac, len(block.samples), None)
            return

        try:
            # put_nowait, not a blocking put(timeout=...): this call runs on
            # the SAME thread that drains InputCapture and updates every
            # stream's last-seen bookkeeping (buf.last_update). A blocking
            # put here doesn't just delay this one chunk -- for as long as
            # it blocks, NOTHING on this thread runs, so InputCapture's own
            # queue (30 slots by default) backs up and starts dropping real,
            # never-yet-processed packets, and every stream's staleness
            # clock freezes (hold_ms and "did the source go quiet" both read
            # last_update). Measured: at a trivial 200 pkt/s with a slow
            # consumer, a 0.2s blocking put() here stalled the thread for
            # 1.75 of every 2 seconds and dropped 85% of arriving packets --
            # a self-inflicted cascade, not a real throughput ceiling.
            # Dropping immediately when the queue is genuinely full keeps
            # this thread free to keep draining and keep bookkeeping fresh,
            # which is strictly better for a downstream consumer that's only
            # briefly behind, and no worse for one that's sustained-overloaded.
            self._out_queue.put_nowait(chunk)
            self.chunks_emitted += 1
        except queue.Full:
            self.packets_dropped += 1
            self._drop_warn_count += 1
            if self._drop_warn_count <= 3 or self._drop_warn_count % 1000 == 0:
                print(f"[Aggregator] Output queue full — chunk dropped (total: {self.packets_dropped})")
            for block in blocks:
                self._log_hold(block.stream_id, "LOST_QUEUE_FULL",
                                block.data_ts_int, block.data_ts_frac, len(block.samples), None)

    def _log_hold(self, sid: int, outcome: str, ts_int: int, ts_frac: int, samples: int,
                  received_at):
        """Record one stream's per-cycle outcome to the hold/loss evidence log.

        Logs TWO different latency numbers, and they answer different
        questions:
          hold_ms          : time.time() [[this machine's clock]] minus the
                              packet's own DIFI timestamp [[stamped on the
                              TRANSMITTER's clock]]. Meaningful only if the
                              two machines' clocks are synchronized -- on
                              unsynchronized VMs this is contaminated by
                              clock skew and can look arbitrarily wrong
                              (inflated, deflated, even negative) with zero
                              relation to real processing time.
          local_latency_ms : time.monotonic() minus the monotonic instant
                              InputCapture itself received the first packet
                              of this chunk (received_at) -- both readings
                              taken on THIS process's own clock. Immune to
                              any cross-machine clock skew; this is the
                              trustworthy number for "how long did the
                              combiner itself take."
        received_at is None for synthetic outcomes (zero-filled or dropped
        for lack of queue room) that have no real captured packet behind them.
        """
        if self._hold_log is None:
            return
        hold_ms = (time.time() - (ts_int + ts_frac / 1e12)) * 1000.0
        local_latency_ms = (time.monotonic() - received_at) * 1000.0 if received_at is not None else ""
        self._hold_log.log(
            wall_clock_str(), "AGGREGATOR", f"0x{sid:08X}", outcome,
            f"{hold_ms:.2f}", ts_int, ts_frac, samples,
            f"{local_latency_ms:.2f}" if local_latency_ms != "" else "",
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

    def get_stream_previews(self, max_samples: int = 4096) -> list:
        """Return [(stream_id, samples, context)] for any stream that has data,
        each trimmed to at most the most recent `max_samples` samples.
        Does NOT consume buffered samples — read-only snapshot for display.
        Includes the still-open (not yet reassembled/closed) group too, so
        the live spectrum doesn't visibly stall while a group waits to close.

        Bounding to a trailing window is essential, not an optimization: a
        caller here (the spectrum display) only ever looks at the last
        seg_len (1024) samples anyway, but concatenating the FULL backlog
        first -- as this used to do -- costs time linear in however much
        data has piled up in buf._packets. Called on a 10Hz timer, that
        turns any transient slowdown into a runaway feedback loop: a
        growing backlog makes each preview call itself slower, stealing
        more CPU from the very threads that would drain that backlog,
        growing it further. Measured: with a 2-second backlog at ~4000
        pkt/s combined, one unbounded call here took 47ms -- at 10Hz, that
        alone is most of a CPU core, before any real packet processing.
        """
        result = []
        for sid, buf in self._buffers.items():
            if buf.context is None:
                continue
            parts = list(buf._open_parts)
            total = sum(len(p) for p in parts)
            for _, _, arr, _ in reversed(buf._packets):
                if total >= max_samples:
                    break
                parts.insert(0, arr)
                total += len(arr)
            if not parts:
                continue
            samples = np.concatenate(parts).astype(np.complex64)
            if len(samples) > max_samples:
                samples = samples[-max_samples:]
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