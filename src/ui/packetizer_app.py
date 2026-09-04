"""
packetizer_app.py
-----------------
DIFI Aggregator — Combiner GUI.

This window is a thin control panel only — it never touches a UDP socket,
a ring buffer, or a dispatch timeline itself. Pressing Listen spawns N
independent ring_capture_main processes (one per configured port, see
ring_pipeline.py) that each write incoming packets directly into a
per-stream shared-memory ring buffer at an O(1) index derived from the
packet's own DIFI timestamp -- no multiprocessing.Queue hand-off, no
sorting. Pressing Forward spawns exactly ONE ring_egress_main process that
reads every stream's buffer TOGETHER and dispatches them aligned to one
common timeline (per this project's own specified architecture -- each
stream can be held for a different real delay, e.g. 100ms/120ms/150ms,
and still be released in sync). See ring_pipeline.py's own module
docstring for the full design and reasoning, and project memory for the
validation history (this replaced an earlier one-worker-per-port design
that raised raw throughput but silently broke that same-timeline alignment
guarantee).

Two-phase operation:
  ▶ Listen   — starts N ring_capture_main processes (ingress only)
  ▶ Forward  — starts the single ring_egress_main process (data flows to
               the Receiver, held/aligned per the "Apply hold" ms)
  ■ Stop     — tears down the whole ring pipeline

Stream rows configure which ports to listen on. Each row's Fwd checkbox
selects whether packets arriving on that port are forwarded — filtering is
by port, pushed to the egress process as a "set_forward_ports" command.
"""

import os
import sys

if not getattr(sys, 'frozen', False):
    _src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _src not in sys.path:
        sys.path.insert(0, _src)

import multiprocessing
import queue as _queue
import time

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QGroupBox, QStatusBar,
    QLineEdit, QSpinBox, QCheckBox,
)

from pipeline_logger    import make_run_dir
from ring_pipeline      import start_ring_ingress, start_ring_egress, stop_ring_egress, stop_ring_pipeline
from gil_friendly_exec  import run_gil_friendly, request_stop
import app_config


class StreamRow(QWidget):
    """
    One listen-stream row.
    Layout: [#] Port:[spinbox] [Fwd☑] [−]

    Signals
    -------
    removed        — user clicked the remove button
    filter_changed — Fwd checkbox toggled (forwarding filter)
    """

    removed        = Signal(object)
    filter_changed = Signal()

    def __init__(self, index: int, default_port: int, fwd_checked: bool = True, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(6)

        self._idx_lbl = QLabel(f"{index}")
        self._idx_lbl.setFixedWidth(18)
        self._idx_lbl.setStyleSheet("color: #888888;")
        lay.addWidget(self._idx_lbl)

        self._port = QSpinBox()
        self._port.setRange(1024, 65535)
        self._port.setValue(default_port)
        self._port.setFixedWidth(100)
        lay.addWidget(self._port)

        self._sid_lbl = QLabel("(waiting…)")
        self._sid_lbl.setStyleSheet("color: #666666; font-size: 11px;")
        self._sid_lbl.setFixedWidth(120)
        lay.addWidget(self._sid_lbl)

        # Activity LED -- driven by a lightweight heartbeat from the worker
        # process (~1/s, just port->stream-id/active booleans, not a full
        # stats poll), not the earlier per-tick Aggregator-state query that
        # ran in the GUI process.
        self._led = QLabel("●")
        self._led.setStyleSheet("color: #444444; font-size: 18px;")
        self._led.setFixedWidth(22)
        lay.addWidget(self._led)

        # Fwd checkbox — controls forwarding to Receiver. Always enabled:
        # filtering is by port, decided up front, not by the stream ID
        # this row's LED/label display (which just reflects what the
        # worker last reported, for operator visibility).
        self._fwd_cb = QCheckBox()
        self._fwd_cb.setChecked(fwd_checked)
        self._fwd_cb.setToolTip("Forward packets arriving on this port to the Receiver")
        self._fwd_cb.setFixedWidth(36)
        self._fwd_cb.stateChanged.connect(lambda _: self.filter_changed.emit())
        lay.addWidget(self._fwd_cb)

        self._remove_btn = QPushButton("−")
        self._remove_btn.setFixedSize(26, 26)
        self._remove_btn.setToolTip("Remove this stream")
        self._remove_btn.clicked.connect(lambda: self.removed.emit(self))
        lay.addWidget(self._remove_btn)

        lay.addStretch()

    # ── accessors ──────────────────────────────────────────────────────────

    def set_index(self, n: int):
        self._idx_lbl.setText(str(n))

    def port(self) -> int:
        return self._port.value()

    def forwarded_port(self) -> int | None:
        """Return this row's port if Fwd is checked, else None."""
        return self._port.value() if self._fwd_cb.isChecked() else None

    def set_stream_id(self, sid: int | None):
        if sid is None:
            self._sid_lbl.setText("(waiting…)")
            self._sid_lbl.setStyleSheet("color: #666666; font-size: 11px;")
        else:
            self._sid_lbl.setText(f"0x{sid:08X}")
            self._sid_lbl.setStyleSheet("color: #aaaaaa; font-size: 11px;")

    def set_active(self, active: bool):
        color = "#00cc44" if active else "#444444"
        self._led.setStyleSheet(f"color: {color}; font-size: 18px;")

    def set_locked(self, locked: bool):
        self._port.setEnabled(not locked)
        # Remove button and Fwd checkbox always stay enabled while listening.


class PacketizerWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("DIFI Combiner")
        self.setMinimumSize(900, 480)
        self._listening   = False   # ring ingress processes are up and capturing
        self._forwarding  = False   # the ring egress process is also running
        # 2026-09-04, rebuilt per the project's own specified architecture
        # (Base/whatsapp from itay.docx -- direct guidance from the academic
        # mentor): "the core of the project is dealing with UNIFIED TIMING
        # in the Aggregator." An earlier version of this file ran one
        # fully-independent worker_main process PER PORT to raise raw
        # throughput -- real-GUI testing did confirm that removed the old
        # shared multiprocessing.Queue bottleneck, but it also silently
        # dropped the project's actual core requirement: N independently-
        # delayed/jittered streams must still be held and released on ONE
        # common timeline (e.g. stream1 delayed 100ms, stream2 120ms,
        # stream3 150ms, aligned to the SAME output instant) -- that needs
        # one process seeing every stream together, which independent
        # per-port workers structurally cannot do.
        #
        # ring_pipeline.py is the replacement: N independent INGRESS
        # processes (still one per port -- this part of the earlier
        # parallelism was correct and stays) write each packet directly
        # into a per-stream shared-memory ring buffer at an O(1) index
        # derived from its own DIFI timestamp (see ring_buffer.py) --
        # avoiding multiprocessing.Queue's real per-call IPC cost entirely,
        # not just spreading it across more workers. Exactly ONE unified
        # egress process then polls every stream's buffer together and
        # dispatches in aligned timestamp order, per the spec. Validated
        # against the mentor's own exact test scenario (3 streams, 100/120/
        # 150ms delay, up to 30ms random jitter, 200ms target buffer):
        # 1.8% loss backend-only, down from 78-92% before any of this
        # session's fixes -- see project memory for the full chain.
        self._ring_handles = None   # set by _listen(), see ring_pipeline.start_ring_ingress()
        self._stream_id_by_port: dict = {}
        self._run_dir     = None    # log folder for the current Listen session
        self._stream_rows: list = []
        # See app_config.py -- loaded here (before _build_ui() creates any
        # widget) so every field, including the stream-row ports below,
        # can seed its initial value from last-saved settings.
        self._cfg = app_config.load("Combiner")
        self._build_ui()

        in_cfg = app_config.Section(self._cfg, "Combiner.Input")
        saved_ports = in_cfg.get_str("ports", "50001,50002")
        try:
            ports = [int(p) for p in saved_ports.split(",") if p.strip()]
        except ValueError:
            ports = []
        ports = ports or [50001, 50002]
        # Default (if never saved) = every port forwarded, matching
        # StreamRow's own default-checked Fwd box.
        fwd_default = ",".join(str(p) for p in ports)
        try:
            fwd_ports = {int(p) for p in in_cfg.get_str("fwd_ports", fwd_default).split(",") if p.strip()}
        except ValueError:
            fwd_ports = set(ports)
        for port in ports:
            self._add_stream_row(port, fwd_checked=port in fwd_ports)

        # Opt-in, env-var-gated autostart -- see transmitter_app.py's
        # identical hook for why (drive the REAL windowed GUI process
        # without unreliable pywinauto/UIA clicking). No effect unless
        # DIFI_AUTOSTART=1, so normal interactive use is unchanged.
        if os.environ.get("DIFI_AUTOSTART") == "1":
            QTimer.singleShot(300, self._autostart_from_env)

    def _autostart_from_env(self):
        env = os.environ
        ports_str = env.get("DIFI_PORTS")
        if ports_str:
            new_ports = [int(p) for p in ports_str.split(",") if p.strip()]
            while len(self._stream_rows) > 1:
                self._remove_stream_row(self._stream_rows[-1])
            if new_ports:
                self._stream_rows[0]._port.setValue(new_ports[0])
                for p in new_ports[1:]:
                    self._add_stream_row(p)
        if env.get("DIFI_CHUNK_SIZE"):
            self._chunk.setValue(int(env["DIFI_CHUNK_SIZE"]))
        if env.get("DIFI_HOLD_MS") is not None:
            hold_ms = int(float(env["DIFI_HOLD_MS"]))
            if hold_ms > 0:
                self._hold_enabled_cb.setChecked(True)
                self._hold_ms.setValue(hold_ms)
            else:
                self._hold_enabled_cb.setChecked(False)
        if env.get("DIFI_DEST_IP"):
            self._dest_ip.setText(env["DIFI_DEST_IP"])
        if env.get("DIFI_DEST_PORT"):
            self._dest_port.setValue(int(env["DIFI_DEST_PORT"]))
        self._listen_btn.click()
        if env.get("DIFI_AUTOFORWARD", "1") == "1":
            QTimer.singleShot(800, self._forward_btn.click)
        if env.get("DIFI_AUTOSTOP_AFTER_S"):
            QTimer.singleShot(int(float(env["DIFI_AUTOSTOP_AFTER_S"]) * 1000), self.close)

    # ── UI construction ────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        left_lay = QVBoxLayout(central)
        left_lay.setSpacing(8)

        # Inputs
        in_box  = QGroupBox("Inputs  (listen for streams from Transmitter VMs)")
        in_vlay = QVBoxLayout(in_box)
        in_vlay.setSpacing(4)

        hdr = QHBoxLayout()
        hdr.setSpacing(6)
        hdr.addSpacing(26)
        for lbl, w in [("Port", 100), ("Stream ID (auto)", 120), ("●", 22), ("Fwd", 36)]:
            l = QLabel(lbl)
            l.setFixedWidth(w)
            l.setStyleSheet("color: #888888; font-size: 11px;")
            hdr.addWidget(l)
        hdr.addStretch()
        in_vlay.addLayout(hdr)

        self._rows_container = QWidget()
        self._rows_layout    = QVBoxLayout(self._rows_container)
        self._rows_layout.setSpacing(2)
        self._rows_layout.setContentsMargins(2, 2, 2, 2)
        self._rows_layout.addStretch()
        in_vlay.addWidget(self._rows_container)

        self._add_btn = QPushButton("＋  Add Stream")
        self._add_btn.setFixedHeight(28)
        self._add_btn.clicked.connect(lambda: self._add_stream_row())
        in_vlay.addWidget(self._add_btn)

        chunk_row = QHBoxLayout()
        chunk_row.addWidget(QLabel("Chunk size:"))
        self._chunk = QSpinBox()
        self._chunk.setRange(64, 65536)
        self._chunk.setSingleStep(256)
        # See transmitter_app.py's DEFAULT_SAMPLES_PER_PKT comment -- this
        # directly sets the sample count of every re-encoded DIFI packet
        # this Combiner sends to the Receiver (WAN mode / hold_ms>0 only --
        # LAN relay mode at hold_ms=0 forwards the Transmitter's own raw
        # bytes unchanged and ignores this field entirely, see
        # combiner_worker.py's _relay_loop()), so the same MTU-fragmentation
        # math applies to the Combiner->Receiver leg independent of whatever
        # packet size the Transmitter used on the way in.
        # 2026-09-03: DIFI_Standard_1.3.0_Final.pdf's own jumbo-frame
        # provision is a 9000-byte MTU (p.15/p.21) -- 2200 samples fits
        # under that (2200*4+56=8856 bytes) with headroom to spare. Only
        # correct once the VM's LAN is actually configured for jumbo frames
        # end-to-end; drop back to ~360 for standard 1500-byte Ethernet.
        in_cfg = app_config.Section(self._cfg, "Combiner.Input")
        self._chunk.setValue(in_cfg.get_int("chunk_size", 2200))
        self._chunk.setSuffix(" samples")
        self._chunk.setFixedWidth(150)
        self._chunk.setToolTip(
            "Samples per outgoing DIFI packet (Combiner -> Receiver, WAN\n"
            "mode / Reorder hold > 0 only -- ignored in LAN relay mode).\n"
            "Wire size = samples*4 + 56 bytes. Keep under your link's actual\n"
            "MTU or packets get IP-fragmented -- the DIFI standard itself\n"
            "requires fragmentation not be produced at the source. Default\n"
            "(2200) fits under the standard's own 9000-byte jumbo-frame MTU;\n"
            "use ~360 instead for standard 1500-byte Ethernet."
        )
        chunk_row.addWidget(self._chunk)
        chunk_row.addStretch()
        in_vlay.addLayout(chunk_row)
        left_lay.addWidget(in_box)

        # Network / Jitter Buffer
        net_box    = QGroupBox("Network / Jitter Buffer")
        net_layout = QHBoxLayout(net_box)
        net_cfg = app_config.Section(self._cfg, "Combiner.Network")

        # Separate on/off checkbox from the ms value itself -- lets an
        # operator flip between LAN (0ms) and WAN mode without having to
        # remember or re-type whatever hold value they'd tuned, and without
        # the spinbox's own value ever meaning anything different: unchecked
        # always means hold=0 regardless of what number is sitting in the
        # box, exactly as if it were 0 ms.
        self._hold_enabled_cb = QCheckBox("Apply hold:")
        self._hold_enabled_cb.setChecked(net_cfg.get_bool("hold_enabled", False))
        self._hold_enabled_cb.setToolTip(
            "Unchecked = LAN pass-through (hold=0), regardless of the ms\n"
            "value below. Checked = WAN mode, holding for that many ms.\n"
            "A checked box with 0 ms behaves exactly like unchecked."
        )
        net_layout.addWidget(self._hold_enabled_cb)

        self._hold_ms = QSpinBox()
        self._hold_ms.setRange(0, 2000)
        self._hold_ms.setValue(net_cfg.get_int("hold_ms", 0))
        self._hold_ms.setSuffix(" ms")
        self._hold_ms.setFixedWidth(90)
        self._hold_ms.setEnabled(self._hold_enabled_cb.isChecked())
        self._hold_enabled_cb.toggled.connect(self._hold_ms.setEnabled)
        self._hold_ms.setToolTip(
            "0 ms = LAN pass-through: lowest latency, but packets are\n"
            "forwarded exactly as received, with NO reordering at all --\n"
            "if the link has any real jitter, packets can arrive (and be\n"
            "forwarded) out of order, both within a stream and between\n"
            "streams. Verified 2026-09-02: with 50-150ms of injected\n"
            "jitter, 0ms mode showed up to ~94% sequence gaps and ~49%\n"
            "cross-stream ordering violations at the Receiver.\n\n"
            "> 0 ms = WAN mode: holds each stream's packets for this long\n"
            "after arrival, then releases in true DIFI-timestamp order --\n"
            "both within each stream and correctly interleaved across all\n"
            "streams, as long as this comfortably exceeds the actual\n"
            "delay+jitter on the link. Verified 2026-09-02: 200ms hold\n"
            "against 50-150ms of injected jitter gave 0% loss, 0 sequence\n"
            "gaps, 0% cross-stream ordering violations (2- and 3-stream).\n\n"
            "Only takes effect when \"Apply hold\" (left) is checked."
        )
        net_layout.addWidget(self._hold_ms)
        net_layout.addStretch()
        left_lay.addWidget(net_box)

        # Output
        out_box  = QGroupBox("Output  (to Receiver VM)")
        out_vlay = QVBoxLayout(out_box)

        out_cfg = app_config.Section(self._cfg, "Combiner.Output")
        ip_row = QHBoxLayout()
        ip_row.addWidget(QLabel("Receiver VM IP:"))
        self._dest_ip = QLineEdit(out_cfg.get_str("dest_ip", "127.0.0.1"))
        self._dest_ip.setPlaceholderText("e.g. 192.168.1.30")
        self._dest_ip.setFixedWidth(160)
        ip_row.addWidget(self._dest_ip)
        ip_row.addStretch()
        out_vlay.addLayout(ip_row)

        port_row = QHBoxLayout()
        port_row.addWidget(QLabel("Destination port:"))
        self._dest_port = QSpinBox()
        self._dest_port.setRange(1024, 65535)
        self._dest_port.setValue(out_cfg.get_int("dest_port", 50010))
        self._dest_port.setFixedWidth(110)
        port_row.addWidget(self._dest_port)
        port_row.addStretch()
        out_vlay.addLayout(port_row)
        left_lay.addWidget(out_box)

        # 2026-09-04: this used to be a "Throughput" box with two labels
        # (Capture in / Forward out) re-painted on every "stats" heartbeat
        # from the workers -- removed on request: during an actual run,
        # the ONLY thing that should still visibly update per-packet-ish
        # activity is each StreamRow's own LED (green/grey), which is
        # already the cheapest possible signal (a single stylesheet swap
        # driven by a ~1/s heartbeat, not a per-packet count). Numeric
        # counters that repaint their text on every tick cost real (if
        # small) Qt layout/repaint work for no operational benefit once
        # the LED already tells the operator "this stream is alive" --
        # and per-stream/session totals are always in the CSV logs
        # afterward if actually needed. See _poll_status()/
        # _handle_capture_status()/_handle_egress_status() -- they still
        # drain the status queues (required either way, or a worker
        # process would eventually block trying to put onto a full one),
        # they just no longer feed a live-repainting label.

        left_lay.addStretch()

        # Buttons: Listen | Forward | Stop | Save Config
        btn_row = QHBoxLayout()
        self._listen_btn  = QPushButton("▶  Listen")
        self._forward_btn = QPushButton("▶  Forward")
        self._stop_btn    = QPushButton("■  Stop")
        self._save_btn    = QPushButton("💾  Save Config")
        self._forward_btn.setEnabled(False)
        self._stop_btn.setEnabled(False)
        for btn in (self._listen_btn, self._forward_btn, self._stop_btn, self._save_btn):
            btn.setFixedHeight(36)
        self._listen_btn.clicked.connect(self._listen)
        self._forward_btn.clicked.connect(self._on_forward_clicked)
        self._stop_btn.clicked.connect(self._stop)
        # Explicit save independent of Listen -- lets an operator type the
        # Receiver VM IP/ports/etc. and write the ini for this VM without
        # actually starting the pipeline (e.g. pre-staging several VMs).
        self._save_btn.setToolTip(
            "Write the current settings to Combiner.ini next to this exe,\n"
            "without starting Listen -- useful for pre-configuring a VM."
        )
        self._save_btn.clicked.connect(self._save_config)
        btn_row.addWidget(self._listen_btn)
        btn_row.addWidget(self._forward_btn)
        btn_row.addWidget(self._stop_btn)
        btn_row.addWidget(self._save_btn)
        left_lay.addLayout(btn_row)

        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Ready — add streams and press Listen")

        # Drains the worker's status queue -- state-transition events only
        # (listening/forwarding/stopped/errors), not a live stats poll. This
        # is the one thing still ticking on the GUI thread, and it does
        # nothing but a non-blocking Queue read, so it doesn't reintroduce
        # the contention this whole split exists to avoid.
        self._status_timer = QTimer()
        self._status_timer.setInterval(200)
        self._status_timer.timeout.connect(self._poll_status)
        self._status_timer.start()

    # ── dynamic row management ─────────────────────────────────────────────

    def _add_stream_row(self, port: int = None, fwd_checked: bool = True):
        n = len(self._stream_rows) + 1
        if port is None:
            port = (max(r.port() for r in self._stream_rows) + 1) if self._stream_rows else 50001
        row = StreamRow(index=n, default_port=port, fwd_checked=fwd_checked)
        row.removed.connect(self._remove_stream_row)
        row.filter_changed.connect(self._on_filter_changed)
        row.set_locked(self._listening)
        self._rows_layout.insertWidget(self._rows_layout.count() - 1, row)
        self._stream_rows.append(row)
        # Adding a port while already Listening isn't supported yet in the
        # ring-pipeline design (would need to spawn one more ring_capture_main
        # process into the running session) -- Add stays available before
        # Listen only; StreamRow's own remove button is likewise only
        # meaningful pre-Listen now. Configure ports, then press Listen.

    def _remove_stream_row(self, row: StreamRow):
        if len(self._stream_rows) <= 1:
            return
        self._rows_layout.removeWidget(row)
        row.deleteLater()
        self._stream_rows.remove(row)
        for i, r in enumerate(self._stream_rows, start=1):
            r.set_index(i)

    def _set_locked(self, locked: bool):
        for row in self._stream_rows:
            row.set_locked(locked)
        self._add_btn.setEnabled(not locked)
        self._chunk.setEnabled(not locked)
        self._hold_enabled_cb.setEnabled(not locked)
        # Only re-enable the ms spinbox on unlock if the checkbox is
        # actually checked -- otherwise unlocking would re-enable it even
        # though it's currently meaningless (unchecked = hold is 0).
        self._hold_ms.setEnabled(not locked and self._hold_enabled_cb.isChecked())

    # ── IPC helpers ────────────────────────────────────────────────────────

    def _poll_status(self):
        h = self._ring_handles
        if h is None:
            return
        while True:
            try:
                msg = h["gui_status_q"].get_nowait()
            except _queue.Empty:
                break
            self._handle_capture_status(msg)
        if h["egress_status_q"] is not None:
            while True:
                try:
                    msg = h["egress_status_q"].get_nowait()
                except _queue.Empty:
                    break
                self._handle_egress_status(msg)

    def _handle_capture_status(self, msg: dict):
        # "stats" (a ~1/s heartbeat with cumulative counts) is intentionally
        # ignored here -- see the removed "Throughput" box's note in
        # _build_ui(). Only "stream_discovered" still touches the UI: it
        # sets the Stream ID text and lights a row's LED, but each fires at
        # most once per stream, not on a repeating timer.
        if msg.get("status") == "stream_discovered":
            self._stream_id_by_port[msg["port"]] = msg["stream_id"]
            for row in self._stream_rows:
                if row.port() == msg["port"]:
                    row.set_stream_id(msg["stream_id"])
                    row.set_active(True)

    def _handle_egress_status(self, msg: dict):
        pass   # nothing left to do with "stats" -- see _handle_capture_status

    # ── lifecycle ──────────────────────────────────────────────────────────

    def _effective_hold_ms(self) -> float:
        """The hold value actually sent to the worker -- 0 whenever "Apply
        hold" is unchecked, regardless of what's in the spinbox, so an
        operator never has to remember/re-zero a previously-tuned value
        just to switch to LAN mode."""
        return self._hold_ms.value() if self._hold_enabled_cb.isChecked() else 0.0

    def _save_config(self):
        """See app_config.py -- called on Listen/Forward/the Save Config
        button and on window close so the exe remembers its own last
        settings across restarts instead of the operator re-typing them
        every run. The Receiver VM IP/port here especially: a mismatch
        is exactly what silently sent a whole test run's data nowhere
        (caught only after the fact from CSV counts, see the 2026-09-03
        0ms-hold run)."""
        in_cfg = app_config.Section(self._cfg, "Combiner.Input")
        in_cfg.set("ports", ",".join(str(r.port()) for r in self._stream_rows))
        in_cfg.set("fwd_ports", ",".join(str(r.port()) for r in self._stream_rows if r.forwarded_port() is not None))
        in_cfg.set("chunk_size", self._chunk.value())

        net_cfg = app_config.Section(self._cfg, "Combiner.Network")
        net_cfg.set("hold_ms", self._hold_ms.value())
        net_cfg.set("hold_enabled", self._hold_enabled_cb.isChecked())

        out_cfg = app_config.Section(self._cfg, "Combiner.Output")
        out_cfg.set("dest_ip", self._dest_ip.text().strip())
        out_cfg.set("dest_port", self._dest_port.value())

        app_config.save("Combiner", self._cfg)

    def _listen(self):
        """Start N ring_capture_main processes, one per configured port --
        see ring_pipeline.py's own module docstring for the architecture.
        No egress yet (Forward starts that)."""
        if self._listening:
            return

        ports = [r.port() for r in self._stream_rows]
        if len(set(ports)) != len(ports):
            self._status.showMessage("Error: duplicate listen ports")
            return

        self._save_config()

        self._run_dir = make_run_dir("Combiner")
        self._ring_handles = start_ring_ingress(ports, self._run_dir)
        self._stream_id_by_port = {}

        self._listening = True
        self._set_locked(True)
        self._listen_btn.setEnabled(False)
        self._forward_btn.setEnabled(True)
        self._stop_btn.setEnabled(True)
        self._status.showMessage(f"Listening on ports {ports}  |  log: {self._run_dir}")

    def _on_forward_clicked(self):
        """Toggle forwarding on/off — Forward button handler."""
        if self._forwarding:
            self._stop_forward()
        else:
            self._forward()

    def _forward(self):
        """Start the single unified egress process -- see
        ring_pipeline.py's start_ring_egress()."""
        if not self._listening or self._forwarding:
            return

        dest_ip   = self._dest_ip.text().strip()
        dest_port = self._dest_port.value()
        self._save_config()
        target_delay_ms = self._effective_hold_ms()
        start_ring_egress(self._ring_handles, dest_ip, dest_port, target_delay_ms)
        self._apply_forward_filter()

        self._forwarding = True
        self._forward_btn.setText("■  Stop Fwd")
        self._dest_ip.setEnabled(False)
        self._dest_port.setEnabled(False)
        self._status.showMessage(f"Starting forward  →  {dest_ip}:{dest_port}  |  hold={target_delay_ms:.0f}ms")

    def _stop_forward(self):
        """Stop forwarding only — keep listening (ingress keeps running)."""
        if not self._forwarding:
            return
        stop_ring_egress(self._ring_handles)
        self._forwarding = False
        self._forward_btn.setText("▶  Forward")
        self._dest_ip.setEnabled(True)
        self._dest_port.setEnabled(True)
        self._status.showMessage(f"Forwarding stopped — still listening | log: {self._run_dir}")

    def _stop(self):
        """Stop everything and tear down the ring pipeline."""
        if not self._listening:
            return
        if self._ring_handles is not None:
            stop_ring_pipeline(self._ring_handles)
        self._ring_handles = None
        self._reset_ui_to_stopped()

    def _reset_ui_to_stopped(self):
        for row in self._stream_rows:
            row.set_stream_id(None)
            row.set_active(False)
        self._listening  = False
        self._forwarding = False
        self._run_dir    = None
        self._set_locked(False)
        self._listen_btn.setEnabled(True)
        self._forward_btn.setText("▶  Forward")
        self._forward_btn.setEnabled(False)
        self._stop_btn.setEnabled(False)
        self._dest_ip.setEnabled(True)
        self._dest_port.setEnabled(True)

    # ── forwarding filter ──────────────────────────────────────────────────

    def _on_filter_changed(self):
        """A Fwd checkbox toggled -- push the updated port filter to the
        single egress process."""
        if not self._forwarding:
            return
        self._apply_forward_filter()

    def _apply_forward_filter(self):
        h = self._ring_handles
        if h is None or h["egress_cmd_q"] is None:
            return
        h["egress_cmd_q"].put({"cmd": "set_forward_ports", "forward_ports": self._current_forward_ports()})

    def _current_forward_ports(self) -> list | None:
        """Return the ports to forward, or None to forward everything
        (all rows currently have Fwd checked)."""
        all_checked = all(r.forwarded_port() is not None for r in self._stream_rows)
        if all_checked:
            return None
        return [r.forwarded_port() for r in self._stream_rows if r.forwarded_port() is not None]

    def closeEvent(self, event):
        self._stop()
        self._save_config()
        event.accept()
        # See gil_friendly_exec.py: app.lastWindowClosed alone was NOT
        # reliably ending the polling loop -- confirmed as a real bug (the
        # process kept running as an orphaned background task after the
        # window closed, still bound to its UDP ports via SO_REUSEADDR,
        # silently competing with the next run for the same packets and
        # CPU). Calling request_stop() directly here, at the one place a
        # close is unambiguously happening, doesn't depend on that signal
        # firing correctly under manual polling at all.
        request_stop(QApplication.instance())


def main():
    # Required on Windows before any other multiprocessing call in a
    # PyInstaller --onefile build -- without it, spawning a ring_capture_main/
    # ring_egress_main child process re-executes this whole frozen exe from
    # the top instead of running that entry point, which recurses into
    # spawning another GUI window instead of doing the child's job.
    multiprocessing.freeze_support()

    from logging_setup import setup_frozen_file_logging
    log_path = setup_frozen_file_logging("Combiner")

    app = QApplication(sys.argv)
    win = PacketizerWindow()
    if log_path:
        win._status.showMessage(f"Logging to {log_path}")
    win.show()
    # See gil_friendly_exec.py: app.exec()'s native Windows event loop was
    # measured, via direct A/B test on this exact app, to starve every
    # other thread in this process (0.0% packet loss with this polling
    # loop vs 41-46% with app.exec(), everything else identical).
    app.lastWindowClosed.connect(lambda: request_stop(app))
    sys.exit(run_gil_friendly(app))


if __name__ == "__main__":
    main()
