"""
receiver_app.py
---------------
DIFI Aggregator — Receiver GUI.

Runs on the Receiver VM.
Listens for the unified DIFI stream and displays a live FFT spectrum.
"""

import os
import sys
import time

if not getattr(sys, 'frozen', False):
    _src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _src not in sys.path:
        sys.path.insert(0, _src)

import numpy as np

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QDoubleSpinBox, QPushButton,
    QGroupBox, QStatusBar, QSpinBox, QSplitter,
)
import pyqtgraph as pg

from modules.receiver import DifiReceiver
from ui.freq_input    import FreqInput
from pipeline_logger  import make_run_dir, wall_clock_str, AsyncPacketLogger as PacketLogger
from gil_friendly_exec import run_gil_friendly, request_stop
import app_config


_STREAM_COLORS = [
    (100, 220, 255),  # stream index 0 — cyan
    (255, 170,  50),  # stream index 1 — orange
    (100, 255, 100),  # stream index 2 — lime
    (255, 100, 255),  # stream index 3+ — magenta
]

# How long a stream may go quiet before the UI calls it inactive/stopped.
# Pure display debounce, not a data-path timeout — see packetizer_app.py's
# identical constant for the full rationale.
STREAM_STALE_S = 1.0


def _stream_color(sid: int):
    return pg.mkPen(_STREAM_COLORS[(sid - 1) % len(_STREAM_COLORS)], width=1)


_HANN_CACHE: dict = {}   # seg_len -> (window, sum(window)) -- recomputing per call/stream/tick was pure waste


def _hann(seg_len: int):
    cached = _HANN_CACHE.get(seg_len)
    if cached is None:
        w = np.hanning(seg_len)
        cached = (w, float(np.sum(w)))
        _HANN_CACHE[seg_len] = cached
    return cached


def _stream_fft(iq, ctx, seg_len: int = 1024):
    """Single-window Hann FFT magnitude spectrum for one IQ buffer."""
    w, w_amp = _hann(seg_len)
    n     = min(len(iq), seg_len)
    seg   = iq[-n:].copy()
    if n < seg_len:
        seg = np.pad(seg, (0, seg_len - n))
    X      = np.fft.fftshift(np.fft.fft(seg * w))
    mag_db = 20.0 * np.log10(np.abs(X) / w_amp + 1e-7)
    freqs  = np.fft.fftshift(
        np.fft.fftfreq(seg_len, d=1.0 / ctx.sample_rate_hz)
    ) + ctx.rf_ref_freq_hz
    return freqs, mag_db


class ReceiverStreamRow(QWidget):
    """
    One discovered-stream status card, stacked over three lines:
        ● 0x00000001                          stopped
          fs 20.000 MHz
          RF 490.000 MHz

    Read-only — the receiver only auto-discovers streams from incoming
    Context packets, there's nothing here for the user to configure.
    The LED and status text are the whole point: a stream that stops
    sending must visibly go from "live" to "stopped", not just sit in a
    flat list forever looking identical to an active one.

    Two lines, not a column table: the left panel this lives in is capped
    at 300px, and "0x00000001 | 20.000 MHz | 490.000 MHz | stopped" simply
    doesn't fit in that width at any reasonable font size -- cramming it
    sideways just moves the collision around instead of fixing it.
    """

    def __init__(self, sid: int, parent=None):
        super().__init__(parent)
        self.sid = sid
        outer = QVBoxLayout(self)
        outer.setContentsMargins(2, 3, 2, 3)
        outer.setSpacing(1)

        top = QHBoxLayout()
        top.setSpacing(6)
        self._led = QLabel("●")
        self._led.setFixedWidth(14)
        top.addWidget(self._led)
        self._sid_lbl = QLabel(f"0x{sid:08X}")
        top.addWidget(self._sid_lbl)
        top.addStretch()
        self._status_lbl = QLabel("")
        self._status_lbl.setStyleSheet("color: #cc5555; font-size: 11px;")
        top.addWidget(self._status_lbl)
        outer.addLayout(top)

        def _detail_line():
            row = QHBoxLayout()
            row.setContentsMargins(20, 0, 0, 0)
            lbl = QLabel("—")
            lbl.setStyleSheet("color: #888888; font-size: 11px;")
            row.addWidget(lbl)
            row.addStretch()
            outer.addLayout(row)
            return lbl

        self._fs_lbl = _detail_line()
        self._rf_lbl = _detail_line()

    def update_state(self, ctx, active: bool):
        if ctx is not None:
            self._fs_lbl.setText(f"fs {ctx.sample_rate_hz / 1e6:.3f} MHz")
            self._rf_lbl.setText(f"RF {ctx.rf_ref_freq_hz / 1e6:.3f} MHz")
        color = "#00cc44" if active else "#888888"
        self._led.setStyleSheet(f"color: {color}; font-size: 16px;")
        self._sid_lbl.setStyleSheet("color: #dddddd;" if active else "color: #888888;")
        self._status_lbl.setText("" if active else "stopped")


class ReceiverWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("DIFI Receiver")
        self.setMinimumSize(1050, 620)
        self._receiver = None
        self._running  = False
        self._recv_log = None
        # See app_config.py -- loaded here (before _build_ui() creates any
        # widget) so the bind port can seed its initial value from the
        # last-saved settings instead of a hardcoded default.
        self._cfg = app_config.load("Receiver")
        self._build_ui()

        # Opt-in, env-var-gated autostart -- see transmitter_app.py's
        # identical hook for why. No effect unless DIFI_AUTOSTART=1.
        if os.environ.get("DIFI_AUTOSTART") == "1":
            QTimer.singleShot(300, self._autostart_from_env)

    def _autostart_from_env(self):
        env = os.environ
        if env.get("DIFI_PORT"):
            self._port.setValue(int(env["DIFI_PORT"]))
        self._start_btn.click()
        if env.get("DIFI_AUTOSTOP_AFTER_S"):
            QTimer.singleShot(int(float(env["DIFI_AUTOSTOP_AFTER_S"]) * 1000), self.close)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # ── left panel ──
        left = QWidget()
        left.setMaximumWidth(300)
        left_layout = QVBoxLayout(left)
        left_layout.setSpacing(8)

        # Listen
        listen_box  = QGroupBox("Listen")
        listen_grid = QGridLayout(listen_box)
        net_cfg = app_config.Section(self._cfg, "Receiver.Network")
        listen_grid.addWidget(QLabel("Bind port:"), 0, 0)
        self._port = QSpinBox()
        self._port.setRange(1024, 65535)
        self._port.setValue(net_cfg.get_int("port", 50010))
        self._port.setFixedWidth(110)
        port_w = QWidget()
        port_l = QHBoxLayout(port_w)
        port_l.setContentsMargins(0, 0, 0, 0)
        port_l.addWidget(self._port)
        port_l.addStretch()
        listen_grid.addWidget(port_w, 0, 1)
        left_layout.addWidget(listen_box)

        # Display controls
        disp_box  = QGroupBox("Display")
        disp_grid = QGridLayout(disp_box)
        disp_cfg  = app_config.Section(self._cfg, "Receiver.Display")

        disp_grid.addWidget(QLabel("Center:"), 0, 0)
        self._center = FreqInput(default_hz=disp_cfg.get_float("center_hz", 2.5e6))
        disp_grid.addWidget(self._center, 0, 1)

        disp_grid.addWidget(QLabel("Span:"), 1, 0)
        self._span = FreqInput(default_hz=disp_cfg.get_float("span_hz", 5e6))
        disp_grid.addWidget(self._span, 1, 1)

        disp_grid.addWidget(QLabel("Amplitude:"), 2, 0)
        self._amp_top = QDoubleSpinBox()
        self._amp_top.setRange(-200, 50)
        self._amp_top.setDecimals(1)
        self._amp_top.setSingleStep(10)
        self._amp_top.setValue(disp_cfg.get_float("amp_top_db", -10))
        self._amp_top.setSuffix(" dB")
        disp_grid.addWidget(self._amp_top, 2, 1)

        disp_grid.addWidget(QLabel("dB / div:"), 3, 0)
        self._db_div = QDoubleSpinBox()
        self._db_div.setRange(1, 100)
        self._db_div.setDecimals(1)
        self._db_div.setValue(disp_cfg.get_float("db_div_db", 10))
        self._db_div.setSuffix(" dB")
        disp_grid.addWidget(self._db_div, 3, 1)

        auto_btn = QPushButton("Auto")
        auto_btn.clicked.connect(self._auto_display)
        disp_grid.addWidget(auto_btn, 4, 0, 1, 2)

        self._center.changed.connect(self._apply_range)
        self._span.changed.connect(self._apply_range)
        self._amp_top.valueChanged.connect(self._apply_range)
        self._db_div.valueChanged.connect(self._apply_range)

        left_layout.addWidget(disp_box)

        # Statistics
        stats_box  = QGroupBox("Statistics")
        stats_grid = QGridLayout(stats_box)
        self._lbl_data  = QLabel("0")
        self._lbl_ctx   = QLabel("0")
        self._lbl_errs  = QLabel("0")
        self._lbl_rate  = QLabel("—")
        stats_grid.addWidget(QLabel("Data packets:"),   0, 0)
        stats_grid.addWidget(self._lbl_data,            0, 1)
        stats_grid.addWidget(QLabel("Context packets:"), 1, 0)
        stats_grid.addWidget(self._lbl_ctx,             1, 1)
        stats_grid.addWidget(QLabel("Parse errors:"),   2, 0)
        stats_grid.addWidget(self._lbl_errs,            2, 1)
        stats_grid.addWidget(QLabel("Rate:"),           3, 0)
        stats_grid.addWidget(self._lbl_rate,            3, 1)
        left_layout.addWidget(stats_box)
        self._last_rate_count = 0
        self._last_rate_bytes = 0
        self._last_rate_time  = None
        self._last_dropped_seen = 0    # rx.raw_q_dropped as of the last _tick() -- see _tick()'s own note
        self._drop_warning_active = False

        # Streams — one card per discovered stream ID, LED shows live/stopped
        streams_box  = QGroupBox("Streams")
        streams_vlay = QVBoxLayout(streams_box)
        streams_vlay.setSpacing(4)
        self._stream_rows_container = QWidget()
        self._stream_rows_layout    = QVBoxLayout(self._stream_rows_container)
        self._stream_rows_layout.setSpacing(6)
        self._stream_rows_layout.setContentsMargins(0, 0, 0, 0)
        streams_vlay.addWidget(self._stream_rows_container)
        left_layout.addWidget(streams_box)
        self._stream_rows: dict = {}   # stream_id -> ReceiverStreamRow

        left_layout.addStretch()

        btn_row = QHBoxLayout()
        self._start_btn = QPushButton("▶  Start")
        self._stop_btn  = QPushButton("■  Stop")
        self._save_btn  = QPushButton("💾  Save Config")
        self._stop_btn.setEnabled(False)
        self._start_btn.setFixedHeight(36)
        self._stop_btn.setFixedHeight(36)
        self._save_btn.setFixedHeight(36)
        self._start_btn.clicked.connect(self._start)
        self._stop_btn.clicked.connect(self._stop)
        # Explicit save independent of Start -- lets an operator set the
        # bind port and write the ini for this VM without actually
        # starting the receiver (e.g. pre-staging VMs).
        self._save_btn.setToolTip(
            "Write the current settings to SysConfig.ini next to this exe,\n"
            "without starting Start -- useful for pre-configuring a VM."
        )
        self._save_btn.clicked.connect(self._save_config)
        btn_row.addWidget(self._start_btn)
        btn_row.addWidget(self._stop_btn)
        btn_row.addWidget(self._save_btn)
        left_layout.addLayout(btn_row)

        splitter.addWidget(left)

        # ── right panel — spectrum ──
        right        = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 4, 4, 4)

        self._plot = pg.PlotWidget()
        self._plot.setLabel("bottom", "Frequency", units="Hz")
        self._plot.setLabel("left",   "Magnitude", units="dB")
        self._plot.showGrid(x=True, y=True, alpha=0.3)
        self._plot.enableAutoRange(axis="xy", enable=False)
        self._plot.getPlotItem().getViewBox().enableAutoRange(enable=False)
        # Standard PyQtGraph real-time-plot settings -- cheapen every
        # setData() call on the 100ms tick (skip rendering points outside
        # the current view, and thin points beyond the pixel resolution
        # of the plot itself, which a 1024-point spectrum can exceed on a
        # narrow window). Small on their own, but stack with the snapshot-
        # copy and Hann-window fixes to shrink the UI thread's per-tick
        # budget overall -- the goal is fewer/shorter windows where a
        # loaded system can stall this thread and freeze the display.
        self._plot.setClipToView(True)
        self._plot.setDownsampling(mode="peak", auto=True)
        self._plot.setYRange(-110, -10, padding=0)
        self._plot.setXRange(0, 5e6, padding=0)
        self._curves: dict = {}  # stream_id → PlotDataItem
        self._y_range_applied = False

        self._ref_line = pg.InfiniteLine(
            angle=0, movable=False,
            pen=pg.mkPen("y", width=1, style=Qt.PenStyle.DashLine),
        )
        self._plot.addItem(self._ref_line)

        # Keep display spinboxes in sync when user pans/zooms with mouse
        self._plot.getPlotItem().getViewBox().sigRangeChanged.connect(
            lambda vb, ranges: self._sync_viewport_to_spinboxes(ranges)
        )

        right_layout.addWidget(self._plot)
        splitter.addWidget(right)
        splitter.setSizes([290, 760])

        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Ready — press Start to begin receiving")

        self._timer = QTimer()
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._tick)

    # ── lifecycle ──────────────────────────────────────────────────────────

    def _save_config(self):
        """See app_config.py -- called on Start/the Save Config button and
        on window close so the exe remembers its own last settings across
        restarts instead of the operator re-typing them every run."""
        net_cfg = app_config.Section(self._cfg, "Receiver.Network")
        net_cfg.set("port", self._port.value())

        disp_cfg = app_config.Section(self._cfg, "Receiver.Display")
        disp_cfg.set("center_hz", self._center.value_hz())
        disp_cfg.set("span_hz", self._span.value_hz())
        disp_cfg.set("amp_top_db", self._amp_top.value())
        disp_cfg.set("db_div_db", self._db_div.value())

        app_config.save("Receiver", self._cfg)

    def _start(self):
        if self._running:
            return
        self._save_config()
        port           = self._port.value()
        self._plot.setTitle(f"Receiver Input — port {port}")
        run_dir        = make_run_dir("Receiver")
        self._recv_log = PacketLogger(
            run_dir, "data_received.csv",
            # 2026-09-04: rf_ref_hz/sample_rate_hz/bandwidth_hz added so
            # this log alone can confirm DIFI Context info (what a
            # Context packet actually carries -- RF reference frequency,
            # sample rate, bandwidth) survived the Combiner correctly,
            # without needing a separate raw-wire sniffer script each
            # time. Populated on CONTEXT rows; blank on DATA rows (a DATA
            # packet doesn't carry these fields itself).
            # active: blank for DATA/CONTEXT rows; True/False on a STATUS
            # row -- logged only on a TRANSITION (see _tick()) so this
            # directly answers "when did the GUI show this stream as
            # disconnected" against the same timeline as the packet rows,
            # instead of needing a screenshot to know when that happened.
            ["wall_clock", "stream_id", "pkt_type", "seq", "difi_ts_int",
             "difi_ts_frac", "samples", "seq_gap", "first_i", "first_q",
             "rf_ref_hz", "sample_rate_hz", "bandwidth_hz", "active"],
        )
        self._stream_active_logged: dict = {}   # stream_id -> last active state WRITTEN to the log
        self._receiver = DifiReceiver(port=port, packet_logger=self._recv_log)
        self._receiver.start()
        self._running = True
        self._y_range_applied = False
        self._last_rate_count = 0
        self._last_rate_bytes = 0
        self._last_rate_time  = time.monotonic()
        self._last_dropped_seen = 0
        self._drop_warning_active = False
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._timer.start()
        self._apply_range()
        # Auto-center the display on whatever RF frequency the incoming
        # signal actually reports, instead of leaving it at the fixed
        # 2.5 MHz default -- a real signal well outside that default span
        # (any RF far from a few MHz) would otherwise show nothing at all
        # unless the user manually pressed Auto first. _auto_display()
        # already retries every 500ms until a context packet arrives, so
        # calling it here is safe even before any data has shown up yet.
        self._auto_display()
        self._status.showMessage(f"Listening on 0.0.0.0:{port} | log: {run_dir}")

    def _stop(self):
        if not self._running:
            return
        self._timer.stop()
        self._receiver.stop()
        self._receiver = None
        if self._recv_log:
            self._recv_log.close()
            self._recv_log = None
        self._running  = False
        for c in self._curves.values():
            c.setData([], [])
        self._curves.clear()
        for row in self._stream_rows.values():
            self._stream_rows_layout.removeWidget(row)
            row.deleteLater()
        self._stream_rows.clear()
        self._lbl_rate.setText("—")
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._status.showMessage("Stopped")

    # ── update loop ────────────────────────────────────────────────────────

    def _tick(self):
        if not self._receiver:
            return

        rx = self._receiver

        # Always update counters so the user can see data flowing
        self._lbl_data.setText(f"{rx.data_received:,}")
        self._lbl_ctx.setText(f"{rx.context_received:,}")
        self._lbl_errs.setText(str(rx.parse_errors))

        now = time.monotonic()
        dt  = now - self._last_rate_time if self._last_rate_time else 0.0
        if dt > 0:
            rate = (rx.data_received - self._last_rate_count) / dt
            mbps = (rx.bytes_received - self._last_rate_bytes) / dt / 1e6
            self._lbl_rate.setText(f"{rate:,.0f} pkt/s  |  {mbps:,.2f} MB/s")
        self._last_rate_count = rx.data_received
        self._last_rate_bytes = rx.bytes_received
        self._last_rate_time  = now

        snaps = rx.get_stream_snapshots(tail_samples=1024)   # matches _stream_fft's seg_len

        # Per-stream status rows — LED + fs/RF, explicitly marked "stopped"
        # once a stream goes stale instead of lingering in a flat list
        # indistinguishable from a still-live one.
        last_seen     = rx.stream_last_seen()
        now           = time.monotonic()
        stale_cutoff  = now - STREAM_STALE_S   # matches the spectrum-curve cutoff below
        remove_cutoff = now - 15.0             # stopped long enough to drop the row entirely

        for sid, (_iq, ctx_s) in snaps.items():
            last = last_seen.get(sid, 0)
            if last < remove_cutoff:
                continue
            if sid not in self._stream_rows:
                row = ReceiverStreamRow(sid)
                self._stream_rows_layout.addWidget(row)
                self._stream_rows[sid] = row
            active = last >= stale_cutoff
            self._stream_rows[sid].update_state(ctx_s, active=active)
            # 2026-09-04: log a STATUS row on every active<->stale
            # TRANSITION (not every tick -- that would be 10/s of noise)
            # so "when did this stream show as disconnected" is answered
            # directly from the CSV, on the same timeline as the DATA/
            # CONTEXT rows, instead of only from a screenshot's timestamp.
            if self._recv_log is not None and self._stream_active_logged.get(sid) != active:
                self._stream_active_logged[sid] = active
                self._recv_log.log(
                    wall_clock_str(), f"0x{sid:08X}", "STATUS", "",
                    "", "", "", "", "", "", "", "", "", active,
                )

        for sid in list(self._stream_rows):
            last = last_seen.get(sid, 0)
            if last < remove_cutoff:
                row = self._stream_rows.pop(sid)
                self._stream_rows_layout.removeWidget(row)
                row.deleteLater()

        # Wait for at least one context before drawing
        if not snaps or all(c is None for _, c in snaps.values()):
            self._status.showMessage(
                f"Listening | data={rx.data_received:,} | waiting for context packet…"
            )
            return

        # Per-stream spectra — one curve per stream ID
        for sid, (iq, ctx_s) in snaps.items():
            if ctx_s is None or len(iq) == 0:
                continue
            if last_seen.get(sid, 0) < stale_cutoff:
                if sid in self._curves:
                    self._curves[sid].setData([], [])
                continue
            if sid not in self._curves:
                self._curves[sid] = self._plot.plot([], [], pen=_stream_color(sid))
            f, m = _stream_fft(iq, ctx_s)
            self._curves[sid].setData(f, m)

        if not self._y_range_applied:
            self._apply_range()
            self._y_range_applied = True

        n_streams = len([c for _, c in snaps.values() if c is not None])
        ctx = rx.context
        fs_str = f"{ctx.sample_rate_hz / 1e6:.3f} MHz" if ctx else "?"

        # 2026-09-05: surfaces rx.raw_q_dropped -- confirmed directly this
        # matters: a real Combiner->Receiver run measured ~12% of arriving
        # packets never counted here, entirely because THIS process's own
        # processing thread fell behind (DifiDataPacket.from_bytes() is
        # the ~44x-more-expensive full parse, unavoidable here since the
        # rolling display buffer needs real decoded IQ, not just header
        # fields -- see DifiReceiver's own note). The Combiner's own
        # ingress/egress counters stayed ~98-100% healthy the entire time,
        # so its equivalent warning correctly never fired -- it has no way
        # to know this is happening, since UDP gives it no feedback from
        # this side at all. Without this, that loss looked like nothing
        # was wrong anywhere.
        dropped_now = rx.raw_q_dropped
        d_dropped = dropped_now - self._last_dropped_seen
        self._last_dropped_seen = dropped_now
        if d_dropped > 0:
            self._drop_warning_active = True
            self._status.showMessage(
                f"WARNING: this Receiver is falling behind the arrival rate -- "
                f"{dropped_now:,} packet(s) dropped so far ({d_dropped:,} in the "
                f"last {self._timer.interval()/1000:.1f}s). The display/parsing work "
                f"can't keep up with incoming data."
            )
            return
        if self._drop_warning_active:
            self._drop_warning_active = False
            self._status.showMessage(
                f"Listening -- recovered, no further drops (total this session: {dropped_now:,})"
            )
            return

        self._status.showMessage(
            f"Listening | data={rx.data_received:,} | "
            f"streams={n_streams} | fs={fs_str}"
        )

    # ── helpers ────────────────────────────────────────────────────────────

    def _apply_range(self):
        center  = self._center.value_hz()
        span    = self._span.value_hz()
        amp_top = self._amp_top.value()
        db_div  = self._db_div.value()
        self._plot.setXRange(center - span / 2, center + span / 2, padding=0)
        self._plot.setYRange(amp_top - db_div * 10, amp_top, padding=0)
        self._ref_line.setValue(amp_top)

    def _auto_display(self):
        if not self._receiver:
            return
        snaps = self._receiver.get_stream_snapshots(tail_samples=0)   # only context objects are used here
        contexts = [c for _, c in snaps.values() if c is not None]
        if not contexts:
            QTimer.singleShot(500, self._auto_display)
            return
        # Center on the mean RF of all active streams; span covers all of them
        rf_refs = [c.rf_ref_freq_hz for c in contexts]
        fs_vals = [c.sample_rate_hz for c in contexts]
        center  = sum(rf_refs) / len(rf_refs)
        span    = (max(rf_refs) - min(rf_refs)) + max(fs_vals)
        span    = max(span, max(fs_vals))
        self._center.set_hz(center)
        self._span.set_hz(span)
        self._amp_top.setValue(-10.0)
        self._db_div.setValue(10.0)
        self._apply_range()

    def _sync_viewport_to_spinboxes(self, _ranges=None):
        """Keep display spinboxes in sync when user pans/zooms the plot."""
        [[x_lo, x_hi], [y_lo, y_hi]] = (
            self._plot.getPlotItem().getViewBox().viewRange()
        )
        if x_hi > x_lo:
            self._center.set_hz((x_lo + x_hi) / 2.0)
            self._span.set_hz(x_hi - x_lo)
        if y_hi > y_lo:
            self._amp_top.blockSignals(True)
            self._db_div.blockSignals(True)
            self._amp_top.setValue(y_hi)
            self._db_div.setValue((y_hi - y_lo) / 10.0)
            self._amp_top.blockSignals(False)
            self._db_div.blockSignals(False)
            self._ref_line.setValue(y_hi)

    def closeEvent(self, event):
        self._stop()
        self._save_config()
        event.accept()
        # See gil_friendly_exec.py / packetizer_app.py's identical fix --
        # app.lastWindowClosed alone was not reliably ending the polling
        # loop, leaving an orphaned background process after the window
        # closed. Calling request_stop() directly here doesn't depend on
        # that signal firing correctly under manual polling.
        request_stop(QApplication.instance())


def main():
    from logging_setup import setup_frozen_file_logging
    log_path = setup_frozen_file_logging("Receiver")

    # See packetizer_app.py's identical fix.
    pg.setConfigOptions(antialias=False)
    app = QApplication(sys.argv)
    win = ReceiverWindow()
    if log_path:
        win._status.showMessage(f"Logging to {log_path}")
    win.show()
    # See gil_friendly_exec.py -- app.exec()'s native Windows event loop
    # was measured to starve every other thread in the process; this
    # polling loop avoids that.
    app.lastWindowClosed.connect(lambda: request_stop(app))
    sys.exit(run_gil_friendly(app))


if __name__ == "__main__":
    main()
