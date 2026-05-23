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


_STREAM_COLORS = [
    (100, 220, 255),  # stream index 0 — cyan
    (255, 170,  50),  # stream index 1 — orange
    (100, 255, 100),  # stream index 2 — lime
    (255, 100, 255),  # stream index 3+ — magenta
]


def _stream_color(sid: int):
    return pg.mkPen(_STREAM_COLORS[(sid - 1) % len(_STREAM_COLORS)], width=1)


def _stream_fft(iq, ctx, seg_len: int = 1024):
    """Single-window Hann FFT magnitude spectrum for one IQ buffer."""
    w     = np.hanning(seg_len)
    w_amp = float(np.sum(w))
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


class ReceiverWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("DIFI Receiver")
        self.setMinimumSize(1050, 620)
        self._receiver = None
        self._running  = False
        self._build_ui()

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
        listen_grid.addWidget(QLabel("Bind port:"), 0, 0)
        self._port = QSpinBox()
        self._port.setRange(1024, 65535)
        self._port.setValue(50010)
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

        disp_grid.addWidget(QLabel("Center:"), 0, 0)
        self._center = FreqInput(default_hz=2.5e6)
        disp_grid.addWidget(self._center, 0, 1)

        disp_grid.addWidget(QLabel("Span:"), 1, 0)
        self._span = FreqInput(default_hz=5e6)
        disp_grid.addWidget(self._span, 1, 1)

        disp_grid.addWidget(QLabel("Amplitude:"), 2, 0)
        self._amp_top = QDoubleSpinBox()
        self._amp_top.setRange(-200, 50)
        self._amp_top.setDecimals(1)
        self._amp_top.setSingleStep(10)
        self._amp_top.setValue(-10)
        self._amp_top.setSuffix(" dB")
        disp_grid.addWidget(self._amp_top, 2, 1)

        disp_grid.addWidget(QLabel("dB / div:"), 3, 0)
        self._db_div = QDoubleSpinBox()
        self._db_div.setRange(1, 100)
        self._db_div.setDecimals(1)
        self._db_div.setValue(10)
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
        self._lbl_fs    = QLabel("—")
        self._lbl_rf    = QLabel("—")
        self._lbl_sid   = QLabel("—")
        stats_grid.addWidget(QLabel("Data packets:"),   0, 0)
        stats_grid.addWidget(self._lbl_data,            0, 1)
        stats_grid.addWidget(QLabel("Context packets:"), 1, 0)
        stats_grid.addWidget(self._lbl_ctx,             1, 1)
        stats_grid.addWidget(QLabel("Parse errors:"),   2, 0)
        stats_grid.addWidget(self._lbl_errs,            2, 1)
        stats_grid.addWidget(QLabel("Sample rate:"),    3, 0)
        stats_grid.addWidget(self._lbl_fs,              3, 1)
        stats_grid.addWidget(QLabel("RF reference:"),   4, 0)
        stats_grid.addWidget(self._lbl_rf,              4, 1)
        stats_grid.addWidget(QLabel("Stream ID:"),      5, 0)
        stats_grid.addWidget(self._lbl_sid,             5, 1)
        left_layout.addWidget(stats_box)

        left_layout.addStretch()

        btn_row = QHBoxLayout()
        self._start_btn = QPushButton("▶  Start")
        self._stop_btn  = QPushButton("■  Stop")
        self._stop_btn.setEnabled(False)
        self._start_btn.setFixedHeight(36)
        self._stop_btn.setFixedHeight(36)
        self._start_btn.clicked.connect(self._start)
        self._stop_btn.clicked.connect(self._stop)
        btn_row.addWidget(self._start_btn)
        btn_row.addWidget(self._stop_btn)
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

    def _start(self):
        if self._running:
            return
        port           = self._port.value()
        self._receiver = DifiReceiver(port=port)
        self._receiver.start()
        self._running = True
        self._y_range_applied = False
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._timer.start()
        self._apply_range()
        self._status.showMessage(f"Listening on 0.0.0.0:{port}")

    def _stop(self):
        if not self._running:
            return
        self._timer.stop()
        self._receiver.stop()
        self._receiver = None
        self._running  = False
        for c in self._curves.values():
            c.setData([], [])
        self._curves.clear()
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

        snaps = rx.get_stream_snapshots()

        # Pick any available context for stats labels
        ctx = rx.context
        if ctx:
            self._lbl_fs.setText(f"{ctx.sample_rate_hz / 1e6:.3f} MHz")
            self._lbl_rf.setText(f"{ctx.rf_ref_freq_hz / 1e6:.3f} MHz")

        # Show all active stream IDs
        if snaps:
            sid_str = ", ".join(f"0x{s:08X}" for s in snaps)
            self._lbl_sid.setText(sid_str)

        # Wait for at least one context before drawing
        if not snaps or all(c is None for _, c in snaps.values()):
            self._status.showMessage(
                f"Listening | data={rx.data_received:,} | waiting for context packet…"
            )
            return

        # Per-stream spectra — one curve per stream ID
        last_seen    = rx.stream_last_seen()
        stale_cutoff = time.monotonic() - 3.0
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
        fs_str = f"{ctx.sample_rate_hz / 1e6:.3f} MHz" if ctx else "?"
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
        snaps = self._receiver.get_stream_snapshots()
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
        event.accept()


def main():
    pg.setConfigOptions(antialias=True)
    app = QApplication(sys.argv)
    win = ReceiverWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
