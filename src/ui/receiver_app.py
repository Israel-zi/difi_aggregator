"""
receiver_app.py
---------------
DIFI Aggregator — Receiver GUI.

Runs on the Receiver VM.
Listens for the unified DIFI stream and displays a live FFT spectrum.
"""

import os
import sys

_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src not in sys.path:
    sys.path.insert(0, _src)

import numpy as np

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QDoubleSpinBox, QComboBox, QPushButton,
    QGroupBox, QStatusBar, QSpinBox, QSplitter,
)
import pyqtgraph as pg

from modules.receiver import DifiReceiver

UNIT_MUL    = {"Hz": 1.0, "kHz": 1_000.0, "MHz": 1_000_000.0, "GHz": 1_000_000_000.0}
UNIT_LABELS = ["Hz", "kHz", "MHz", "GHz"]


class FreqInput(QWidget):
    changed = Signal()

    def __init__(self, default_hz: float = 1e6, parent=None):
        super().__init__(parent)
        if default_hz >= 1e9:
            unit, val = "GHz", default_hz / 1e9
        elif default_hz >= 1e6:
            unit, val = "MHz", default_hz / 1e6
        elif default_hz >= 1e3:
            unit, val = "kHz", default_hz / 1e3
        else:
            unit, val = "Hz", default_hz

        self._spin = QDoubleSpinBox()
        self._spin.setDecimals(3)
        self._spin.setRange(0.0, 999_999.999)
        self._spin.setValue(val)
        self._spin.setFixedWidth(115)

        self._unit = QComboBox()
        self._unit.addItems(UNIT_LABELS)
        self._unit.setCurrentText(unit)
        self._unit.setFixedWidth(70)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._spin)
        lay.addWidget(self._unit)

        self._spin.valueChanged.connect(self.changed)
        self._unit.currentIndexChanged.connect(self.changed)

    def value_hz(self) -> float:
        return self._spin.value() * UNIT_MUL[self._unit.currentText()]


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

        splitter = QSplitter(Qt.Horizontal)
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
        self._port.setFixedWidth(90)
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
        self._plot.setYRange(-110, -10, padding=0)
        self._plot.setXRange(0, 5e6, padding=0)
        self._curve = self._plot.plot([], [], pen=pg.mkPen("c", width=1))

        self._ref_line = pg.InfiniteLine(
            angle=0, movable=False,
            pen=pg.mkPen("y", width=1, style=Qt.DashLine),
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
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._timer.start()
        self._apply_range()   # immediately restore Y range and spinbox-tracked X range
        self._status.showMessage(f"Listening on 0.0.0.0:{port}")

        QTimer.singleShot(600, self._auto_display)

    def _stop(self):
        if not self._running:
            return
        self._timer.stop()
        self._receiver.stop()
        self._receiver = None
        self._running  = False
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._status.showMessage("Stopped")

    # ── update loop ────────────────────────────────────────────────────────

    def _tick(self):
        if not self._receiver:
            return

        rx  = self._receiver
        ctx = rx.context

        # Always update counters so the user can see data flowing
        self._lbl_data.setText(f"{rx.data_received:,}")
        self._lbl_ctx.setText(f"{rx.context_received:,}")
        self._lbl_errs.setText(str(rx.parse_errors))

        if ctx:
            self._lbl_fs.setText(f"{ctx.sample_rate_hz / 1e6:.3f} MHz")
            self._lbl_rf.setText(f"{ctx.rf_ref_freq_hz / 1e6:.3f} MHz")
            self._lbl_sid.setText(f"0x{ctx.stream_id:08X}")

        # Wait for context before drawing — RF shift is unknown without it
        if ctx is None:
            self._status.showMessage(
                f"Listening | data={rx.data_received:,} | waiting for context packet…"
            )
            return

        rf_ref = ctx.rf_ref_freq_hz
        fs     = ctx.sample_rate_hz

        iq = rx.get_iq_snapshot()
        n  = len(iq)
        if n == 0:
            return

        # FFT — full complex IQ spectrum (-fs/2 to +fs/2) shifted to RF
        window = np.hanning(n)
        X      = np.fft.fftshift(np.fft.fft(iq * window))
        freqs  = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / fs)) + rf_ref
        mag_db = 20 * np.log10(np.abs(X) / n + 1e-12)
        self._curve.setData(freqs, mag_db)

        self._status.showMessage(
            f"Listening | data={rx.data_received:,} | "
            f"fs={fs / 1e6:.3f} MHz"
            + (f" | RF={rf_ref/1e6:.3f} MHz" if rf_ref else "")
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
        ctx = self._receiver.context
        if ctx is None:
            # Context not yet received — retry after 500 ms
            QTimer.singleShot(500, self._auto_display)
            return
        fs     = ctx.sample_rate_hz
        rf_ref = ctx.rf_ref_freq_hz
        center = rf_ref
        span   = fs

        def pick_unit(hz):
            a = abs(hz)
            if a >= 1e9: return "GHz", hz / 1e9
            if a >= 1e6: return "MHz", hz / 1e6
            if a >= 1e3: return "kHz", hz / 1e3
            return "Hz", hz

        s_unit, s_val = pick_unit(span)
        c_unit, c_val = (s_unit, 0.0) if center == 0.0 else pick_unit(center)

        self._center._unit.setCurrentText(c_unit)
        self._center._spin.setValue(c_val)
        self._span._unit.setCurrentText(s_unit)
        self._span._spin.setValue(s_val)
        self._amp_top.setValue(-10.0)
        self._db_div.setValue(10.0)
        self._apply_range()   # force-apply even if spinbox values didn't change

    def _sync_viewport_to_spinboxes(self, ranges):
        """Keep X-axis display spinboxes in sync when user pans/zooms the plot."""
        x_lo, x_hi = ranges[0]
        if x_hi <= x_lo:
            return
        center_hz = (x_lo + x_hi) / 2.0
        span_hz   = x_hi - x_lo

        def unit_val(hz):
            a = abs(hz)
            if a >= 1e9: return "GHz", hz / 1e9
            if a >= 1e6: return "MHz", hz / 1e6
            if a >= 1e3: return "kHz", hz / 1e3
            return "Hz", hz

        c_unit, c_val = unit_val(center_hz)
        s_unit, s_val = unit_val(span_hz)

        for widget, unit, val in [
            (self._center, c_unit, c_val),
            (self._span,   s_unit, s_val),
        ]:
            widget._spin.blockSignals(True)
            widget._unit.blockSignals(True)
            widget._unit.setCurrentText(unit)
            widget._spin.setValue(val)
            widget._spin.blockSignals(False)
            widget._unit.blockSignals(False)

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
