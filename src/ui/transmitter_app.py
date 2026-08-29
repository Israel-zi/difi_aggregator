"""
transmitter_app.py
------------------
DIFI Aggregator -- Transmitter GUI.

Runs on the Transmitter VM.
Controls a single DifiGenerator and sends its stream to the Combiner VM.
"""

import os
import sys

if not getattr(sys, 'frozen', False):
    _src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _src not in sys.path:
        sys.path.insert(0, _src)

import threading

import numpy as np
import scipy.signal as sp_sig

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QDoubleSpinBox, QPushButton,
    QGroupBox, QStatusBar, QLineEdit, QSpinBox, QButtonGroup, QRadioButton,
    QSplitter,
)
import pyqtgraph as pg

from modules.generator import DifiGenerator, SIGNAL_CW, SIGNAL_BW, SIGNAL_OFF
from ui.freq_input     import FreqInput
from pipeline_logger   import make_run_dir, PacketLogger


class TransmitterWindow(QMainWindow):

    SAMPLES_PER_PKT = 1024
    BIT_DEPTH       = 16

    def __init__(self):
        super().__init__()
        self.setWindowTitle("DIFI Transmitter")
        self.setMinimumSize(900, 520)
        self._running = False
        self._gen     = None
        self._sent_log = None
        self._build_ui()

    def _build_ui(self):
        central  = QWidget()
        self.setCentralWidget(central)
        root     = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # ── left panel: controls ──────────────────────────────────────────
        left     = QWidget()
        left.setMaximumWidth(400)
        left_lay = QVBoxLayout(left)
        left_lay.setSpacing(8)

        # ── Network ──
        net_box  = QGroupBox("Network")
        net_grid = QGridLayout(net_box)

        net_grid.addWidget(QLabel("Combiner VM IP:"), 0, 0)
        self._dest_ip = QLineEdit("127.0.0.1")
        self._dest_ip.setPlaceholderText("e.g. 192.168.1.20")
        self._dest_ip.setFixedWidth(160)
        net_grid.addWidget(self._dest_ip, 0, 1)

        net_grid.addWidget(QLabel("Dest port:"), 1, 0)
        self._dest_port = QSpinBox()
        self._dest_port.setRange(1024, 65535)
        self._dest_port.setValue(50001)
        self._dest_port.setFixedWidth(110)
        port_w = QWidget()
        port_l = QHBoxLayout(port_w)
        port_l.setContentsMargins(0, 0, 0, 0)
        port_l.addWidget(self._dest_port)
        port_l.addStretch()
        net_grid.addWidget(port_w, 1, 1)

        net_grid.addWidget(QLabel("Stream ID:"), 2, 0)
        self._stream_id = QLineEdit("0x00000001")
        self._stream_id.setFixedWidth(120)
        sid_w = QWidget()
        sid_l = QHBoxLayout(sid_w)
        sid_l.setContentsMargins(0, 0, 0, 0)
        sid_l.addWidget(self._stream_id)
        sid_l.addStretch()
        net_grid.addWidget(sid_w, 2, 1)

        net_grid.addWidget(QLabel("Sim delay:"), 3, 0)
        self._sim_delay = QDoubleSpinBox()
        self._sim_delay.setRange(0, 5000)
        self._sim_delay.setDecimals(0)
        self._sim_delay.setValue(0)
        self._sim_delay.setSuffix(" ms")
        self._sim_delay.setFixedWidth(110)
        self._sim_delay.setToolTip(
            "Fixed simulated one-way network delay for this modem's path\n"
            "to the Combiner VM (e.g. 100/120/150 ms)."
        )
        net_grid.addWidget(self._sim_delay, 3, 1)

        net_grid.addWidget(QLabel("Sim jitter max:"), 4, 0)
        self._sim_jitter = QDoubleSpinBox()
        self._sim_jitter.setRange(0, 1000)
        self._sim_jitter.setDecimals(0)
        self._sim_jitter.setValue(0)
        self._sim_jitter.setSuffix(" ms")
        self._sim_jitter.setFixedWidth(110)
        self._sim_jitter.setToolTip(
            "Extra random delay on top of Sim delay, uniform in\n"
            "[0, this value] per packet."
        )
        net_grid.addWidget(self._sim_jitter, 4, 1)

        left_lay.addWidget(net_box)

        # ── Signal ──
        sig_box  = QGroupBox("Signal")
        sig_grid = QGridLayout(sig_box)

        sig_grid.addWidget(QLabel("Sample rate:"), 0, 0)
        self._fs = FreqInput(default_hz=10e6)
        sig_grid.addWidget(self._fs, 0, 1)

        sig_grid.addWidget(QLabel("Signal type:"), 1, 0)
        type_w   = QWidget()
        type_lay = QHBoxLayout(type_w)
        type_lay.setContentsMargins(0, 0, 0, 0)
        self._cw_rb  = QRadioButton("CW")
        self._bw_rb  = QRadioButton("BW")
        self._off_rb = QRadioButton("OFF")
        self._cw_rb.setChecked(True)
        grp = QButtonGroup(self)
        grp.addButton(self._cw_rb)
        grp.addButton(self._bw_rb)
        grp.addButton(self._off_rb)
        type_lay.addWidget(self._cw_rb)
        type_lay.addWidget(self._bw_rb)
        type_lay.addWidget(self._off_rb)
        type_lay.addStretch()
        sig_grid.addWidget(type_w, 1, 1)

        sig_grid.addWidget(QLabel("RF Frequency:"), 2, 0)
        self._tone = FreqInput(default_hz=1e6)
        sig_grid.addWidget(self._tone, 2, 1)

        sig_grid.addWidget(QLabel("Bandwidth:"), 3, 0)
        self._bw = FreqInput(default_hz=1e6)
        self._bw.setEnabled(False)
        sig_grid.addWidget(self._bw, 3, 1)

        sig_grid.addWidget(QLabel("RF reference:"), 4, 0)
        self._rf = FreqInput(default_hz=0)
        sig_grid.addWidget(self._rf, 4, 1)

        sig_grid.addWidget(QLabel("Amplitude:"), 5, 0)
        self._amp = QDoubleSpinBox()
        self._amp.setRange(-100.0, 0.0)
        self._amp.setDecimals(1)
        self._amp.setSingleStep(1.0)
        self._amp.setValue(-20.0)
        self._amp.setSuffix(" dBm")
        sig_grid.addWidget(self._amp, 5, 1)

        self._stat = QLabel("Idle")
        self._stat.setStyleSheet("color: #888888;")
        sig_grid.addWidget(self._stat, 6, 0, 1, 2)

        for rb in (self._cw_rb, self._bw_rb, self._off_rb):
            rb.toggled.connect(lambda checked: self._bw.setEnabled(self._bw_rb.isChecked()))

        # live update generator while running, and refresh spectrum preview
        for rb in (self._cw_rb, self._bw_rb, self._off_rb):
            rb.toggled.connect(self._on_param_changed)
        self._tone.changed.connect(self._on_param_changed)
        self._bw.changed.connect(self._on_param_changed)
        self._rf.changed.connect(self._on_param_changed)
        self._amp.valueChanged.connect(self._on_param_changed)
        self._fs.changed.connect(self._on_param_changed)
        self._sim_delay.valueChanged.connect(self._on_param_changed)
        self._sim_jitter.valueChanged.connect(self._on_param_changed)

        left_lay.addWidget(sig_box)
        left_lay.addStretch()

        # ── Start / Stop ──
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
        left_lay.addLayout(btn_row)

        splitter.addWidget(left)

        # ── right panel: display controls + spectrum ──────────────────────
        right     = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(4, 4, 4, 4)
        right_lay.setSpacing(4)

        disp_box  = QGroupBox("Display")
        disp_vlay = QVBoxLayout(disp_box)
        disp_vlay.setSpacing(4)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Center:"))
        self._disp_center = FreqInput(default_hz=1e6)
        row1.addWidget(self._disp_center)
        row1.addSpacing(16)
        row1.addWidget(QLabel("Span:"))
        self._disp_span = FreqInput(default_hz=10e6)
        row1.addWidget(self._disp_span)
        row1.addStretch()
        disp_vlay.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Amplitude:"))
        self._disp_amp = QDoubleSpinBox()
        self._disp_amp.setRange(-200, 50)
        self._disp_amp.setDecimals(1)
        self._disp_amp.setSingleStep(10)
        self._disp_amp.setValue(-10)
        self._disp_amp.setSuffix(" dB")
        self._disp_amp.setFixedWidth(120)
        row2.addWidget(self._disp_amp)
        row2.addSpacing(16)
        row2.addWidget(QLabel("dB / div:"))
        self._disp_dbdiv = QDoubleSpinBox()
        self._disp_dbdiv.setRange(1, 100)
        self._disp_dbdiv.setDecimals(1)
        self._disp_dbdiv.setValue(10)
        self._disp_dbdiv.setSuffix(" dB")
        self._disp_dbdiv.setFixedWidth(110)
        row2.addWidget(self._disp_dbdiv)
        auto_btn = QPushButton("Auto")
        auto_btn.setFixedWidth(60)
        auto_btn.clicked.connect(self._auto_display)
        row2.addWidget(auto_btn)
        row2.addStretch()
        disp_vlay.addLayout(row2)

        self._disp_center.changed.connect(self._apply_range)
        self._disp_span.changed.connect(self._apply_range)
        self._disp_amp.valueChanged.connect(self._apply_range)
        self._disp_dbdiv.valueChanged.connect(self._apply_range)

        right_lay.addWidget(disp_box)

        self._plot = pg.PlotWidget(title="Transmitter Output")
        self._plot.setLabel("bottom", "Frequency", units="Hz")
        self._plot.setLabel("left",   "Magnitude", units="dB")
        self._plot.showGrid(x=True, y=True, alpha=0.3)
        self._plot.enableAutoRange(axis="xy", enable=False)
        self._plot.getPlotItem().getViewBox().enableAutoRange(enable=False)
        self._curve = self._plot.plot([], [], pen=pg.mkPen((100, 220, 255), width=1))

        self._ref_line = pg.InfiniteLine(
            angle=0, movable=False,
            pen=pg.mkPen("y", width=1, style=Qt.PenStyle.DashLine),
        )
        self._plot.addItem(self._ref_line)

        self._plot.getPlotItem().getViewBox().sigRangeChanged.connect(
            lambda vb, ranges: self._sync_viewport_to_spinboxes()
        )

        right_lay.addWidget(self._plot)
        splitter.addWidget(right)
        splitter.setSizes([380, 520])

        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Ready — enter Combiner IP and press Start")

        self._timer = QTimer()
        self._timer.setInterval(200)
        self._timer.timeout.connect(self._tick)

        self._apply_range()
        self._update_spectrum()

    # ── helpers ────────────────────────────────────────────────────────────

    def _signal_type(self) -> str:
        if self._cw_rb.isChecked():  return SIGNAL_CW
        if self._bw_rb.isChecked():  return SIGNAL_BW
        return SIGNAL_OFF

    def _stream_id_int(self) -> int:
        """Parse stream ID from the text field. Raises ValueError on bad input."""
        return int(self._stream_id.text().strip(), 16)

    def _rf_ref(self) -> float:
        rf_ref = self._rf.value_hz()
        if rf_ref == 0.0 and abs(self._tone.value_hz()) > self._fs.value_hz() / 2.0:
            return self._tone.value_hz()
        return rf_ref

    # ── lifecycle ──────────────────────────────────────────────────────────

    def _start(self):
        if self._running:
            return

        ip     = self._dest_ip.text().strip()
        fs     = self._fs.value_hz()
        rf_ref = self._rf_ref()
        tone_bb = self._tone.value_hz() - rf_ref

        try:
            sid = self._stream_id_int()
        except ValueError:
            self._status.showMessage("Invalid Stream ID — use hex e.g. 0x00000001")
            return

        run_dir = make_run_dir("Transmitter")
        self._sent_log = PacketLogger(
            run_dir, "data_sent.csv",
            ["wall_clock", "stream_id", "pkt_type", "seq", "difi_ts_int",
             "difi_ts_frac", "samples", "dest_ip", "dest_port"],
        )

        self._gen = DifiGenerator(
            stream_id       = sid,
            tone_hz         = tone_bb,
            signal_type     = self._signal_type(),
            dest_host       = ip,
            dest_port       = self._dest_port.value(),
            sample_rate_hz  = fs,
            samples_per_pkt = self.SAMPLES_PER_PKT,
            bit_depth       = self.BIT_DEPTH,
            rf_ref_freq_hz  = rf_ref,
            bandwidth_hz    = self._bw.value_hz(),
            ref_level_dbm   = self._amp.value(),
            sim_delay_ms    = self._sim_delay.value(),
            sim_jitter_ms   = self._sim_jitter.value(),
            packet_logger   = self._sent_log,
        )

        pkt_rate = fs / self.SAMPLES_PER_PKT
        threading.Thread(
            target=self._gen.run,
            kwargs=dict(packet_rate_hz=pkt_rate),
            daemon=True,
        ).start()

        self._running = True
        self._fs.setEnabled(False)
        self._dest_ip.setEnabled(False)
        self._dest_port.setEnabled(False)
        self._stream_id.setEnabled(False)
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._timer.start()

        port = self._dest_port.value()
        self._plot.setTitle(f"Transmitter Output — port {port}")
        self._status.showMessage(
            f"Sending to {ip}:{port} | "
            f"stream=0x{sid:08X} | fs={fs/1e6:.2f} MHz | "
            f"type={self._signal_type()} | RF={self._tone.value_hz()/1e6:.3f} MHz | "
            f"log: {run_dir}"
        )

    def _stop(self):
        if not self._running:
            return
        self._timer.stop()
        if self._gen:
            self._gen.close()
            self._gen = None
        if self._sent_log:
            self._sent_log.close()
            self._sent_log = None
        self._stat.setText("Idle")
        self._stat.setStyleSheet("color: #888888;")
        self._running = False
        self._fs.setEnabled(True)
        self._dest_ip.setEnabled(True)
        self._dest_port.setEnabled(True)
        self._stream_id.setEnabled(True)
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._plot.setTitle("Transmitter Output")
        self._status.showMessage("Stopped")

    def _tick(self):
        if not self._running or not self._gen:
            return
        self._stat.setText(f"Running — {self._gen.pkt_count:,} pkts sent")
        self._stat.setStyleSheet("color: #00cc44;")
        self._update_spectrum()

    def _on_param_changed(self, *_):
        """Called when any signal parameter changes — live-update generator and spectrum."""
        if self._running and self._gen:
            rf_ref = self._rf_ref()
            self._gen.update_params(
                tone_hz        = self._tone.value_hz() - rf_ref,
                signal_type    = self._signal_type(),
                bandwidth_hz   = self._bw.value_hz(),
                rf_ref_freq_hz = rf_ref,
                ref_level_dbm  = self._amp.value(),
                sim_delay_ms   = self._sim_delay.value(),
                sim_jitter_ms  = self._sim_jitter.value(),
            )
        self._update_spectrum()

    # ── spectrum ───────────────────────────────────────────────────────────

    def _update_spectrum(self):
        """Compute and display the expected signal spectrum from current UI parameters.

        Adds a small synthetic noise floor, freshly randomized on every call.
        A noiseless ideal tone (or BW noise redrawn from a fixed seed, as
        this used to do) is bit-for-bit identical on every redraw, which
        reads as a frozen/dead display even while actively transmitting —
        real spectrum analyzers visibly jitter tick-to-tick because of their
        own noise floor, and this display should look the same way.
        """
        fs       = self._fs.value_hz()
        rf_ref   = self._rf_ref()
        tone_bb  = self._tone.value_hz() - rf_ref
        sig_type = self._signal_type()
        amp_dbm  = self._amp.value()
        bw       = self._bw.value_hz()

        if fs <= 0:
            return

        seg_len = 1024
        t = np.arange(seg_len) / fs

        if sig_type == SIGNAL_OFF:
            self._curve.setData([], [])
            return

        rng     = np.random.default_rng()   # fresh noise each tick -> visibly live display
        amp_lin = 10 ** (amp_dbm / 20.0)

        if sig_type == SIGNAL_CW:
            iq = np.exp(1j * 2 * np.pi * tone_bb * t).astype(np.complex64)
            iq *= amp_lin
        else:  # BW
            noise  = (rng.standard_normal(seg_len) + 1j * rng.standard_normal(seg_len)).astype(np.complex64)
            nyq    = fs / 2.0
            cutoff = max(min(bw / 2.0 / nyq, 0.499), 1e-4)
            fir    = sp_sig.firwin(101, cutoff)
            fi     = sp_sig.lfilter(fir, [1.0], noise.real)
            fq     = sp_sig.lfilter(fir, [1.0], noise.imag)
            iq     = (fi + 1j * fq).astype(np.complex64)
            iq    *= np.exp(1j * 2 * np.pi * tone_bb * t)
            iq    *= amp_lin

        # Synthetic noise floor ~45 dB below the signal peak.
        floor_amp = amp_lin * 10 ** (-45.0 / 20.0)
        iq = iq + floor_amp * (
            rng.standard_normal(seg_len) + 1j * rng.standard_normal(seg_len)
        ).astype(np.complex64)

        w      = np.hanning(seg_len)
        w_amp  = float(np.sum(w))
        X      = np.fft.fftshift(np.fft.fft(iq * w))
        mag_db = 20.0 * np.log10(np.abs(X) / w_amp + 1e-7)
        freqs  = np.fft.fftshift(np.fft.fftfreq(seg_len, d=1.0 / fs)) + rf_ref

        self._curve.setData(freqs, mag_db)

    # ── display helpers ────────────────────────────────────────────────────

    def _apply_range(self):
        center  = self._disp_center.value_hz()
        span    = self._disp_span.value_hz()
        amp_top = self._disp_amp.value()
        db_div  = self._disp_dbdiv.value()
        self._plot.setXRange(center - span / 2, center + span / 2, padding=0)
        self._plot.setYRange(amp_top - db_div * 10, amp_top, padding=0)
        self._ref_line.setValue(amp_top)

    def _auto_display(self):
        fs     = self._fs.value_hz()
        rf_ref = self._rf_ref()
        self._disp_center.set_hz(rf_ref if rf_ref != 0.0 else self._tone.value_hz())
        self._disp_span.set_hz(fs)
        self._disp_amp.setValue(-10.0)
        self._disp_dbdiv.setValue(10.0)
        self._apply_range()

    def _sync_viewport_to_spinboxes(self):
        [[x_lo, x_hi], [y_lo, y_hi]] = (
            self._plot.getPlotItem().getViewBox().viewRange()
        )
        if x_hi > x_lo:
            self._disp_center.set_hz((x_lo + x_hi) / 2.0)
            self._disp_span.set_hz(x_hi - x_lo)
        if y_hi > y_lo:
            self._disp_amp.blockSignals(True)
            self._disp_dbdiv.blockSignals(True)
            self._disp_amp.setValue(y_hi)
            self._disp_dbdiv.setValue((y_hi - y_lo) / 10.0)
            self._disp_amp.blockSignals(False)
            self._disp_dbdiv.blockSignals(False)
            self._ref_line.setValue(y_hi)

    def closeEvent(self, event):
        self._stop()
        event.accept()


def main():
    from logging_setup import setup_frozen_file_logging
    log_path = setup_frozen_file_logging("Transmitter")

    pg.setConfigOptions(antialias=True)
    app = QApplication(sys.argv)
    win = TransmitterWindow()
    if log_path:
        win._status.showMessage(f"Logging to {log_path}")
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
