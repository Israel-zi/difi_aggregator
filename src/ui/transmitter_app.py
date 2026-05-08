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

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QDoubleSpinBox, QPushButton,
    QGroupBox, QStatusBar, QLineEdit, QSpinBox, QButtonGroup, QRadioButton,
)
import pyqtgraph as pg

from modules.generator import DifiGenerator, SIGNAL_CW, SIGNAL_BW, SIGNAL_OFF
from ui.freq_input     import FreqInput


class TransmitterWindow(QMainWindow):

    SAMPLES_PER_PKT = 1024
    BIT_DEPTH       = 16

    def __init__(self):
        super().__init__()
        self.setWindowTitle("DIFI Transmitter")
        self.setMinimumSize(420, 500)
        self._running = False
        self._gen     = None
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(8)

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

        root.addWidget(net_box)

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
            rb.toggled.connect(lambda: self._bw.setEnabled(self._bw_rb.isChecked()))

        # live update while running
        for rb in (self._cw_rb, self._bw_rb, self._off_rb):
            rb.toggled.connect(self._live_update)
        self._tone.changed.connect(self._live_update)
        self._bw.changed.connect(self._live_update)
        self._rf.changed.connect(self._live_update)
        self._amp.valueChanged.connect(self._live_update)

        root.addWidget(sig_box)
        root.addStretch()

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
        root.addLayout(btn_row)

        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Ready — enter Combiner IP and press Start")

        self._timer = QTimer()
        self._timer.setInterval(500)
        self._timer.timeout.connect(self._tick)

    # ── helpers ────────────────────────────────────────────────────────────

    def _signal_type(self) -> str:
        if self._cw_rb.isChecked():  return SIGNAL_CW
        if self._bw_rb.isChecked():  return SIGNAL_BW
        return SIGNAL_OFF

    def _stream_id_int(self) -> int:
        try:
            return int(self._stream_id.text().strip(), 16)
        except ValueError:
            return 0x00000001

    def _rf_ref(self) -> float:
        rf_ref = self._rf.value_hz()
        if rf_ref == 0.0 and abs(self._tone.value_hz()) > self._fs.value_hz() / 2.0:
            return self._tone.value_hz()
        return rf_ref

    # ── lifecycle ──────────────────────────────────────────────────────────

    def _start(self):
        if self._running:
            return

        ip      = self._dest_ip.text().strip()
        fs      = self._fs.value_hz()
        rf_ref  = self._rf_ref()
        tone_bb = self._tone.value_hz() - rf_ref

        try:
            sid = self._stream_id_int()
        except ValueError:
            self._status.showMessage("Invalid Stream ID — use hex e.g. 0x00000001")
            return

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
        self._status.showMessage(
            f"Sending to {ip}:{self._dest_port.value()} | "
            f"stream=0x{sid:08X} | fs={fs/1e6:.2f} MHz | "
            f"type={self._signal_type()} | RF={self._tone.value_hz()/1e6:.3f} MHz"
        )

    def _stop(self):
        if not self._running:
            return
        self._timer.stop()
        if self._gen:
            self._gen.close()
            self._gen = None
        self._stat.setText("Idle")
        self._stat.setStyleSheet("color: #888888;")
        self._running = False
        self._fs.setEnabled(True)
        self._dest_ip.setEnabled(True)
        self._dest_port.setEnabled(True)
        self._stream_id.setEnabled(True)
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._status.showMessage("Stopped")

    def _tick(self):
        if not self._running or not self._gen:
            return
        self._stat.setText(f"Running — {self._gen.pkt_count:,} pkts sent")
        self._stat.setStyleSheet("color: #00cc44;")

    def _live_update(self):
        if not self._running or not self._gen:
            return
        rf_ref = self._rf_ref()
        self._gen.update_params(
            tone_hz        = self._tone.value_hz() - rf_ref,
            signal_type    = self._signal_type(),
            bandwidth_hz   = self._bw.value_hz(),
            rf_ref_freq_hz = rf_ref,
            ref_level_dbm  = self._amp.value(),
        )

    def closeEvent(self, event):
        self._stop()
        event.accept()


def main():
    pg.setConfigOptions(antialias=True)
    app = QApplication(sys.argv)
    win = TransmitterWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
