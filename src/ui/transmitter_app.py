"""
transmitter_app.py
------------------
DIFI Aggregator — Transmitter GUI.

Runs on the Transmitter VM.
Controls two DifiGenerators and sends their streams to the Packetizer VM.
"""

import os
import sys

_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src not in sys.path:
    sys.path.insert(0, _src)

import threading

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QDoubleSpinBox, QComboBox, QPushButton,
    QGroupBox, QStatusBar, QLineEdit, QSpinBox, QButtonGroup, QRadioButton,
)
import pyqtgraph as pg

from modules.generator import DifiGenerator, SIGNAL_CW, SIGNAL_BW, SIGNAL_OFF

UNIT_MUL    = {"Hz": 1.0, "kHz": 1_000.0, "MHz": 1_000_000.0, "GHz": 1_000_000_000.0}
UNIT_LABELS = ["Hz", "kHz", "MHz", "GHz"]


class FreqInput(QWidget):
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
        self._spin.setRange(0.001, 999_999.999)
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

    def value_hz(self) -> float:
        return self._spin.value() * UNIT_MUL[self._unit.currentText()]


class GeneratorPanel(QGroupBox):
    def __init__(self, n: int, default_tone_hz: float, default_port: int, parent=None):
        super().__init__(f"Generator {n}  —  Stream 0x0000000{n}", parent)
        grid = QGridLayout(self)

        grid.addWidget(QLabel("Dest port:"), 0, 0)
        self._port = QSpinBox()
        self._port.setRange(1024, 65535)
        self._port.setValue(default_port)
        self._port.setFixedWidth(90)
        port_row = QWidget()
        port_lay = QHBoxLayout(port_row)
        port_lay.setContentsMargins(0, 0, 0, 0)
        port_lay.addWidget(self._port)
        port_lay.addStretch()
        grid.addWidget(port_row, 0, 1)

        grid.addWidget(QLabel("Signal type:"), 1, 0)
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
        grid.addWidget(type_w, 1, 1)

        grid.addWidget(QLabel("Tone / Center:"), 2, 0)
        self._tone = FreqInput(default_hz=default_tone_hz)
        grid.addWidget(self._tone, 2, 1)

        grid.addWidget(QLabel("Bandwidth:"), 3, 0)
        self._bw = FreqInput(default_hz=1e6)
        grid.addWidget(self._bw, 3, 1)

        grid.addWidget(QLabel("RF reference:"), 4, 0)
        self._rf = FreqInput(default_hz=437e6)
        grid.addWidget(self._rf, 4, 1)

        grid.addWidget(QLabel("Amplitude:"), 5, 0)
        self._amp = QDoubleSpinBox()
        self._amp.setRange(-100.0, 0.0)
        self._amp.setDecimals(1)
        self._amp.setSingleStep(1.0)
        self._amp.setValue(-20.0)
        self._amp.setSuffix(" dBm")
        grid.addWidget(self._amp, 5, 1)

        self._stat = QLabel("Idle")
        self._stat.setStyleSheet("color: #888888;")
        grid.addWidget(self._stat, 6, 0, 1, 2)

        for rb in (self._cw_rb, self._bw_rb, self._off_rb):
            rb.toggled.connect(lambda: self._bw.setEnabled(self._bw_rb.isChecked()))
        self._bw.setEnabled(False)

    def port(self)           -> int:   return self._port.value()
    def signal_type(self)    -> str:
        if self._cw_rb.isChecked():  return SIGNAL_CW
        if self._bw_rb.isChecked():  return SIGNAL_BW
        return SIGNAL_OFF
    def tone_hz(self)        -> float: return self._tone.value_hz()
    def bandwidth_hz(self)   -> float: return self._bw.value_hz()
    def rf_ref_freq_hz(self) -> float: return self._rf.value_hz()
    def amplitude_dbm(self)  -> float: return self._amp.value()

    def set_status(self, running: bool, pkt_count: int = 0):
        if running:
            self._stat.setText(f"Running — {pkt_count:,} pkts sent")
            self._stat.setStyleSheet("color: #00cc44;")
        else:
            self._stat.setText("Idle")
            self._stat.setStyleSheet("color: #888888;")


class TransmitterWindow(QMainWindow):

    SAMPLES_PER_PKT = 1024
    BIT_DEPTH       = 16

    def __init__(self):
        super().__init__()
        self.setWindowTitle("DIFI Transmitter")
        self.setMinimumSize(680, 560)
        self._running = False
        self._gen1    = None
        self._gen2    = None
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(8)

        # ── network & timing ──
        net_box  = QGroupBox("Network & Timing")
        net_grid = QGridLayout(net_box)

        net_grid.addWidget(QLabel("Packetizer VM IP:"), 0, 0)
        self._dest_ip = QLineEdit("127.0.0.1")
        self._dest_ip.setPlaceholderText("e.g. 192.168.1.20")
        self._dest_ip.setFixedWidth(160)
        net_grid.addWidget(self._dest_ip, 0, 1)

        net_grid.addWidget(QLabel("Sample rate:"), 0, 2)
        self._fs = FreqInput(default_hz=10e6)
        net_grid.addWidget(self._fs, 0, 3)
        root.addWidget(net_box)

        # ── generator panels side by side ──
        panels = QHBoxLayout()
        self._panel1 = GeneratorPanel(1, default_tone_hz=1e6, default_port=50001)
        self._panel2 = GeneratorPanel(2, default_tone_hz=2e6, default_port=50002)
        panels.addWidget(self._panel1)
        panels.addWidget(self._panel2)
        root.addLayout(panels)

        root.addStretch()

        # ── start / stop ──
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
        self._status.showMessage("Ready — enter Packetizer IP and press Start")

        self._timer = QTimer()
        self._timer.setInterval(500)
        self._timer.timeout.connect(self._tick)

    def _start(self):
        if self._running:
            return

        ip       = self._dest_ip.text().strip()
        fs       = self._fs.value_hz()
        pkt_rate = fs / self.SAMPLES_PER_PKT

        self._gen1 = DifiGenerator(
            stream_id       = 0x00000001,
            tone_hz         = self._panel1.tone_hz(),
            signal_type     = self._panel1.signal_type(),
            dest_host       = ip,
            dest_port       = self._panel1.port(),
            sample_rate_hz  = fs,
            samples_per_pkt = self.SAMPLES_PER_PKT,
            bit_depth       = self.BIT_DEPTH,
            rf_ref_freq_hz  = self._panel1.rf_ref_freq_hz(),
            bandwidth_hz    = self._panel1.bandwidth_hz(),
            ref_level_dbm   = self._panel1.amplitude_dbm(),
        )
        self._gen2 = DifiGenerator(
            stream_id       = 0x00000002,
            tone_hz         = self._panel2.tone_hz(),
            signal_type     = self._panel2.signal_type(),
            dest_host       = ip,
            dest_port       = self._panel2.port(),
            sample_rate_hz  = fs,
            samples_per_pkt = self.SAMPLES_PER_PKT,
            bit_depth       = self.BIT_DEPTH,
            rf_ref_freq_hz  = self._panel2.rf_ref_freq_hz(),
            bandwidth_hz    = self._panel2.bandwidth_hz(),
            ref_level_dbm   = self._panel2.amplitude_dbm(),
        )

        threading.Thread(
            target=self._gen1.run,
            kwargs=dict(packet_rate_hz=pkt_rate),
            daemon=True,
        ).start()
        threading.Thread(
            target=self._gen2.run,
            kwargs=dict(packet_rate_hz=pkt_rate),
            daemon=True,
        ).start()

        self._running = True
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._timer.start()
        self._status.showMessage(
            f"Sending to {ip} | "
            f"fs={fs / 1e6:.2f} MHz | "
            f"pkt_rate={pkt_rate:.1f} pkt/s"
        )

    def _stop(self):
        if not self._running:
            return
        self._timer.stop()
        if self._gen1:
            self._gen1.close()
        if self._gen2:
            self._gen2.close()
        self._panel1.set_status(False)
        self._panel2.set_status(False)
        self._running = False
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._status.showMessage("Stopped")

    def _tick(self):
        if not self._running:
            return
        c1 = self._gen1._pkt_count if self._gen1 else 0
        c2 = self._gen2._pkt_count if self._gen2 else 0
        self._panel1.set_status(True, c1)
        self._panel2.set_status(True, c2)
        self._status.showMessage(
            f"Gen 1: {c1:,} pkts | Gen 2: {c2:,} pkts"
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
