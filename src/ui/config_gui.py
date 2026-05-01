"""
config_gui.py
-------------
DIFI Aggregator — Configuration GUI.
Supports CW and BW signal modes, full frequency/BW/amplitude/sample-rate control.
"""

import sys
import threading
import time

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QDoubleSpinBox, QComboBox, QPushButton, QGroupBox,
    QStatusBar, QMainWindow, QSplitter, QButtonGroup, QRadioButton,
)
import pyqtgraph as pg

from modules.generator     import DifiGenerator, SIGNAL_CW, SIGNAL_BW
from modules.input_capture import InputCapture
from modules.aggregator    import Aggregator
from modules.packetizer    import Packetizer
from modules.sender        import DifiSender
from modules.receiver      import DifiReceiver


# ─────────────────────────────────────────────
# Frequency input widget (value + Hz/kHz/MHz/GHz)
# ─────────────────────────────────────────────

UNIT_MUL = {"Hz": 1.0, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9}


class FreqInput(QWidget):
    def __init__(self, default_hz: float = 1e6, max_hz: float = None, parent=None):
        super().__init__(parent)
        self._max_hz = max_hz

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
        self._spin.setMinimumWidth(110)

        self._unit = QComboBox()
        self._unit.addItems(list(UNIT_MUL.keys()))
        self._unit.setCurrentText(unit)
        self._unit.setMinimumWidth(60)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._spin)
        lay.addWidget(self._unit)

    def value_hz(self) -> float:
        return self._spin.value() * UNIT_MUL[self._unit.currentText()]


# ─────────────────────────────────────────────
# Generator config panel
# ─────────────────────────────────────────────

class GeneratorPanel(QGroupBox):
    """All controls for one Generator."""

    def __init__(self, n: int, parent=None):
        super().__init__(f"Generator {n}  —  Stream 0x0000000{n}", parent)
        self.n = n
        grid   = QGridLayout(self)

        # Signal type
        grid.addWidget(QLabel("Signal type:"), 0, 0)
        type_w   = QWidget()
        type_lay = QHBoxLayout(type_w)
        type_lay.setContentsMargins(0, 0, 0, 0)
        self._cw_rb = QRadioButton("CW")
        self._bw_rb = QRadioButton("BW")
        self._cw_rb.setChecked(True)
        self._grp   = QButtonGroup(self)
        self._grp.addButton(self._cw_rb)
        self._grp.addButton(self._bw_rb)
        type_lay.addWidget(self._cw_rb)
        type_lay.addWidget(self._bw_rb)
        type_lay.addStretch()
        grid.addWidget(type_w, 0, 1)

        # Sample rate
        grid.addWidget(QLabel("Sample rate:"), 1, 0)
        self._fs = FreqInput(default_hz=10e6)
        grid.addWidget(self._fs, 1, 1)

        # Tone / center frequency
        grid.addWidget(QLabel("Tone / Center freq:"), 2, 0)
        self._tone = FreqInput(default_hz=1e6 if n == 1 else 2e6)
        grid.addWidget(self._tone, 2, 1)

        # Bandwidth
        grid.addWidget(QLabel("Bandwidth:"), 3, 0)
        self._bw = FreqInput(default_hz=1e6)
        grid.addWidget(self._bw, 3, 1)

        # RF reference
        grid.addWidget(QLabel("RF reference:"), 4, 0)
        self._rf = FreqInput(default_hz=1e9)
        grid.addWidget(self._rf, 4, 1)

        # Amplitude
        grid.addWidget(QLabel("Amplitude:"), 5, 0)
        self._amp = QDoubleSpinBox()
        self._amp.setRange(-100.0, 0.0)
        self._amp.setDecimals(1)
        self._amp.setSingleStep(1.0)
        self._amp.setValue(-20.0)
        self._amp.setSuffix(" dBm")
        grid.addWidget(self._amp, 5, 1)

        # BW row enabled only when BW selected
        self._cw_rb.toggled.connect(self._on_type_change)
        self._on_type_change()

    def _on_type_change(self):
        is_bw = self._bw_rb.isChecked()
        self._bw.setEnabled(is_bw)

    # ── getters ────────────────────────────────────────────────────────────

    def signal_type(self) -> str:
        return SIGNAL_CW if self._cw_rb.isChecked() else SIGNAL_BW

    def sample_rate_hz(self) -> float:
        return self._fs.value_hz()

    def tone_hz(self) -> float:
        return self._tone.value_hz()

    def bandwidth_hz(self) -> float:
        return self._bw.value_hz()

    def rf_ref_freq_hz(self) -> float:
        return self._rf.value_hz()

    def amplitude_dbm(self) -> float:
        return self._amp.value()


# ─────────────────────────────────────────────
# Main window
# ─────────────────────────────────────────────

class MainWindow(QMainWindow):

    SAMPLES_PER_PKT  = 1024
    BIT_DEPTH        = 16
    CAPTURE_PORTS    = [50001, 50002]
    EXPECTED_STREAMS = [0x00000001, 0x00000002]
    RECEIVER_PORT    = 50010

    def __init__(self):
        super().__init__()
        self.setWindowTitle("DIFI Aggregator PoC")
        self.setMinimumSize(980, 640)
        self._pipeline_running = False
        self._modules          = {}
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root     = QVBoxLayout(central)
        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        # left — controls
        left        = QWidget()
        left.setMaximumWidth(380)
        left_layout = QVBoxLayout(left)
        self._panel1 = GeneratorPanel(1)
        self._panel2 = GeneratorPanel(2)
        left_layout.addWidget(self._panel1)
        left_layout.addWidget(self._panel2)
        left_layout.addStretch()
        left_layout.addWidget(self._build_buttons())
        splitter.addWidget(left)

        # right — spectrum
        right        = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(QLabel("<b>Aggregated Stream — Live Spectrum</b>"))
        self._plot = pg.PlotWidget()
        self._plot.setLabel("bottom", "Frequency", units="Hz")
        self._plot.setLabel("left",   "Magnitude", units="dB")
        self._plot.showGrid(x=True, y=True, alpha=0.3)
        self._plot.setYRange(-80, 10)
        self._curve = self._plot.plot([], [], pen=pg.mkPen("c", width=1))
        right_layout.addWidget(self._plot)
        splitter.addWidget(right)
        splitter.setSizes([360, 620])

        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Ready — configure and press Start")

        self._timer = QTimer()
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._update_spectrum)

    def _build_buttons(self) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        self._start_btn = QPushButton("▶  Start")
        self._stop_btn  = QPushButton("■  Stop")
        self._stop_btn.setEnabled(False)
        self._start_btn.setFixedHeight(36)
        self._stop_btn.setFixedHeight(36)
        self._start_btn.clicked.connect(self._start_pipeline)
        self._stop_btn.clicked.connect(self._stop_pipeline)
        h.addWidget(self._start_btn)
        h.addWidget(self._stop_btn)
        return w

    # ── pipeline ───────────────────────────────────────────────────────────

    def _start_pipeline(self):
        if self._pipeline_running:
            return

        p1 = self._panel1
        p2 = self._panel2

        gen1 = DifiGenerator(
            stream_id      = 0x00000001,
            tone_hz        = p1.tone_hz(),
            signal_type    = p1.signal_type(),
            dest_port      = 50001,
            sample_rate_hz = p1.sample_rate_hz(),
            samples_per_pkt = self.SAMPLES_PER_PKT,
            bit_depth      = self.BIT_DEPTH,
            rf_ref_freq_hz = p1.rf_ref_freq_hz(),
            bandwidth_hz   = p1.bandwidth_hz(),
            ref_level_dbm  = p1.amplitude_dbm(),
        )
        gen2 = DifiGenerator(
            stream_id      = 0x00000002,
            tone_hz        = p2.tone_hz(),
            signal_type    = p2.signal_type(),
            dest_port      = 50002,
            sample_rate_hz = p2.sample_rate_hz(),
            samples_per_pkt = self.SAMPLES_PER_PKT,
            bit_depth      = self.BIT_DEPTH,
            rf_ref_freq_hz = p2.rf_ref_freq_hz(),
            bandwidth_hz   = p2.bandwidth_hz(),
            ref_level_dbm  = p2.amplitude_dbm(),
        )

        capture    = InputCapture(ports=self.CAPTURE_PORTS)
        aggregator = Aggregator(
            capture          = capture,
            expected_streams = self.EXPECTED_STREAMS,
            chunk_size       = self.SAMPLES_PER_PKT,
        )
        packetizer = Packetizer(aggregator=aggregator)
        sender     = DifiSender(packetizer=packetizer, dest_port=self.RECEIVER_PORT)
        receiver   = DifiReceiver(port=self.RECEIVER_PORT)

        self._modules = dict(
            gen1=gen1, gen2=gen2,
            capture=capture, aggregator=aggregator,
            packetizer=packetizer, sender=sender, receiver=receiver,
        )

        receiver.start();  time.sleep(0.05)
        capture.start();   time.sleep(0.05)
        aggregator.start()
        packetizer.start()
        sender.start()

        rate1 = p1.sample_rate_hz() / self.SAMPLES_PER_PKT
        rate2 = p2.sample_rate_hz() / self.SAMPLES_PER_PKT
        threading.Thread(
            target=gen1.run, kwargs=dict(packet_rate_hz=rate1), daemon=True
        ).start()
        threading.Thread(
            target=gen2.run, kwargs=dict(packet_rate_hz=rate2), daemon=True
        ).start()

        self._pipeline_running = True
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._timer.start()
        self._status.showMessage(
            f"Running — "
            f"Gen1: {p1.signal_type()} {p1.tone_hz()/1e6:.3f}MHz  |  "
            f"Gen2: {p2.signal_type()} {p2.tone_hz()/1e6:.3f}MHz"
        )

    def _stop_pipeline(self):
        if not self._pipeline_running:
            return
        self._timer.stop()
        m = self._modules
        m["sender"].stop()
        m["packetizer"].stop()
        m["aggregator"].stop()
        m["capture"].stop()
        m["receiver"].stop()
        m["gen1"].close()
        m["gen2"].close()
        self._pipeline_running = False
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._status.showMessage("Stopped")

    # ── spectrum ───────────────────────────────────────────────────────────

    def _update_spectrum(self):
        rx = self._modules.get("receiver")
        if rx is None:
            return
        iq = rx.get_iq_snapshot()
        fs = rx.get_sample_rate()
        n  = len(iq)
        if n == 0:
            return

        window = np.hanning(n)
        X      = np.fft.fftshift(np.fft.fft(iq * window))
        freqs  = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / fs))
        mag_db = 20 * np.log10(np.abs(X) / n + 1e-12)

        self._curve.setData(freqs, mag_db)
        self._plot.setXRange(freqs[0], freqs[-1], padding=0)

        agg = self._modules.get("aggregator")
        self._status.showMessage(
            f"Running — "
            f"data={rx.data_received}  ctx={rx.context_received}  "
            f"chunks={agg.chunks_emitted if agg else 0}"
        )

    def closeEvent(self, event):
        self._stop_pipeline()
        event.accept()


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main():
    pg.setConfigOptions(antialias=True)
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()