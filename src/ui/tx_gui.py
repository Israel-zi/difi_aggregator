import sys
import time
import numpy as np

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QDoubleSpinBox, QPushButton, QMessageBox
)

import pyqtgraph as pg


class TxGui(QWidget):
    """
    Simple TX GUI:
    - Generates complex CW samples in real-time (no networking yet)
    - Shows live spectrum
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("DIFI Aggregator - TX (Live Spectrum)")

        # Signal parameters
        self.fs = 48_000.0          # sample rate for demo (Hz)
        self.n = 4096               # FFT size / buffer size
        self.phase = 0.0

        # UI widgets
        self.tone1_box = QDoubleSpinBox()
        self.tone1_box.setRange(0.0, self.fs / 2 - 1)
        self.tone1_box.setDecimals(1)
        self.tone1_box.setValue(2000.0)
        self.tone1_box.setSuffix(" Hz")

        self.tone2_box = QDoubleSpinBox()
        self.tone2_box.setRange(0.0, self.fs / 2 - 1)
        self.tone2_box.setDecimals(1)
        self.tone2_box.setValue(6000.0)
        self.tone2_box.setSuffix(" Hz")

        self.amp_box = QDoubleSpinBox()
        self.amp_box.setRange(0.0, 10.0)
        self.amp_box.setDecimals(3)
        self.amp_box.setSingleStep(0.1)
        self.amp_box.setValue(0.8)

        self.start_btn = QPushButton("Start")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)

        # Plot
        pg.setConfigOptions(antialias=True)
        self.plot = pg.PlotWidget(title="Live Spectrum (Magnitude, dB)")
        self.plot.setLabel("bottom", "Frequency", units="Hz")
        self.plot.setLabel("left", "Magnitude", units="dB")
        self.plot.showGrid(x=True, y=True)

        self.curve = self.plot.plot([], [])

        # Layout
        top = QHBoxLayout()
        top.addWidget(QLabel("Tone 1:"))
        top.addWidget(self.tone1_box)
        top.addSpacing(12)
        top.addWidget(QLabel("Tone 2:"))
        top.addWidget(self.tone2_box)
        top.addSpacing(12)
        top.addWidget(QLabel("Amplitude:"))
        top.addWidget(self.amp_box)
        top.addStretch()
        top.addWidget(self.start_btn)
        top.addWidget(self.stop_btn)

        root = QVBoxLayout()
        root.addLayout(top)
        root.addWidget(self.plot)
        self.setLayout(root)

        # Timer for live updates
        self.timer = QTimer(self)
        self.timer.setInterval(50)  # ~20 FPS
        self.timer.timeout.connect(self.update_spectrum)

        self.start_btn.clicked.connect(self.start)
        self.stop_btn.clicked.connect(self.stop)

        # Precompute FFT frequency axis
        self.freqs = np.fft.fftshift(np.fft.fftfreq(self.n, d=1.0 / self.fs))

        # Only show 0..Fs/2 (positive freqs) for cleaner display
        self.pos_mask = self.freqs >= 0
        self.freqs_pos = self.freqs[self.pos_mask]

        self.last_t = time.time()

    def start(self):
        if self.tone1_box.value() >= self.fs / 2 or self.tone2_box.value() >= self.fs / 2:
            QMessageBox.warning(self, "Invalid frequency", "Tone frequencies must be < Fs/2.")
            return

        self.timer.start()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def stop(self):
        self.timer.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def update_spectrum(self):
        f1 = float(self.tone1_box.value())
        f2 = float(self.tone2_box.value())
        a = float(self.amp_box.value())

        t = np.arange(self.n) / self.fs

        # Two-tone complex baseband
        x1 = 0.5 * a * np.exp(1j * (2.0 * np.pi * f1 * t + self.phase))
        x2 = 0.5 * a * np.exp(1j * (2.0 * np.pi * f2 * t + self.phase))
        x = x1 + x2

        # Advance phase (keep continuity) - use average tone just for phase continuity
        favg = 0.5 * (f1 + f2)
        self.phase = (self.phase + 2.0 * np.pi * favg * (self.n / self.fs)) % (2.0 * np.pi)

        X = np.fft.fftshift(np.fft.fft(x))
        mag = np.abs(X)
        mag_db = 20.0 * np.log10(np.maximum(mag, 1e-12))

        self.curve.setData(self.freqs_pos, mag_db[self.pos_mask])


def main():
    app = QApplication(sys.argv)
    w = TxGui()
    w.resize(1000, 600)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()