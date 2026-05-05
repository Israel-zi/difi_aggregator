"""
config_gui.py
-------------
DIFI Aggregator — Configuration GUI.
"""

import sys
import threading
import time

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QDoubleSpinBox, QSpinBox, QComboBox, QPushButton, QGroupBox,
    QStatusBar, QMainWindow, QSplitter, QButtonGroup, QRadioButton,
    QFrame,
)
import pyqtgraph as pg

from modules.generator     import DifiGenerator, SIGNAL_CW, SIGNAL_BW, SIGNAL_OFF  # noqa: F401
from modules.input_capture import InputCapture
from modules.aggregator    import Aggregator
from modules.packetizer    import Packetizer
from modules.sender        import DifiSender
from modules.receiver      import DifiReceiver

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
        self._spin.setKeyboardTracking(False)  # only emit on Enter/focus-leave, not per-keystroke

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


class GeneratorPanel(QGroupBox):
    changed = Signal()

    def __init__(self, n: int, parent=None):
        super().__init__(f"Generator {n}  —  Stream 0x0000000{n}", parent)
        grid = QGridLayout(self)

        grid.addWidget(QLabel("Signal type:"), 0, 0)
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
        grid.addWidget(type_w, 0, 1)

        grid.addWidget(QLabel("RF Frequency:"), 1, 0)
        self._tone = FreqInput(default_hz=1e6 if n == 1 else 2e6)
        grid.addWidget(self._tone, 1, 1)

        grid.addWidget(QLabel("Bandwidth:"), 2, 0)
        self._bw = FreqInput(default_hz=1e6)
        grid.addWidget(self._bw, 2, 1)

        grid.addWidget(QLabel("RF reference:"), 3, 0)
        self._rf = FreqInput(default_hz=0)
        grid.addWidget(self._rf, 3, 1)

        grid.addWidget(QLabel("Amplitude:"), 4, 0)
        self._amp = QDoubleSpinBox()
        self._amp.setRange(-100.0, 0.0)
        self._amp.setDecimals(1)
        self._amp.setSingleStep(1.0)
        self._amp.setValue(-20.0)
        self._amp.setSuffix(" dBm")
        grid.addWidget(self._amp, 4, 1)

        for rb in (self._cw_rb, self._bw_rb, self._off_rb):
            rb.toggled.connect(lambda: self._bw.setEnabled(self._bw_rb.isChecked()))
        self._bw.setEnabled(False)

        # emit changed whenever any control is modified
        for rb in (self._cw_rb, self._bw_rb, self._off_rb):
            rb.toggled.connect(self.changed)
        self._tone.changed.connect(self.changed)
        self._bw.changed.connect(self.changed)
        self._rf.changed.connect(self.changed)
        self._amp.valueChanged.connect(self.changed)

    def signal_type(self)    -> str:   return SIGNAL_CW if self._cw_rb.isChecked() else (SIGNAL_BW if self._bw_rb.isChecked() else "OFF")
    def tone_hz(self)        -> float: return self._tone.value_hz()
    def bandwidth_hz(self)   -> float: return self._bw.value_hz()
    def rf_ref_freq_hz(self) -> float: return self._rf.value_hz()
    def amplitude_dbm(self)  -> float: return self._amp.value()


class MainWindow(QMainWindow):

    SAMPLES_PER_PKT  = 1024
    BIT_DEPTH        = 16
    CAPTURE_PORTS    = [50001, 50002]
    EXPECTED_STREAMS = [0x00000001, 0x00000002]
    RECEIVER_PORT    = 50010

    def __init__(self):
        super().__init__()
        self.setWindowTitle("DIFI Aggregator PoC")
        self.setMinimumSize(1100, 700)
        self._pipeline_running = False
        self._modules          = {}

        # sweep state
        self._sweep_active             = False
        self._sweep_positions          = np.array([])
        self._sweep_pos_idx            = 0
        self._sweep_composite_freqs    = np.array([])
        self._sweep_composite_data_raw = np.array([])
        self._sweep_sort_idx           = np.array([], dtype=int)
        self._sweep_snap_acc           = np.array([])
        self._sweep_snap_count         = 0

        self._build_ui()
        self._panel1.changed.connect(self._live_update_generators)
        self._panel2.changed.connect(self._live_update_generators)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root     = QVBoxLayout(central)
        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        # ── left panel ──
        left        = QWidget()
        left.setMaximumWidth(420)
        left_layout = QVBoxLayout(left)

        fs_box    = QGroupBox("Shared Sample Rate (both generators)")
        fs_layout = QHBoxLayout(fs_box)
        fs_layout.addWidget(QLabel("Sample rate:"))
        self._shared_fs = FreqInput(default_hz=10e6)
        fs_layout.addWidget(self._shared_fs)
        left_layout.addWidget(fs_box)

        self._panel1 = GeneratorPanel(1)
        self._panel2 = GeneratorPanel(2)
        left_layout.addWidget(self._panel1)
        left_layout.addWidget(self._panel2)
        left_layout.addWidget(self._build_sweep_controls())
        left_layout.addStretch()
        left_layout.addWidget(self._build_buttons())
        splitter.addWidget(left)

        # ── right panel ──
        right        = QWidget()
        right_layout = QVBoxLayout(right)

        # display controls bar
        right_layout.addWidget(self._build_display_controls())

        # spectrum plot
        self._plot = pg.PlotWidget()
        self._plot.setLabel("bottom", "Frequency", units="Hz")
        self._plot.setLabel("left",   "Magnitude", units="dB")
        self._plot.showGrid(x=True, y=True, alpha=0.3)
        self._plot.enableAutoRange(axis="xy", enable=False)
        self._plot.getPlotItem().getViewBox().enableAutoRange(enable=False)
        self._plot.setYRange(-110, -10, padding=0)
        self._plot.setXRange(0, 5e6, padding=0)
        self._curve = self._plot.plot([], [], pen=pg.mkPen("c", width=1))
        self._y_range_applied = False   # re-enforce Y after first real data

        # reference line at amplitude level
        self._ref_line = pg.InfiniteLine(
            angle=0, movable=False,
            pen=pg.mkPen("y", width=1, style=Qt.DashLine)
        )
        self._plot.addItem(self._ref_line)

        # Keep display spinboxes in sync when user pans/zooms with mouse
        self._plot.getPlotItem().getViewBox().sigRangeChanged.connect(
            lambda vb, ranges: self._sync_viewport_to_spinboxes(ranges)
        )

        right_layout.addWidget(self._plot)
        splitter.addWidget(right)
        splitter.setSizes([400, 700])

        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Ready — configure and press Start")

        self._timer = QTimer()
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._update_spectrum)

    def _build_display_controls(self) -> QWidget:
        box    = QGroupBox("Display")
        vlay   = QVBoxLayout(box)
        vlay.setSpacing(4)

        # row 1 — frequency axis
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Center:"))
        self._disp_center = FreqInput(default_hz=2.5e6)
        row1.addWidget(self._disp_center)
        row1.addSpacing(16)
        row1.addWidget(QLabel("Span:"))
        self._disp_span = FreqInput(default_hz=5e6)
        row1.addWidget(self._disp_span)
        row1.addStretch()
        auto_btn = QPushButton("Auto")
        auto_btn.setFixedWidth(60)
        auto_btn.clicked.connect(self._auto_display)
        row1.addWidget(auto_btn)
        vlay.addLayout(row1)

        # row 2 — amplitude axis
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
        row2.addWidget(QLabel("dB/div:"))
        self._disp_dbdiv = QDoubleSpinBox()
        self._disp_dbdiv.setRange(1, 100)
        self._disp_dbdiv.setDecimals(1)
        self._disp_dbdiv.setSingleStep(1)
        self._disp_dbdiv.setValue(10)
        self._disp_dbdiv.setSuffix(" dB")
        self._disp_dbdiv.setFixedWidth(110)
        row2.addWidget(self._disp_dbdiv)
        row2.addStretch()
        vlay.addLayout(row2)

        # apply range immediately whenever any display control changes
        self._disp_center.changed.connect(self._apply_range)
        self._disp_span.changed.connect(self._apply_range)
        self._disp_amp.valueChanged.connect(self._apply_range)
        self._disp_dbdiv.valueChanged.connect(self._apply_range)

        return box

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

    def _build_sweep_controls(self) -> QWidget:
        box  = QGroupBox("Spectrum Sweep")
        vlay = QVBoxLayout(box)
        grid = QGridLayout()
        vlay.addLayout(grid)

        grid.addWidget(QLabel("Start:"), 0, 0)
        self._sw_start = FreqInput(default_hz=0.0)
        grid.addWidget(self._sw_start, 0, 1)

        grid.addWidget(QLabel("Stop:"), 1, 0)
        self._sw_stop = FreqInput(default_hz=100e6)
        grid.addWidget(self._sw_stop, 1, 1)

        grid.addWidget(QLabel("Step:"), 2, 0)
        self._sw_step = FreqInput(default_hz=10e6)
        grid.addWidget(self._sw_step, 2, 1)

        grid.addWidget(QLabel("Dwell:"), 3, 0)
        self._sw_dwell = QSpinBox()
        self._sw_dwell.setRange(1, 50)
        self._sw_dwell.setValue(1)
        self._sw_dwell.setSuffix(" frames")
        grid.addWidget(self._sw_dwell, 3, 1)

        btn_row = QHBoxLayout()
        self._sw_run_btn  = QPushButton("▶  Sweep")
        self._sw_stop_btn = QPushButton("■  Stop Sweep")
        self._sw_stop_btn.setEnabled(False)
        self._sw_run_btn.clicked.connect(self._start_sweep)
        self._sw_stop_btn.clicked.connect(self._stop_sweep)
        btn_row.addWidget(self._sw_run_btn)
        btn_row.addWidget(self._sw_stop_btn)
        vlay.addLayout(btn_row)

        return box

    def _rf_ref_for(self, panel) -> float:
        """
        Effective LO for a generator panel.

        When RF Reference is left at 0 but RF Frequency is beyond the
        Nyquist limit (fs/2), the signal would alias to baseband.
        In that case treat RF Frequency as the LO so the generator
        produces a tone at 0 Hz baseband, and the context packet carries
        the correct absolute carrier frequency.  The display then shows
        the signal at its intended RF position automatically.

        If RF Reference is explicitly set to a non-zero value, use it as-is.
        """
        rf_ref = panel.rf_ref_freq_hz()
        if rf_ref == 0.0 and abs(panel.tone_hz()) > self._shared_fs.value_hz() / 2.0:
            return panel.tone_hz()
        return rf_ref

    def _start_pipeline(self):
        if self._pipeline_running:
            return

        p1 = self._panel1
        p2 = self._panel2
        fs = self._shared_fs.value_hz()

        rf_ref1  = self._rf_ref_for(p1)
        rf_ref2  = self._rf_ref_for(p2)
        tone1_bb = p1.tone_hz() - rf_ref1
        tone2_bb = p2.tone_hz() - rf_ref2

        gen1 = DifiGenerator(
            stream_id       = 0x00000001,
            tone_hz         = tone1_bb,
            signal_type     = p1.signal_type(),
            dest_port       = 50001,
            sample_rate_hz  = fs,
            samples_per_pkt = self.SAMPLES_PER_PKT,
            bit_depth       = self.BIT_DEPTH,
            rf_ref_freq_hz  = rf_ref1,
            bandwidth_hz    = p1.bandwidth_hz(),
            ref_level_dbm   = p1.amplitude_dbm(),
        )
        gen2 = DifiGenerator(
            stream_id       = 0x00000002,
            tone_hz         = tone2_bb,
            signal_type     = p2.signal_type(),
            dest_port       = 50002,
            sample_rate_hz  = fs,
            samples_per_pkt = self.SAMPLES_PER_PKT,
            bit_depth       = self.BIT_DEPTH,
            rf_ref_freq_hz  = rf_ref2,
            bandwidth_hz    = p2.bandwidth_hz(),
            ref_level_dbm   = p2.amplitude_dbm(),
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

        # Warn if any baseband tone still exceeds Nyquist after LO auto-assignment
        nyquist = fs / 2.0
        alias_warnings = []
        if abs(tone1_bb) >= nyquist:
            alias_warnings.append(f"Gen1 RF={p1.tone_hz()/1e6:.3f}MHz bb={tone1_bb/1e6:.3f}MHz")
        if abs(tone2_bb) >= nyquist:
            alias_warnings.append(f"Gen2 RF={p2.tone_hz()/1e6:.3f}MHz bb={tone2_bb/1e6:.3f}MHz")
        if alias_warnings:
            self._status.showMessage(
                f"⚠ Tone exceeds Nyquist ({nyquist/1e6:.3f}MHz): "
                + ", ".join(alias_warnings)
                + " — set RF Reference to the correct LO"
            )

        pkt_rate = fs / self.SAMPLES_PER_PKT
        threading.Thread(target=gen1.run, kwargs=dict(packet_rate_hz=pkt_rate), daemon=True).start()
        threading.Thread(target=gen2.run, kwargs=dict(packet_rate_hz=pkt_rate), daemon=True).start()

        self._pipeline_running = True
        self._y_range_applied  = False      # will re-enforce Y after first real data
        self._shared_fs.setEnabled(False)   # lock sample rate while running
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._timer.start()
        self._apply_range()   # immediately restore Y range and spinbox-tracked X range

        self._status.showMessage(
            f"Running — fs={fs/1e6:.1f}MHz | "
            f"Gen1:{p1.signal_type()} {p1.tone_hz()/1e6:.3f}MHz | "
            f"Gen2:{p2.signal_type()} {p2.tone_hz()/1e6:.3f}MHz"
        )

    def _stop_pipeline(self):
        if not self._pipeline_running:
            return
        if self._sweep_active:
            self._stop_sweep()
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
        self._shared_fs.setEnabled(True)    # unlock sample rate
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._status.showMessage("Stopped")

    def _update_spectrum(self):
        rx = self._modules.get("receiver")
        if rx is None:
            return
        iq  = rx.get_iq_snapshot()
        # always use the configured sample rate (not receiver's default)
        fs  = self._shared_fs.value_hz()
        n   = len(iq)
        if n == 0:
            return

        # Use the combined stream's actual sample rate from the received context.
        # After freq-stitching, new_fs may be larger than the generator's orig_fs.
        ctx    = rx.context
        fs     = ctx.sample_rate_hz      if ctx else self._shared_fs.value_hz()
        rf_ref = ctx.rf_ref_freq_hz      if ctx else self._panel1.rf_ref_freq_hz()

        # FFT — full complex IQ spectrum
        window = np.hanning(n)
        X      = np.fft.fftshift(np.fft.fft(iq * window))
        mag_db = 20 * np.log10(np.abs(X) / n + 1e-7)

        if self._sweep_active:
            # Use the known LO position directly — don't wait for context to propagate
            lo_hz = self._sweep_positions[self._sweep_pos_idx]
            freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / fs)) + lo_hz
            self._sweep_step(freqs, mag_db)
        else:
            freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / fs)) + rf_ref
            self._curve.setData(freqs, mag_db)

        # On first real data: auto-fit frequency axis to the actual received
        # signal band (uses combined center + fs from context).  This handles
        # the freq-stitched case where the combined stream may be at a
        # different center / sample-rate than the individual generators.
        if not self._y_range_applied:
            if ctx and not self._sweep_active:
                self._auto_display()
            else:
                self._apply_range()
            self._y_range_applied = True

        if not self._sweep_active:
            agg    = self._modules.get("aggregator")
            ctx    = rx.context
            rf_ref = ctx.rf_ref_freq_hz if ctx else 0
            rf_str = f" | RF={rf_ref/1e6:.3f} MHz" if rf_ref else ""
            self._status.showMessage(
                f"Running — fs={fs/1e6:.3f} MHz{rf_str} | "
                f"data={rx.data_received}  chunks={agg.chunks_emitted if agg else 0}"
            )

    def _live_update_generators(self):
        if not self._pipeline_running:
            return
        gen1 = self._modules.get("gen1")
        gen2 = self._modules.get("gen2")
        if not gen1 or not gen2:
            return
        p1, p2   = self._panel1, self._panel2
        rf_ref1  = self._rf_ref_for(p1)
        rf_ref2  = self._rf_ref_for(p2)
        gen1.update_params(
            tone_hz        = p1.tone_hz() - rf_ref1,
            signal_type    = p1.signal_type(),
            bandwidth_hz   = p1.bandwidth_hz(),
            rf_ref_freq_hz = rf_ref1,
            ref_level_dbm  = p1.amplitude_dbm(),
        )
        gen2.update_params(
            tone_hz        = p2.tone_hz() - rf_ref2,
            signal_type    = p2.signal_type(),
            bandwidth_hz   = p2.bandwidth_hz(),
            rf_ref_freq_hz = rf_ref2,
            ref_level_dbm  = p2.amplitude_dbm(),
        )
        rx = self._modules.get("receiver")
        if rx:
            rx.flush()   # clear stale IQ so display refreshes on the next tick

    def _apply_range(self):
        center  = self._disp_center.value_hz()
        span    = self._disp_span.value_hz()
        amp_top = self._disp_amp.value()
        db_div  = self._disp_dbdiv.value()
        self._plot.setXRange(center - span / 2, center + span / 2, padding=0)
        self._plot.setYRange(amp_top - db_div * 10, amp_top, padding=0)
        self._ref_line.setValue(amp_top)

    def _auto_display(self):
        """Fit display to the combined stream's actual band: rf_ref ± fs/2."""
        if not self._pipeline_running:
            return
        rx  = self._modules.get("receiver")
        ctx = rx.context if rx else None
        if ctx is None:
            QTimer.singleShot(300, self._auto_display)
            return

        # Use the received context — after freq-stitching this carries the
        # correct combined center frequency and combined sample rate.
        center = ctx.rf_ref_freq_hz
        span   = ctx.sample_rate_hz

        def pick_unit(hz):
            a = abs(hz)
            if a >= 1e9: return "GHz", hz / 1e9
            if a >= 1e6: return "MHz", hz / 1e6
            if a >= 1e3: return "kHz", hz / 1e3
            return "Hz", hz

        s_unit, s_val = pick_unit(span)
        # when center is zero, reuse span's unit so spinbox doesn't overflow
        c_unit, c_val = (s_unit, 0.0) if center == 0.0 else pick_unit(center)

        self._disp_center._unit.setCurrentText(c_unit)
        self._disp_center._spin.setValue(c_val)
        self._disp_span._unit.setCurrentText(s_unit)
        self._disp_span._spin.setValue(s_val)
        self._disp_amp.setValue(-10.0)
        self._disp_dbdiv.setValue(10.0)
        self._apply_range()

    def _smart_auto_display(self):
        """Auto-fit only when the signal is outside the current viewport."""
        if not self._pipeline_running:
            return
        rx  = self._modules.get("receiver")
        ctx = rx.context if rx else None
        if ctx is None:
            QTimer.singleShot(300, self._smart_auto_display)
            return
        lo = self._disp_center.value_hz() - self._disp_span.value_hz() / 2
        hi = self._disp_center.value_hz() + self._disp_span.value_hz() / 2
        t1 = self._panel1.tone_hz()
        t2 = self._panel2.tone_hz()
        if not (lo <= t1 <= hi and lo <= t2 <= hi):
            self._auto_display()

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
            (self._disp_center, c_unit, c_val),
            (self._disp_span,   s_unit, s_val),
        ]:
            widget._spin.blockSignals(True)
            widget._unit.blockSignals(True)
            widget._unit.setCurrentText(unit)
            widget._spin.setValue(val)
            widget._spin.blockSignals(False)
            widget._unit.blockSignals(False)

    def _start_sweep(self):
        if not self._pipeline_running:
            self._status.showMessage("Start the pipeline first, then start sweep.")
            return
        start = self._sw_start.value_hz()
        stop  = self._sw_stop.value_hz()
        step  = self._sw_step.value_hz()
        if step <= 0 or stop <= start:
            self._status.showMessage("Sweep: stop must be > start and step must be > 0.")
            return
        self._sweep_positions          = np.arange(start, stop + step * 0.5, step)
        self._sweep_pos_idx            = 0
        self._sweep_snap_count         = 0
        self._sweep_snap_acc           = np.array([])
        self._sweep_composite_freqs    = np.array([])
        self._sweep_composite_data_raw = np.array([])
        self._sweep_active             = True
        self._sw_run_btn.setEnabled(False)
        self._sw_stop_btn.setEnabled(True)
        self._timer.setInterval(25)   # fast sweep ticks
        self._move_lo_to(self._sweep_positions[0])

        # Auto-fit display to the full sweep range
        self._set_disp_range((start + stop) / 2.0, stop - start + step)

    def _set_disp_range(self, center_hz: float, span_hz: float):
        def pick_unit(hz):
            a = abs(hz)
            if a >= 1e9: return "GHz", hz / 1e9
            if a >= 1e6: return "MHz", hz / 1e6
            if a >= 1e3: return "kHz", hz / 1e3
            return "Hz", hz
        c_unit, c_val = pick_unit(center_hz)
        s_unit, s_val = pick_unit(span_hz)
        self._disp_center._unit.setCurrentText(c_unit)
        self._disp_center._spin.setValue(c_val)
        self._disp_span._unit.setCurrentText(s_unit)
        self._disp_span._spin.setValue(s_val)
        self._apply_range()

    def _stop_sweep(self):
        self._sweep_active = False
        self._timer.setInterval(100)  # restore normal display rate
        self._sw_run_btn.setEnabled(True)
        self._sw_stop_btn.setEnabled(False)

    def _move_lo_to(self, lo_hz: float):
        gen1    = self._modules.get("gen1")
        gen2    = self._modules.get("gen2")
        p1, p2  = self._panel1, self._panel2
        nyquist = self._shared_fs.value_hz() / 2.0

        for gen, p in [(gen1, p1), (gen2, p2)]:
            if not gen:
                continue
            tone_bb = p.tone_hz() - lo_hz
            if abs(tone_bb) >= nyquist:
                # Signal is outside this LO position's Nyquist band.
                # Output silence so the sweep composite shows noise floor here,
                # not an aliased copy of the signal at the wrong frequency.
                gen.update_params(
                    tone_hz        = 0.0,
                    signal_type    = SIGNAL_OFF,
                    bandwidth_hz   = p.bandwidth_hz(),
                    rf_ref_freq_hz = lo_hz,
                    ref_level_dbm  = p.amplitude_dbm(),
                )
            else:
                gen.update_params(
                    tone_hz        = tone_bb,
                    signal_type    = p.signal_type(),
                    bandwidth_hz   = p.bandwidth_hz(),
                    rf_ref_freq_hz = lo_hz,
                    ref_level_dbm  = p.amplitude_dbm(),
                )

        rx = self._modules.get("receiver")
        if rx:
            rx.flush()

    def _sweep_step(self, freqs: np.ndarray, mag_db: np.ndarray):
        dwell  = self._sw_dwell.value()
        n_pos  = len(self._sweep_positions)
        n_bins = len(freqs)

        # Accumulate (peak-hold) at current LO position
        if self._sweep_snap_count == 0:
            self._sweep_snap_acc = mag_db.copy()
        else:
            self._sweep_snap_acc = np.maximum(self._sweep_snap_acc, mag_db)
        self._sweep_snap_count += 1

        if self._sweep_snap_count < dwell:
            return  # still dwelling at this position

        pos_idx = self._sweep_pos_idx

        # Re-initialise composite arrays at the start of each pass
        if pos_idx == 0:
            self._sweep_composite_freqs    = np.zeros(n_pos * n_bins)
            self._sweep_composite_data_raw = np.full(n_pos * n_bins, -140.0)

        # Store snapshot for this position
        sl = slice(pos_idx * n_bins, (pos_idx + 1) * n_bins)
        self._sweep_composite_freqs[sl]    = freqs
        self._sweep_composite_data_raw[sl] = self._sweep_snap_acc

        # Advance counters
        self._sweep_pos_idx   += 1
        self._sweep_snap_count  = 0
        self._sweep_snap_acc    = np.array([])

        # Render whatever has been filled so far — live, position by position
        n_filled  = self._sweep_pos_idx * n_bins
        partial_f = self._sweep_composite_freqs[:n_filled]
        partial_m = self._sweep_composite_data_raw[:n_filled]
        sort_idx  = np.argsort(partial_f)
        self._curve.setData(partial_f[sort_idx], partial_m[sort_idx])

        # Wrap around to position 0 when the pass is complete
        if self._sweep_pos_idx >= n_pos:
            self._sweep_pos_idx = 0

        self._move_lo_to(self._sweep_positions[self._sweep_pos_idx])

        n_done = self._sweep_pos_idx
        self._status.showMessage(
            f"Sweeping — position {n_done + 1}/{n_pos} "
            f"@ {self._sweep_positions[n_done] / 1e6:.3f} MHz"
        )

    def closeEvent(self, event):
        self._stop_pipeline()
        event.accept()


def main():
    pg.setConfigOptions(antialias=True)
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()