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
    QLabel, QDoubleSpinBox, QSpinBox, QPushButton, QGroupBox,
    QStatusBar, QMainWindow, QSplitter, QButtonGroup, QRadioButton,
)
import pyqtgraph as pg

from modules.generator     import DifiGenerator, SIGNAL_CW, SIGNAL_BW, SIGNAL_OFF
from modules.input_capture import InputCapture, JitterBuffer
from modules.aggregator    import Aggregator
from modules.packetizer    import Packetizer
from modules.sender        import DifiSender
from modules.receiver      import DifiReceiver
from ui.freq_input         import FreqInput


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


class GeneratorPanel(QGroupBox):
    changed = Signal()

    def __init__(self, n: int, default_signal_type: str = SIGNAL_CW, parent=None):
        super().__init__(f"Generator {n}  —  Stream 0x0000000{n}", parent)
        grid = QGridLayout(self)

        grid.addWidget(QLabel("Signal type:"), 0, 0)
        type_w   = QWidget()
        type_lay = QHBoxLayout(type_w)
        type_lay.setContentsMargins(0, 0, 0, 0)
        self._cw_rb  = QRadioButton("CW")
        self._bw_rb  = QRadioButton("BW")
        self._off_rb = QRadioButton("OFF")
        if default_signal_type == SIGNAL_BW:
            self._bw_rb.setChecked(True)
        elif default_signal_type == SIGNAL_OFF:
            self._off_rb.setChecked(True)
        else:
            self._cw_rb.setChecked(True)
        self._grp = QButtonGroup(self)
        self._grp.addButton(self._cw_rb)
        self._grp.addButton(self._bw_rb)
        self._grp.addButton(self._off_rb)
        type_lay.addWidget(self._cw_rb)
        type_lay.addWidget(self._bw_rb)
        type_lay.addWidget(self._off_rb)
        type_lay.addStretch()
        grid.addWidget(type_w, 0, 1)

        grid.addWidget(QLabel("RF Frequency:"), 1, 0)
        self._tone = FreqInput(default_hz=n * 1e6)
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
            rb.toggled.connect(lambda checked: self._bw.setEnabled(self._bw_rb.isChecked()))
        self._bw.setEnabled(default_signal_type == SIGNAL_BW)

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
    CAPTURE_PORTS    = [50001, 50002, 50003]
    EXPECTED_STREAMS = [0x00000001, 0x00000002, 0x00000003]
    RECEIVER_PORT    = 50010

    def __init__(self):
        super().__init__()
        self.setWindowTitle("DIFI Aggregator PoC")
        self.setMinimumSize(1100, 820)
        self._pipeline_running = False
        self._modules          = {}

        self._build_ui()
        self._panel1.changed.connect(self._live_update_generators)
        self._panel2.changed.connect(self._live_update_generators)
        self._panel3.changed.connect(self._live_update_generators)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root     = QVBoxLayout(central)
        splitter = QSplitter(Qt.Orientation.Horizontal)
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

        net_box    = QGroupBox("Network / Jitter Buffer")
        net_layout = QHBoxLayout(net_box)
        net_layout.addWidget(QLabel("Reorder hold:"))
        self._hold_ms = QSpinBox()
        self._hold_ms.setRange(0, 2000)
        self._hold_ms.setValue(0)
        self._hold_ms.setSuffix(" ms")
        self._hold_ms.setFixedWidth(90)
        self._hold_ms.setToolTip(
            "0 ms = LAN mode (pass-through, no added latency).\n"
            "Set to the expected one-way WAN jitter (e.g. 100-300 ms)\n"
            "so that out-of-order packets from each generator are\n"
            "sorted by timestamp before reaching the aggregator."
        )
        net_layout.addWidget(self._hold_ms)
        net_layout.addStretch()
        left_layout.addWidget(net_box)

        self._panel1 = GeneratorPanel(1)
        self._panel2 = GeneratorPanel(2)
        self._panel3 = GeneratorPanel(3, default_signal_type=SIGNAL_OFF)
        left_layout.addWidget(self._panel1)
        left_layout.addWidget(self._panel2)
        left_layout.addWidget(self._panel3)
        left_layout.addStretch()
        left_layout.addWidget(self._build_buttons())
        splitter.addWidget(left)

        # ── right panel — two stacked spectrum plots ──
        right        = QWidget()
        right_layout = QVBoxLayout(right)

        right_layout.addWidget(self._build_display_controls())

        # ── Plot 1: aggregator (per-stream, pre-pipeline) ──
        self._plot = pg.PlotWidget(
            title="Combiner Input — ports 50001/50002/50003"
        )
        self._plot.setLabel("bottom", "Frequency", units="Hz")
        self._plot.setLabel("left",   "Magnitude", units="dB")
        self._plot.showGrid(x=True, y=True, alpha=0.3)
        self._plot.enableAutoRange(axis="xy", enable=False)
        self._plot.getPlotItem().getViewBox().enableAutoRange(enable=False)
        self._plot.setYRange(-110, -10, padding=0)
        self._plot.setXRange(0, 5e6, padding=0)
        self._curves1: dict = {}  # stream_id → PlotDataItem
        self._ref_line = pg.InfiniteLine(
            angle=0, movable=False,
            pen=pg.mkPen("y", width=1, style=Qt.PenStyle.DashLine)
        )
        self._plot.addItem(self._ref_line)
        # Annotation showing data provenance
        self._plot1_label = pg.TextItem(
            text="source: Aggregator.last_chunk  (float32, no encoding)",
            color=(100, 200, 255), anchor=(0, 1),
        )
        self._plot1_label.setPos(0, 0)
        self._plot.addItem(self._plot1_label)

        # ── Plot 2: DIFI receiver (post-pipeline, per-stream) ──
        self._plot2 = pg.PlotWidget(
            title="Receiver Input — port 50010"
        )
        self._plot2.setLabel("bottom", "Frequency", units="Hz")
        self._plot2.setLabel("left",   "Magnitude", units="dB")
        self._plot2.showGrid(x=True, y=True, alpha=0.3)
        self._plot2.enableAutoRange(axis="xy", enable=False)
        self._plot2.getPlotItem().getViewBox().enableAutoRange(enable=False)
        self._plot2.setYRange(-110, -10, padding=0)
        self._curves2: dict = {}  # stream_id → PlotDataItem
        self._ref_line2 = pg.InfiniteLine(
            angle=0, movable=False,
            pen=pg.mkPen("y", width=1, style=Qt.PenStyle.DashLine)
        )
        self._plot2.addItem(self._ref_line2)
        # Annotation updated live with packet counter — proves data is from receiver
        self._plot2_label = pg.TextItem(
            text="waiting for receiver packets…",
            color=(255, 220, 80), anchor=(0, 1),
        )
        self._plot2_label.setPos(0, 0)
        self._plot2.addItem(self._plot2_label)

        # Link X-axes: panning/zooming either plot moves both
        self._plot2.setXLink(self._plot)

        # Sync display spinboxes when the user pans/zooms plot 1
        self._plot.getPlotItem().getViewBox().sigRangeChanged.connect(
            lambda vb, ranges: self._sync_viewport_to_spinboxes(ranges)
        )

        plots_splitter = QSplitter(Qt.Orientation.Vertical)
        plots_splitter.addWidget(self._plot)
        plots_splitter.addWidget(self._plot2)
        plots_splitter.setSizes([300, 300])

        right_layout.addWidget(plots_splitter)
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
        p3 = self._panel3
        fs = self._shared_fs.value_hz()

        rf_ref1  = self._rf_ref_for(p1)
        rf_ref2  = self._rf_ref_for(p2)
        rf_ref3  = self._rf_ref_for(p3)
        tone1_bb = p1.tone_hz() - rf_ref1
        tone2_bb = p2.tone_hz() - rf_ref2
        tone3_bb = p3.tone_hz() - rf_ref3

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
        gen3 = DifiGenerator(
            stream_id       = 0x00000003,
            tone_hz         = tone3_bb,
            signal_type     = p3.signal_type(),
            dest_port       = 50003,
            sample_rate_hz  = fs,
            samples_per_pkt = self.SAMPLES_PER_PKT,
            bit_depth       = self.BIT_DEPTH,
            rf_ref_freq_hz  = rf_ref3,
            bandwidth_hz    = p3.bandwidth_hz(),
            ref_level_dbm   = p3.amplitude_dbm(),
        )

        capture    = InputCapture(ports=self.CAPTURE_PORTS)
        jitter     = JitterBuffer(capture, hold_ms=self._hold_ms.value())
        aggregator = Aggregator(
            capture          = jitter,
            expected_streams = self.EXPECTED_STREAMS,
            chunk_size       = self.SAMPLES_PER_PKT,
        )
        packetizer = Packetizer(aggregator=aggregator)
        sender     = DifiSender(packetizer=packetizer, dest_port=self.RECEIVER_PORT)
        receiver   = DifiReceiver(port=self.RECEIVER_PORT)

        self._modules = dict(
            gen1=gen1, gen2=gen2, gen3=gen3,
            capture=capture, jitter=jitter, aggregator=aggregator,
            packetizer=packetizer, sender=sender, receiver=receiver,
        )

        receiver.start();  time.sleep(0.05)
        capture.start();   time.sleep(0.05)
        jitter.start()
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
        if abs(tone3_bb) >= nyquist:
            alias_warnings.append(f"Gen3 RF={p3.tone_hz()/1e6:.3f}MHz bb={tone3_bb/1e6:.3f}MHz")
        if alias_warnings:
            self._status.showMessage(
                f"⚠ Tone exceeds Nyquist ({nyquist/1e6:.3f}MHz): "
                + ", ".join(alias_warnings)
                + " — set RF Reference to the correct LO"
            )

        pkt_rate = fs / self.SAMPLES_PER_PKT
        threading.Thread(target=gen1.run, kwargs=dict(packet_rate_hz=pkt_rate), daemon=True).start()
        threading.Thread(target=gen2.run, kwargs=dict(packet_rate_hz=pkt_rate), daemon=True).start()
        threading.Thread(target=gen3.run, kwargs=dict(packet_rate_hz=pkt_rate), daemon=True).start()

        self._pipeline_running = True
        self._shared_fs.setEnabled(False)   # lock sample rate while running
        self._hold_ms.setEnabled(False)     # lock jitter budget while running
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._timer.start()

        self._status.showMessage(
            f"Running — fs={fs/1e6:.1f}MHz | "
            f"Gen1:{p1.signal_type()} {p1.tone_hz()/1e6:.3f}MHz | "
            f"Gen2:{p2.signal_type()} {p2.tone_hz()/1e6:.3f}MHz | "
            f"Gen3:{p3.signal_type()} {p3.tone_hz()/1e6:.3f}MHz"
        )

    def _stop_pipeline(self):
        if not self._pipeline_running:
            return
        self._timer.stop()
        m = self._modules
        m["sender"].stop()
        m["packetizer"].stop()
        m["aggregator"].stop()
        m["jitter"].stop()
        m["capture"].stop()
        m["receiver"].stop()
        m["gen1"].close()
        m["gen2"].close()
        m["gen3"].close()
        self._pipeline_running = False
        self._shared_fs.setEnabled(True)    # unlock sample rate
        self._hold_ms.setEnabled(True)      # unlock jitter budget
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._status.showMessage("Stopped")

    def _update_spectrum(self):
        agg = self._modules.get("aggregator")
        if agg is None:
            return
        chunk = agg.last_chunk
        if chunk is None:
            return

        # ── Plot 1: aggregator — per-stream spectra (generator inputs) ─────────
        active_sids1 = []
        for s in chunk.streams:
            if s.context is None or len(s.samples) == 0:
                continue
            sid = s.stream_id
            active_sids1.append(f"0x{sid:08X}")
            if sid not in self._curves1:
                self._curves1[sid] = self._plot.plot([], [], pen=_stream_color(sid))
            f, m = _stream_fft(s.samples, s.context)
            self._curves1[sid].setData(f, m)
        if active_sids1:
            self._plot1_label.setText(
                f"source: Aggregator.last_chunk  (float32, no encoding)  |  "
                f"chunks: {agg.chunks_emitted:,}  |  streams: {' + '.join(active_sids1)}"
            )

        # ── Plot 2: DIFI receiver — per-stream spectra (combiner output) ────────
        rx = self._modules.get("receiver")
        if rx:
            snaps = rx.get_stream_snapshots()
            for sid, (iq, ctx_s) in snaps.items():
                if ctx_s is None or len(iq) == 0:
                    continue
                if sid not in self._curves2:
                    self._curves2[sid] = self._plot2.plot([], [], pen=_stream_color(sid))
                f, m = _stream_fft(iq, ctx_s)
                self._curves2[sid].setData(f, m)
            sid_list = " + ".join(f"0x{s:08X}" for s in snaps)
            self._plot2_label.setText(
                f"source: DifiReceiver  (int16→float32)  |  "
                f"UDP pkts received: {rx.data_received:,}  |  "
                f"streams: {sid_list if sid_list else 'none yet'}"
            )

        # ── Status bar: streams + pipeline latency ────────────────────────────
        active_los = sorted(
            s.context.rf_ref_freq_hz for s in chunk.streams
            if np.any(s.samples != 0)
        )
        rf_str = (
            " | streams: " + ", ".join(f"{lo/1e6:.3f} MHz" for lo in active_los)
            if active_los else ""
        )
        chunk_age_ms = (time.monotonic() - chunk.created_at) * 1000
        fs0 = chunk.streams[0].context.sample_rate_hz
        rx_pkts = rx.data_received if rx else 0
        self._status.showMessage(
            f"Running — fs={fs0/1e6:.1f} MHz{rf_str} | "
            f"chunks={agg.chunks_emitted}  rx={rx_pkts}  "
            f"latency≈{chunk_age_ms:.1f} ms"
        )

    def _live_update_generators(self):
        if not self._pipeline_running:
            return
        gen1 = self._modules.get("gen1")
        gen2 = self._modules.get("gen2")
        gen3 = self._modules.get("gen3")
        if not gen1 or not gen2 or not gen3:
            return
        p1, p2, p3 = self._panel1, self._panel2, self._panel3
        rf_ref1  = self._rf_ref_for(p1)
        rf_ref2  = self._rf_ref_for(p2)
        rf_ref3  = self._rf_ref_for(p3)
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
        gen3.update_params(
            tone_hz        = p3.tone_hz() - rf_ref3,
            signal_type    = p3.signal_type(),
            bandwidth_hz   = p3.bandwidth_hz(),
            rf_ref_freq_hz = rf_ref3,
            ref_level_dbm  = p3.amplitude_dbm(),
        )
        # Drain pipeline queues so stale chunks with old context don't reach the
        # receiver after parameter changes, then flush the receiver IQ buffer.
        agg = self._modules.get("aggregator")
        if agg:
            agg.flush_queue()
        pktzr = self._modules.get("packetizer")
        if pktzr:
            pktzr.flush_queue()
        rx = self._modules.get("receiver")
        if rx:
            rx.flush()

    def _apply_range(self):
        center  = self._disp_center.value_hz()
        span    = self._disp_span.value_hz()
        amp_top = self._disp_amp.value()
        db_div  = self._disp_dbdiv.value()
        x_lo, x_hi = center - span / 2, center + span / 2
        y_lo, y_hi = amp_top - db_div * 10, amp_top
        self._plot.setXRange(x_lo, x_hi, padding=0)
        self._plot.setYRange(y_lo, y_hi, padding=0)
        self._ref_line.setValue(amp_top)
        # plot2 X is linked; only need to set Y independently
        self._plot2.setYRange(y_lo, y_hi, padding=0)
        self._ref_line2.setValue(amp_top)

    def _sync_viewport_to_spinboxes(self, _ranges=None):
        """Keep display spinboxes in sync when user pans/zooms the plot."""
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
            self._plot2.setYRange(y_lo, y_hi, padding=0)
            self._ref_line2.setValue(y_hi)


    def _set_disp_range(self, center_hz: float, span_hz: float):
        self._disp_center.set_hz(center_hz)
        self._disp_span.set_hz(span_hz)
        self._apply_range()


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