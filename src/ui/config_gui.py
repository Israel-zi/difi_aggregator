"""
config_gui.py
-------------
DIFI Aggregator — Configuration GUI.
"""

import sys
import queue
import threading
import time

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal, QRegularExpression
from PySide6.QtGui import QRegularExpressionValidator
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QDoubleSpinBox, QSpinBox, QPushButton, QGroupBox,
    QStatusBar, QMainWindow, QSplitter, QButtonGroup, QRadioButton, QTabWidget,
)
import pyqtgraph as pg

from modules.generator     import DifiGenerator, SIGNAL_CW, SIGNAL_BW, SIGNAL_OFF
from modules.input_capture import InputCapture, JitterBuffer, PortListener
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
    return pg.mkPen(_STREAM_COLORS[sid % len(_STREAM_COLORS)], width=1)


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


class GeneratorPanel(QWidget):
    changed = Signal()

    def __init__(self, default_signal_type: str = SIGNAL_CW,
                 default_port: int = 50000, default_stream_id: int = 1,
                 default_tone_hz: float = 1e6, parent=None):
        super().__init__(parent)
        grid = QGridLayout(self)

        grid.addWidget(QLabel("Stream ID:"), 0, 0)
        self._stream_id = QLineEdit(f"0x{default_stream_id:08X}")
        self._stream_id.setValidator(
            QRegularExpressionValidator(QRegularExpression(r"0[xX][0-9A-Fa-f]{1,8}"))
        )
        self._stream_id.setFixedWidth(110)
        grid.addWidget(self._stream_id, 0, 1)

        grid.addWidget(QLabel("Signal type:"), 1, 0)
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
        grid.addWidget(type_w, 1, 1)

        grid.addWidget(QLabel("RF Frequency:"), 2, 0)
        self._tone = FreqInput(default_hz=default_tone_hz)
        grid.addWidget(self._tone, 2, 1)

        grid.addWidget(QLabel("Bandwidth:"), 3, 0)
        self._bw = FreqInput(default_hz=1e6)
        grid.addWidget(self._bw, 3, 1)

        grid.addWidget(QLabel("RF reference:"), 4, 0)
        self._rf = FreqInput(default_hz=0)
        grid.addWidget(self._rf, 4, 1)

        grid.addWidget(QLabel("Amplitude:"), 5, 0)
        self._amp = QDoubleSpinBox()
        self._amp.setRange(-100.0, 0.0)
        self._amp.setDecimals(1)
        self._amp.setSingleStep(1.0)
        self._amp.setValue(-20.0)
        self._amp.setSuffix(" dBm")
        self._amp.setFixedWidth(140)
        grid.addWidget(self._amp, 5, 1)

        grid.addWidget(QLabel("UDP Port:"), 6, 0)
        self._port = QSpinBox()
        self._port.setRange(1, 65535)
        self._port.setValue(default_port)
        self._port.setFixedWidth(140)
        self._port.setKeyboardTracking(False)
        grid.addWidget(self._port, 6, 1)

        grid.setRowStretch(7, 1)

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
        self._port.valueChanged.connect(self.changed)
        self._stream_id.editingFinished.connect(self.changed)

    def signal_type(self)    -> str:   return SIGNAL_CW if self._cw_rb.isChecked() else (SIGNAL_BW if self._bw_rb.isChecked() else "OFF")
    def tone_hz(self)        -> float: return self._tone.value_hz()
    def bandwidth_hz(self)   -> float: return self._bw.value_hz()
    def rf_ref_freq_hz(self) -> float: return self._rf.value_hz()
    def amplitude_dbm(self)  -> float: return self._amp.value()
    def port(self)           -> int:   return self._port.value()

    def stream_id(self) -> int:
        try:
            return int(self._stream_id.text(), 16)
        except ValueError:
            return 0


class MainWindow(QMainWindow):

    SAMPLES_PER_PKT  = 1024
    BIT_DEPTH        = 16

    def __init__(self):
        super().__init__()
        self.setWindowTitle("DIFI Aggregator PoC")
        self.setMinimumSize(1100, 820)
        self._pipeline_running = False
        self._modules          = {}
        self._gen_panels        = []
        self._pipeline_warning  = ""
        self._agg_standalone_bind_errors = {}

        self._build_ui()

        self._add_generator(default_signal_type=SIGNAL_OFF)
        self._add_agg_port_row(default_port=self._next_default_agg_port())

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root     = QVBoxLayout(central)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # ── left panel ──
        left        = QWidget()
        left.setMaximumWidth(480)
        left_layout = QVBoxLayout(left)

        fs_box    = QGroupBox("Shared Sample Rate (both generators)")
        fs_layout = QHBoxLayout(fs_box)
        fs_layout.addWidget(QLabel("Sample rate:"))
        self._shared_fs = FreqInput(default_hz=10e6)
        self._shared_fs.changed.connect(self._live_update_generators)
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
        self._hold_ms.setKeyboardTracking(False)
        self._hold_ms.setToolTip(
            "0 ms = LAN mode (pass-through, no added latency).\n"
            "Set to the expected one-way WAN jitter (e.g. 100-300 ms)\n"
            "so that out-of-order packets from each generator are\n"
            "sorted by timestamp before reaching the aggregator."
        )
        self._hold_ms.valueChanged.connect(self._on_hold_ms_changed)
        net_layout.addWidget(self._hold_ms)
        net_layout.addStretch()
        left_layout.addWidget(net_box)

        gen_box    = QGroupBox("Generators")
        gen_layout = QVBoxLayout(gen_box)
        self._gen_tabs = QTabWidget()
        gen_layout.addWidget(self._gen_tabs)
        gen_btn_row = QHBoxLayout()
        self._add_gen_btn    = QPushButton("+  Add Generator")
        self._remove_gen_btn = QPushButton("−  Remove Generator")
        self._add_gen_btn.clicked.connect(lambda: self._add_generator())
        self._remove_gen_btn.clicked.connect(self._remove_generator)
        gen_btn_row.addWidget(self._add_gen_btn)
        gen_btn_row.addWidget(self._remove_gen_btn)
        gen_layout.addLayout(gen_btn_row)
        left_layout.addWidget(gen_box)

        agg_box     = QGroupBox("Aggregator")
        agg_vlayout = QVBoxLayout(agg_box)
        agg_vlayout.setContentsMargins(12, 18, 12, 14)
        agg_vlayout.setSpacing(12)

        self._agg_ports_widget = QWidget()
        self._agg_ports_layout = QVBoxLayout(self._agg_ports_widget)
        self._agg_ports_layout.setContentsMargins(0, 0, 0, 0)
        self._agg_ports_layout.setSpacing(8)
        self._agg_port_rows          = []
        self._agg_port_labels        = []
        self._agg_ports              = []
        self._agg_port_remove_btns   = []
        self._agg_port_listen_btns   = []
        self._agg_port_listeners     = []
        self._agg_port_status_labels = []
        self._agg_port_active        = []
        agg_vlayout.addWidget(self._agg_ports_widget)

        agg_btn_row = QHBoxLayout()
        self._add_port_btn = QPushButton("+  Add Port")
        self._add_port_btn.clicked.connect(self._add_agg_port)
        agg_btn_row.addWidget(self._add_port_btn)
        agg_btn_row.addStretch()
        agg_vlayout.addLayout(agg_btn_row)

        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("Aggregator → Receiver:"))
        self._port_out = QSpinBox()
        self._port_out.setRange(1, 65535)
        self._port_out.setValue(50010)
        self._port_out.setFixedWidth(140)
        self._port_out.setKeyboardTracking(False)
        self._port_out.valueChanged.connect(self._on_port_out_changed)
        out_row.addWidget(self._port_out)
        out_row.addStretch()
        agg_vlayout.addLayout(out_row)

        left_layout.addWidget(agg_box)

        self._port_test_timer = QTimer()
        self._port_test_timer.setInterval(300)
        self._port_test_timer.timeout.connect(self._update_port_status_labels)
        self._port_test_timer.start()

        left_layout.addWidget(self._build_buttons())
        splitter.addWidget(left)

        # ── right panel — two stacked spectrum plots ──
        right        = QWidget()
        right_layout = QVBoxLayout(right)

        right_layout.addWidget(self._build_display_controls())

        # ── Plot 1: aggregator (per-stream, pre-pipeline) ──
        self._plot = pg.PlotWidget(title="Aggregator Input")
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
        self._plot2 = pg.PlotWidget(title="Receiver Input")
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
        self._timer.start()   # always running — shows live data whenever any is flowing, blank otherwise

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

    # ── generator tab management ────────────────────────────────────────────

    def _next_default_port(self) -> int:
        used = {p.port() for p in self._gen_panels}
        port = 50001
        while port in used:
            port += 1
        return port

    def _next_default_stream_id(self) -> int:
        used = {p.stream_id() for p in self._gen_panels}
        sid = 1
        while sid in used:
            sid += 1
        return sid

    def _renumber_tabs(self):
        for i in range(self._gen_tabs.count()):
            self._gen_tabs.setTabText(i, f"Generator {i + 1}")

    def _renumber_agg_ports(self):
        for i, label in enumerate(self._agg_port_labels):
            label.setText(f"Input port {i + 1}:")

    def _add_agg_port_row(self, default_port: int, active: bool = False):
        row_widget = QWidget()
        row = QHBoxLayout(row_widget)
        row.setContentsMargins(0, 0, 0, 0)
        label = QLabel(f"Input port {len(self._agg_ports) + 1}:")
        spin  = QSpinBox()
        spin.setRange(1, 65535)
        spin.setValue(default_port)
        spin.setFixedWidth(140)
        spin.setKeyboardTracking(False)
        spin._prev_value = default_port
        spin.setToolTip(
            "Port the Aggregator listens on for this stream.\n"
            "Does not have to match the generator's UDP Port —\n"
            "set it differently to simulate a generator whose\n"
            "traffic the Aggregator isn't picking up."
        )
        spin.valueChanged.connect(lambda new_val, s=spin: self._on_agg_port_value_changed(s, new_val))
        listen_btn = QPushButton("▶")
        listen_btn.setFixedWidth(28)
        listen_btn.setToolTip("Start listening on this port")
        listen_btn.clicked.connect(lambda: self._toggle_port_listener(row_widget))
        remove_btn = QPushButton("✕")
        remove_btn.setFixedWidth(28)
        remove_btn.setToolTip("Remove this input port")
        remove_btn.clicked.connect(lambda: self._remove_specific_agg_port(row_widget))
        status_label = QLabel("")
        status_label.setStyleSheet("color: #4CAF50;")
        row.addWidget(label)
        row.addWidget(spin)
        row.addWidget(listen_btn)
        row.addWidget(remove_btn)
        row.addWidget(status_label)
        row.addStretch()
        self._agg_ports_layout.addWidget(row_widget)
        self._agg_port_rows.append(row_widget)
        self._agg_port_labels.append(label)
        self._agg_ports.append(spin)
        self._agg_port_remove_btns.append(remove_btn)
        self._agg_port_listen_btns.append(listen_btn)
        self._agg_port_listeners.append(None)
        self._agg_port_status_labels.append(status_label)
        self._agg_port_active.append(False)
        self._update_agg_remove_enabled()
        if active:
            self._set_port_active(len(self._agg_ports) - 1, True)

    def _remove_agg_port_row(self, idx: int):
        port = self._agg_ports[idx].value()
        if self._agg_port_active[idx]:
            if self._pipeline_running:
                capture = self._modules.get("capture")
                if capture:
                    capture.remove_port(port)
                    capture.bind_errors.pop(port, None)
            else:
                self._stop_standalone_listener(idx)
        self._agg_standalone_bind_errors.pop(port, None)
        self._agg_port_active.pop(idx)
        self._agg_port_listeners.pop(idx)
        self._agg_port_listen_btns.pop(idx)
        self._agg_port_status_labels.pop(idx)
        row_widget = self._agg_port_rows.pop(idx)
        self._agg_ports_layout.removeWidget(row_widget)
        row_widget.deleteLater()
        self._agg_port_labels.pop(idx)
        self._agg_ports.pop(idx)
        self._agg_port_remove_btns.pop(idx)
        self._renumber_agg_ports()
        self._update_agg_remove_enabled()

    def _set_port_active(self, idx: int, want_active: bool):
        """Turn listening on/off for one Aggregator port row.

        Works identically whether the main pipeline is running or not: while
        running it drives the real InputCapture (add_port/remove_port); while
        stopped it drives a standalone PortListener so the row still shows
        live status. Either way this is the single source of truth for
        whether a port is being listened to.
        """
        if self._agg_port_active[idx] == want_active:
            return
        port = self._agg_ports[idx].value()
        btn  = self._agg_port_listen_btns[idx]
        if want_active:
            if self._pipeline_running:
                capture = self._modules.get("capture")
                try:
                    capture.add_port(port)
                    capture.bind_errors.pop(port, None)
                except OSError as exc:
                    capture.bind_errors[port] = str(exc)
                    self._status.showMessage(f"⚠ Couldn't listen on port {port}: {exc}")
            else:
                try:
                    listener = PortListener(port=port, out_queue=queue.Queue(maxsize=50))
                    listener.start()
                    self._agg_port_listeners[idx] = listener
                    self._agg_standalone_bind_errors.pop(port, None)
                except OSError as exc:
                    self._agg_standalone_bind_errors[port] = str(exc)
                    self._status.showMessage(f"⚠ Couldn't listen on port {port}: {exc}")
            self._agg_port_active[idx] = True
            btn.setText("■")
            btn.setToolTip(f"Listening on port {port} — click to pause")
        else:
            if self._pipeline_running:
                capture = self._modules.get("capture")
                if capture:
                    capture.remove_port(port)
                    capture.bind_errors.pop(port, None)
            else:
                self._stop_standalone_listener(idx)
                self._agg_standalone_bind_errors.pop(port, None)
            self._agg_port_active[idx] = False
            btn.setText("▶")
            btn.setToolTip("Start listening on this port")
            self._agg_port_status_labels[idx].setText("")

    def _stop_standalone_listener(self, idx: int):
        listener = self._agg_port_listeners[idx]
        if listener is not None:
            listener.stop()
            listener.join(timeout=1.0)
            self._agg_port_listeners[idx] = None

    def _resume_standalone_listener(self, idx: int):
        port = self._agg_ports[idx].value()
        try:
            listener = PortListener(port=port, out_queue=queue.Queue(maxsize=50))
            listener.start()
            self._agg_port_listeners[idx] = listener
            self._agg_standalone_bind_errors.pop(port, None)
        except OSError as exc:
            self._agg_standalone_bind_errors[port] = str(exc)

    def _on_agg_port_value_changed(self, spin, new_value):
        idx = self._agg_ports.index(spin)
        old_value = getattr(spin, "_prev_value", new_value)
        spin._prev_value = new_value
        if old_value == new_value or not self._agg_port_active[idx]:
            return
        if self._pipeline_running:
            capture = self._modules.get("capture")
            if capture:
                capture.remove_port(old_value)
                capture.bind_errors.pop(old_value, None)
                try:
                    capture.add_port(new_value)
                    capture.bind_errors.pop(new_value, None)
                except OSError as exc:
                    capture.bind_errors[new_value] = str(exc)
                    self._status.showMessage(f"⚠ Couldn't listen on port {new_value}: {exc}")
        else:
            self._stop_standalone_listener(idx)
            self._agg_standalone_bind_errors.pop(old_value, None)
            try:
                listener = PortListener(port=new_value, out_queue=queue.Queue(maxsize=50))
                listener.start()
                self._agg_port_listeners[idx] = listener
                self._agg_standalone_bind_errors.pop(new_value, None)
            except OSError as exc:
                self._agg_standalone_bind_errors[new_value] = str(exc)
                self._status.showMessage(f"⚠ Couldn't listen on port {new_value}: {exc}")

    def _on_port_out_changed(self, value):
        if self._pipeline_running:
            sender   = self._modules.get("sender")
            receiver = self._modules.get("receiver")
            if sender:
                sender.set_dest_port(value)
            if receiver:
                try:
                    receiver.rebind(value)
                except OSError as exc:
                    self._status.showMessage(f"⚠ Couldn't bind receiver to port {value}: {exc}")

    def _on_hold_ms_changed(self, value):
        if self._pipeline_running:
            jitter = self._modules.get("jitter")
            if jitter:
                jitter.set_hold_ms(value)

    def _toggle_port_listener(self, row_widget):
        idx = self._agg_port_rows.index(row_widget)
        self._set_port_active(idx, not self._agg_port_active[idx])

    def _update_port_status_labels(self):
        """Refresh each Aggregator port row's status label.

        A row only shows anything if it's toggled active (▶ pressed). While
        the real pipeline is running this reflects the actual InputCapture
        listener (live packet count, or "bind failed"); while stopped it
        reflects the standalone listener started for that same toggle — same
        button, same state, whether the pipeline is running or not.
        """
        capture = self._modules.get("capture") if self._pipeline_running else None
        for i, (spin, label) in enumerate(zip(self._agg_ports, self._agg_port_status_labels)):
            if not self._agg_port_active[i]:
                label.setText("")
                continue
            port = spin.value()
            if capture:
                if port in capture.bind_errors:
                    label.setStyleSheet("color: #E53935;")
                    label.setText("bind failed")
                else:
                    stats = capture.port_stats()
                    label.setStyleSheet("color: #4CAF50;")
                    label.setText(f"{stats.get(port, 0)} pkts")
            elif port in self._agg_standalone_bind_errors:
                label.setStyleSheet("color: #E53935;")
                label.setText("bind failed")
            else:
                listener = self._agg_port_listeners[i]
                if listener is not None:
                    label.setStyleSheet("color: #4CAF50;")
                    label.setText(f"{listener.stats['data_received']} pkts")

    def _update_agg_remove_enabled(self):
        enabled = len(self._agg_ports) > 1
        for btn in self._agg_port_remove_btns:
            btn.setEnabled(enabled)

    def _stop_all_standalone_listeners(self):
        """Stop every standalone PortListener without changing active/button state.

        Used to release ports before Start hands them off to the real
        InputCapture, and for final cleanup on close.
        """
        for idx in range(len(self._agg_port_listeners)):
            self._stop_standalone_listener(idx)

    def _add_agg_port(self):
        self._add_agg_port_row(default_port=self._next_default_agg_port())

    def _remove_specific_agg_port(self, row_widget):
        if len(self._agg_ports) <= 1:
            return
        idx = self._agg_port_rows.index(row_widget)
        self._remove_agg_port_row(idx)

    def _next_default_agg_port(self) -> int:
        used = {sb.value() for sb in self._agg_ports}
        port = 50001
        while port in used:
            port += 1
        return port

    def _build_generator(self, panel) -> DifiGenerator:
        rf_ref  = self._rf_ref_for(panel)
        tone_bb = panel.tone_hz() - rf_ref
        return DifiGenerator(
            stream_id       = panel.stream_id(),
            tone_hz         = tone_bb,
            signal_type     = panel.signal_type(),
            dest_port       = panel.port(),
            sample_rate_hz  = self._shared_fs.value_hz(),
            samples_per_pkt = self.SAMPLES_PER_PKT,
            bit_depth       = self.BIT_DEPTH,
            rf_ref_freq_hz  = rf_ref,
            bandwidth_hz    = panel.bandwidth_hz(),
            ref_level_dbm   = panel.amplitude_dbm(),
        )

    def _start_generator_thread(self, gen: DifiGenerator):
        pkt_rate = self._shared_fs.value_hz() / self.SAMPLES_PER_PKT
        threading.Thread(target=gen.run, kwargs=dict(packet_rate_hz=pkt_rate), daemon=True).start()

    def _active_stream_ids(self) -> set:
        """Stream IDs of generators that are actually transmitting (not OFF).

        The Aggregator's fixed mode waits for ALL expected streams before
        emitting any chunk — an OFF generator sends nothing at all, so it
        must be excluded or the whole pipeline would stall forever.
        """
        return {p.stream_id() for p in self._gen_panels if p.signal_type() != SIGNAL_OFF}

    def _add_generator(self, default_signal_type: str = SIGNAL_OFF):
        idx  = len(self._gen_panels) + 1
        port = self._next_default_port()
        sid  = self._next_default_stream_id()
        panel = GeneratorPanel(
            default_signal_type = default_signal_type,
            default_port        = port,
            default_stream_id   = sid,
            default_tone_hz     = idx * 1e6,
        )
        panel.changed.connect(self._live_update_generators)
        self._gen_panels.append(panel)
        self._gen_tabs.addTab(panel, f"Generator {idx}")
        self._gen_tabs.setCurrentWidget(panel)
        self._renumber_tabs()
        self._remove_gen_btn.setEnabled(len(self._gen_panels) > 1)
        if self._pipeline_running:
            gen = self._build_generator(panel)
            self._modules["gens"].append(gen)
            self._start_generator_thread(gen)
            self._modules["aggregator"].update_stream_filter(self._active_stream_ids())

    def _remove_generator(self):
        if len(self._gen_panels) <= 1:
            return
        idx = self._gen_tabs.currentIndex()
        if idx < 0:
            return
        panel = self._gen_panels.pop(idx)
        self._gen_tabs.removeTab(idx)
        panel.deleteLater()
        self._renumber_tabs()
        self._remove_gen_btn.setEnabled(len(self._gen_panels) > 1)
        if self._pipeline_running:
            gen = self._modules["gens"].pop(idx)
            gen.close()
            self._modules["aggregator"].update_stream_filter(self._active_stream_ids())

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

        panels    = self._gen_panels
        # Only ports currently toggled on (▶) are listened to — Start respects
        # whatever's currently configured, it doesn't force-listen on everything added.
        agg_ports = [sb.value() for sb, active in zip(self._agg_ports, self._agg_port_active) if active]
        sids      = [p.stream_id() for p in panels]
        port_out  = self._port_out.value()

        listen_ports = agg_ports + [port_out]
        if len(set(listen_ports)) != len(listen_ports):
            self._status.showMessage(
                "⚠ Aggregator listening ports must be unique — "
                "two or more input/receiver ports are the same"
            )
            return
        if len(set(sids)) != len(sids):
            self._status.showMessage(
                "⚠ Stream IDs must be unique — two or more generators share one"
            )
            return

        self._stop_all_standalone_listeners()   # hand active ports off to the real InputCapture

        fs = self._shared_fs.value_hz()

        gens = [self._build_generator(panel) for panel in panels]

        capture    = InputCapture(ports=agg_ports)
        jitter     = JitterBuffer(capture, hold_ms=self._hold_ms.value())
        aggregator = Aggregator(
            capture          = jitter,
            expected_streams = self._active_stream_ids(),
            chunk_size       = self.SAMPLES_PER_PKT,
        )
        packetizer = Packetizer(aggregator=aggregator)
        sender     = DifiSender(packetizer=packetizer, dest_port=port_out)

        try:
            receiver = DifiReceiver(port=port_out)
            receiver.start()
        except OSError as exc:
            self._status.showMessage(f"⚠ Couldn't bind receiver to port {port_out}: {exc}")
            return
        time.sleep(0.05)

        self._modules = dict(
            gens=gens,
            capture=capture, jitter=jitter, aggregator=aggregator,
            packetizer=packetizer, sender=sender, receiver=receiver,
        )

        capture.start();   time.sleep(0.05)
        jitter.start()
        aggregator.start()
        packetizer.start()
        sender.start()

        status_extra = ""
        if capture.bind_errors:
            failed = ", ".join(f"{p} ({err})" for p, err in capture.bind_errors.items())
            status_extra = f" | ⚠ Failed to listen on: {failed}"

        # Warn if any baseband tone still exceeds Nyquist after LO auto-assignment.
        # Stored persistently (not just shown once) since the recurring spectrum
        # status update below would otherwise silently overwrite it within ~100ms.
        nyquist = fs / 2.0
        alias_warnings = []
        for i, (panel, gen) in enumerate(zip(panels, gens), start=1):
            if abs(gen.tone_hz) >= nyquist:
                alias_warnings.append(
                    f"Gen{i} RF={panel.tone_hz()/1e6:.3f}MHz bb={gen.tone_hz/1e6:.3f}MHz"
                )
        self._pipeline_warning = (
            f" | ⚠ Tone exceeds Nyquist ({nyquist/1e6:.3f}MHz): "
            + ", ".join(alias_warnings)
            + " — set RF Reference to the correct LO"
        ) if alias_warnings else ""
        status_extra += self._pipeline_warning

        for gen in gens:
            self._start_generator_thread(gen)

        self._pipeline_running = True
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)

        gen_str = " | ".join(
            f"Gen{i}:{p.signal_type()} {p.tone_hz()/1e6:.3f}MHz" for i, p in enumerate(panels, start=1)
        )
        self._status.showMessage(f"Running — fs={fs/1e6:.1f}MHz | {gen_str}{status_extra}")

    def _stop_pipeline(self, resume_standalone: bool = True):
        if not self._pipeline_running:
            return
        m = self._modules
        m["sender"].stop()
        m["packetizer"].stop()
        m["aggregator"].stop()
        m["jitter"].stop()
        m["capture"].stop()
        m["receiver"].stop()
        for gen in m["gens"]:
            gen.close()
        self._modules = {}
        self._pipeline_running = False
        self._pipeline_warning = ""
        self._clear_plots()
        if resume_standalone:
            # Ports that were active stay "live" via standalone listeners even
            # while stopped, and are handed straight back to Start next time.
            for idx in range(len(self._agg_port_active)):
                if self._agg_port_active[idx]:
                    self._resume_standalone_listener(idx)
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._status.showMessage("Stopped")

    def _clear_plots(self):
        for curve in self._curves1.values():
            curve.setData([], [])
        for curve in self._curves2.values():
            curve.setData([], [])
        self._plot1_label.setText("source: Aggregator.last_chunk  (float32, no encoding)")
        self._plot2_label.setText("waiting for receiver packets…")

    # How long since the last fresh data before a plot/stream is treated as
    # dead and cleared, rather than left showing a frozen last snapshot.
    STALE_DATA_TIMEOUT_S = 1.5

    def _update_spectrum(self):
        agg = self._modules.get("aggregator")
        if agg is None:
            self._clear_plots()
            return
        chunk = agg.last_chunk
        if chunk is None:
            return
        now = time.monotonic()
        if now - chunk.created_at > self.STALE_DATA_TIMEOUT_S:
            # Aggregator hasn't produced anything new in a while — nothing is
            # actually flowing right now, so don't keep showing the old chunk.
            self._clear_plots()
            return

        # ── Plot 1: aggregator — per-stream spectra (generator inputs) ─────────
        active_sids1 = []
        seen_sids1 = set()
        for s in chunk.streams:
            if s.context is None or len(s.samples) == 0:
                continue
            sid = s.stream_id
            seen_sids1.add(sid)
            active_sids1.append(f"0x{sid:08X}")
            if sid not in self._curves1:
                self._curves1[sid] = self._plot.plot([], [], pen=_stream_color(sid))
            f, m = _stream_fft(s.samples, s.context)
            self._curves1[sid].setData(f, m)
        # Clear curves for any stream that's no longer part of the latest chunk
        # (e.g. a generator was switched OFF) instead of leaving it frozen.
        for sid, curve in self._curves1.items():
            if sid not in seen_sids1:
                curve.setData([], [])
        self._plot1_label.setText(
            f"source: Aggregator.last_chunk  (float32, no encoding)  |  "
            f"chunks: {agg.chunks_emitted:,}  |  streams: {' + '.join(active_sids1)}"
            if active_sids1 else "source: Aggregator.last_chunk  (float32, no encoding)"
        )

        # ── Plot 2: DIFI receiver — per-stream spectra (combiner output) ────────
        rx = self._modules.get("receiver")
        if rx:
            snaps      = rx.get_stream_snapshots()
            last_seen  = rx.stream_last_seen()
            active_sids2 = []
            seen_sids2   = set()
            for sid, (iq, ctx_s) in snaps.items():
                if ctx_s is None or len(iq) == 0:
                    continue
                if now - last_seen.get(sid, 0.0) > self.STALE_DATA_TIMEOUT_S:
                    continue   # stream has gone silent — don't keep redrawing its last snapshot
                seen_sids2.add(sid)
                active_sids2.append(f"0x{sid:08X}")
                if sid not in self._curves2:
                    self._curves2[sid] = self._plot2.plot([], [], pen=_stream_color(sid))
                f, m = _stream_fft(iq, ctx_s)
                self._curves2[sid].setData(f, m)
            for sid, curve in self._curves2.items():
                if sid not in seen_sids2:
                    curve.setData([], [])
            sid_list = " + ".join(active_sids2)
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
        seq_errors = rx.seq_errors if rx else 0
        dropped    = agg.packets_dropped
        loss_str = f"  seq_errors={seq_errors}  dropped={dropped}" if (seq_errors or dropped) else ""
        self._status.showMessage(
            f"Running — fs={fs0/1e6:.1f} MHz{rf_str} | "
            f"chunks={agg.chunks_emitted}  rx={rx_pkts}  "
            f"latency≈{chunk_age_ms:.1f} ms{loss_str}{self._pipeline_warning}"
        )

    def _live_update_generators(self):
        if not self._pipeline_running:
            return
        gens = self._modules.get("gens")
        if not gens or len(gens) != len(self._gen_panels):
            return
        fs = self._shared_fs.value_hz()
        for gen, panel in zip(gens, self._gen_panels):
            rf_ref = self._rf_ref_for(panel)
            gen.update_params(
                tone_hz        = panel.tone_hz() - rf_ref,
                signal_type    = panel.signal_type(),
                bandwidth_hz   = panel.bandwidth_hz(),
                rf_ref_freq_hz = rf_ref,
                ref_level_dbm  = panel.amplitude_dbm(),
                sample_rate_hz = fs,
                dest_port      = panel.port(),
                stream_id      = panel.stream_id(),
            )
        # Drain pipeline queues so stale chunks with old context don't reach the
        # receiver after parameter changes, then flush the receiver IQ buffer.
        agg = self._modules.get("aggregator")
        if agg:
            agg.update_stream_filter(self._active_stream_ids())
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
        self._stop_pipeline(resume_standalone=False)
        self._stop_all_standalone_listeners()
        event.accept()


def main():
    pg.setConfigOptions(antialias=True)
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
