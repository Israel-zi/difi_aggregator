"""
packetizer_app.py
-----------------
DIFI Aggregator — Combiner GUI.

Runs on the Combiner VM.
Receives DIFI streams from Transmitter VMs on configurable listen ports,
re-packetizes them preserving original Stream IDs (read from the DIFI
packet header — not configured here), and forwards all streams to a single
destination port on the Receiver VM.

Includes a live spectrum display of the aggregated streams.
"""

import os
import sys

if not getattr(sys, 'frozen', False):
    _src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _src not in sys.path:
        sys.path.insert(0, _src)

import time
import numpy as np

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QGroupBox, QStatusBar,
    QLineEdit, QSpinBox, QDoubleSpinBox, QScrollArea, QFrame, QSplitter,
)
import pyqtgraph as pg

from modules.input_capture import InputCapture
from modules.aggregator    import Aggregator
from modules.packetizer    import Packetizer
from modules.sender        import DifiSender
from ui.freq_input         import FreqInput


class StreamRow(QWidget):
    """One listen-stream row: index + port + auto-discovered stream ID + LED + remove."""

    removed = Signal(object)

    def __init__(self, index: int, default_port: int, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(6)

        self._idx_lbl = QLabel(f"{index}")
        self._idx_lbl.setFixedWidth(18)
        self._idx_lbl.setStyleSheet("color: #888888;")
        lay.addWidget(self._idx_lbl)

        lay.addWidget(QLabel("Port:"))
        self._port = QSpinBox()
        self._port.setRange(1024, 65535)
        self._port.setValue(default_port)
        self._port.setFixedWidth(110)
        lay.addWidget(self._port)

        self._sid_lbl = QLabel("(waiting…)")
        self._sid_lbl.setStyleSheet("color: #666666; font-size: 11px;")
        self._sid_lbl.setFixedWidth(110)
        lay.addWidget(self._sid_lbl)

        self._led = QLabel("●")
        self._led.setStyleSheet("color: #444444; font-size: 18px;")
        lay.addWidget(self._led)

        self._remove_btn = QPushButton("−")
        self._remove_btn.setFixedSize(26, 26)
        self._remove_btn.setToolTip("Remove this stream")
        self._remove_btn.clicked.connect(lambda: self.removed.emit(self))
        lay.addWidget(self._remove_btn)

        lay.addStretch()

    def set_index(self, n: int):
        self._idx_lbl.setText(str(n))

    def port(self) -> int:
        return self._port.value()

    def set_stream_id(self, sid: int):
        self._sid_lbl.setText(f"0x{sid:08X}")
        self._sid_lbl.setStyleSheet("color: #aaaaaa; font-size: 11px;")

    def set_active(self, active: bool):
        self._led.setStyleSheet(f"color: {'#00cc44' if active else '#444444'}; font-size: 18px;")

    def set_locked(self, locked: bool):
        self._port.setEnabled(not locked)
        self._remove_btn.setEnabled(not locked)


class PacketizerWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("DIFI Combiner")
        self.setMinimumSize(960, 560)
        self._running     = False
        self._modules     = {}
        self._stream_rows: list = []
        self._build_ui()
        self._add_stream_row(50001)
        self._add_stream_row(50002)

    # ── UI construction ────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        # ── left panel: controls ──────────────────────────────────────────
        left     = QWidget()
        left.setMaximumWidth(340)
        left_lay = QVBoxLayout(left)
        left_lay.setSpacing(8)

        # Inputs
        in_box  = QGroupBox("Inputs  (listen for streams from Transmitter VMs)")
        in_vlay = QVBoxLayout(in_box)
        in_vlay.setSpacing(4)

        hdr = QHBoxLayout()
        hdr.addSpacing(22)
        for lbl, w in [("Port", 100), ("Stream ID (auto)", 110), ("Live", 30)]:
            l = QLabel(lbl)
            l.setFixedWidth(w)
            l.setStyleSheet("color: #888888; font-size: 11px;")
            hdr.addWidget(l)
        hdr.addStretch()
        in_vlay.addLayout(hdr)

        self._rows_container = QWidget()
        self._rows_layout    = QVBoxLayout(self._rows_container)
        self._rows_layout.setSpacing(2)
        self._rows_layout.setContentsMargins(2, 2, 2, 2)
        self._rows_layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(self._rows_container)
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(70)
        scroll.setMaximumHeight(180)
        scroll.setFrameShape(QFrame.StyledPanel)
        in_vlay.addWidget(scroll)

        self._add_btn = QPushButton("＋  Add Stream")
        self._add_btn.setFixedHeight(28)
        self._add_btn.clicked.connect(lambda: self._add_stream_row())
        in_vlay.addWidget(self._add_btn)

        chunk_row = QHBoxLayout()
        chunk_row.addWidget(QLabel("Chunk size:"))
        self._chunk = QSpinBox()
        self._chunk.setRange(64, 65536)
        self._chunk.setSingleStep(256)
        self._chunk.setValue(1024)
        self._chunk.setSuffix(" samples")
        self._chunk.setFixedWidth(150)
        chunk_row.addWidget(self._chunk)
        chunk_row.addStretch()
        in_vlay.addLayout(chunk_row)
        left_lay.addWidget(in_box)

        # Output
        out_box  = QGroupBox("Output  (to Receiver VM)")
        out_vlay = QVBoxLayout(out_box)

        ip_row = QHBoxLayout()
        ip_row.addWidget(QLabel("Receiver VM IP:"))
        self._dest_ip = QLineEdit("127.0.0.1")
        self._dest_ip.setPlaceholderText("e.g. 192.168.1.30")
        self._dest_ip.setFixedWidth(160)
        ip_row.addWidget(self._dest_ip)
        ip_row.addStretch()
        out_vlay.addLayout(ip_row)

        port_row = QHBoxLayout()
        port_row.addWidget(QLabel("Destination port:"))
        self._dest_port = QSpinBox()
        self._dest_port.setRange(1024, 65535)
        self._dest_port.setValue(50010)
        self._dest_port.setFixedWidth(110)
        port_row.addWidget(self._dest_port)
        port_row.addStretch()
        out_vlay.addLayout(port_row)
        left_lay.addWidget(out_box)

        # Stats
        stats_box  = QGroupBox("Statistics")
        stats_vlay = QVBoxLayout(stats_box)

        row_a = QHBoxLayout()
        row_a.addWidget(QLabel("Chunks emitted:"))
        self._lbl_chunks = QLabel("0")
        row_a.addWidget(self._lbl_chunks)
        row_a.addSpacing(16)
        row_a.addWidget(QLabel("Pkts forwarded:"))
        self._lbl_pkts = QLabel("0")
        row_a.addWidget(self._lbl_pkts)
        row_a.addStretch()
        stats_vlay.addLayout(row_a)

        row_b = QHBoxLayout()
        row_b.addWidget(QLabel("Drops:"))
        self._lbl_drops = QLabel("0")
        row_b.addWidget(self._lbl_drops)
        row_b.addSpacing(16)
        row_b.addWidget(QLabel("Chunk rate:"))
        self._lbl_rate = QLabel("—")
        row_b.addWidget(self._lbl_rate)
        row_b.addStretch()
        stats_vlay.addLayout(row_b)
        left_lay.addWidget(stats_box)

        left_lay.addStretch()

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

        # Display controls
        disp_box  = QGroupBox("Display")
        disp_vlay = QVBoxLayout(disp_box)
        disp_vlay.setSpacing(4)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Center:"))
        self._disp_center = FreqInput(default_hz=2.5e6)
        row1.addWidget(self._disp_center)
        row1.addSpacing(16)
        row1.addWidget(QLabel("Span:"))
        self._disp_span = FreqInput(default_hz=5e6)
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
        row2.addStretch()
        disp_vlay.addLayout(row2)

        self._disp_center.changed.connect(self._apply_range)
        self._disp_span.changed.connect(self._apply_range)
        self._disp_amp.valueChanged.connect(self._apply_range)
        self._disp_dbdiv.valueChanged.connect(self._apply_range)

        right_lay.addWidget(disp_box)

        # Spectrum plot
        self._plot = pg.PlotWidget(
            title="Combiner — incoming streams (pre-encode)"
        )
        self._plot.setLabel("bottom", "Frequency", units="Hz")
        self._plot.setLabel("left",   "Magnitude", units="dB")
        self._plot.showGrid(x=True, y=True, alpha=0.3)
        self._plot.enableAutoRange(axis="xy", enable=False)
        self._plot.getPlotItem().getViewBox().enableAutoRange(enable=False)
        self._curve = self._plot.plot([], [], pen=pg.mkPen("c", width=1))
        self._ref_line = pg.InfiniteLine(
            angle=0, movable=False,
            pen=pg.mkPen("y", width=1, style=Qt.DashLine),
        )
        self._plot.addItem(self._ref_line)

        self._plot.getPlotItem().getViewBox().sigRangeChanged.connect(
            lambda vb, ranges: self._sync_viewport_to_spinboxes(ranges)
        )

        right_lay.addWidget(self._plot)
        splitter.addWidget(right)
        splitter.setSizes([320, 640])

        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Ready — add streams and press Start")

        self._stats_timer = QTimer()
        self._stats_timer.setInterval(500)
        self._stats_timer.timeout.connect(self._tick)

        self._spec_timer = QTimer()
        self._spec_timer.setInterval(100)
        self._spec_timer.timeout.connect(self._update_spectrum)

        self._prev_chunks = 0
        self._prev_tick_t = 0.0

        # Apply initial range from spinbox defaults
        self._apply_range()

    # ── dynamic row management ─────────────────────────────────────────────

    def _add_stream_row(self, port: int = None):
        n = len(self._stream_rows) + 1
        if port is None:
            port = (max(r.port() for r in self._stream_rows) + 1) if self._stream_rows else 50001
        row = StreamRow(index=n, default_port=port)
        row.removed.connect(self._remove_stream_row)
        row.set_locked(self._running)
        self._rows_layout.insertWidget(self._rows_layout.count() - 1, row)
        self._stream_rows.append(row)

    def _remove_stream_row(self, row: StreamRow):
        if self._running or len(self._stream_rows) <= 1:
            return
        self._rows_layout.removeWidget(row)
        row.deleteLater()
        self._stream_rows.remove(row)
        for i, r in enumerate(self._stream_rows, start=1):
            r.set_index(i)

    def _set_locked(self, locked: bool):
        for row in self._stream_rows:
            row.set_locked(locked)
        self._add_btn.setEnabled(not locked)
        self._chunk.setEnabled(not locked)
        self._dest_ip.setEnabled(not locked)
        self._dest_port.setEnabled(not locked)
        self._start_btn.setEnabled(not locked)
        self._stop_btn.setEnabled(locked)

    # ── lifecycle ──────────────────────────────────────────────────────────

    def _start(self):
        if self._running:
            return

        ports     = [r.port() for r in self._stream_rows]
        chunk_size = self._chunk.value()
        dest_ip    = self._dest_ip.text().strip()
        dest_port  = self._dest_port.value()

        if len(set(ports)) != len(ports):
            self._status.showMessage("Error: duplicate listen ports")
            return

        capture    = InputCapture(ports=ports)
        aggregator = Aggregator(
            capture          = capture,
            expected_streams = None,
            expected_count   = len(ports),
            chunk_size       = chunk_size,
        )
        packetizer = Packetizer(aggregator=aggregator)
        sender     = DifiSender(
            packetizer = packetizer,
            dest_host  = dest_ip,
            dest_port  = dest_port,
        )

        self._modules = dict(capture=capture, aggregator=aggregator,
                             packetizer=packetizer, sender=sender)

        capture.start()
        time.sleep(0.05)
        aggregator.start()
        packetizer.start()
        sender.start()

        self._running     = True
        self._prev_chunks = 0
        self._prev_tick_t = time.monotonic()
        self._set_locked(True)
        self._stats_timer.start()
        self._spec_timer.start()
        self._apply_range()

        self._status.showMessage(
            f"Running — listening on ports {ports}  →  {dest_ip}:{dest_port}  |  "
            f"discovering {len(ports)} stream(s)…"
        )

    def _stop(self):
        if not self._running:
            return
        self._stats_timer.stop()
        self._spec_timer.stop()
        m = self._modules
        m["sender"].stop()
        m["packetizer"].stop()
        m["aggregator"].stop()
        m["capture"].stop()
        for row in self._stream_rows:
            row.set_active(False)
        self._curve.setData([], [])
        self._modules = {}
        self._running = False
        self._set_locked(False)
        self._status.showMessage("Stopped")

    # ── update loops ───────────────────────────────────────────────────────

    def _tick(self):
        if not self._running:
            return
        agg = self._modules.get("aggregator")
        pkt = self._modules.get("packetizer")
        snd = self._modules.get("sender")
        if not agg:
            return

        chunks = agg.chunks_emitted
        now    = time.monotonic()
        dt     = now - self._prev_tick_t
        rate   = (chunks - self._prev_chunks) / dt if dt > 0 else 0.0
        self._prev_chunks = chunks
        self._prev_tick_t = now

        self._lbl_chunks.setText(f"{chunks:,}")
        self._lbl_pkts.setText(f"{snd.packets_sent:,}" if snd else "0")
        self._lbl_drops.setText(str(agg.packets_dropped + (pkt.packets_dropped if pkt else 0)))
        self._lbl_rate.setText(f"{rate:.1f} chunks/s")

        last_seen   = agg.stream_last_seen()
        cutoff      = now - 3.0
        active_sids = sorted(last_seen.keys())
        for i, row in enumerate(self._stream_rows):
            if i < len(active_sids):
                sid = active_sids[i]
                row.set_stream_id(sid)
                row.set_active(last_seen[sid] >= cutoff)
            else:
                row.set_active(False)

    def _update_spectrum(self):
        agg = self._modules.get("aggregator")
        if not agg:
            return

        # Use a fully emitted chunk when available; fall back to whatever
        # is currently buffered per stream so the spectrum shows even when
        # waiting for all expected streams to connect.
        chunk = agg.last_chunk
        if chunk is not None:
            stream_data = [(s.stream_id, s.samples, s.context) for s in chunk.streams]
        else:
            stream_data = agg.get_stream_previews()

        if not stream_data:
            return

        all_freqs = []
        all_mags  = []
        for _sid, samples, ctx in stream_data:
            n = len(samples)
            if n == 0:
                continue
            fs     = ctx.sample_rate_hz
            lo     = ctx.rf_ref_freq_hz
            window = np.hanning(n)
            X      = np.fft.fftshift(np.fft.fft(samples * window))
            mag_db = 20 * np.log10(np.abs(X) / n + 1e-7)
            freqs  = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / fs)) + lo
            all_freqs.append(freqs)
            all_mags.append(mag_db)

        if all_freqs:
            cf  = np.concatenate(all_freqs)
            cm  = np.concatenate(all_mags)
            idx = np.argsort(cf)
            self._curve.setData(cf[idx], cm[idx])

    # ── display helpers ────────────────────────────────────────────────────

    def _apply_range(self):
        center  = self._disp_center.value_hz()
        span    = self._disp_span.value_hz()
        amp_top = self._disp_amp.value()
        db_div  = self._disp_dbdiv.value()
        self._plot.setXRange(center - span / 2, center + span / 2, padding=0)
        self._plot.setYRange(amp_top - db_div * 10, amp_top, padding=0)
        self._ref_line.setValue(amp_top)

    def _sync_viewport_to_spinboxes(self, ranges):
        x_lo, x_hi = ranges[0]
        if x_hi <= x_lo:
            return
        self._disp_center.set_hz((x_lo + x_hi) / 2.0)
        self._disp_span.set_hz(x_hi - x_lo)

    def closeEvent(self, event):
        self._stop()
        event.accept()


def main():
    pg.setConfigOptions(antialias=True)
    app = QApplication(sys.argv)
    win = PacketizerWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
