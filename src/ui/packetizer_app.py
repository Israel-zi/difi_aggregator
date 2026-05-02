"""
packetizer_app.py
-----------------
DIFI Aggregator — Packetizer GUI.

Runs on the Packetizer VM.
Receives two DIFI streams from the Transmitter VMs, aggregates them into a
single unified stream, and forwards it to the Receiver VM.
"""

import os
import sys

_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src not in sys.path:
    sys.path.insert(0, _src)

import time

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QGroupBox, QStatusBar,
    QLineEdit, QSpinBox,
)
import pyqtgraph as pg

from modules.input_capture import InputCapture
from modules.aggregator    import Aggregator
from modules.packetizer    import Packetizer
from modules.sender        import DifiSender


def _parse_hex(text: str) -> int:
    return int(text.strip(), 16)


class StreamRow(QWidget):
    """One input stream row: port + stream ID + live status LED."""

    def __init__(self, n: int, default_port: int, default_sid: str, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        lay.addWidget(QLabel(f"Stream {n}"))

        lay.addWidget(QLabel("Port:"))
        self._port = QSpinBox()
        self._port.setRange(1024, 65535)
        self._port.setValue(default_port)
        self._port.setFixedWidth(80)
        lay.addWidget(self._port)

        lay.addWidget(QLabel("Stream ID:"))
        self._sid = QLineEdit(default_sid)
        self._sid.setFixedWidth(120)
        lay.addWidget(self._sid)

        self._led = QLabel("●")
        self._led.setStyleSheet("color: #444444; font-size: 20px;")
        lay.addWidget(self._led)

        lay.addStretch()

    def port(self) -> int:
        return self._port.value()

    def stream_id(self) -> int:
        try:
            return _parse_hex(self._sid.text())
        except ValueError:
            return 0

    def set_active(self, active: bool):
        color = "#00cc44" if active else "#444444"
        self._led.setStyleSheet(f"color: {color}; font-size: 20px;")


class PacketizerWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("DIFI Packetizer")
        self.setMinimumSize(520, 520)
        self._running = False
        self._modules = {}
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(8)

        # ── inputs ──
        in_box  = QGroupBox("Inputs  (from Transmitter VMs)")
        in_grid = QGridLayout(in_box)

        self._row1 = StreamRow(1, default_port=50001, default_sid="0x00000001")
        self._row2 = StreamRow(2, default_port=50002, default_sid="0x00000002")
        in_grid.addWidget(self._row1, 0, 0)
        in_grid.addWidget(self._row2, 1, 0)

        in_grid.addWidget(QLabel("Chunk size:"), 2, 0)
        chunk_w = QWidget()
        chunk_l = QHBoxLayout(chunk_w)
        chunk_l.setContentsMargins(0, 0, 0, 0)
        self._chunk = QSpinBox()
        self._chunk.setRange(64, 65536)
        self._chunk.setSingleStep(256)
        self._chunk.setValue(1024)
        self._chunk.setSuffix(" samples")
        self._chunk.setFixedWidth(150)
        chunk_l.addWidget(self._chunk)
        chunk_l.addStretch()
        in_grid.addWidget(chunk_w, 2, 0)
        root.addWidget(in_box)

        # ── output ──
        out_box  = QGroupBox("Output  (to Receiver VM)")
        out_grid = QGridLayout(out_box)

        out_grid.addWidget(QLabel("Receiver VM IP:"), 0, 0)
        self._dest_ip = QLineEdit("127.0.0.1")
        self._dest_ip.setPlaceholderText("e.g. 192.168.1.30")
        self._dest_ip.setFixedWidth(160)
        out_grid.addWidget(self._dest_ip, 0, 1)

        out_grid.addWidget(QLabel("Port:"), 1, 0)
        self._dest_port = QSpinBox()
        self._dest_port.setRange(1024, 65535)
        self._dest_port.setValue(50010)
        self._dest_port.setFixedWidth(90)
        dest_port_w = QWidget()
        dest_port_l = QHBoxLayout(dest_port_w)
        dest_port_l.setContentsMargins(0, 0, 0, 0)
        dest_port_l.addWidget(self._dest_port)
        dest_port_l.addStretch()
        out_grid.addWidget(dest_port_w, 1, 1)

        out_grid.addWidget(QLabel("Unified Stream ID:"), 2, 0)
        self._unified_sid = QLineEdit("0xAA000000")
        self._unified_sid.setFixedWidth(120)
        unified_w = QWidget()
        unified_l = QHBoxLayout(unified_w)
        unified_l.setContentsMargins(0, 0, 0, 0)
        unified_l.addWidget(self._unified_sid)
        unified_l.addStretch()
        out_grid.addWidget(unified_w, 2, 1)
        root.addWidget(out_box)

        # ── statistics ──
        stats_box  = QGroupBox("Statistics")
        stats_grid = QGridLayout(stats_box)

        self._lbl_chunks = QLabel("0")
        self._lbl_pkts   = QLabel("0")
        self._lbl_drops  = QLabel("0")
        self._lbl_rate   = QLabel("—")

        stats_grid.addWidget(QLabel("Chunks emitted:"),    0, 0)
        stats_grid.addWidget(self._lbl_chunks,             0, 1)
        stats_grid.addWidget(QLabel("Packets forwarded:"), 1, 0)
        stats_grid.addWidget(self._lbl_pkts,               1, 1)
        stats_grid.addWidget(QLabel("Drops:"),             2, 0)
        stats_grid.addWidget(self._lbl_drops,              2, 1)
        stats_grid.addWidget(QLabel("Chunk rate:"),        3, 0)
        stats_grid.addWidget(self._lbl_rate,               3, 1)
        root.addWidget(stats_box)

        root.addStretch()

        # ── buttons ──
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
        self._status.showMessage("Ready — configure and press Start")

        self._timer = QTimer()
        self._timer.setInterval(500)
        self._timer.timeout.connect(self._tick)
        self._prev_chunks = 0
        self._prev_tick_t = 0.0

    def _start(self):
        if self._running:
            return

        try:
            ports      = [self._row1.port(), self._row2.port()]
            stream_ids = [self._row1.stream_id(), self._row2.stream_id()]
            chunk_size = self._chunk.value()
            dest_ip    = self._dest_ip.text().strip()
            dest_port  = self._dest_port.value()
            unified_id = _parse_hex(self._unified_sid.text())
        except ValueError as e:
            self._status.showMessage(f"Config error: {e}")
            return

        capture    = InputCapture(ports=ports)
        aggregator = Aggregator(
            capture          = capture,
            expected_streams = stream_ids,
            chunk_size       = chunk_size,
        )
        packetizer = Packetizer(
            aggregator        = aggregator,
            unified_stream_id = unified_id,
        )
        sender = DifiSender(
            packetizer = packetizer,
            dest_host  = dest_ip,
            dest_port  = dest_port,
        )

        self._modules = dict(
            capture=capture, aggregator=aggregator,
            packetizer=packetizer, sender=sender,
        )
        self._stream_ids = stream_ids

        capture.start()
        time.sleep(0.05)
        aggregator.start()
        packetizer.start()
        sender.start()

        self._running = True
        self._prev_chunks = 0
        self._prev_tick_t = time.monotonic()
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._timer.start()
        self._status.showMessage(
            f"Forwarding to {dest_ip}:{dest_port}  |  "
            f"unified stream_id=0x{unified_id:08X}"
        )

    def _stop(self):
        if not self._running:
            return
        self._timer.stop()
        m = self._modules
        m["sender"].stop()
        m["packetizer"].stop()
        m["aggregator"].stop()
        m["capture"].stop()
        self._row1.set_active(False)
        self._row2.set_active(False)
        self._running = False
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._status.showMessage("Stopped")

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
        drops = agg.packets_dropped + (pkt.packets_dropped if pkt else 0)
        self._lbl_drops.setText(str(drops))
        self._lbl_rate.setText(f"{rate:.1f} chunks/s")

        # stream LED: green if seen in the last 3 seconds
        last_seen = agg.stream_last_seen()
        cutoff    = time.monotonic() - 3.0
        rows      = {self._stream_ids[0]: self._row1, self._stream_ids[1]: self._row2}
        for sid, row in rows.items():
            t = last_seen.get(sid)
            row.set_active(t is not None and t >= cutoff)

    def closeEvent(self, event):
        self._stop()
        event.accept()


def main():
    app = QApplication(sys.argv)
    win = PacketizerWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
