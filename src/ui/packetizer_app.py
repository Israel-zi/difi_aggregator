"""
packetizer_app.py
-----------------
DIFI Aggregator — Combiner GUI.

This window is a thin control panel only — it never touches a UDP socket
or an aggregation buffer itself. Pressing Listen spawns the actual packet
pipeline (InputCapture + JitterBuffer + Aggregator + Packetizer) in a
SEPARATE OS PROCESS (see combiner_worker.py) and talks to it over
multiprocessing.Queue. Rate-sweep A/B testing found that running those
worker threads in-process alongside this Qt window hit a hard throughput
ceiling around 2000-2400 pkt/s, while the identical threads run with no Qt
involved at all stayed lossless past 4000 pkt/s -- so the fix is to not
share a process with Qt at all, not to keep tuning around it.

This also means there is no live per-packet/per-chunk stats display here
any more -- that data lives in the worker process, and the whole point is
not adding a channel that pulls it back out every few hundred ms. The CSV
logs (written by the worker, path shown in the status bar) are the
source of truth for what happened during a run.

Two-phase operation:
  ▶ Listen   — starts the worker process, which starts InputCapture +
               JitterBuffer + Aggregator
  ▶ Forward  — tells the worker to additionally start Packetizer (data
               flows to the Receiver)
  ■ Stop     — tells the worker to stop everything and exits its process

Stream rows configure which ports to listen on. Each row's Fwd checkbox
selects whether packets arriving on that port are forwarded — filtering is
by port (what this window actually knows), translated to the discovered
stream ID inside the worker process.
"""

import os
import sys

if not getattr(sys, 'frozen', False):
    _src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _src not in sys.path:
        sys.path.insert(0, _src)

import multiprocessing
import queue as _queue

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QGroupBox, QStatusBar,
    QLineEdit, QSpinBox, QCheckBox,
)

from pipeline_logger    import make_run_dir
from combiner_worker    import worker_main
from gil_friendly_exec  import run_gil_friendly, request_stop


class StreamRow(QWidget):
    """
    One listen-stream row.
    Layout: [#] Port:[spinbox] [Fwd☑] [−]

    Signals
    -------
    removed        — user clicked the remove button
    filter_changed — Fwd checkbox toggled (forwarding filter)
    """

    removed        = Signal(object)
    filter_changed = Signal()

    def __init__(self, index: int, default_port: int, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(6)

        self._idx_lbl = QLabel(f"{index}")
        self._idx_lbl.setFixedWidth(18)
        self._idx_lbl.setStyleSheet("color: #888888;")
        lay.addWidget(self._idx_lbl)

        self._port = QSpinBox()
        self._port.setRange(1024, 65535)
        self._port.setValue(default_port)
        self._port.setFixedWidth(100)
        lay.addWidget(self._port)

        self._sid_lbl = QLabel("(waiting…)")
        self._sid_lbl.setStyleSheet("color: #666666; font-size: 11px;")
        self._sid_lbl.setFixedWidth(120)
        lay.addWidget(self._sid_lbl)

        # Activity LED -- driven by a lightweight heartbeat from the worker
        # process (~1/s, just port->stream-id/active booleans, not a full
        # stats poll), not the earlier per-tick Aggregator-state query that
        # ran in the GUI process.
        self._led = QLabel("●")
        self._led.setStyleSheet("color: #444444; font-size: 18px;")
        self._led.setFixedWidth(22)
        lay.addWidget(self._led)

        # Fwd checkbox — controls forwarding to Receiver. Always enabled:
        # filtering is by port, decided up front, not by the stream ID
        # this row's LED/label display (which just reflects what the
        # worker last reported, for operator visibility).
        self._fwd_cb = QCheckBox()
        self._fwd_cb.setChecked(True)
        self._fwd_cb.setToolTip("Forward packets arriving on this port to the Receiver")
        self._fwd_cb.setFixedWidth(36)
        self._fwd_cb.stateChanged.connect(lambda _: self.filter_changed.emit())
        lay.addWidget(self._fwd_cb)

        self._remove_btn = QPushButton("−")
        self._remove_btn.setFixedSize(26, 26)
        self._remove_btn.setToolTip("Remove this stream")
        self._remove_btn.clicked.connect(lambda: self.removed.emit(self))
        lay.addWidget(self._remove_btn)

        lay.addStretch()

    # ── accessors ──────────────────────────────────────────────────────────

    def set_index(self, n: int):
        self._idx_lbl.setText(str(n))

    def port(self) -> int:
        return self._port.value()

    def forwarded_port(self) -> int | None:
        """Return this row's port if Fwd is checked, else None."""
        return self._port.value() if self._fwd_cb.isChecked() else None

    def set_stream_id(self, sid: int | None):
        if sid is None:
            self._sid_lbl.setText("(waiting…)")
            self._sid_lbl.setStyleSheet("color: #666666; font-size: 11px;")
        else:
            self._sid_lbl.setText(f"0x{sid:08X}")
            self._sid_lbl.setStyleSheet("color: #aaaaaa; font-size: 11px;")

    def set_active(self, active: bool):
        color = "#00cc44" if active else "#444444"
        self._led.setStyleSheet(f"color: {color}; font-size: 18px;")

    def set_locked(self, locked: bool):
        self._port.setEnabled(not locked)
        # Remove button and Fwd checkbox always stay enabled while listening.


class PacketizerWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("DIFI Combiner")
        self.setMinimumSize(900, 480)
        self._listening   = False   # worker process is up and capturing
        self._forwarding  = False   # worker is also forwarding to Receiver
        self._proc        = None    # multiprocessing.Process (worker)
        self._cmd_q       = None
        self._status_q    = None
        self._run_dir     = None    # log folder for the current Listen session
        self._stream_rows: list = []
        self._build_ui()
        self._add_stream_row(50001)
        self._add_stream_row(50002)

    # ── UI construction ────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        left_lay = QVBoxLayout(central)
        left_lay.setSpacing(8)

        # Inputs
        in_box  = QGroupBox("Inputs  (listen for streams from Transmitter VMs)")
        in_vlay = QVBoxLayout(in_box)
        in_vlay.setSpacing(4)

        hdr = QHBoxLayout()
        hdr.setSpacing(6)
        hdr.addSpacing(26)
        for lbl, w in [("Port", 100), ("Stream ID (auto)", 120), ("●", 22), ("Fwd", 36)]:
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
        in_vlay.addWidget(self._rows_container)

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

        # Network / Jitter Buffer
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
        left_lay.addWidget(net_box)

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

        left_lay.addStretch()

        # Buttons: Listen | Forward | Stop
        btn_row = QHBoxLayout()
        self._listen_btn  = QPushButton("▶  Listen")
        self._forward_btn = QPushButton("▶  Forward")
        self._stop_btn    = QPushButton("■  Stop")
        self._forward_btn.setEnabled(False)
        self._stop_btn.setEnabled(False)
        for btn in (self._listen_btn, self._forward_btn, self._stop_btn):
            btn.setFixedHeight(36)
        self._listen_btn.clicked.connect(self._listen)
        self._forward_btn.clicked.connect(self._on_forward_clicked)
        self._stop_btn.clicked.connect(self._stop)
        btn_row.addWidget(self._listen_btn)
        btn_row.addWidget(self._forward_btn)
        btn_row.addWidget(self._stop_btn)
        left_lay.addLayout(btn_row)

        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Ready — add streams and press Listen")

        # Drains the worker's status queue -- state-transition events only
        # (listening/forwarding/stopped/errors), not a live stats poll. This
        # is the one thing still ticking on the GUI thread, and it does
        # nothing but a non-blocking Queue read, so it doesn't reintroduce
        # the contention this whole split exists to avoid.
        self._status_timer = QTimer()
        self._status_timer.setInterval(200)
        self._status_timer.timeout.connect(self._poll_status)
        self._status_timer.start()

    # ── dynamic row management ─────────────────────────────────────────────

    def _add_stream_row(self, port: int = None):
        n = len(self._stream_rows) + 1
        if port is None:
            port = (max(r.port() for r in self._stream_rows) + 1) if self._stream_rows else 50001
        row = StreamRow(index=n, default_port=port)
        row.removed.connect(self._remove_stream_row)
        row.filter_changed.connect(self._on_filter_changed)
        row.set_locked(self._listening)
        self._rows_layout.insertWidget(self._rows_layout.count() - 1, row)
        self._stream_rows.append(row)

        if self._listening:
            self._send_cmd(cmd="add_port", port=port)

    def _remove_stream_row(self, row: StreamRow):
        if len(self._stream_rows) <= 1:
            return
        if self._listening:
            self._send_cmd(cmd="remove_port", port=row.port())
        self._rows_layout.removeWidget(row)
        row.deleteLater()
        self._stream_rows.remove(row)
        for i, r in enumerate(self._stream_rows, start=1):
            r.set_index(i)

    def _set_locked(self, locked: bool):
        for row in self._stream_rows:
            row.set_locked(locked)
        # Add button stays enabled — ports can be added while listening.
        self._chunk.setEnabled(not locked)
        self._hold_ms.setEnabled(not locked)

    # ── IPC helpers ────────────────────────────────────────────────────────

    def _send_cmd(self, **kwargs):
        if self._cmd_q is not None:
            self._cmd_q.put(kwargs)

    def _poll_status(self):
        if self._status_q is None:
            return
        while True:
            try:
                msg = self._status_q.get_nowait()
            except _queue.Empty:
                return
            self._handle_status(msg)

    def _handle_status(self, msg: dict):
        kind = msg.get("status")
        if kind == "listening":
            self._status.showMessage(
                f"Listening on ports {msg['ports']}  |  log: {msg['run_dir']}"
            )
        elif kind == "listen_error":
            self._status.showMessage(f"Listen failed: {msg['error']}")
            self._teardown_process()
            self._reset_ui_to_stopped()
        elif kind == "port_error":
            self._status.showMessage(f"Port {msg['port']} failed: {msg['error']}")
        elif kind == "forwarding":
            host, port = msg["dest"]
            self._status.showMessage(
                f"Running  →  {host}:{port}  |  log: {self._run_dir}"
            )
        elif kind == "forward_stopped":
            self._status.showMessage(f"Forwarding stopped — still listening | log: {self._run_dir}")
        elif kind == "stopped":
            s = msg.get("summary", {})
            self._status.showMessage(
                f"Stopped | received={s.get('data_received', 0):,} "
                f"chunks={s.get('chunks_emitted', 0):,} "
                f"drops={s.get('packets_dropped', 0):,} | log: {msg.get('run_dir', '')}"
            )
        elif kind == "activity":
            ports = msg.get("ports", {})
            for row in self._stream_rows:
                info = ports.get(row.port())
                if info is None:
                    row.set_stream_id(None)
                    row.set_active(False)
                else:
                    row.set_stream_id(info["stream_id"])
                    row.set_active(info["active"])

    # ── lifecycle ──────────────────────────────────────────────────────────

    def _listen(self):
        """Spawn the worker process and tell it to start capturing (no
        forwarding yet)."""
        if self._listening:
            return

        ports      = [r.port() for r in self._stream_rows]
        chunk_size = self._chunk.value()

        if len(set(ports)) != len(ports):
            self._status.showMessage("Error: duplicate listen ports")
            return

        self._run_dir = make_run_dir("Combiner")
        self._cmd_q    = multiprocessing.Queue()
        self._status_q = multiprocessing.Queue()
        self._proc = multiprocessing.Process(
            target=worker_main, args=(self._cmd_q, self._status_q), daemon=True,
        )
        self._proc.start()
        self._send_cmd(
            cmd="listen", ports=ports, chunk_size=chunk_size,
            hold_ms=self._hold_ms.value(), run_dir=self._run_dir,
        )

        self._listening = True
        self._set_locked(True)
        self._listen_btn.setEnabled(False)
        self._forward_btn.setEnabled(True)
        self._stop_btn.setEnabled(True)
        self._status.showMessage(f"Starting — ports {ports}  |  log: {self._run_dir}")

    def _on_forward_clicked(self):
        """Toggle forwarding on/off — Forward button handler."""
        if self._forwarding:
            self._stop_forward()
        else:
            self._forward()

    def _forward(self):
        """Tell the worker to start forwarding the aggregated stream."""
        if not self._listening or self._forwarding:
            return

        dest_ip   = self._dest_ip.text().strip()
        dest_port = self._dest_port.value()
        self._send_cmd(
            cmd="forward", dest_host=dest_ip, dest_port=dest_port,
            forward_ports=self._current_forward_ports(),
        )

        self._forwarding = True
        self._forward_btn.setText("■  Stop Fwd")
        self._dest_ip.setEnabled(False)
        self._dest_port.setEnabled(False)
        self._status.showMessage(f"Starting forward  →  {dest_ip}:{dest_port}")

    def _stop_forward(self):
        """Stop forwarding only — keep listening."""
        if not self._forwarding:
            return
        self._send_cmd(cmd="stop_forward")
        self._forwarding = False
        self._forward_btn.setText("▶  Forward")
        self._dest_ip.setEnabled(True)
        self._dest_port.setEnabled(True)

    def _stop(self):
        """Stop everything and tear down the worker process."""
        if not self._listening:
            return

        self._send_cmd(cmd="shutdown")
        self._teardown_process()
        self._reset_ui_to_stopped()

    def _teardown_process(self):
        """Join the worker process, force-terminating it if it doesn't exit
        promptly -- see closeEvent for why this must never leave an
        orphaned background process still bound to its ports."""
        if self._proc is not None:
            self._proc.join(timeout=3.0)
            if self._proc.is_alive():
                self._proc.terminate()
                self._proc.join(timeout=2.0)
        self._proc     = None
        self._cmd_q    = None
        self._status_q = None

    def _reset_ui_to_stopped(self):
        for row in self._stream_rows:
            row.set_stream_id(None)
            row.set_active(False)
        self._listening  = False
        self._forwarding = False
        self._run_dir    = None
        self._set_locked(False)
        self._listen_btn.setEnabled(True)
        self._forward_btn.setText("▶  Forward")
        self._forward_btn.setEnabled(False)
        self._stop_btn.setEnabled(False)
        self._dest_ip.setEnabled(True)
        self._dest_port.setEnabled(True)

    # ── forwarding filter ──────────────────────────────────────────────────

    def _on_filter_changed(self):
        """Push the updated port-based forward filter to the worker when a
        Fwd checkbox is toggled."""
        if not self._forwarding:
            return
        self._send_cmd(cmd="set_forward_ports", forward_ports=self._current_forward_ports())

    def _current_forward_ports(self) -> list | None:
        """Return the ports to forward, or None to forward everything
        (all rows currently have Fwd checked)."""
        all_checked = all(r.forwarded_port() is not None for r in self._stream_rows)
        if all_checked:
            return None
        return [r.forwarded_port() for r in self._stream_rows if r.forwarded_port() is not None]

    def closeEvent(self, event):
        self._stop()
        event.accept()
        # See gil_friendly_exec.py: app.lastWindowClosed alone was NOT
        # reliably ending the polling loop -- confirmed as a real bug (the
        # process kept running as an orphaned background task after the
        # window closed, still bound to its UDP ports via SO_REUSEADDR,
        # silently competing with the next run for the same packets and
        # CPU). Calling request_stop() directly here, at the one place a
        # close is unambiguously happening, doesn't depend on that signal
        # firing correctly under manual polling at all.
        request_stop(QApplication.instance())


def main():
    # Required on Windows before any other multiprocessing call in a
    # PyInstaller --onefile build -- without it, spawning the worker
    # process re-executes this whole frozen exe from the top instead of
    # running combiner_worker.worker_main, which recurses into spawning
    # another GUI window instead of doing the worker's job.
    multiprocessing.freeze_support()

    from logging_setup import setup_frozen_file_logging
    log_path = setup_frozen_file_logging("Combiner")

    app = QApplication(sys.argv)
    win = PacketizerWindow()
    if log_path:
        win._status.showMessage(f"Logging to {log_path}")
    win.show()
    # See gil_friendly_exec.py: app.exec()'s native Windows event loop was
    # measured, via direct A/B test on this exact app, to starve every
    # other thread in this process (0.0% packet loss with this polling
    # loop vs 41-46% with app.exec(), everything else identical).
    app.lastWindowClosed.connect(lambda: request_stop(app))
    sys.exit(run_gil_friendly(app))


if __name__ == "__main__":
    main()
