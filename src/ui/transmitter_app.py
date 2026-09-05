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
import time

import numpy as np
import scipy.signal as sp_sig

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QDoubleSpinBox, QPushButton,
    QGroupBox, QStatusBar, QLineEdit, QSpinBox, QButtonGroup, QRadioButton,
    QSplitter, QComboBox,
)
import pyqtgraph as pg

from modules.generator import DifiGenerator, SIGNAL_CW, SIGNAL_BW, SIGNAL_OFF, SIGNAL_PATTERN
from ui.freq_input     import FreqInput
from pipeline_logger   import make_run_dir, AsyncPacketLogger as PacketLogger
from gil_friendly_exec import run_gil_friendly, request_stop
import app_config


class TransmitterWindow(QMainWindow):

    # 2026-09-01 field evidence: the real Combiner VM run lost ~71-95% of
    # packets even at modest offered rates while CPU/RAM/NIC throughput all
    # sat idle -- and every local (loopback) A/B test of the pipeline code
    # itself stayed lossless. The one thing loopback categorically cannot
    # reproduce is IP fragmentation: at the OLD default of 1024 samples/pkt,
    # each DIFI Data packet is 1024*4 + 28 (DIFI header) = 4124 bytes of UDP
    # payload -- 4152 bytes on the wire with IP/UDP headers, vs. the
    # standard 1500-byte Ethernet MTU. That forces every single packet into
    # 3 IP fragments, and losing any ONE fragment silently drops the whole
    # packet; Windows' IP reassembly also has a fixed-size concurrent slot
    # table, which is a plausible fit for the measured ~600-700 pkt/s
    # ceiling holding flat regardless of a 2441/4883/11719 pkt/s offered
    # rate (see tools/mtu_safe_packet_test.py). 360 samples/pkt kept the
    # whole packet at 1496 bytes on the wire -- under the standard 1500
    # MTU, so it never fragmented.
    #
    # 2026-09-03: DIFI_Standard_1.3.0_Final.pdf (Base/) confirms the
    # standard's own jumbo-frame provision is a 9000-byte MTU (p.15: "Jumbo
    # frames with maximum transmission unit of 9000 bytes"; p.21/p.26 cap
    # data packets at "maximum Ethernet packet size of 9000 bytes"), with
    # fixed overhead of IP(20)+UDP(8)+VITA(28)=56 bytes -- same formula this
    # GUI already used for the 1500-byte case. The standard also explicitly
    # requires fragmentation NOT be produced at the source ("IPv4
    # fragmentation shall not be produced at the source... acceptable for
    # the Sink to discard fragmented packets") -- so whatever MTU the link
    # actually runs, samples/pkt must stay under it, same reasoning as
    # before just against a configurable ceiling instead of a hardcoded
    # 1500. Default MTU here (9000) and default samples/pkt (2200, leaving
    # ~144 bytes of headroom under the 8944-byte jumbo cap for VLAN tagging
    # etc.) both assume the operator has already configured jumbo frames
    # end-to-end per the standard -- see the MTU field's own tooltip.
    DEFAULT_SAMPLES_PER_PKT = 2200
    DEFAULT_MTU_BYTES       = 9000
    BIT_DEPTH       = 16

    # Standard delay/jitter profiles for the Sim delay/Sim jitter fields
    # below -- (delay_ms, jitter_ms). Not DIFI-standard values (the DIFI
    # standard itself doesn't specify any -- see 2026-09-03 discussion);
    # these are the general engineering rule (hold/jitter-buffer sizing
    # should comfortably exceed real link jitter) applied to typical named
    # link classes, so an operator can pick a realistic scenario without
    # having to already know reasonable delay/jitter numbers for it.
    # "Typical WAN/Internet" (100/50) is the one pair actually validated
    # end-to-end earlier this project (0% loss, 0% ordering violations at
    # hold_ms=200). "Manual" (the default, preserving old behavior) leaves
    # the two spinboxes directly editable instead of preset-and-locked.
    DELAY_PRESETS = {
        "Manual":                   None,
        "LAN (clean)":              (2.0, 1.0),
        "Good WAN":                 (25.0, 10.0),
        "Typical WAN/Internet":     (100.0, 50.0),
        "Degraded/congested link":  (150.0, 60.0),
        "Satellite (GEO)":          (250.0, 15.0),
    }

    def __init__(self):
        super().__init__()
        self.setWindowTitle("DIFI Transmitter")
        self.setMinimumSize(900, 520)
        self._running = False
        self._gen     = None
        self._sent_log = None
        self._last_rate_count = 0
        self._last_rate_bytes = 0
        self._last_rate_time  = None
        # See app_config.py -- loaded here (before _build_ui() creates any
        # widget) so every field below can seed its initial value from the
        # last-saved settings instead of a hardcoded default.
        self._cfg = app_config.load("Transmitter")
        self._build_ui()

        # Opt-in, env-var-gated autostart -- lets a script drive the REAL
        # windowed GUI process (real Qt event loop, real window, real OS
        # scheduling/priority behavior) without pywinauto/UIA clicking,
        # which is unreliable in some environments. No effect unless
        # DIFI_AUTOSTART=1 is set, so normal interactive use is unchanged.
        if os.environ.get("DIFI_AUTOSTART") == "1":
            QTimer.singleShot(300, self._autostart_from_env)

    def _autostart_from_env(self):
        env = os.environ
        if env.get("DIFI_DEST_IP"):
            self._dest_ip.setText(env["DIFI_DEST_IP"])
        if env.get("DIFI_DEST_PORT"):
            self._dest_port.setValue(int(env["DIFI_DEST_PORT"]))
        if env.get("DIFI_STREAM_ID"):
            self._stream_id.setText(env["DIFI_STREAM_ID"])
        if env.get("DIFI_SAMPLES_PER_PKT"):
            self._samples_per_pkt.setValue(int(env["DIFI_SAMPLES_PER_PKT"]))
        # 2026-09-05: these all used to call set_hz() with its default
        # emit=False, so the "changed" signal never fired here -- confirmed
        # directly: the "Expected: N pkt/s" preview kept showing whatever
        # rate was last SAVED to SysConfig, not the rate this autostart run
        # actually configured (e.g. a real 20MHz test still showed the old
        # "2,273 pkt/s" from a previously-saved 5MHz session). The actual
        # generator itself was never affected -- _start() reads self._fs.
        # value_hz() live, not a cached signal-driven value -- so this was
        # a real but purely cosmetic bug, not a functional one. emit=True
        # here makes the preview (and _on_param_changed) fire the same way
        # a manual edit in the GUI would.
        if env.get("DIFI_SAMPLE_RATE_HZ"):
            self._fs.set_hz(float(env["DIFI_SAMPLE_RATE_HZ"]), emit=True)
        if env.get("DIFI_SIGNAL_TYPE") == "BW":
            self._bw_rb.setChecked(True)
        elif env.get("DIFI_SIGNAL_TYPE") == "CW":
            self._cw_rb.setChecked(True)
        if env.get("DIFI_BANDWIDTH_HZ"):
            self._bw.set_hz(float(env["DIFI_BANDWIDTH_HZ"]), emit=True)
        if env.get("DIFI_TONE_HZ"):
            self._tone.set_hz(float(env["DIFI_TONE_HZ"]), emit=True)
        if env.get("DIFI_RF_REF_HZ"):
            self._rf.set_hz(float(env["DIFI_RF_REF_HZ"]), emit=True)
        if env.get("DIFI_SIM_DELAY_MS"):
            self._sim_delay.setValue(float(env["DIFI_SIM_DELAY_MS"]))
        if env.get("DIFI_SIM_JITTER_MS"):
            self._sim_jitter.setValue(float(env["DIFI_SIM_JITTER_MS"]))
        self._start_btn.click()
        if env.get("DIFI_AUTOSTOP_AFTER_S"):
            QTimer.singleShot(int(float(env["DIFI_AUTOSTOP_AFTER_S"]) * 1000), self.close)

    def _build_ui(self):
        central  = QWidget()
        self.setCentralWidget(central)
        root     = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # ── left panel: controls ──────────────────────────────────────────
        left     = QWidget()
        left.setMaximumWidth(400)
        left_lay = QVBoxLayout(left)
        left_lay.setSpacing(8)

        # See app_config.py's module docstring for why this exists: the
        # same Combiner IP/port/stream ID/samples-per-pkt get re-typed
        # every run otherwise, and that's a real, previously-observed
        # source of mistakes (a Combiner run whose Forward destination
        # didn't match the real Receiver, only caught after the fact).
        net_cfg = app_config.Section(self._cfg, "Transmitter.Network")

        # ── Network ──
        net_box  = QGroupBox("Network")
        net_grid = QGridLayout(net_box)

        net_grid.addWidget(QLabel("Combiner VM IP:"), 0, 0)
        self._dest_ip = QLineEdit(net_cfg.get_str("dest_ip", "127.0.0.1"))
        self._dest_ip.setPlaceholderText("e.g. 192.168.1.20")
        self._dest_ip.setFixedWidth(160)
        net_grid.addWidget(self._dest_ip, 0, 1)

        net_grid.addWidget(QLabel("Dest port:"), 1, 0)
        self._dest_port = QSpinBox()
        self._dest_port.setRange(1024, 65535)
        self._dest_port.setValue(net_cfg.get_int("dest_port", 50001))
        self._dest_port.setFixedWidth(110)
        port_w = QWidget()
        port_l = QHBoxLayout(port_w)
        port_l.setContentsMargins(0, 0, 0, 0)
        port_l.addWidget(self._dest_port)
        port_l.addStretch()
        net_grid.addWidget(port_w, 1, 1)

        net_grid.addWidget(QLabel("Stream ID:"), 2, 0)
        self._stream_id = QLineEdit(net_cfg.get_str("stream_id", "0x00000001"))
        self._stream_id.setFixedWidth(120)
        sid_w = QWidget()
        sid_l = QHBoxLayout(sid_w)
        sid_l.setContentsMargins(0, 0, 0, 0)
        sid_l.addWidget(self._stream_id)
        sid_l.addStretch()
        net_grid.addWidget(sid_w, 2, 1)

        net_grid.addWidget(QLabel("Link MTU:"), 3, 0)
        self._mtu = QSpinBox()
        self._mtu.setRange(500, 9216)
        self._mtu.setValue(net_cfg.get_int("mtu", self.DEFAULT_MTU_BYTES))
        self._mtu.setSuffix(" bytes")
        self._mtu.setFixedWidth(110)
        self._mtu.setToolTip(
            "The actual Ethernet MTU configured on the path to the Combiner\n"
            "VM -- just informs the fragmentation warning below, sends\n"
            "nothing over the wire. Standard Ethernet is 1500; the DIFI\n"
            "standard's own jumbo-frame provision (DIFI_Standard_1.3.0_Final.pdf,\n"
            "p.15/p.21) is 9000. Set this to whatever the VM's NIC/vSwitch is\n"
            "actually configured for -- if any hop on the path is still at\n"
            "1500 while this says 9000, packets fragment (or get dropped)\n"
            "instead of arriving faster."
        )
        net_grid.addWidget(self._mtu, 3, 1)

        net_grid.addWidget(QLabel("Samples/pkt:"), 4, 0)
        self._samples_per_pkt = QSpinBox()
        self._samples_per_pkt.setRange(16, 4096)
        self._samples_per_pkt.setValue(net_cfg.get_int("samples_per_pkt", self.DEFAULT_SAMPLES_PER_PKT))
        self._samples_per_pkt.setFixedWidth(110)
        self._samples_per_pkt.setToolTip(
            "IQ samples per DIFI Data packet. Wire size = samples*4 + 56 bytes\n"
            "(DIFI + IP/UDP headers). Must stay under the Link MTU above or\n"
            "every packet gets IP-fragmented, and losing any one fragment\n"
            "silently drops the whole packet -- the DIFI standard itself\n"
            "requires fragmentation not be produced at the source. Default\n"
            "(2200) fits under the standard's own 9000-byte jumbo-frame MTU\n"
            "with headroom to spare; drop this back to ~360 if the link is\n"
            "still standard 1500-byte Ethernet."
        )
        net_grid.addWidget(self._samples_per_pkt, 4, 1)

        net_grid.addWidget(QLabel("Delay profile:"), 5, 0)
        self._delay_preset = QComboBox()
        self._delay_preset.addItems(list(self.DELAY_PRESETS.keys()))
        saved_preset = net_cfg.get_str("delay_preset", "Manual")
        if saved_preset not in self.DELAY_PRESETS:
            saved_preset = "Manual"
        self._delay_preset.setCurrentText(saved_preset)
        self._delay_preset.setFixedWidth(180)
        self._delay_preset.setToolTip(
            "Standard delay/jitter profiles for typical link classes --\n"
            "picks both fields below for you. \"Manual\" (default) leaves\n"
            "them directly editable instead. Not DIFI-standard values (the\n"
            "standard itself doesn't specify any) -- \"Typical WAN/Internet\"\n"
            "(100/50 ms) is the one pair actually validated end-to-end on\n"
            "this project (0% loss, 0% ordering violations at 200ms hold)."
        )
        net_grid.addWidget(self._delay_preset, 5, 1)

        net_grid.addWidget(QLabel("Sim delay:"), 6, 0)
        self._sim_delay = QDoubleSpinBox()
        self._sim_delay.setRange(0, 5000)
        self._sim_delay.setDecimals(0)
        self._sim_delay.setValue(net_cfg.get_float("sim_delay_ms", 0))
        self._sim_delay.setSuffix(" ms")
        self._sim_delay.setFixedWidth(110)
        self._sim_delay.setToolTip(
            "Fixed simulated one-way network delay for this modem's path\n"
            "to the Combiner VM (e.g. 100/120/150 ms). Only editable when\n"
            "Delay profile above is set to \"Manual\"."
        )
        net_grid.addWidget(self._sim_delay, 6, 1)

        net_grid.addWidget(QLabel("Sim jitter max:"), 7, 0)
        self._sim_jitter = QDoubleSpinBox()
        self._sim_jitter.setRange(0, 1000)
        self._sim_jitter.setDecimals(0)
        self._sim_jitter.setValue(net_cfg.get_float("sim_jitter_ms", 0))
        self._sim_jitter.setSuffix(" ms")
        self._sim_jitter.setFixedWidth(110)
        self._sim_jitter.setToolTip(
            "Extra random delay on top of Sim delay, uniform in\n"
            "[0, this value] per packet. Only editable when Delay profile\n"
            "above is set to \"Manual\"."
        )
        net_grid.addWidget(self._sim_jitter, 7, 1)

        # Apply the loaded/default preset immediately (locks the two
        # spinboxes and sets their values, unless "Manual") -- then wire
        # future changes.
        self._on_delay_preset_changed(self._delay_preset.currentText())
        self._delay_preset.currentTextChanged.connect(self._on_delay_preset_changed)

        left_lay.addWidget(net_box)

        sig_cfg = app_config.Section(self._cfg, "Transmitter.Signal")

        # ── Signal ──
        sig_box  = QGroupBox("Signal")
        sig_grid = QGridLayout(sig_box)

        sig_grid.addWidget(QLabel("Sample rate:"), 0, 0)
        self._fs = FreqInput(default_hz=sig_cfg.get_float("sample_rate_hz", 10e6))
        sig_grid.addWidget(self._fs, 0, 1)

        sig_grid.addWidget(QLabel("Signal type:"), 1, 0)
        type_w   = QWidget()
        type_lay = QHBoxLayout(type_w)
        type_lay.setContentsMargins(0, 0, 0, 0)
        self._cw_rb      = QRadioButton("CW")
        self._bw_rb      = QRadioButton("BW")
        self._pattern_rb = QRadioButton("PATTERN")
        self._off_rb     = QRadioButton("OFF")
        {"CW": self._cw_rb, "BW": self._bw_rb, "PATTERN": self._pattern_rb,
         "OFF": self._off_rb}.get(sig_cfg.get_str("signal_type", "CW"), self._cw_rb).setChecked(True)
        grp = QButtonGroup(self)
        grp.addButton(self._cw_rb)
        grp.addButton(self._bw_rb)
        grp.addButton(self._pattern_rb)
        grp.addButton(self._off_rb)
        type_lay.addWidget(self._cw_rb)
        type_lay.addWidget(self._bw_rb)
        type_lay.addWidget(self._pattern_rb)
        type_lay.addWidget(self._off_rb)
        type_lay.addStretch()
        sig_grid.addWidget(type_w, 1, 1)

        sig_grid.addWidget(QLabel("RF Frequency:"), 2, 0)
        self._tone = FreqInput(default_hz=sig_cfg.get_float("tone_hz", 1e6))
        sig_grid.addWidget(self._tone, 2, 1)

        sig_grid.addWidget(QLabel("Bandwidth:"), 3, 0)
        self._bw = FreqInput(default_hz=sig_cfg.get_float("bandwidth_hz", 1e6))
        # Matches whatever signal_type was just loaded/checked above -- the
        # toggled-signal wiring further below only fires on a future change,
        # not for a checked state already set at construction.
        self._bw.setEnabled(self._bw_rb.isChecked())
        sig_grid.addWidget(self._bw, 3, 1)

        sig_grid.addWidget(QLabel("RF reference:"), 4, 0)
        self._rf = FreqInput(default_hz=sig_cfg.get_float("rf_ref_hz", 0))
        sig_grid.addWidget(self._rf, 4, 1)

        sig_grid.addWidget(QLabel("Amplitude:"), 5, 0)
        self._amp = QDoubleSpinBox()
        self._amp.setRange(-100.0, 0.0)
        self._amp.setDecimals(1)
        self._amp.setSingleStep(1.0)
        self._amp.setValue(sig_cfg.get_float("ref_level_dbm", -20.0))
        self._amp.setSuffix(" dBm")
        sig_grid.addWidget(self._amp, 5, 1)

        self._stat = QLabel("Idle")
        self._stat.setStyleSheet("color: #888888;")
        sig_grid.addWidget(self._stat, 6, 0, 1, 2)

        # Computed (not measured) rate, from Sample rate / Samples-per-pkt --
        # visible before Start is even pressed, so the operator can see the
        # expected packets/s and MB/s for a given configuration up front,
        # not just the live measured rate once running.
        self._rate_preview = QLabel()
        self._rate_preview.setStyleSheet("color: #6699cc;")
        sig_grid.addWidget(self._rate_preview, 7, 0, 1, 2)

        for rb in (self._cw_rb, self._bw_rb, self._pattern_rb, self._off_rb):
            rb.toggled.connect(lambda checked: self._bw.setEnabled(self._bw_rb.isChecked()))

        # live update generator while running, and refresh spectrum preview
        for rb in (self._cw_rb, self._bw_rb, self._pattern_rb, self._off_rb):
            rb.toggled.connect(self._on_param_changed)
        self._tone.changed.connect(self._on_param_changed)
        self._bw.changed.connect(self._on_param_changed)
        self._rf.changed.connect(self._on_param_changed)
        self._amp.valueChanged.connect(self._on_param_changed)
        self._fs.changed.connect(self._on_param_changed)
        self._sim_delay.valueChanged.connect(self._on_param_changed)
        self._sim_jitter.valueChanged.connect(self._on_param_changed)
        self._fs.changed.connect(self._update_rate_preview)
        self._samples_per_pkt.valueChanged.connect(self._update_rate_preview)
        self._mtu.valueChanged.connect(self._update_rate_preview)

        left_lay.addWidget(sig_box)
        left_lay.addStretch()

        # ── Start / Stop / Save Config ──
        btn_row = QHBoxLayout()
        self._start_btn = QPushButton("▶  Start")
        self._stop_btn  = QPushButton("■  Stop")
        self._save_btn  = QPushButton("💾  Save Config")
        self._stop_btn.setEnabled(False)
        self._start_btn.setFixedHeight(36)
        self._stop_btn.setFixedHeight(36)
        self._save_btn.setFixedHeight(36)
        self._start_btn.clicked.connect(self._start)
        self._stop_btn.clicked.connect(self._stop)
        # Explicit save independent of Start -- lets an operator type the
        # Combiner VM IP/stream ID/etc. and write the ini for this VM
        # without actually starting transmission (e.g. pre-staging VMs).
        self._save_btn.setToolTip(
            "Write the current settings to SysConfig.ini next to this exe,\n"
            "without starting Start -- useful for pre-configuring a VM."
        )
        self._save_btn.clicked.connect(self._save_config)
        btn_row.addWidget(self._start_btn)
        btn_row.addWidget(self._stop_btn)
        btn_row.addWidget(self._save_btn)
        left_lay.addLayout(btn_row)

        splitter.addWidget(left)

        # ── right panel: display controls + spectrum ──────────────────────
        right     = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(4, 4, 4, 4)
        right_lay.setSpacing(4)

        disp_box  = QGroupBox("Display")
        disp_vlay = QVBoxLayout(disp_box)
        disp_vlay.setSpacing(4)
        disp_cfg  = app_config.Section(self._cfg, "Transmitter.Display")

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Center:"))
        self._disp_center = FreqInput(default_hz=disp_cfg.get_float("center_hz", 1e6))
        row1.addWidget(self._disp_center)
        row1.addSpacing(16)
        row1.addWidget(QLabel("Span:"))
        self._disp_span = FreqInput(default_hz=disp_cfg.get_float("span_hz", 10e6))
        row1.addWidget(self._disp_span)
        row1.addStretch()
        disp_vlay.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Amplitude:"))
        self._disp_amp = QDoubleSpinBox()
        self._disp_amp.setRange(-200, 50)
        self._disp_amp.setDecimals(1)
        self._disp_amp.setSingleStep(10)
        self._disp_amp.setValue(disp_cfg.get_float("amp_top_db", -10))
        self._disp_amp.setSuffix(" dB")
        self._disp_amp.setFixedWidth(120)
        row2.addWidget(self._disp_amp)
        row2.addSpacing(16)
        row2.addWidget(QLabel("dB / div:"))
        self._disp_dbdiv = QDoubleSpinBox()
        self._disp_dbdiv.setRange(1, 100)
        self._disp_dbdiv.setDecimals(1)
        self._disp_dbdiv.setValue(disp_cfg.get_float("db_div_db", 10))
        self._disp_dbdiv.setSuffix(" dB")
        self._disp_dbdiv.setFixedWidth(110)
        row2.addWidget(self._disp_dbdiv)
        auto_btn = QPushButton("Auto")
        auto_btn.setFixedWidth(60)
        auto_btn.clicked.connect(self._auto_display)
        row2.addWidget(auto_btn)
        row2.addStretch()
        disp_vlay.addLayout(row2)

        self._disp_center.changed.connect(self._apply_range)
        self._disp_span.changed.connect(self._apply_range)
        self._disp_amp.valueChanged.connect(self._apply_range)
        self._disp_dbdiv.valueChanged.connect(self._apply_range)

        right_lay.addWidget(disp_box)

        self._plot = pg.PlotWidget(title="Transmitter Output")
        self._plot.setLabel("bottom", "Frequency", units="Hz")
        self._plot.setLabel("left",   "Magnitude", units="dB")
        self._plot.showGrid(x=True, y=True, alpha=0.3)
        self._plot.enableAutoRange(axis="xy", enable=False)
        self._plot.getPlotItem().getViewBox().enableAutoRange(enable=False)
        self._curve = self._plot.plot([], [], pen=pg.mkPen((100, 220, 255), width=1))

        self._ref_line = pg.InfiniteLine(
            angle=0, movable=False,
            pen=pg.mkPen("y", width=1, style=Qt.PenStyle.DashLine),
        )
        self._plot.addItem(self._ref_line)

        self._plot.getPlotItem().getViewBox().sigRangeChanged.connect(
            lambda vb, ranges: self._sync_viewport_to_spinboxes()
        )

        right_lay.addWidget(self._plot)
        splitter.addWidget(right)
        splitter.setSizes([380, 520])

        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Ready — enter Combiner IP and press Start")

        self._timer = QTimer()
        self._timer.setInterval(200)
        self._timer.timeout.connect(self._tick)

        self._apply_range()
        self._update_spectrum()
        self._update_rate_preview()

    def _update_rate_preview(self, *_):
        """Computed (not measured) packets/s and MB/s for the CURRENTLY
        configured Sample rate / Samples-per-pkt -- updates live as either
        changes, whether or not the generator is running, so the operator
        can see the expected load up front. samples*4+56 matches the same
        on-wire-byte formula used throughout this project (DIFI header +
        IP/UDP headers); flags packets that will be IP-fragmented against
        the Link MTU field above (not a hardcoded 1500 -- see that field's
        own tooltip for why it's configurable) since that's a real, silent
        loss risk on any real (non-loopback) network -- see
        capture_worker.py's docstring."""
        fs  = self._fs.value_hz()
        n   = self._samples_per_pkt.value()
        mtu = self._mtu.value()
        if fs <= 0 or n <= 0:
            self._rate_preview.setText("")
            return
        pkt_rate   = fs / n
        wire_bytes = n * 4 + 56
        mbps       = pkt_rate * wire_bytes / 1e6
        frag_warn  = f"  ⚠ exceeds {mtu}-byte MTU, will fragment" if wire_bytes > mtu else ""
        self._rate_preview.setText(
            f"Expected: {pkt_rate:,.0f} pkt/s  |  {mbps:,.2f} MB/s  ({wire_bytes} bytes/pkt){frag_warn}"
        )

    # ── helpers ────────────────────────────────────────────────────────────

    def _signal_type(self) -> str:
        if self._cw_rb.isChecked():      return SIGNAL_CW
        if self._bw_rb.isChecked():      return SIGNAL_BW
        if self._pattern_rb.isChecked(): return SIGNAL_PATTERN
        return SIGNAL_OFF

    def _stream_id_int(self) -> int:
        """Parse stream ID from the text field. Raises ValueError on bad input."""
        return int(self._stream_id.text().strip(), 16)

    def _on_delay_preset_changed(self, name: str):
        """Apply a standard delay/jitter profile (locking the two
        spinboxes to it), or unlock them for direct entry on "Manual"."""
        preset = self.DELAY_PRESETS.get(name)
        if preset is None:
            self._sim_delay.setEnabled(True)
            self._sim_jitter.setEnabled(True)
            return
        delay_ms, jitter_ms = preset
        self._sim_delay.setValue(delay_ms)
        self._sim_jitter.setValue(jitter_ms)
        self._sim_delay.setEnabled(False)
        self._sim_jitter.setEnabled(False)

    def _rf_ref(self) -> float:
        rf_ref = self._rf.value_hz()
        if rf_ref == 0.0 and abs(self._tone.value_hz()) > self._fs.value_hz() / 2.0:
            return self._tone.value_hz()
        return rf_ref

    def _save_config(self):
        """See app_config.py -- called on Start and on window close so the
        exe remembers its own last settings across restarts instead of
        the operator re-typing them (Combiner IP especially) every run."""
        net_cfg = app_config.Section(self._cfg, "Transmitter.Network")
        net_cfg.set("dest_ip", self._dest_ip.text().strip())
        net_cfg.set("dest_port", self._dest_port.value())
        net_cfg.set("stream_id", self._stream_id.text().strip())
        net_cfg.set("mtu", self._mtu.value())
        net_cfg.set("samples_per_pkt", self._samples_per_pkt.value())
        net_cfg.set("sim_delay_ms", self._sim_delay.value())
        net_cfg.set("sim_jitter_ms", self._sim_jitter.value())
        net_cfg.set("delay_preset", self._delay_preset.currentText())

        sig_cfg = app_config.Section(self._cfg, "Transmitter.Signal")
        sig_cfg.set("sample_rate_hz", self._fs.value_hz())
        sig_cfg.set("signal_type", self._signal_type())
        sig_cfg.set("tone_hz", self._tone.value_hz())
        sig_cfg.set("bandwidth_hz", self._bw.value_hz())
        sig_cfg.set("rf_ref_hz", self._rf.value_hz())
        sig_cfg.set("ref_level_dbm", self._amp.value())

        disp_cfg = app_config.Section(self._cfg, "Transmitter.Display")
        disp_cfg.set("center_hz", self._disp_center.value_hz())
        disp_cfg.set("span_hz", self._disp_span.value_hz())
        disp_cfg.set("amp_top_db", self._disp_amp.value())
        disp_cfg.set("db_div_db", self._disp_dbdiv.value())

        app_config.save("Transmitter", self._cfg)

    # ── lifecycle ──────────────────────────────────────────────────────────

    def _start(self):
        if self._running:
            return

        ip     = self._dest_ip.text().strip()
        fs     = self._fs.value_hz()
        rf_ref = self._rf_ref()
        tone_bb = self._tone.value_hz() - rf_ref

        try:
            sid = self._stream_id_int()
        except ValueError:
            self._status.showMessage("Invalid Stream ID — use hex e.g. 0x00000001")
            return

        self._save_config()

        run_dir = make_run_dir("Transmitter")
        self._sent_log = PacketLogger(
            run_dir, "data_sent.csv",
            ["wall_clock", "stream_id", "pkt_type", "seq", "difi_ts_int",
             "difi_ts_frac", "samples", "dest_ip", "dest_port", "first_i", "first_q"],
        )

        samples_per_pkt = self._samples_per_pkt.value()
        self._gen = DifiGenerator(
            stream_id       = sid,
            tone_hz         = tone_bb,
            signal_type     = self._signal_type(),
            dest_host       = ip,
            dest_port       = self._dest_port.value(),
            sample_rate_hz  = fs,
            samples_per_pkt = samples_per_pkt,
            bit_depth       = self.BIT_DEPTH,
            rf_ref_freq_hz  = rf_ref,
            bandwidth_hz    = self._bw.value_hz(),
            ref_level_dbm   = self._amp.value(),
            sim_delay_ms    = self._sim_delay.value(),
            sim_jitter_ms   = self._sim_jitter.value(),
            packet_logger   = self._sent_log,
        )

        pkt_rate = fs / samples_per_pkt
        threading.Thread(
            target=self._gen.run,
            kwargs=dict(packet_rate_hz=pkt_rate),
            daemon=True,
        ).start()

        self._running = True
        self._last_rate_count = 0
        self._last_rate_bytes = 0
        self._last_rate_time  = time.monotonic()
        self._fs.setEnabled(False)
        self._dest_ip.setEnabled(False)
        self._dest_port.setEnabled(False)
        self._stream_id.setEnabled(False)
        self._samples_per_pkt.setEnabled(False)
        self._mtu.setEnabled(False)
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._timer.start()

        port = self._dest_port.value()
        self._plot.setTitle(f"Transmitter Output — port {port}")
        # Auto-center the display on the RF frequency/sample rate actually
        # configured, instead of leaving it at the fixed 1 MHz default --
        # unlike the Receiver's version this needs no external data, since
        # this window already knows its own signal's real parameters.
        self._auto_display()
        self._status.showMessage(
            f"Sending to {ip}:{port} | "
            f"stream=0x{sid:08X} | fs={fs/1e6:.2f} MHz | "
            f"type={self._signal_type()} | RF={self._tone.value_hz()/1e6:.3f} MHz | "
            f"log: {run_dir}"
        )

    def _stop(self):
        if not self._running:
            return
        self._timer.stop()
        if self._gen:
            self._gen.close()
            self._gen = None
        if self._sent_log:
            self._sent_log.close()
            self._sent_log = None
        self._stat.setText("Idle")
        self._stat.setStyleSheet("color: #888888;")
        self._running = False
        self._fs.setEnabled(True)
        self._dest_ip.setEnabled(True)
        self._dest_port.setEnabled(True)
        self._stream_id.setEnabled(True)
        self._samples_per_pkt.setEnabled(True)
        self._mtu.setEnabled(True)
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._plot.setTitle("Transmitter Output")
        self._status.showMessage("Stopped")

    def _tick(self):
        if not self._running or not self._gen:
            return

        now   = time.monotonic()
        count = self._gen.pkt_count
        nbytes = self._gen.bytes_sent
        dt    = now - self._last_rate_time if self._last_rate_time else 0.0
        rate  = (count - self._last_rate_count) / dt if dt > 0 else 0.0
        mbps  = (nbytes - self._last_rate_bytes) / dt / 1e6 if dt > 0 else 0.0
        self._last_rate_count = count
        self._last_rate_bytes = nbytes
        self._last_rate_time  = now

        errs = self._gen.send_errors
        if errs:
            # Real send failures at the OS level -- these packets are already
            # in data_sent.csv (logged before dispatch) but never left this
            # machine. Surfaced in red since the windowed EXE's console
            # prints go to a log file the user won't see until after Stop.
            self._stat.setText(
                f"Running — {count:,} pkts sent ({rate:,.0f} pkt/s | {mbps:,.2f} MB/s) | "
                f"⚠ {errs:,} sendto() FAILED")
            self._stat.setStyleSheet("color: #ff4444;")
        else:
            self._stat.setText(
                f"Running — {count:,} pkts sent ({rate:,.0f} pkt/s | {mbps:,.2f} MB/s)")
            self._stat.setStyleSheet("color: #00cc44;")
        # Opt-in test hook: skip the spectrum redraw (a real np.fft.fft()
        # over a freshly-synthesized 1024-sample segment, in this SAME
        # process as the sending thread, every 200ms) to A/B whether it's
        # contributing to the gap between DifiGenerator's own isolated
        # per-packet ceiling and what a real Transmitter GUI process
        # actually achieves. No effect unless set.
        if os.environ.get("DIFI_DISABLE_SPECTRUM_TIMER") != "1":
            self._update_spectrum()

    def _on_param_changed(self, *_):
        """Called when any signal parameter changes — live-update generator and spectrum."""
        if self._running and self._gen:
            rf_ref = self._rf_ref()
            self._gen.update_params(
                tone_hz        = self._tone.value_hz() - rf_ref,
                signal_type    = self._signal_type(),
                bandwidth_hz   = self._bw.value_hz(),
                rf_ref_freq_hz = rf_ref,
                ref_level_dbm  = self._amp.value(),
                sim_delay_ms   = self._sim_delay.value(),
                sim_jitter_ms  = self._sim_jitter.value(),
            )
        self._update_spectrum()

    # ── spectrum ───────────────────────────────────────────────────────────

    def _update_spectrum(self):
        """Compute and display the expected signal spectrum from current UI parameters.

        Adds a small synthetic noise floor, freshly randomized on every call.
        A noiseless ideal tone (or BW noise redrawn from a fixed seed, as
        this used to do) is bit-for-bit identical on every redraw, which
        reads as a frozen/dead display even while actively transmitting —
        real spectrum analyzers visibly jitter tick-to-tick because of their
        own noise floor, and this display should look the same way.
        """
        fs       = self._fs.value_hz()
        rf_ref   = self._rf_ref()
        tone_bb  = self._tone.value_hz() - rf_ref
        sig_type = self._signal_type()
        amp_dbm  = self._amp.value()
        bw       = self._bw.value_hz()

        if fs <= 0:
            return

        seg_len = 1024
        t = np.arange(seg_len) / fs

        if sig_type == SIGNAL_OFF:
            self._curve.setData([], [])
            return

        rng     = np.random.default_rng()   # fresh noise each tick -> visibly live display
        amp_lin = 10 ** (amp_dbm / 20.0)

        if sig_type == SIGNAL_CW:
            iq = np.exp(1j * 2 * np.pi * tone_bb * t).astype(np.complex64)
            iq *= amp_lin
        elif sig_type == SIGNAL_PATTERN:
            from modules.generator import PATTERN_PERIOD
            idx = np.arange(seg_len)
            iq  = (((idx % PATTERN_PERIOD) / PATTERN_PERIOD) * 2.0 - 1.0).astype(np.complex64)
            iq *= amp_lin
        else:  # BW
            noise  = (rng.standard_normal(seg_len) + 1j * rng.standard_normal(seg_len)).astype(np.complex64)
            nyq    = fs / 2.0
            cutoff = max(min(bw / 2.0 / nyq, 0.499), 1e-4)
            fir    = sp_sig.firwin(101, cutoff)
            fi     = sp_sig.lfilter(fir, [1.0], noise.real)
            fq     = sp_sig.lfilter(fir, [1.0], noise.imag)
            iq     = (fi + 1j * fq).astype(np.complex64)
            iq    *= np.exp(1j * 2 * np.pi * tone_bb * t)
            iq    *= amp_lin

        # Synthetic noise floor ~45 dB below the signal peak.
        floor_amp = amp_lin * 10 ** (-45.0 / 20.0)
        iq = iq + floor_amp * (
            rng.standard_normal(seg_len) + 1j * rng.standard_normal(seg_len)
        ).astype(np.complex64)

        w      = np.hanning(seg_len)
        w_amp  = float(np.sum(w))
        X      = np.fft.fftshift(np.fft.fft(iq * w))
        mag_db = 20.0 * np.log10(np.abs(X) / w_amp + 1e-7)
        freqs  = np.fft.fftshift(np.fft.fftfreq(seg_len, d=1.0 / fs)) + rf_ref

        self._curve.setData(freqs, mag_db)

    # ── display helpers ────────────────────────────────────────────────────

    def _apply_range(self):
        center  = self._disp_center.value_hz()
        span    = self._disp_span.value_hz()
        amp_top = self._disp_amp.value()
        db_div  = self._disp_dbdiv.value()
        self._plot.setXRange(center - span / 2, center + span / 2, padding=0)
        self._plot.setYRange(amp_top - db_div * 10, amp_top, padding=0)
        self._ref_line.setValue(amp_top)

    def _auto_display(self):
        fs     = self._fs.value_hz()
        rf_ref = self._rf_ref()
        self._disp_center.set_hz(rf_ref if rf_ref != 0.0 else self._tone.value_hz())
        self._disp_span.set_hz(fs)
        self._disp_amp.setValue(-10.0)
        self._disp_dbdiv.setValue(10.0)
        self._apply_range()

    def _sync_viewport_to_spinboxes(self):
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

    def closeEvent(self, event):
        self._stop()
        self._save_config()
        event.accept()
        # See gil_friendly_exec.py / packetizer_app.py's identical fix --
        # app.lastWindowClosed alone was not reliably ending the polling
        # loop, leaving an orphaned background process after the window
        # closed. Calling request_stop() directly here doesn't depend on
        # that signal firing correctly under manual polling.
        request_stop(QApplication.instance())


def main():
    from logging_setup import setup_frozen_file_logging
    log_path = setup_frozen_file_logging("Transmitter")

    # See packetizer_app.py's identical fix: antialias=True is expensive on
    # a frequently-redrawn curve and was never actually needed here.
    pg.setConfigOptions(antialias=False)
    app = QApplication(sys.argv)
    win = TransmitterWindow()
    if log_path:
        win._status.showMessage(f"Logging to {log_path}")
    win.show()
    # See gil_friendly_exec.py -- app.exec()'s native Windows event loop
    # was measured to starve every other thread in the process (the
    # generator/dispatcher threads here); this polling loop avoids that.
    app.lastWindowClosed.connect(lambda: request_stop(app))
    sys.exit(run_gil_friendly(app))


if __name__ == "__main__":
    main()
