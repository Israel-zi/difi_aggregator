"""
freq_input.py
-------------
Shared frequency-input widget used by all DIFI GUI apps.
"""

from PySide6.QtCore import Signal, QEvent
from PySide6.QtWidgets import QWidget, QHBoxLayout, QDoubleSpinBox, QComboBox

UNIT_MUL    = {"Hz": 1.0, "kHz": 1_000.0, "MHz": 1_000_000.0, "GHz": 1_000_000_000.0}
UNIT_LABELS = ["Hz", "kHz", "MHz", "GHz"]
_UNIT_STEP  = {"Hz": 1_000.0, "kHz": 1.0, "MHz": 1.0, "GHz": 0.001}


class FreqInput(QWidget):
    """
    Frequency spin-box with an adjacent unit combobox.

    Features
    --------
    - Auto-normalizes on commit: 1750 MHz → 1.75 GHz, 0.95 GHz → 950 MHz.
    - Preserves Hz value when the unit is changed manually or via keyboard.
    - Keyboard shortcuts while the spinbox has focus: h=Hz  k=kHz  m=MHz  g=GHz.
    - Scroll-wheel step ≈ 1 MHz regardless of active unit.
    """

    changed = Signal()

    _KEY_UNIT = {'h': 'Hz', 'k': 'kHz', 'm': 'MHz', 'g': 'GHz'}

    def __init__(self, default_hz: float = 1e6, parent=None):
        super().__init__(parent)
        unit, val = self._pick_unit(default_hz)

        self._spin = QDoubleSpinBox()
        self._spin.setDecimals(3)
        self._spin.setRange(0.0, 999_999.999)
        self._spin.setValue(val)
        self._spin.setSingleStep(_UNIT_STEP[unit])
        self._spin.setFixedWidth(115)
        self._spin.setKeyboardTracking(False)

        self._unit = QComboBox()
        self._unit.addItems(UNIT_LABELS)
        self._unit.setCurrentText(unit)
        self._unit.setFixedWidth(70)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._spin)
        lay.addWidget(self._unit)

        self._active_unit = unit

        self._spin.valueChanged.connect(self._on_spin_changed)
        self._unit.currentIndexChanged.connect(self._on_unit_changed)
        # Key events go to the internal QLineEdit, not the QDoubleSpinBox itself.
        self._spin.lineEdit().installEventFilter(self)

    # ── helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _pick_unit(hz: float) -> tuple:
        """Return (unit_str, display_value) for the most readable representation."""
        a = abs(hz)
        if a >= 1e9:   return "GHz", hz / 1e9
        if a >= 1e6:   return "MHz", hz / 1e6
        if a >= 1e3:   return "kHz", hz / 1e3
        return "Hz", hz

    def _apply_unit(self, unit: str, hz: float):
        """Write unit + converted value to widgets without triggering signals."""
        self._active_unit = unit
        self._unit.blockSignals(True)
        self._unit.setCurrentText(unit)
        self._unit.blockSignals(False)
        self._spin.blockSignals(True)
        self._spin.setValue(hz / UNIT_MUL[unit])
        self._spin.blockSignals(False)
        self._spin.setSingleStep(_UNIT_STEP[unit])

    # ── unit handling ──────────────────────────────────────────────────────

    def _on_spin_changed(self, value: float):
        """Auto-normalize unit on commit (e.g. 1750 MHz → 1.75 GHz)."""
        hz = value * UNIT_MUL[self._active_unit]
        if hz > 0:
            best, _ = self._pick_unit(hz)
            if best != self._active_unit:
                self._apply_unit(best, hz)
        self.changed.emit()

    def _on_unit_changed(self):
        """Combobox changed by user: preserve Hz value, update spinbox display."""
        new_unit = self._unit.currentText()
        if new_unit == self._active_unit:
            return
        hz = self._spin.value() * UNIT_MUL[self._active_unit]
        self._apply_unit(new_unit, hz)
        self.changed.emit()

    def set_unit(self, unit: str):
        """Programmatically change unit while keeping the Hz value the same."""
        if unit == self._unit.currentText():
            return
        self._apply_unit(unit, self.value_hz())
        self.changed.emit()

    # ── keyboard shortcut (h / k / m / g) ─────────────────────────────────

    def eventFilter(self, obj, event):
        if obj is self._spin.lineEdit() and event.type() == QEvent.Type.KeyPress:
            unit = self._KEY_UNIT.get(event.text().lower())
            if unit:
                self.set_unit(unit)
                return True
        return super().eventFilter(obj, event)

    # ── value access / mutation ────────────────────────────────────────────

    def value_hz(self) -> float:
        return self._spin.value() * UNIT_MUL[self._unit.currentText()]

    def set_hz(self, hz: float, emit: bool = False):
        """Set from absolute Hz, auto-selecting the best display unit."""
        unit, _ = self._pick_unit(hz)
        self._apply_unit(unit, hz)
        if emit:
            self.changed.emit()
