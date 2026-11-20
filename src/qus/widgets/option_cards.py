"""
Option cards and toggle switch shared by the segmentation type and config type screens.

A 2x2 grid of clickable cards (title, description, badge, optional extra widget such as a
dropdown) with single selection, plus an animated toggle switch and the "Include spectrum check
after segmentation" row that slides/fades in.
"""

from typing import Dict, List, Optional

from PyQt6.QtCore import (QEasingCurve, QParallelAnimationGroup, QPropertyAnimation, QRectF,
                          QSize, Qt, pyqtProperty, pyqtSignal)
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtWidgets import (QAbstractButton, QFrame, QGraphicsOpacityEffect, QGridLayout,
                             QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget)

CARD_BG = "#353535"
CARD_BORDER = "#4a4a4a"
CARD_HOVER_BORDER = "#7c3aff"
CARD_SELECTED_BORDER = "#9e55ff"
CARD_SELECTED_BG = "#2e1a5a"

CARD_STYLE = f"""
QFrame#optionCard {{
    background: {CARD_BG};
    border: 2px solid {CARD_BORDER};
    border-radius: 12px;
}}
QFrame#optionCard:hover {{
    border: 2px solid {CARD_HOVER_BORDER};
}}
QFrame#optionCard[selected="true"] {{
    background: {CARD_SELECTED_BG};
    border: 2px solid {CARD_SELECTED_BORDER};
}}
QFrame#optionCard[available="false"] {{
    border: 2px solid #3d3d3d;
}}
QLabel {{ background: transparent; border: 0px; }}
QLabel#cardTitle {{ color: white; font-size: 20px; font-weight: bold; }}
QLabel#cardDescription {{ color: rgb(200, 200, 200); font-size: 14px; }}
QComboBox {{
    color: white; background: rgb(60, 60, 60); border: 1px solid rgb(100, 100, 100);
    border-radius: 6px; padding: 3px 8px; font-size: 14px;
}}
QComboBox QAbstractItemView {{
    color: white; background: rgb(50, 50, 50); border: 1px solid rgb(120, 120, 120);
    selection-color: white; selection-background-color: rgb(90, 37, 255); outline: 0px;
}}
QLabel#cardNote {{ color: rgb(255, 216, 77); font-size: 12px; }}
QLabel#cardBadge {{
    color: rgb(220, 205, 255);
    background: rgba(158, 85, 255, 0.22);
    border: 1px solid rgba(158, 85, 255, 0.55);
    border-radius: 8px;
    padding: 2px 8px;
    font-size: 12px;
}}
QFrame#optionCard[available="false"] QLabel#cardTitle,
QFrame#optionCard[available="false"] QLabel#cardDescription,
QFrame#optionCard[available="false"] QLabel#cardBadge {{ color: rgb(110, 110, 110); }}
"""


class OptionCard(QFrame):
    """One clickable option: title, description, badge, and an optional extra widget (e.g. a
    dropdown) between the description and the badge."""

    clicked = pyqtSignal(str)  # key

    def __init__(self, key: str, title: str, description: str, badge: str,
                 parent: Optional[QWidget] = None, extra: Optional[QWidget] = None):
        super().__init__(parent)
        self.key = key
        self.setObjectName("optionCard")
        self.setStyleSheet(CARD_STYLE)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(QSize(260, 130))
        self.setProperty("selected", False)
        self.setProperty("available", True)

        title_label = QLabel(title)
        title_label.setObjectName("cardTitle")
        description_label = QLabel(description)
        description_label.setObjectName("cardDescription")
        description_label.setWordWrap(True)
        self._description = description_label
        self._note = QLabel()
        self._note.setObjectName("cardNote")
        self._note.setWordWrap(True)
        self._note.hide()
        badge_label = QLabel(badge)
        badge_label.setObjectName("cardBadge")

        badge_row = QHBoxLayout()
        badge_row.addWidget(badge_label)
        badge_row.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(8)
        layout.addWidget(title_label)
        layout.addWidget(description_label, 1)
        if extra is not None:
            layout.addWidget(extra)
        layout.addWidget(self._note)
        layout.addLayout(badge_row)

    def set_description(self, text: str) -> None:
        self._description.setText(text)

    def set_note(self, text: str) -> None:
        """Small highlighted note under the description (hidden when empty)."""
        self._note.setText(text)
        self._note.setVisible(bool(text))

    def set_selected(self, selected: bool) -> None:
        self._set_property("selected", selected)

    def set_available(self, available: bool, reason: str = "") -> None:
        self._set_property("available", available)
        self.setEnabled(available)
        self.setCursor(Qt.CursorShape.PointingHandCursor if available else Qt.CursorShape.ArrowCursor)
        self.setToolTip("" if available else reason)

    def is_available(self) -> bool:
        return bool(self.property("available"))

    def _set_property(self, name: str, value: bool) -> None:
        self.setProperty(name, value)
        self.style().unpolish(self)
        self.style().polish(self)
        for child in self.findChildren(QLabel):
            child.style().unpolish(child)
            child.style().polish(child)

    def mouseReleaseEvent(self, event) -> None:
        if self.isEnabled() and event.button() == Qt.MouseButton.LeftButton \
                and self.rect().contains(event.position().toPoint()):
            self.clicked.emit(self.key)
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:
        if self.isEnabled() and event.key() in (Qt.Key.Key_Space, Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.clicked.emit(self.key)
            return
        super().keyPressEvent(event)


class OptionCardGrid(QWidget):
    """2x2 grid of OptionCards with single selection. options: dicts with key, title,
    description, badge and optionally extra (a QWidget shown on the card)."""

    selection_changed = pyqtSignal(str)  # key

    def __init__(self, options: List[dict], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._cards: Dict[str, OptionCard] = {}
        self._selected: Optional[str] = None
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(18)
        for i, opt in enumerate(options):
            card = OptionCard(opt["key"], opt["title"], opt["description"], opt["badge"], self,
                              extra=opt.get("extra"))
            card.clicked.connect(self.select)
            self._cards[opt["key"]] = card
            grid.addWidget(card, i // 2, i % 2)
        grid.setColumnStretch(0, 1)   # equal columns, whatever the description lengths
        grid.setColumnStretch(1, 1)

    def select(self, key: Optional[str]) -> None:
        if key is not None and (key not in self._cards or not self._cards[key].is_available()):
            return
        self._selected = key
        for k, card in self._cards.items():
            card.set_selected(k == key)
        self.selection_changed.emit(key or "")

    def selected(self) -> Optional[str]:
        return self._selected

    def card(self, key: str) -> OptionCard:
        return self._cards[key]

    def set_available(self, key: str, available: bool, reason: str = "") -> None:
        self._cards[key].set_available(available, reason)
        if not available and self._selected == key:
            self.select(None)


class ToggleSwitch(QAbstractButton):
    """Small animated on/off switch."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(46, 24)
        self._offset = 0.0
        self._anim = QPropertyAnimation(self, b"offset", self)
        self._anim.setDuration(140)
        self._anim.setEasingCurve(QEasingCurve.Type.InOutCubic)
        self.toggled.connect(self._animate)

    def get_offset(self) -> float:
        return self._offset

    def set_offset(self, value: float) -> None:
        self._offset = value
        self.update()

    offset = pyqtProperty(float, fget=get_offset, fset=set_offset)

    def setChecked(self, checked: bool) -> None:  # noqa: N802 (Qt API)
        super().setChecked(checked)
        self._anim.stop()
        self.set_offset(1.0 if checked else 0.0)

    def _animate(self, checked: bool) -> None:
        self._anim.stop()
        self._anim.setStartValue(self._offset)
        self._anim.setEndValue(1.0 if checked else 0.0)
        self._anim.start()

    def sizeHint(self) -> QSize:
        return QSize(46, 24)

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        off, on = QColor(85, 85, 85), QColor(CARD_SELECTED_BORDER)
        t = self._offset
        track = QColor(int(off.red() + (on.red() - off.red()) * t),
                       int(off.green() + (on.green() - off.green()) * t),
                       int(off.blue() + (on.blue() - off.blue()) * t))
        p.setBrush(track)
        r = QRectF(self.rect())
        p.drawRoundedRect(r, r.height() / 2, r.height() / 2)
        d = r.height() - 6
        p.setBrush(QColor("white"))
        p.drawEllipse(QRectF(3 + t * (r.width() - d - 6), 3, d, d))


class SpectrumCheckToggleRow(QWidget):
    """'Include spectrum check after segmentation' row that slides/fades in and out."""

    TEXT = "Include spectrum check after segmentation"

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        frame = QFrame(self)
        frame.setObjectName("toggleRow")
        frame.setStyleSheet(f"""
            QFrame#toggleRow {{ background: {CARD_BG}; border: 1px solid {CARD_BORDER}; border-radius: 10px; }}
            QLabel {{ color: white; font-size: 15px; background: transparent; border: 0px; }}
        """)
        label = QLabel(self.TEXT)
        self.switch = ToggleSwitch()
        label.mousePressEvent = lambda _e: self.switch.toggle()  # clicking the text flips it too
        row = QHBoxLayout(frame)
        row.setContentsMargins(18, 10, 18, 10)
        row.addWidget(label)
        row.addStretch(1)
        row.addWidget(self.switch)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(frame)

        self._full_height = frame.sizeHint().height()
        # fixed height = sizeHint clamped by maximumHeight, which the slide animation drives
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self._opacity = QGraphicsOpacityEffect(self)
        self._opacity.setOpacity(0.0)
        self.setGraphicsEffect(self._opacity)
        self.setMaximumHeight(0)
        self._shown = False

        self._height_anim = QPropertyAnimation(self, b"maximumHeight", self)
        self._fade_anim = QPropertyAnimation(self._opacity, b"opacity", self)
        for anim in (self._height_anim, self._fade_anim):
            anim.setDuration(220)
            anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._group = QParallelAnimationGroup(self)
        self._group.addAnimation(self._height_anim)
        self._group.addAnimation(self._fade_anim)

    def is_checked(self) -> bool:
        return self._shown and self.switch.isChecked()

    def set_shown(self, shown: bool, animate: bool = True) -> None:
        if shown == self._shown:
            return
        self._shown = shown
        if shown:
            self.switch.setChecked(False)  # defaults to off each time it appears
        self._group.stop()
        end_h, end_o = (self._full_height, 1.0) if shown else (0, 0.0)
        if not animate:
            self.setMaximumHeight(end_h)
            self._opacity.setOpacity(end_o)
            return
        self._height_anim.setStartValue(self.maximumHeight())
        self._height_anim.setEndValue(end_h)
        self._fade_anim.setStartValue(self._opacity.opacity())
        self._fade_anim.setEndValue(end_o)
        self._group.start()


# names used by the segmentation type screen
SegTypeCard = OptionCard
SegTypeCardGrid = OptionCardGrid
