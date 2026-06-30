"""
viz/editor.py  —  XLSum Dataset Editor

PyQt6 GUI for browsing and inspecting the xlsum judge training data.

Usage:
    python viz/editor.py
    python viz/editor.py --split train
"""

from __future__ import annotations
import argparse, json, pathlib, sys
from PyQt6.QtCore import Qt, QSortFilterProxyModel, QStringListModel
from PyQt6.QtGui import (
    QColor, QFont, QFontDatabase, QKeySequence, QPalette, QShortcut,
    QTextCharFormat, QTextCursor,
)
from PyQt6.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QHBoxLayout, QLabel,
    QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
    QPlainTextEdit, QPushButton, QSizePolicy, QSlider, QSplitter,
    QStackedWidget, QStatusBar, QTabWidget, QVBoxLayout, QWidget,
)

ROOT     = pathlib.Path(__file__).parent.parent
DATA_DIR = ROOT / "data/processed"

DIMS   = ["(all)", "faithfulness", "coverage", "naturalness", "coherence"]
LANGS  = ["(all)", "ar", "cs", "de", "es", "fr", "hi", "id", "it",
          "ja", "ko", "ru", "sv", "tr", "zh"]
LANG_NAMES = {
    "ar": "Arabic",    "cs": "Czech",      "de": "German",    "es": "Spanish",
    "fr": "French",    "hi": "Hindi",      "id": "Indonesian","it": "Italian",
    "ja": "Japanese",  "ko": "Korean",     "ru": "Russian",   "sv": "Swedish",
    "tr": "Turkish",   "zh": "Chinese",
}

SCORE_COLORS = {
    1: "#e74c3c", 2: "#e67e22", 3: "#f39c12",
    4: "#f1c40f", 5: "#2ecc71", 6: "#27ae60", 7: "#1abc9c",
}


# ── Utility ───────────────────────────────────────────────────────────────────

def _bold(widget):
    f = widget.font(); f.setBold(True); widget.setFont(f)


def _load_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _score_badge(score: int) -> str:
    color = SCORE_COLORS.get(score, "#888")
    return f'<span style="background:{color};color:white;padding:2px 8px;border-radius:4px;font-weight:bold;">{score}</span>'


# ── Widgets ───────────────────────────────────────────────────────────────────

class ScoreBar(QWidget):
    """Horizontal colored bar showing a 1–7 score."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        self._cells: list[QLabel] = []
        for i in range(1, 8):
            cell = QLabel(str(i))
            cell.setAlignment(Qt.AlignmentFlag.AlignCenter)
            cell.setFixedSize(32, 32)
            cell.setStyleSheet("border-radius: 4px; font-weight: bold;")
            self._cells.append(cell)
            layout.addWidget(cell)
        layout.addStretch()

    def set_score(self, score: int | None):
        for i, cell in enumerate(self._cells, 1):
            if score is not None and i == score:
                color = SCORE_COLORS.get(i, "#888")
                cell.setStyleSheet(
                    f"background:{color};color:white;border-radius:4px;font-weight:bold;"
                )
            else:
                cell.setStyleSheet(
                    "background:#e0e0e0;color:#888;border-radius:4px;font-weight:bold;"
                )


class OverviewTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        def row(label_text, widget):
            h = QHBoxLayout()
            lbl = QLabel(label_text)
            lbl.setFixedWidth(110)
            _bold(lbl)
            h.addWidget(lbl)
            h.addWidget(widget)
            layout.addLayout(h)
            return widget

        self._taskid  = QLabel()
        self._taskid.setWordWrap(True)
        self._taskid.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        row("Task ID:", self._taskid)

        self._lang    = QLabel(); row("Language:", self._lang)
        self._model   = QLabel(); row("Gen. Model:", self._model)
        self._dim     = QLabel(); row("Dimension:", self._dim)

        score_lbl = QLabel("Human Score:")
        score_lbl.setFixedWidth(110)
        _bold(score_lbl)
        self._score_bar = ScoreBar()
        h = QHBoxLayout()
        h.addWidget(score_lbl)
        h.addWidget(self._score_bar)
        layout.addLayout(h)

        layout.addStretch()

    def load(self, record: dict):
        m = record["meta"]
        lang_code = m.get("language", "")
        lang_name = LANG_NAMES.get(lang_code, lang_code)
        self._taskid.setText(m.get("taskid", ""))
        self._lang.setText(f"{lang_name} ({lang_code})")
        self._model.setText(m.get("gen_model", ""))
        dim = m.get("dimension", "")
        self._dim.setText(dim.capitalize())
        self._score_bar.set_score(m.get("human_score"))


class ReviewsTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._editor = QPlainTextEdit()
        self._editor.setReadOnly(True)
        layout.addWidget(self._editor)

    def load(self, record: dict):
        user_msg = next(
            (m["content"] for m in record["messages"] if m["role"] == "user"), ""
        )
        # Extract just the reviews block
        if "Reviews:" in user_msg and "Summary:" in user_msg:
            reviews_block = user_msg.split("Reviews:")[1].split("Summary:")[0].strip()
        else:
            reviews_block = user_msg
        self._editor.setPlainText(reviews_block)

    def set_font(self, font: QFont):
        self._editor.setFont(font)


class SummaryTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._dim_label = QLabel()
        _bold(self._dim_label)
        layout.addWidget(self._dim_label)

        self._summary = QPlainTextEdit()
        self._summary.setReadOnly(True)
        layout.addWidget(self._summary)

    def load(self, record: dict):
        m = record["meta"]
        user_msg = next(
            (msg["content"] for msg in record["messages"] if msg["role"] == "user"), ""
        )
        summary_text = ""
        if "Summary:" in user_msg:
            after = user_msg.split("Summary:")[1]
            # stop at the score question line
            summary_text = after.split("\n\n")[0].strip()

        self._dim_label.setText(f"Scoring dimension: {m.get('dimension','').capitalize()}")
        self._summary.setPlainText(summary_text)

    def set_font(self, font: QFont):
        self._summary.setFont(font)


class RawTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._editor = QPlainTextEdit()
        self._editor.setReadOnly(True)
        self._editor.setFont(QFont("Courier New", 10))
        layout.addWidget(self._editor)

    def load(self, record: dict):
        self._editor.setPlainText(json.dumps(record, ensure_ascii=False, indent=2))


# ── Main Window ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self, initial_split: str = "train"):
        super().__init__()
        self._all_records: list[dict] = []
        self._filtered: list[dict] = []
        self._current_record: dict | None = None

        self.setWindowTitle("XLSum Dataset Editor")
        self.resize(1280, 820)
        self._build_ui()
        self._load_split(initial_split)

        QShortcut(QKeySequence("Ctrl+E"), self).activated.connect(self._export_filtered)
        QShortcut(QKeySequence("Ctrl+F"), self).activated.connect(
            lambda: self._search_edit.setFocus()
        )

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(splitter)

        # ── Left panel ────────────────────────────────────────────────────────
        left = QWidget()
        left.setMinimumWidth(220)
        left.setMaximumWidth(300)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(6, 6, 6, 6)
        ll.setSpacing(4)

        title = QLabel("XLSum Records")
        _bold(title)
        ll.addWidget(title)

        # Split selector
        self._split_combo = QComboBox()
        self._split_combo.addItems(["train", "dev", "test"])
        self._split_combo.currentTextChanged.connect(self._load_split)
        ll.addWidget(self._split_combo)

        ll.addWidget(self._hsep())

        # Filters
        filter_lbl = QLabel("Filters")
        _bold(filter_lbl)
        ll.addWidget(filter_lbl)

        self._lang_combo = QComboBox()
        self._lang_combo.addItems(LANGS)
        self._lang_combo.currentIndexChanged.connect(self._apply_filters)
        ll.addWidget(QLabel("Language:"))
        ll.addWidget(self._lang_combo)

        self._dim_combo = QComboBox()
        self._dim_combo.addItems(DIMS)
        self._dim_combo.currentIndexChanged.connect(self._apply_filters)
        ll.addWidget(QLabel("Dimension:"))
        ll.addWidget(self._dim_combo)

        ll.addWidget(QLabel("Min score:"))
        score_row = QHBoxLayout()
        self._min_score = QSlider(Qt.Orientation.Horizontal)
        self._min_score.setRange(1, 7)
        self._min_score.setValue(1)
        self._min_score.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._min_score_label = QLabel("1")
        self._min_score_label.setFixedWidth(16)
        self._min_score.valueChanged.connect(
            lambda v: (self._min_score_label.setText(str(v)), self._apply_filters())
        )
        score_row.addWidget(self._min_score)
        score_row.addWidget(self._min_score_label)
        ll.addLayout(score_row)

        ll.addWidget(self._hsep())

        # Search
        self._search_edit = QLineEdit()
        self._search_edit.setPlaceholderText("Search taskid / model… (Ctrl+F)")
        self._search_edit.textChanged.connect(self._apply_filters)
        ll.addWidget(self._search_edit)

        # Record list
        self._count_label = QLabel("0 records")
        ll.addWidget(self._count_label)

        self._record_list = QListWidget()
        self._record_list.currentRowChanged.connect(self._on_row_changed)
        ll.addWidget(self._record_list)

        # Buttons
        export_btn = QPushButton("Export filtered → JSONL  (Ctrl+E)")
        export_btn.clicked.connect(self._export_filtered)
        ll.addWidget(export_btn)

        self._dark_mode = False
        self._dark_btn = QPushButton("Dark mode")
        self._dark_btn.clicked.connect(self._toggle_dark)
        ll.addWidget(self._dark_btn)

        splitter.addWidget(left)

        # ── Right panel ───────────────────────────────────────────────────────
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 4, 0, 0)
        rl.setSpacing(2)

        self._record_label = QLabel("No record selected")
        self._record_label.setContentsMargins(8, 0, 0, 0)
        _bold(self._record_label)
        rl.addWidget(self._record_label)

        self._stack = QStackedWidget()

        placeholder = QLabel("Select a record from the left panel.")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder.setStyleSheet("color: #888; font-style: italic;")
        self._stack.addWidget(placeholder)   # index 0

        self._tabs = QTabWidget()
        self._overview_tab = OverviewTab()
        self._reviews_tab  = ReviewsTab()
        self._summary_tab  = SummaryTab()
        self._raw_tab      = RawTab()

        self._tabs.addTab(self._overview_tab, "Overview")
        self._tabs.addTab(self._reviews_tab,  "Reviews")
        self._tabs.addTab(self._summary_tab,  "Summary")
        self._tabs.addTab(self._raw_tab,      "Raw JSON")
        self._stack.addWidget(self._tabs)    # index 1

        rl.addWidget(self._stack)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        self.setStatusBar(QStatusBar())

    def _hsep(self) -> QWidget:
        sep = QWidget(); sep.setFixedHeight(1)
        sep.setStyleSheet("background:#ccc;")
        return sep

    # ── Data loading ──────────────────────────────────────────────────────────

    def _load_split(self, split: str):
        path = DATA_DIR / f"{split}.jsonl"
        self._all_records = _load_jsonl(path)
        self._lang_combo.setCurrentIndex(0)
        self._dim_combo.setCurrentIndex(0)
        self._min_score.setValue(1)
        self._search_edit.clear()
        self._apply_filters()
        self.statusBar().showMessage(
            f"Loaded {len(self._all_records)} records from {path.name}"
        )

    # ── Filtering ─────────────────────────────────────────────────────────────

    def _apply_filters(self):
        lang      = self._lang_combo.currentText()
        dim       = self._dim_combo.currentText()
        min_score = self._min_score.value()
        query     = self._search_edit.text().lower()

        self._filtered = [
            r for r in self._all_records
            if (lang == "(all)" or r["meta"].get("language") == lang)
            and (dim  == "(all)" or r["meta"].get("dimension") == dim)
            and r["meta"].get("human_score", 0) >= min_score
            and (not query or query in r["meta"].get("taskid", "").lower()
                           or query in r["meta"].get("gen_model", "").lower())
        ]

        self._record_list.blockSignals(True)
        self._record_list.clear()
        for r in self._filtered:
            m     = r["meta"]
            score = m.get("human_score", "?")
            label = f"[{score}] {m.get('dimension','')[:3].upper()}·{m.get('language','')}  {m.get('gen_model','')}"
            item  = QListWidgetItem(label)
            color = SCORE_COLORS.get(score, "#888")
            item.setForeground(QColor(color))
            self._record_list.addItem(item)
        self._record_list.blockSignals(False)

        self._count_label.setText(
            f"{len(self._filtered)} / {len(self._all_records)} records"
        )
        self._stack.setCurrentIndex(0)
        self._record_label.setText("No record selected")

    # ── Record display ────────────────────────────────────────────────────────

    def _on_row_changed(self, row: int):
        if row < 0 or row >= len(self._filtered):
            return
        record = self._filtered[row]
        self._current_record = record
        m = record["meta"]

        lang_code = m.get("language", "")
        self._record_label.setText(
            f"{m.get('taskid','')}"
        )

        self._overview_tab.load(record)
        self._reviews_tab.load(record)
        self._summary_tab.load(record)
        self._raw_tab.load(record)

        self._stack.setCurrentIndex(1)
        self.statusBar().showMessage(
            f"{m.get('gen_model','')} · {m.get('dimension','')} · "
            f"{LANG_NAMES.get(lang_code, lang_code)} · score={m.get('human_score','?')}"
        )

    # ── Export ────────────────────────────────────────────────────────────────

    def _export_filtered(self):
        if not self._filtered:
            QMessageBox.warning(self, "Export", "No records to export.")
            return
        dest, _ = QFileDialog.getSaveFileName(
            self, "Export filtered records", str(DATA_DIR / "filtered.jsonl"),
            "JSONL (*.jsonl);;All files (*)"
        )
        if not dest:
            return
        with open(dest, "w", encoding="utf-8") as f:
            for r in self._filtered:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        QMessageBox.information(
            self, "Done", f"Exported {len(self._filtered)} records →\n{dest}"
        )
        self.statusBar().showMessage(f"Exported {len(self._filtered)} records")

    # ── Dark mode ─────────────────────────────────────────────────────────────

    def _toggle_dark(self):
        self._dark_mode = not self._dark_mode
        app = QApplication.instance()
        if self._dark_mode:
            palette = QPalette()
            palette.setColor(QPalette.ColorRole.Window,          QColor("#1e1e1e"))
            palette.setColor(QPalette.ColorRole.WindowText,      QColor("#d4d4d4"))
            palette.setColor(QPalette.ColorRole.Base,            QColor("#252526"))
            palette.setColor(QPalette.ColorRole.AlternateBase,   QColor("#2d2d30"))
            palette.setColor(QPalette.ColorRole.Text,            QColor("#d4d4d4"))
            palette.setColor(QPalette.ColorRole.Button,          QColor("#3c3c3c"))
            palette.setColor(QPalette.ColorRole.ButtonText,      QColor("#d4d4d4"))
            palette.setColor(QPalette.ColorRole.Highlight,       QColor("#264f78"))
            palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
            app.setPalette(palette)
            self._dark_btn.setText("Light mode")
        else:
            app.setPalette(app.style().standardPalette())
            self._dark_btn.setText("Dark mode")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=["train", "dev", "test"])
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont("Segoe UI", 10))

    win = MainWindow(initial_split=args.split)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
