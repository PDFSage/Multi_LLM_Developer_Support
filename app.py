"""PyQt6 chat client supporting streamed responses from Gemini and OpenAI."""
import sys, os, json
from enum import Enum
from dataclasses import dataclass, field
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSettings
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTextEdit,
    QLineEdit,
    QPushButton,
    QListWidget,
    QListWidgetItem,
    QSplitter,
    QDialog,
    QFormLayout,
    QLabel,
    QComboBox,
    QDialogButtonBox,
)
from PyQt6.QtGui import QTextCursor


class Provider(Enum):
    GEMINI = "Gemini"
    OPENAI = "OpenAI"


@dataclass
class Chat:
    title: str
    messages: list = field(default_factory=list)


class StreamWorker(QThread):
    chunk = pyqtSignal(str)
    done = pyqtSignal()

    def __init__(self, provider: Provider, key: str, prompt: str, history: list):
        super().__init__()
        self.provider = provider
        self.key = key
        self.prompt = prompt
        self.history = history

    def run(self):
        if self.provider is Provider.GEMINI:
            from google import genai
            from google.genai import types

            client = genai.Client(
                api_key=self.key,
                http_options=types.HttpOptions(api_version="v1alpha"),
            )
            text = "\n".join([f'{m["role"]}: {m["content"]}' for m in self.history] + [f"user: {self.prompt}"])
            for c in client.models.generate_content_stream(
                model="gemini-2.0-flash-001", contents=text
            ):
                if getattr(c, "text", ""):
                    self.chunk.emit(c.text)
        else:
            from openai import OpenAI

            client = OpenAI(api_key=self.key)
            stream = client.chat.completions.create(
                model="gpt-4o",
                messages=self.history + [{"role": "user", "content": self.prompt}],
                stream=True,
            )
            for e in stream:
                d = e.choices[0].delta
                if getattr(d, "content", None):
                    self.chunk.emit(d.content)
        self.done.emit()


class SettingsDialog(QDialog):
    def __init__(self, settings: QSettings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.settings = settings
        layout = QFormLayout(self)
        self.provider_box = QComboBox()
        self.provider_box.addItems([p.value for p in Provider])
        self.provider_box.setCurrentText(self.settings.value("provider", Provider.GEMINI.value))
        self.gemini_edit = QLineEdit(self.settings.value("gemini_api_key", ""))
        self.openai_edit = QLineEdit(self.settings.value("openai_api_key", ""))
        self.openai_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.gemini_edit.setEchoMode(QLineEdit.EchoMode.Password)
        layout.addRow(QLabel("Provider"), self.provider_box)
        layout.addRow(QLabel("Gemini API Key"), self.gemini_edit)
        layout.addRow(QLabel("OpenAI API Key"), self.openai_edit)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def accept(self):
        self.settings.setValue("provider", self.provider_box.currentText())
        self.settings.setValue("gemini_api_key", self.gemini_edit.text())
        self.settings.setValue("openai_api_key", self.openai_edit.text())
        super().accept()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Chat")
        self.settings = QSettings("chat", "client")
        self.chats: list[Chat] = []
        self.current = None
        self.workers: list[StreamWorker] = []
        self.sidebar = QListWidget()
        self.sidebar.itemClicked.connect(self.load_chat)
        self.new_btn = QPushButton("+")
        self.new_btn.clicked.connect(self.new_chat)
        left = QVBoxLayout()
        left.addWidget(self.new_btn)
        left.addWidget(self.sidebar)
        left_w = QWidget()
        left_w.setLayout(left)
        self.display = QTextEdit()
        self.display.setReadOnly(True)
        self.input = QLineEdit()
        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self.send)
        bottom = QHBoxLayout()
        bottom.addWidget(self.input)
        bottom.addWidget(self.send_btn)
        right = QVBoxLayout()
        right.addWidget(self.display)
        rb = QWidget()
        rb.setLayout(bottom)
        right.addWidget(rb)
        right_w = QWidget()
        right_w.setLayout(right)
        splitter = QSplitter()
        splitter.addWidget(left_w)
        splitter.addWidget(right_w)
        self.setCentralWidget(splitter)
        men = self.menuBar().addMenu("File")
        act = men.addAction("Settings")
        act.triggered.connect(self.open_settings)
        self.new_chat()

    def open_settings(self):
        SettingsDialog(self.settings, self).exec()

    def new_chat(self):
        chat = Chat(title=f"Chat {len(self.chats)+1}")
        self.chats.append(chat)
        item = QListWidgetItem(chat.title)
        self.sidebar.addItem(item)
        self.sidebar.setCurrentItem(item)
        self.load_chat(item)

    def load_chat(self, item):
        idx = self.sidebar.row(item)
        self.current = self.chats[idx]
        self.refresh()

    def refresh(self):
        self.display.clear()
        for m in self.current.messages:
            self.display.append(f'{m["role"]}: {m["content"]}')

    def finish_worker(self, worker: StreamWorker):
        if worker in self.workers:
            self.workers.remove(worker)
        self.current.messages.append({"role": "assistant", "content": self.ai_buffer})

    def send(self):
        txt = self.input.text().strip()
        if not txt or self.current is None:
            return
        self.current.messages.append({"role": "user", "content": txt})
        self.display.append(f"user: {txt}")
        self.input.clear()
        prov = Provider(self.settings.value("provider", Provider.GEMINI.value))
        key_name = "gemini_api_key" if prov is Provider.GEMINI else "openai_api_key"
        key = self.settings.value(key_name, "")
        worker = StreamWorker(prov, key, txt, self.current.messages[:-1])
        self.ai_buffer = ""
        worker.chunk.connect(self.append_ai)
        worker.done.connect(lambda w=worker: self.finish_worker(w))
        worker.finished.connect(worker.deleteLater)
        self.workers.append(worker)
        worker.start()

    def append_ai(self, s: str):
        first = self.ai_buffer == ""
        self.ai_buffer += s
        if first:
            self.display.append("assistant: ")
        self.display.moveCursor(QTextCursor.MoveOperation.End)
        self.display.insertPlainText(s)
        self.display.verticalScrollBar().setValue(self.display.verticalScrollBar().maximum())

    def closeEvent(self, event):
        for w in self.workers:
            w.quit()
            w.wait()
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
