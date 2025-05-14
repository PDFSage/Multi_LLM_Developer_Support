import sys, os, json, mimetypes, uuid
from enum import Enum
from dataclasses import dataclass, field
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSettings
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QLineEdit, QPushButton, QListWidget, QListWidgetItem,
    QSplitter, QDialog, QFormLayout, QLabel, QComboBox, QDialogButtonBox,
    QFileDialog, QCheckBox
)
from PyQt6.QtGui import QTextCursor
from google.cloud import storage

@dataclass
class ProviderConfig:
    openai_api_key: str
    gemini_api_key: str
    pdfsage_project: str = "pdfsage-5ecef"
    pdfsage_location: str = "us-central1"

from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional, Union
import base64
from openai import OpenAI
from google import genai
from google.genai import types

class StreamingRenderer:
    def __init__(self, iterator: Iterable[str]):
        self._it = iterator
    def __iter__(self) -> Iterator[str]:
        for chunk in self._it:
            yield chunk

class GeminiForDevelopers:
    def __init__(self, api_key: str):
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(api_version="v1alpha"),
        )
    def list_models(self) -> List[str]:
        return [m.name for m in self._client.models.list()]
    def generate(self, contents: Union[str, List[types.Part]], model: str = "gemini-2.0-flash-001", safety: Optional[List[types.SafetySetting]] = None, stream: bool = False) -> Union[str, StreamingRenderer]:
        cfg = types.GenerateContentConfig(safety_settings=safety or [])
        if stream:
            iterator = self._client.models.generate_content_stream(model=model, contents=contents, config=cfg)
            return StreamingRenderer(iterator)
        response = self._client.models.generate_content(model=model, contents=contents, config=cfg)
        return response.text

class GeminiForVertexAIAPI:
    def __init__(self, project: str, location: str, api_key: Optional[str] = None):
        if api_key:
            self._client = genai.Client(api_key=api_key, vertexai=True, project=project, location=location)
        else:
            self._client = genai.Client(vertexai=True, project=project, location=location)
    def list_models(self) -> List[str]:
        return [m.name for m in self._client.models.list()]
    def generate(self, contents: Union[str, List[types.Part]], model: str = "gemini-2.0-flash-001", safety: Optional[List[types.SafetySetting]] = None, stream: bool = False) -> Union[str, StreamingRenderer]:
        cfg = types.GenerateContentConfig(safety_settings=safety or [])
        if stream:
            iterator = self._client.models.generate_content_stream(model=model, contents=contents, config=cfg)
            return StreamingRenderer(iterator)
        response = self._client.models.generate_content(model=model, contents=contents, config=cfg)
        return response.text
    def upload_to_gcs(self, gcs_uri: str, local_path: str, mime_type: str = "application/octet-stream") -> str:
        bucket_name, blob_name = gcs_uri[5:].split("/", 1)
        storage.Client().bucket(bucket_name).blob(blob_name).upload_from_filename(local_path, content_type=mime_type)
        return gcs_uri

class OpenAIChat:
    def __init__(self, api_key: str):
        self._client = OpenAI(api_key=api_key)
    def chat(self, prompt_or_messages: Union[str, List[dict]], model: str = "gpt-4o-mini", stream: bool = False, image_path: Optional[str] = None, mime_type: str = "image/png", **kwargs) -> Union[str, StreamingRenderer]:
        if isinstance(prompt_or_messages, str):
            messages = [{"role": "user", "content": prompt_or_messages}]
        else:
            messages = prompt_or_messages
        if image_path:
            with open(image_path, "rb") as f:
                b64_image = base64.b64encode(f.read()).decode("utf-8")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": messages[0]["content"]},
                        {"type": "input_image", "image_url": f"data:{mime_type};base64,{b64_image}"},
                    ],
                }
            ]
        if stream:
            iterator = self._client.chat.completions.create(model=model, messages=messages, stream=True, **kwargs)
            return StreamingRenderer((c.choices[0].delta.get("content", "") for c in iterator))
        response = self._client.chat.completions.create(model=model, messages=messages, **kwargs)
        return response.choices[0].message.content

def moderation_settings() -> List[types.SafetySetting]:
    return [types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_ONLY_HIGH")]

def part_from_image_uri(uri: str, mime_type: str) -> types.Part:
    return types.Part.from_uri(file_uri=uri, mime_type=mime_type)

def part_from_text(text: str) -> types.Part:
    return types.Part.from_text(text)

class Provider(Enum):
    GEMINI_API = "Gemini API"
    GEMINI_PDFSAGE = "PDFSage Gemini"
    OPENAI = "OpenAI"

@dataclass
class Chat:
    title: str
    messages: list = field(default_factory=list)

class StreamWorker(QThread):
    chunk = pyqtSignal(str)
    done = pyqtSignal()
    def __init__(self, provider: Provider, key: str, prompt: str, history: list, model: str, moderate: bool, images: list[str]):
        super().__init__()
        self.provider, self.key, self.prompt = provider, key, prompt
        self.history, self.model, self.moderate, self.images = history, model, moderate, images
    def _upload_to_gcs(self, local_path: str) -> str:
        client = storage.Client()
        bucket = client.bucket("pdfsage-gemini-data-for-some-reason")
        blob_name = f"{uuid.uuid4().hex}{os.path.splitext(local_path)[1]}"
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(local_path)
        return f"gs://pdfsage-gemini-data-for-some-reason/{blob_name}"
    def run(self):
        if self.provider in (Provider.GEMINI_API, Provider.GEMINI_PDFSAGE):
            from google import genai
            from google.genai import types
            if self.provider is Provider.GEMINI_API:
                client = genai.Client(api_key=self.key, http_options=types.HttpOptions(api_version="v1alpha"))
            else:
                client = genai.Client(api_key=self.key, vertexai=True, project="pdfsage-5ecef", location="us-central1")
            safety = [types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_ONLY_HIGH")] if self.moderate else None
            if self.images:
                parts = [types.Part.from_text(self.prompt)]
                for p in self.images:
                    uri = self._upload_to_gcs(p)
                    mime = mimetypes.guess_type(p)[0] or "application/octet-stream"
                    parts.append(types.Part.from_uri(file_uri=uri, mime_type=mime))
                contents = parts
            else:
                text = "\n".join([f'{m["role"]}: {m["content"]}' for m in self.history] + [f"user: {self.prompt}"])
                contents = text
            if safety:
                cfg = types.GenerateContentConfig(safety_settings=safety)
            else:
                cfg = types.GenerateContentConfig()
            for c in client.models.generate_content_stream(model=self.model, contents=contents, config=cfg):
                if getattr(c, "text", ""):
                    self.chunk.emit(c.text)
        else:
            from openai import OpenAI
            client = OpenAI(api_key=self.key)
            stream = client.chat.completions.create(model=self.model, messages=self.history + [{"role": "user", "content": self.prompt}], stream=True)
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
        self.provider_box.setCurrentText(self.settings.value("provider", Provider.GEMINI_API.value))
        self.gemini_edit = QLineEdit(self.settings.value("gemini_api_key", ""))
        self.pdfsage_edit = QLineEdit(self.settings.value("pdfsage_api_key", ""))
        self.openai_edit = QLineEdit(self.settings.value("openai_api_key", ""))
        self.gemini_model = QLineEdit(self.settings.value("gemini_model", "gemini-2.0-flash-001"))
        self.openai_model = QLineEdit(self.settings.value("openai_model", "gpt-4o"))
        self.moderate_box = QCheckBox("Moderate hate-speech")
        self.moderate_box.setChecked(self.settings.value("gemini_moderate", "false") == "true")
        for e in (self.gemini_edit, self.pdfsage_edit, self.openai_edit):
            e.setEchoMode(QLineEdit.EchoMode.Password)
        layout.addRow(QLabel("Provider"), self.provider_box)
        layout.addRow(QLabel("Gemini API Key"), self.gemini_edit)
        layout.addRow(QLabel("PDFSage Gemini Key"), self.pdfsage_edit)
        layout.addRow(QLabel("Gemini/OpenAI Model"), self.gemini_model)
        layout.addRow(QLabel("OpenAI API Key"), self.openai_edit)
        layout.addRow(self.moderate_box)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    def accept(self):
        self.settings.setValue("provider", self.provider_box.currentText())
        self.settings.setValue("gemini_api_key", self.gemini_edit.text())
        self.settings.setValue("pdfsage_api_key", self.pdfsage_edit.text())
        self.settings.setValue("openai_api_key", self.openai_edit.text())
        self.settings.setValue("gemini_model", self.gemini_model.text())
        self.settings.setValue("openai_model", self.openai_model.text())
        self.settings.setValue("gemini_moderate", "true" if self.moderate_box.isChecked() else "false")
        super().accept()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Chat")
        self.settings = QSettings("chat", "client")
        self.chats: list[Chat] = []
        self.current = None
        self.workers: list[StreamWorker] = []
        self.images: list[str] = []
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
        self.add_image_btn = QPushButton("Add Image")
        self.send_btn.clicked.connect(self.send)
        self.add_image_btn.clicked.connect(self.add_image)
        bottom = QHBoxLayout()
        bottom.addWidget(self.input)
        bottom.addWidget(self.add_image_btn)
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
    def add_image(self):
        file, _ = QFileDialog.getOpenFileName(self, "Choose image")
        if file:
            self.images.append(file)
            self.display.append(f"*added image*: {os.path.basename(file)}")
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
        prov = Provider(self.settings.value("provider", Provider.GEMINI_API.value))
        if prov is Provider.GEMINI_API:
            key = self.settings.value("gemini_api_key", "")
        elif prov is Provider.GEMINI_PDFSAGE:
            key = self.settings.value("pdfsage_api_key", "")
        else:
            key = self.settings.value("openai_api_key", "")
        model_key = "gemini_model" if prov is not Provider.OPENAI else "openai_model"
        model = self.settings.value(model_key, "gemini-2.0-flash-001")
        moderate = self.settings.value("gemini_moderate", "false") == "true"
        worker = StreamWorker(prov, key, txt, self.current.messages[:-1], model, moderate, self.images)
        self.ai_buffer = ""
        worker.chunk.connect(self.append_ai)
        worker.done.connect(lambda w=worker: self.finish_worker(w))
        worker.finished.connect(worker.deleteLater)
        self.workers.append(worker)
        worker.start()
        self.images = []
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
