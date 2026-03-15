from __future__ import annotations

import os
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .config import has_ffmpeg, load_settings, resolve_ffmpeg_path, save_settings
from .constants import (
    DEFAULT_VAD_MAX_SEGMENT_SECONDS,
    DEFAULT_VAD_MIN_SILENCE_MS,
    DEFAULT_VAD_MIN_SPEECH_MS,
    DEFAULT_VAD_SPEECH_PAD_MS,
    DEFAULT_VAD_THRESHOLD,
    MEDIA_EXTENSIONS,
    STATUS_LABELS,
    SUBTITLE_EXTENSIONS,
    SUPPORTED_EXTENSIONS,
    VIDEO_EXTENSIONS,
)
from .media import discover_supported_files
from .models import AppSettings, TaskCancelled, TaskState
from .orchestrator import TaskRunner
from .providers import transcription as transcription_provider
from .utils import new_task_id, normalize_language_code, parse_context_bias


class WorkerSignals(QObject):
    progress = Signal(str, str, int, str)
    finished = Signal(str, bool, str, str, dict)


class DropFrame(QFrame):
    files_dropped = Signal(list)

    def __init__(self) -> None:
        super().__init__()
        self.setAcceptDrops(True)
        self.setObjectName("dropFrame")
        layout = QVBoxLayout(self)
        label = QLabel("将视频/音频/字幕文件或文件夹拖拽到这里")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:  # noqa: N802
        paths: List[str] = []
        for url in event.mimeData().urls():
            local = url.toLocalFile()
            if local:
                paths.append(local)
        if paths:
            self.files_dropped.emit(paths)
        event.acceptProposedAction()


def run_task_worker(
    task_id: str,
    source_path: Path,
    settings: AppSettings,
    signals: WorkerSignals,
    cancel_event: threading.Event,
) -> None:
    runner = TaskRunner(settings)

    def report(status: str, progress: int, message: str) -> None:
        signals.progress.emit(task_id, status, progress, message)

    try:
        outputs = runner.run_task(source_path, report, cancel_event)
        signals.finished.emit(task_id, True, "Completed", "完成", outputs)
    except TaskCancelled as exc:
        signals.finished.emit(task_id, False, "Cancelled", str(exc), {})
    except Exception as exc:
        message = str(exc).strip() or traceback.format_exc(limit=1)
        signals.finished.emit(task_id, False, "Failed", message, {})


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Subtitle Studio 字幕工作台")
        self.resize(1280, 860)

        self.signals = WorkerSignals()
        self.signals.progress.connect(self.on_task_progress)
        self.signals.finished.connect(self.on_task_finished)

        self.executor: Optional[ThreadPoolExecutor] = None
        self.cancel_event = threading.Event()
        self.tasks: Dict[str, TaskState] = {}
        self.path_to_task: Dict[str, str] = {}
        self.active_run_ids: set[str] = set()
        self.completed_run_ids: set[str] = set()
        self.run_progress: Dict[str, int] = {}
        self.futures: Dict[str, Any] = {}
        self.is_running = False

        self.init_ui()
        self.apply_style()
        self.load_settings_into_ui()

    def init_ui(self) -> None:
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(14, 14, 14, 14)
        root_layout.setSpacing(10)

        tabs = QTabWidget()
        tabs.addTab(self._build_task_page(), "任务")
        tabs.addTab(self._build_settings_page(), "设置")
        root_layout.addWidget(tabs)

        self.setCentralWidget(root)

    def _build_task_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self.drop_frame = DropFrame()
        self.drop_frame.files_dropped.connect(self.on_drop_paths)
        layout.addWidget(self.drop_frame)

        import_row = QWidget()
        import_layout = QHBoxLayout(import_row)
        import_layout.setContentsMargins(0, 0, 0, 0)

        self.add_file_btn = QPushButton("添加文件")
        self.add_file_btn.clicked.connect(self.on_add_file)
        self.add_folder_btn = QPushButton("添加文件夹")
        self.add_folder_btn.clicked.connect(self.on_add_folder)
        self.remove_btn = QPushButton("删除所选")
        self.remove_btn.clicked.connect(self.on_remove_selected)
        self.clear_btn = QPushButton("清空列表")
        self.clear_btn.clicked.connect(self.on_clear_all)
        self.start_btn = QPushButton("开始")
        self.start_btn.clicked.connect(self.on_start)
        self.stop_btn = QPushButton("停止")
        self.stop_btn.clicked.connect(self.on_stop)
        self.stop_btn.setEnabled(False)
        self.open_output_btn = QPushButton("打开输出目录")
        self.open_output_btn.clicked.connect(self.on_open_output_dir)

        import_layout.addWidget(self.add_file_btn)
        import_layout.addWidget(self.add_folder_btn)
        import_layout.addWidget(self.remove_btn)
        import_layout.addWidget(self.clear_btn)
        import_layout.addStretch(1)
        import_layout.addWidget(self.start_btn)
        import_layout.addWidget(self.stop_btn)
        import_layout.addWidget(self.open_output_btn)
        layout.addWidget(import_row)

        self.task_table = QTableWidget(0, 5)
        self.task_table.setHorizontalHeaderLabels(["来源文件", "状态", "进度", "输出文件", "消息"])
        self.task_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.task_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.task_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.task_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.task_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.task_table.verticalHeader().setVisible(False)
        self.task_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.task_table.setAlternatingRowColors(True)
        self.task_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.task_table)

        self.total_progress = QProgressBar()
        self.total_progress.setRange(0, 100)
        self.total_progress.setValue(0)
        self.summary_label = QLabel("当前无运行任务")
        layout.addWidget(self.total_progress)
        layout.addWidget(self.summary_label)

        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFixedHeight(140)
        layout.addWidget(self.log_text)
        return page

    def _build_settings_page(self) -> QWidget:
        page = QWidget()
        outer_layout = QVBoxLayout(page)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        outer_layout.addWidget(scroll)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        for group in (
            self._build_transcription_group(),
            self._build_translation_group(),
            self._build_output_group(),
            self._build_preprocess_group(),
        ):
            group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
            layout.addWidget(group)

        actions = QWidget()
        actions.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        actions_layout = QHBoxLayout(actions)
        actions_layout.setContentsMargins(0, 0, 0, 0)
        actions_layout.addStretch(1)
        self.save_settings_btn = QPushButton("保存设置")
        self.save_settings_btn.clicked.connect(self.on_save_settings)
        actions_layout.addWidget(self.save_settings_btn)
        layout.addWidget(actions)
        layout.addStretch(1)
        scroll.setWidget(content)
        return page

    def _build_transcription_group(self) -> QGroupBox:
        group = QGroupBox("转写设置")
        layout = QGridLayout(group)

        self.transcription_provider_combo = QComboBox()
        self.transcription_provider_combo.addItem("Mistral", "mistral")
        self.transcription_provider_combo.addItem("Whisper(OpenAI 兼容)", "whisper_openai_compatible")
        self.transcription_provider_combo.currentIndexChanged.connect(self.on_transcription_provider_changed)

        self.mistral_api_key_input = QLineEdit(os.environ.get("MISTRAL_API_KEY", ""))
        self.mistral_api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.show_mistral_key_checkbox = QCheckBox("显示")
        self.show_mistral_key_checkbox.toggled.connect(
            lambda checked: self.mistral_api_key_input.setEchoMode(
                QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password
            )
        )
        mistral_key_row = QWidget()
        mistral_key_layout = QHBoxLayout(mistral_key_row)
        mistral_key_layout.setContentsMargins(0, 0, 0, 0)
        mistral_key_layout.addWidget(self.mistral_api_key_input)
        mistral_key_layout.addWidget(self.show_mistral_key_checkbox)

        self.mistral_model_combo = QComboBox()
        self.mistral_model_combo.setEditable(True)
        self.mistral_model_combo.addItems(["voxtral-mini-latest", "voxtral-small-latest"])

        self.whisper_base_url_input = QLineEdit("https://api.openai.com/v1")
        self.whisper_model_input = QLineEdit("whisper-1")
        self.whisper_api_key_input = QLineEdit(os.environ.get("OPENAI_API_KEY", ""))
        self.whisper_api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.show_whisper_key_checkbox = QCheckBox("显示")
        self.show_whisper_key_checkbox.toggled.connect(
            lambda checked: self.whisper_api_key_input.setEchoMode(
                QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password
            )
        )
        whisper_key_row = QWidget()
        whisper_key_layout = QHBoxLayout(whisper_key_row)
        whisper_key_layout.setContentsMargins(0, 0, 0, 0)
        whisper_key_layout.addWidget(self.whisper_api_key_input)
        whisper_key_layout.addWidget(self.show_whisper_key_checkbox)

        self.language_mode_combo = QComboBox()
        self.language_mode_combo.addItems(["自动识别", "指定语言"])
        self.language_mode_combo.currentIndexChanged.connect(self.on_language_mode_changed)
        self.language_input = QLineEdit("zh")
        self.timestamp_combo = QComboBox()
        self.timestamp_combo.addItems(["none", "segment", "word"])
        self.timestamp_combo.setCurrentText("segment")
        self.diarize_checkbox = QCheckBox("启用说话人分离（仅 Mistral）")
        self.thread_spin = QSpinBox()
        self.thread_spin.setRange(1, 16)
        self.thread_spin.setValue(3)
        self.context_bias_input = QPlainTextEdit()
        self.context_bias_input.setPlaceholderText("术语提示/上下文偏置，使用逗号或换行分隔")
        self.context_bias_input.setFixedHeight(64)

        layout.addWidget(QLabel("转写提供方"), 0, 0)
        layout.addWidget(self.transcription_provider_combo, 0, 1)
        layout.addWidget(QLabel("Mistral API Key"), 1, 0)
        layout.addWidget(mistral_key_row, 1, 1)
        layout.addWidget(QLabel("Mistral 模型"), 2, 0)
        layout.addWidget(self.mistral_model_combo, 2, 1)
        layout.addWidget(QLabel("Whisper Base URL"), 3, 0)
        layout.addWidget(self.whisper_base_url_input, 3, 1)
        layout.addWidget(QLabel("Whisper API Key"), 4, 0)
        layout.addWidget(whisper_key_row, 4, 1)
        layout.addWidget(QLabel("Whisper 模型"), 5, 0)
        layout.addWidget(self.whisper_model_input, 5, 1)
        layout.addWidget(QLabel("语言模式"), 6, 0)
        layout.addWidget(self.language_mode_combo, 6, 1)
        layout.addWidget(QLabel("指定语言"), 7, 0)
        layout.addWidget(self.language_input, 7, 1)
        layout.addWidget(QLabel("时间戳粒度"), 8, 0)
        layout.addWidget(self.timestamp_combo, 8, 1)
        layout.addWidget(QLabel("任务线程数"), 9, 0)
        layout.addWidget(self.thread_spin, 9, 1)
        layout.addWidget(QLabel("术语提示"), 10, 0)
        layout.addWidget(self.context_bias_input, 10, 1)
        layout.addWidget(self.diarize_checkbox, 11, 0, 1, 2)
        return group

    def _build_translation_group(self) -> QGroupBox:
        group = QGroupBox("翻译设置")
        layout = QGridLayout(group)

        self.translation_mode_combo = QComboBox()
        self.translation_mode_combo.addItem("不翻译", "none")
        self.translation_mode_combo.addItem("Mistral API 翻译", "mistral")
        self.translation_mode_combo.addItem("OpenAI 兼容 API 翻译", "openai")
        self.translation_mode_combo.currentIndexChanged.connect(self.on_translation_mode_changed)

        self.translation_target_input = QLineEdit("zh")
        self.translation_model_input = QLineEdit("mistral-small-latest")
        self.translation_bilingual_checkbox = QCheckBox("SRT 输出双语（原文 + 译文）")
        self.translation_bilingual_checkbox.setChecked(True)
        self.translation_keep_original_checkbox = QCheckBox("翻译后额外输出原文字幕（xxx.orig.srt）")
        self.allow_subtitle_import_checkbox = QCheckBox("允许导入字幕文件并翻译")
        self.allow_subtitle_import_checkbox.setChecked(True)
        self.subtitle_translation_thread_spin = QSpinBox()
        self.subtitle_translation_thread_spin.setRange(1, 16)
        self.subtitle_translation_thread_spin.setValue(3)
        self.translation_openai_base_input = QLineEdit("https://api.openai.com/v1")
        self.translation_openai_key_input = QLineEdit(os.environ.get("OPENAI_API_KEY", ""))
        self.translation_openai_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.show_translation_openai_key_checkbox = QCheckBox("显示")
        self.show_translation_openai_key_checkbox.toggled.connect(
            lambda checked: self.translation_openai_key_input.setEchoMode(
                QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password
            )
        )
        openai_key_row = QWidget()
        openai_key_layout = QHBoxLayout(openai_key_row)
        openai_key_layout.setContentsMargins(0, 0, 0, 0)
        openai_key_layout.addWidget(self.translation_openai_key_input)
        openai_key_layout.addWidget(self.show_translation_openai_key_checkbox)

        layout.addWidget(QLabel("翻译模式"), 0, 0)
        layout.addWidget(self.translation_mode_combo, 0, 1)
        layout.addWidget(QLabel("目标语言"), 1, 0)
        layout.addWidget(self.translation_target_input, 1, 1)
        layout.addWidget(QLabel("翻译模型"), 2, 0)
        layout.addWidget(self.translation_model_input, 2, 1)
        layout.addWidget(self.translation_bilingual_checkbox, 3, 0, 1, 2)
        layout.addWidget(self.translation_keep_original_checkbox, 4, 0, 1, 2)
        layout.addWidget(self.allow_subtitle_import_checkbox, 5, 0, 1, 2)
        layout.addWidget(QLabel("字幕翻译线程数"), 6, 0)
        layout.addWidget(self.subtitle_translation_thread_spin, 6, 1)
        layout.addWidget(QLabel("OpenAI 兼容 Base URL"), 7, 0)
        layout.addWidget(self.translation_openai_base_input, 7, 1)
        layout.addWidget(QLabel("OpenAI 兼容 API Key"), 8, 0)
        layout.addWidget(openai_key_row, 8, 1)
        return group

    def _build_output_group(self) -> QGroupBox:
        group = QGroupBox("输出设置")
        layout = QGridLayout(group)

        self.output_mode_combo = QComboBox()
        self.output_mode_combo.addItem("输出到原文件目录", "source")
        self.output_mode_combo.addItem("输出到指定目录", "custom")
        self.output_mode_combo.currentIndexChanged.connect(self.on_output_mode_changed)
        self.output_dir_input = QLineEdit(str(Path.cwd() / "subtitles"))
        self.output_btn = QPushButton("浏览")
        self.output_btn.clicked.connect(self.on_choose_output_dir)
        output_row = QWidget()
        output_layout = QHBoxLayout(output_row)
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.addWidget(self.output_dir_input)
        output_layout.addWidget(self.output_btn)

        self.save_srt_checkbox = QCheckBox("保存 .srt")
        self.save_srt_checkbox.setChecked(True)
        self.save_lrc_checkbox = QCheckBox("纯音频保存 .lrc")
        self.save_lrc_checkbox.setChecked(True)
        self.save_txt_checkbox = QCheckBox("保存 .txt")
        self.save_txt_checkbox.setChecked(True)
        self.save_json_checkbox = QCheckBox("保存 .json")
        format_row = QWidget()
        format_layout = QHBoxLayout(format_row)
        format_layout.setContentsMargins(0, 0, 0, 0)
        format_layout.addWidget(self.save_srt_checkbox)
        format_layout.addWidget(self.save_lrc_checkbox)
        format_layout.addWidget(self.save_txt_checkbox)
        format_layout.addWidget(self.save_json_checkbox)
        format_layout.addStretch(1)

        layout.addWidget(QLabel("输出目录模式"), 0, 0)
        layout.addWidget(self.output_mode_combo, 0, 1)
        layout.addWidget(QLabel("指定输出目录"), 1, 0)
        layout.addWidget(output_row, 1, 1)
        layout.addWidget(format_row, 2, 0, 1, 2)
        return group

    def _build_preprocess_group(self) -> QGroupBox:
        group = QGroupBox("预处理")
        layout = QGridLayout(group)

        self.enable_vad_checkbox = QCheckBox("启用 Silero VAD 预切分")
        self.enable_vad_checkbox.setChecked(False)
        self.enable_vad_checkbox.toggled.connect(self.on_vad_enabled_changed)

        self.ffmpeg_path_input = QLineEdit()
        self.ffmpeg_path_input.setPlaceholderText("留空则自动检测内置 ffmpeg / FFMPEG_BINARY / PATH")
        self.ffmpeg_browse_btn = QPushButton("浏览")
        self.ffmpeg_browse_btn.clicked.connect(self.on_choose_ffmpeg)
        self.ffmpeg_auto_btn = QPushButton("自动检测")
        self.ffmpeg_auto_btn.clicked.connect(self.on_auto_detect_ffmpeg)
        ffmpeg_row = QWidget()
        ffmpeg_layout = QHBoxLayout(ffmpeg_row)
        ffmpeg_layout.setContentsMargins(0, 0, 0, 0)
        ffmpeg_layout.addWidget(self.ffmpeg_path_input)
        ffmpeg_layout.addWidget(self.ffmpeg_browse_btn)
        ffmpeg_layout.addWidget(self.ffmpeg_auto_btn)

        self.vad_min_speech_spin = QSpinBox()
        self.vad_min_speech_spin.setRange(1, 60_000)
        self.vad_min_speech_spin.setSingleStep(50)
        self.vad_min_speech_spin.setSuffix(" ms")
        self.vad_min_speech_spin.setValue(DEFAULT_VAD_MIN_SPEECH_MS)

        self.vad_min_silence_spin = QSpinBox()
        self.vad_min_silence_spin.setRange(1, 60_000)
        self.vad_min_silence_spin.setSingleStep(50)
        self.vad_min_silence_spin.setSuffix(" ms")
        self.vad_min_silence_spin.setValue(DEFAULT_VAD_MIN_SILENCE_MS)

        self.vad_speech_pad_spin = QSpinBox()
        self.vad_speech_pad_spin.setRange(0, 60_000)
        self.vad_speech_pad_spin.setSingleStep(50)
        self.vad_speech_pad_spin.setSuffix(" ms")
        self.vad_speech_pad_spin.setValue(DEFAULT_VAD_SPEECH_PAD_MS)

        self.vad_max_segment_spin = QSpinBox()
        self.vad_max_segment_spin.setRange(1, 24 * 3600)
        self.vad_max_segment_spin.setSingleStep(30)
        self.vad_max_segment_spin.setSuffix(" s")
        self.vad_max_segment_spin.setValue(DEFAULT_VAD_MAX_SEGMENT_SECONDS)

        self.vad_threshold_spin = QDoubleSpinBox()
        self.vad_threshold_spin.setRange(0.0, 1.0)
        self.vad_threshold_spin.setDecimals(2)
        self.vad_threshold_spin.setSingleStep(0.05)
        self.vad_threshold_spin.setValue(DEFAULT_VAD_THRESHOLD)

        self.vad_controls = [
            self.vad_min_speech_spin,
            self.vad_min_silence_spin,
            self.vad_speech_pad_spin,
            self.vad_max_segment_spin,
            self.vad_threshold_spin,
        ]

        self.ffmpeg_hint_label = QLabel("视频任务与 VAD 预切分需要 ffmpeg；可手动指定，也可留空自动检测。")
        self.ffmpeg_hint_label.setWordWrap(True)

        layout.addWidget(QLabel("ffmpeg"), 0, 0)
        layout.addWidget(ffmpeg_row, 0, 1)
        layout.addWidget(self.ffmpeg_hint_label, 1, 0, 1, 2)
        layout.addWidget(self.enable_vad_checkbox, 2, 0, 1, 2)
        layout.addWidget(QLabel("最短语音"), 3, 0)
        layout.addWidget(self.vad_min_speech_spin, 3, 1)
        layout.addWidget(QLabel("最短静音"), 4, 0)
        layout.addWidget(self.vad_min_silence_spin, 4, 1)
        layout.addWidget(QLabel("语音补边"), 5, 0)
        layout.addWidget(self.vad_speech_pad_spin, 5, 1)
        layout.addWidget(QLabel("单段最长时长"), 6, 0)
        layout.addWidget(self.vad_max_segment_spin, 6, 1)
        layout.addWidget(QLabel("检测阈值"), 7, 0)
        layout.addWidget(self.vad_threshold_spin, 7, 1)
        return group

    def apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow { background: #eef3f8; }
            QGroupBox {
                border: 1px solid #b8c7d9;
                border-radius: 10px;
                margin-top: 10px;
                font-weight: 600;
                background: #f9fbfd;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 6px;
                color: #1e2d3d;
            }
            #dropFrame {
                border: 2px dashed #4f83c2;
                border-radius: 10px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #f4f8fc, stop:1 #e1ecf8);
                min-height: 84px;
            }
            QLabel { color: #1c2b39; }
            QPushButton {
                background: #2e78c7;
                color: white;
                border: none;
                border-radius: 7px;
                padding: 7px 11px;
                font-weight: 600;
            }
            QPushButton:hover { background: #266ab1; }
            QPushButton:disabled {
                background: #9db5cc;
                color: #ebf2f9;
            }
            QLineEdit, QPlainTextEdit, QComboBox, QSpinBox {
                border: 1px solid #b4c4d4;
                border-radius: 6px;
                padding: 5px;
                background: white;
            }
            QTableWidget {
                border: 1px solid #b8c7d9;
                border-radius: 8px;
                background: white;
                alternate-background-color: #f4f7fb;
            }
            QHeaderView::section {
                background: #d9e6f3;
                padding: 6px;
                border: none;
                border-right: 1px solid #bfd0e1;
                color: #1a2a3a;
                font-weight: 600;
            }
            QProgressBar {
                border: 1px solid #9eb4c8;
                border-radius: 6px;
                text-align: center;
                background: #f4f8fc;
            }
            QProgressBar::chunk {
                background: #3f96dd;
                border-radius: 5px;
            }
            """
        )
        self.setFont(QFont("Segoe UI", 10))

    def log(self, message: str) -> None:
        self.log_text.appendPlainText(message)

    def load_settings_into_ui(self) -> None:
        settings = load_settings()
        self.apply_settings_to_ui(settings)
        self.log("已加载本地设置")
        self.on_language_mode_changed()
        self.on_transcription_provider_changed()
        self.on_translation_mode_changed()
        self.on_output_mode_changed()
        self.on_vad_enabled_changed()

    def apply_settings_to_ui(self, settings: AppSettings) -> None:
        self.transcription_provider_combo.setCurrentIndex(
            1 if settings.transcription.provider == "whisper_openai_compatible" else 0
        )
        self.mistral_api_key_input.setText(settings.transcription.mistral.api_key)
        self.mistral_model_combo.setCurrentText(settings.transcription.mistral.model)
        self.whisper_base_url_input.setText(settings.transcription.whisper.base_url)
        self.whisper_api_key_input.setText(settings.transcription.whisper.api_key)
        self.whisper_model_input.setText(settings.transcription.whisper.model)
        self.language_mode_combo.setCurrentIndex(1 if settings.transcription.language_mode == "manual" else 0)
        self.language_input.setText(settings.transcription.language)
        self.timestamp_combo.setCurrentText(settings.transcription.timestamp_granularity)
        self.diarize_checkbox.setChecked(settings.transcription.diarize)
        self.thread_spin.setValue(settings.transcription.thread_count)
        self.context_bias_input.setPlainText(settings.transcription.context_bias)

        self.translation_mode_combo.setCurrentIndex({"none": 0, "mistral": 1, "openai": 2}.get(settings.translation.mode, 0))
        self.translation_target_input.setText(settings.translation.target_language)
        self.translation_model_input.setText(settings.translation.model)
        self.translation_bilingual_checkbox.setChecked(settings.translation.bilingual_srt)
        self.translation_keep_original_checkbox.setChecked(settings.translation.keep_original_srt)
        self.allow_subtitle_import_checkbox.setChecked(settings.translation.allow_subtitle_import)
        self.subtitle_translation_thread_spin.setValue(settings.translation.subtitle_translation_thread_count)
        self.translation_openai_base_input.setText(settings.translation.openai_base_url)
        self.translation_openai_key_input.setText(settings.translation.openai_api_key)

        self.output_mode_combo.setCurrentIndex(1 if settings.output.mode == "custom" else 0)
        self.output_dir_input.setText(str(settings.output.output_dir))
        self.save_srt_checkbox.setChecked(settings.output.save_srt)
        self.save_lrc_checkbox.setChecked(settings.output.save_lrc)
        self.save_txt_checkbox.setChecked(settings.output.save_txt)
        self.save_json_checkbox.setChecked(settings.output.save_json)
        self.ffmpeg_path_input.setText(settings.output.ffmpeg_path)
        self.enable_vad_checkbox.setChecked(settings.vad.enabled)
        self.vad_min_speech_spin.setValue(settings.vad.min_speech_ms)
        self.vad_min_silence_spin.setValue(settings.vad.min_silence_ms)
        self.vad_speech_pad_spin.setValue(settings.vad.speech_pad_ms)
        self.vad_max_segment_spin.setValue(settings.vad.max_segment_seconds)
        self.vad_threshold_spin.setValue(settings.vad.threshold)

    def collect_settings_from_ui(self) -> AppSettings:
        settings = AppSettings()
        settings.transcription.provider = self.transcription_provider_combo.currentData()
        settings.transcription.mistral.api_key = self.mistral_api_key_input.text().strip()
        settings.transcription.mistral.model = self.mistral_model_combo.currentText().strip() or "voxtral-mini-latest"
        settings.transcription.whisper.base_url = self.whisper_base_url_input.text().strip() or "https://api.openai.com/v1"
        settings.transcription.whisper.api_key = self.whisper_api_key_input.text().strip()
        settings.transcription.whisper.model = self.whisper_model_input.text().strip() or "whisper-1"
        settings.transcription.language_mode = "manual" if self.language_mode_combo.currentIndex() == 1 else "auto"
        settings.transcription.language = normalize_language_code(self.language_input.text().strip())
        settings.transcription.timestamp_granularity = self.timestamp_combo.currentText().strip() or "none"
        settings.transcription.diarize = self.diarize_checkbox.isChecked()
        settings.transcription.thread_count = self.thread_spin.value()
        settings.transcription.context_bias = parse_context_bias(self.context_bias_input.toPlainText())

        settings.translation.mode = self.translation_mode_combo.currentData()
        settings.translation.target_language = normalize_language_code(self.translation_target_input.text().strip())
        settings.translation.model = self.translation_model_input.text().strip()
        settings.translation.bilingual_srt = self.translation_bilingual_checkbox.isChecked()
        settings.translation.keep_original_srt = self.translation_keep_original_checkbox.isChecked()
        settings.translation.allow_subtitle_import = self.allow_subtitle_import_checkbox.isChecked()
        settings.translation.subtitle_translation_thread_count = self.subtitle_translation_thread_spin.value()
        settings.translation.openai_base_url = self.translation_openai_base_input.text().strip() or "https://api.openai.com/v1"
        settings.translation.openai_api_key = self.translation_openai_key_input.text().strip()

        settings.output.mode = self.output_mode_combo.currentData()
        settings.output.output_dir = Path(self.output_dir_input.text().strip() or str(Path.cwd() / "subtitles"))
        settings.output.save_srt = self.save_srt_checkbox.isChecked()
        settings.output.save_lrc = self.save_lrc_checkbox.isChecked()
        settings.output.save_txt = self.save_txt_checkbox.isChecked()
        settings.output.save_json = self.save_json_checkbox.isChecked()
        settings.output.ffmpeg_path = resolve_ffmpeg_path(self.ffmpeg_path_input.text())

        settings.vad.enabled = self.enable_vad_checkbox.isChecked()
        settings.vad.min_speech_ms = self.vad_min_speech_spin.value()
        settings.vad.min_silence_ms = self.vad_min_silence_spin.value()
        settings.vad.speech_pad_ms = self.vad_speech_pad_spin.value()
        settings.vad.max_segment_seconds = self.vad_max_segment_spin.value()
        settings.vad.threshold = self.vad_threshold_spin.value()
        return settings

    def on_save_settings(self) -> None:
        try:
            settings = self.collect_settings_from_ui()
            path = save_settings(settings)
            self.log(f"设置已保存到：{path}")
            QMessageBox.information(self, "保存成功", f"设置已保存到：\n{path}")
        except Exception as exc:
            QMessageBox.warning(self, "保存失败", f"无法保存设置：{exc}")

    def on_transcription_provider_changed(self) -> None:
        provider = self.transcription_provider_combo.currentData()
        use_mistral = provider == "mistral"
        use_whisper = provider == "whisper_openai_compatible"
        self.mistral_api_key_input.setEnabled(use_mistral or self.translation_mode_combo.currentData() == "mistral")
        self.show_mistral_key_checkbox.setEnabled(use_mistral or self.translation_mode_combo.currentData() == "mistral")
        self.mistral_model_combo.setEnabled(use_mistral)
        self.whisper_base_url_input.setEnabled(use_whisper)
        self.whisper_api_key_input.setEnabled(use_whisper)
        self.show_whisper_key_checkbox.setEnabled(use_whisper)
        self.whisper_model_input.setEnabled(use_whisper)
        self.diarize_checkbox.setEnabled(use_mistral)
        if not use_mistral:
            self.diarize_checkbox.setChecked(False)

    def on_translation_mode_changed(self) -> None:
        mode = self.translation_mode_combo.currentData()
        enable_translation = mode != "none"
        use_openai = mode == "openai"
        current_model = self.translation_model_input.text().strip()

        self.translation_target_input.setEnabled(enable_translation)
        self.translation_model_input.setEnabled(enable_translation)
        self.translation_bilingual_checkbox.setEnabled(enable_translation)
        self.translation_keep_original_checkbox.setEnabled(enable_translation)
        self.allow_subtitle_import_checkbox.setEnabled(enable_translation)
        self.subtitle_translation_thread_spin.setEnabled(enable_translation)
        self.translation_openai_base_input.setEnabled(use_openai)
        self.translation_openai_key_input.setEnabled(use_openai)
        self.show_translation_openai_key_checkbox.setEnabled(use_openai)

        if mode == "mistral" and current_model in {"", "gpt-4o-mini"}:
            self.translation_model_input.setText("mistral-small-latest")
        if mode == "openai" and current_model in {"", "mistral-small-latest"}:
            self.translation_model_input.setText("gpt-4o-mini")
        self.on_transcription_provider_changed()

    def on_language_mode_changed(self) -> None:
        manual = self.language_mode_combo.currentIndex() == 1
        self.language_input.setEnabled(manual)
        if manual:
            self.language_input.setPlaceholderText("语言代码，例如 zh / en")
        else:
            self.language_input.setPlaceholderText("自动识别时无需填写")

    def on_output_mode_changed(self) -> None:
        custom = self.output_mode_combo.currentData() == "custom"
        self.output_dir_input.setEnabled(custom)
        self.output_btn.setEnabled(custom)

    def on_vad_enabled_changed(self) -> None:
        enabled = self.enable_vad_checkbox.isChecked()
        for widget in self.vad_controls:
            widget.setEnabled(enabled)

    def on_choose_output_dir(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "选择输出目录", self.output_dir_input.text())
        if directory:
            self.output_dir_input.setText(directory)

    def on_choose_ffmpeg(self) -> None:
        start_dir = self.ffmpeg_path_input.text().strip() or str(Path.cwd())
        filters = "可执行文件 (*.exe);;所有文件 (*)" if sys.platform.startswith("win") else "所有文件 (*)"
        file_path, _ = QFileDialog.getOpenFileName(self, "选择 ffmpeg 可执行文件", start_dir, filters)
        if file_path:
            self.ffmpeg_path_input.setText(file_path)

    def on_auto_detect_ffmpeg(self) -> None:
        ffmpeg_path = resolve_ffmpeg_path("")
        if ffmpeg_path:
            self.ffmpeg_path_input.setText(ffmpeg_path)
            self.log(f"已检测到 ffmpeg：{ffmpeg_path}")
            return
        QMessageBox.warning(self, "未找到 ffmpeg", "未检测到 ffmpeg，请手动选择可执行文件路径。")

    def on_drop_paths(self, paths: List[str]) -> None:
        self.add_paths(paths)

    def on_add_file(self) -> None:
        filters = (
            "媒体/字幕文件 (*.mp4 *.mov *.mkv *.avi *.wmv *.webm *.m4v *.flv *.ts "
            "*.mp3 *.wav *.m4a *.aac *.flac *.ogg *.opus *.wma *.srt *.vtt *.txt)"
        )
        files, _ = QFileDialog.getOpenFileNames(self, "选择媒体或字幕文件", str(Path.cwd()), filters)
        if files:
            self.add_paths(files)

    def on_add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "选择文件夹", str(Path.cwd()))
        if folder:
            self.add_paths([folder])

    def normalize_path_key(self, path: Path) -> str:
        try:
            resolved = path.resolve()
        except Exception:
            resolved = path.absolute()
        return str(resolved).casefold()

    def add_paths(self, raw_paths: List[str]) -> None:
        allow_subtitle_import = self.allow_subtitle_import_checkbox.isChecked()
        discovered: List[Path] = []
        skipped_subtitle = 0
        for raw in raw_paths:
            path = Path(raw)
            if path.is_dir():
                for item in discover_supported_files(path):
                    if item.suffix.lower() in SUBTITLE_EXTENSIONS and not allow_subtitle_import:
                        skipped_subtitle += 1
                        continue
                    discovered.append(item)
            elif path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                if path.suffix.lower() in SUBTITLE_EXTENSIONS and not allow_subtitle_import:
                    skipped_subtitle += 1
                    continue
                discovered.append(path)

        if not discovered:
            if skipped_subtitle > 0 and not allow_subtitle_import:
                self.log("字幕导入开关已关闭，已忽略字幕文件")
            else:
                self.log("未找到支持的媒体或字幕文件")
            return

        added = 0
        for path in discovered:
            key = self.normalize_path_key(path)
            if key in self.path_to_task:
                continue
            task_id = new_task_id()
            row = self.task_table.rowCount()
            self.task_table.insertRow(row)
            self.task_table.setItem(row, 0, QTableWidgetItem(str(path)))
            self.task_table.setItem(row, 1, QTableWidgetItem("排队中"))
            progress_bar = QProgressBar()
            progress_bar.setRange(0, 100)
            progress_bar.setValue(0)
            self.task_table.setCellWidget(row, 2, progress_bar)
            self.task_table.setItem(row, 3, QTableWidgetItem("-"))
            self.task_table.setItem(row, 4, QTableWidgetItem("就绪"))

            self.tasks[task_id] = TaskState(task_id=task_id, source_path=path, row=row)
            self.path_to_task[key] = task_id
            added += 1

        self.log(f"已添加 {added} 个文件")
        if skipped_subtitle > 0 and not allow_subtitle_import:
            self.log(f"已忽略 {skipped_subtitle} 个字幕文件（导入开关已关闭）")
        self.update_summary_text()

    def on_remove_selected(self) -> None:
        if self.is_running:
            QMessageBox.information(self, "任务进行中", "任务运行时无法删除行")
            return
        rows = sorted({idx.row() for idx in self.task_table.selectionModel().selectedRows()}, reverse=True)
        if not rows:
            return
        remove_ids = [task_id for task_id, state in self.tasks.items() if state.row in rows]
        for row in rows:
            self.task_table.removeRow(row)
        for task_id in remove_ids:
            self.path_to_task.pop(self.normalize_path_key(self.tasks[task_id].source_path), None)
            self.tasks.pop(task_id, None)
        self.rebuild_row_mapping()
        self.log(f"已删除 {len(rows)} 行所选任务")
        self.update_summary_text()

    def on_clear_all(self) -> None:
        if self.is_running:
            QMessageBox.information(self, "任务进行中", "请先停止运行中的任务再清空")
            return
        self.task_table.setRowCount(0)
        self.tasks.clear()
        self.path_to_task.clear()
        self.active_run_ids.clear()
        self.completed_run_ids.clear()
        self.run_progress.clear()
        self.futures.clear()
        self.total_progress.setValue(0)
        self.update_summary_text()
        self.log("已清空所有任务")

    def rebuild_row_mapping(self) -> None:
        path_to_row: Dict[str, int] = {}
        for row in range(self.task_table.rowCount()):
            item = self.task_table.item(row, 0)
            if not item:
                continue
            path_to_row[self.normalize_path_key(Path(item.text()))] = row
        for task in self.tasks.values():
            key = self.normalize_path_key(task.source_path)
            if key in path_to_row:
                task.row = path_to_row[key]

    def collect_settings(self) -> AppSettings:
        settings = self.collect_settings_from_ui()
        if settings.transcription.language_mode == "manual" and not settings.transcription.language:
            raise RuntimeError("已选择指定语言，请填写有效语言代码，例如 zh / en")
        if settings.translation.mode != "none":
            if not settings.translation.target_language:
                raise RuntimeError("请填写目标语言代码，例如 zh / en / ja")
            if not settings.translation.model:
                raise RuntimeError("请填写翻译模型名称")
        if settings.translation.mode == "openai" and not settings.translation.openai_api_key:
            raise RuntimeError("OpenAI 兼容翻译模式需要填写 API Key")
        if settings.translation.mode == "mistral" and not settings.transcription.mistral.api_key:
            raise RuntimeError("Mistral 翻译模式需要填写 MISTRAL_API_KEY")
        if settings.transcription.provider == "mistral" and not settings.transcription.mistral.api_key:
            raise RuntimeError("Mistral 转写需要填写 MISTRAL_API_KEY")
        if settings.transcription.provider == "whisper_openai_compatible":
            if not settings.transcription.whisper.api_key:
                raise RuntimeError("Whisper 转写需要填写第三方/OpenAI 兼容 API Key")
            if not settings.transcription.whisper.model:
                raise RuntimeError("Whisper 转写需要填写模型名称")
        if not (
            settings.output.save_srt
            or settings.output.save_lrc
            or settings.output.save_txt
            or settings.output.save_json
        ):
            raise RuntimeError("请至少选择一种输出格式")
        if settings.output.mode == "custom":
            settings.output.output_dir.mkdir(parents=True, exist_ok=True)
        if settings.transcription.provider == "mistral" and transcription_provider.Mistral is None:
            raise RuntimeError("缺少依赖：mistralai")
        return settings

    def on_start(self) -> None:
        if self.is_running:
            return
        if not self.tasks:
            QMessageBox.information(self, "没有任务", "请先添加文件")
            return
        try:
            settings = self.collect_settings()
        except Exception as exc:
            QMessageBox.warning(self, "设置无效", str(exc))
            return

        run_ids = [task_id for task_id, task in self.tasks.items() if task.status in {"Queued", "Failed", "Cancelled"}]
        if not run_ids:
            QMessageBox.information(self, "没有可执行任务", "当前没有可运行的排队/失败/取消任务")
            return

        has_media = any(self.tasks[task_id].source_path.suffix.lower() in MEDIA_EXTENSIONS for task_id in run_ids)
        has_subtitle = any(self.tasks[task_id].source_path.suffix.lower() in SUBTITLE_EXTENSIONS for task_id in run_ids)
        has_video = any(self.tasks[task_id].source_path.suffix.lower() in VIDEO_EXTENSIONS for task_id in run_ids)
        requires_ffmpeg = has_video or settings.vad.enabled
        if requires_ffmpeg and not has_ffmpeg(settings.output.ffmpeg_path):
            QMessageBox.warning(
                self,
                "缺少 ffmpeg",
                "视频任务或 VAD 预切分需要可用的 ffmpeg，请在设置页手动选择，或留空后使用自动检测。",
            )
            return
        if has_subtitle and settings.translation.mode == "none":
            QMessageBox.warning(self, "翻译未启用", "导入字幕任务需要启用翻译模式")
            return
        if has_subtitle and not settings.translation.allow_subtitle_import:
            QMessageBox.warning(self, "字幕导入已关闭", "请在设置中开启“允许导入字幕文件并翻译”")
            return
        if settings.transcription.provider == "mistral" and settings.transcription.timestamp_granularity != "none" and settings.transcription.language_mode == "manual":
            self.log("Mistral 启用时间戳粒度后，language 参数将被忽略")

        self.executor = ThreadPoolExecutor(max_workers=settings.transcription.thread_count)
        self.cancel_event.clear()
        self.active_run_ids = set(run_ids)
        self.completed_run_ids.clear()
        self.run_progress = {task_id: 0 for task_id in run_ids}
        self.futures.clear()

        self.is_running = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.remove_btn.setEnabled(False)
        self.clear_btn.setEnabled(False)

        for task_id in run_ids:
            task = self.tasks[task_id]
            self.update_task_row(task_id, "Queued", 0, "等待执行")
            future = self.executor.submit(
                run_task_worker,
                task_id,
                task.source_path,
                settings,
                self.signals,
                self.cancel_event,
            )
            self.futures[task_id] = future

        self.log(f"已启动 {len(run_ids)} 个任务，线程数：{settings.transcription.thread_count}")
        if has_media and settings.transcription.provider == "whisper_openai_compatible":
            self.log("转写后端：Whisper(OpenAI 兼容)")
        self.update_total_progress()
        self.update_summary_text()

    def on_stop(self) -> None:
        if not self.is_running:
            return
        self.cancel_event.set()
        canceled_count = 0
        for task_id, future in self.futures.items():
            if task_id in self.completed_run_ids:
                continue
            if future.cancel():
                canceled_count += 1
                self.mark_task_done(task_id, False, "Cancelled", "启动前已取消", {})
        self.log(f"已请求停止，取消了 {canceled_count} 个排队任务")

    def on_open_output_dir(self) -> None:
        if self.output_mode_combo.currentData() == "source":
            selected_rows = self.task_table.selectionModel().selectedRows()
            if selected_rows:
                item = self.task_table.item(selected_rows[0].row(), 0)
                folder = Path(item.text()).parent if item else Path.cwd()
            elif self.tasks:
                folder = next(iter(self.tasks.values())).source_path.parent
            else:
                folder = Path.cwd()
        else:
            folder = Path(self.output_dir_input.text().strip() or str(Path.cwd() / "subtitles"))
            folder.mkdir(parents=True, exist_ok=True)

        if sys.platform.startswith("win"):
            os.startfile(str(folder))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            import subprocess

            subprocess.Popen(["open", str(folder)])
        else:
            import subprocess

            subprocess.Popen(["xdg-open", str(folder)])

    def update_task_row(self, task_id: str, status: str, progress: int, message: str, outputs: str = "-") -> None:
        task = self.tasks.get(task_id)
        if not task:
            return
        task.status = status
        task.progress = progress
        task.message = message
        row = task.row

        status_item = self.task_table.item(row, 1)
        if status_item:
            status_item.setText(STATUS_LABELS.get(status, status))
        progress_bar = self.task_table.cellWidget(row, 2)
        if isinstance(progress_bar, QProgressBar):
            progress_bar.setValue(progress)
        output_item = self.task_table.item(row, 3)
        if output_item:
            output_item.setText(outputs)
        message_item = self.task_table.item(row, 4)
        if message_item:
            message_item.setText(message)

    def on_task_progress(self, task_id: str, status: str, progress: int, message: str) -> None:
        if task_id not in self.active_run_ids:
            return
        self.run_progress[task_id] = max(0, min(100, progress))
        self.update_task_row(task_id, status, progress, message)
        self.update_total_progress()

    def on_task_finished(self, task_id: str, success: bool, status: str, message: str, outputs: dict) -> None:
        self.mark_task_done(task_id, success, status, message, outputs)

    def mark_task_done(self, task_id: str, success: bool, status: str, message: str, outputs: dict) -> None:
        if task_id not in self.active_run_ids or task_id in self.completed_run_ids:
            return
        self.completed_run_ids.add(task_id)
        self.run_progress[task_id] = 100 if success else self.run_progress.get(task_id, 0)
        output_text = " | ".join(outputs.values()) if outputs else "-"
        display_message = message
        if status == "Failed" and len(display_message) > 180:
            display_message = display_message[:180] + "..."
        final_progress = 100 if success else max(0, self.run_progress.get(task_id, 0))
        self.update_task_row(task_id, status, final_progress, display_message, output_text)

        if success:
            self.log(f"[{task_id[:8]}] 已完成: {self.tasks[task_id].source_path.name}")
        else:
            self.log(f"[{task_id[:8]}] {STATUS_LABELS.get(status, status)}: {self.tasks[task_id].source_path.name} | {message}")
        self.update_total_progress()
        self.update_summary_text()
        if len(self.completed_run_ids) >= len(self.active_run_ids):
            self.finish_run()

    def finish_run(self) -> None:
        finished_ids = list(self.active_run_ids)
        if self.executor:
            self.executor.shutdown(wait=False, cancel_futures=False)
            self.executor = None
        self.is_running = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.remove_btn.setEnabled(True)
        self.clear_btn.setEnabled(True)

        success_count = 0
        failed_count = 0
        canceled_count = 0
        for task_id in finished_ids:
            status = self.tasks[task_id].status
            if status == "Completed":
                success_count += 1
            elif status == "Cancelled":
                canceled_count += 1
            else:
                failed_count += 1
        self.log(f"任务结束：成功={success_count}，失败={failed_count}，取消={canceled_count}")
        self.active_run_ids.clear()
        self.completed_run_ids.clear()
        self.run_progress.clear()
        self.futures.clear()
        self.update_total_progress()
        self.update_summary_text()

    def update_total_progress(self) -> None:
        if not self.active_run_ids:
            self.total_progress.setValue(0)
            return
        value = int(round(sum(self.run_progress.get(task_id, 0) for task_id in self.active_run_ids) / len(self.active_run_ids)))
        self.total_progress.setValue(max(0, min(100, value)))

    def update_summary_text(self) -> None:
        total = len(self.tasks)
        if total == 0:
            self.summary_label.setText("暂无任务")
            return
        if not self.active_run_ids:
            queued = sum(1 for task in self.tasks.values() if task.status == "Queued")
            done = sum(1 for task in self.tasks.values() if task.status == "Completed")
            failed = sum(1 for task in self.tasks.values() if task.status == "Failed")
            canceled = sum(1 for task in self.tasks.values() if task.status == "Cancelled")
            self.summary_label.setText(f"总数={total} | 排队={queued} | 完成={done} | 失败={failed} | 取消={canceled}")
            return
        running = len(self.active_run_ids) - len(self.completed_run_ids)
        self.summary_label.setText(f"当前批次：已完成 {len(self.completed_run_ids)}/{len(self.active_run_ids)} | 运行中={running}")
