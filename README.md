# Subtitle Studio（桌面 GUI）

基于 PySide6 的桌面字幕工具，支持多后端音视频转写、字幕翻译、批量任务和本地桌面打包。

## 功能

- 支持拖拽文件/文件夹
- 可添加单个文件，或递归扫描整个文件夹
- 支持导入字幕文件并直接翻译（`.srt` / `.vtt` / `.txt`）
- 转写后端可切换：
  - `Mistral`
  - `Whisper(OpenAI 兼容)`，支持第三方部署服务
- 可选 `Silero VAD` 预切分
- 可配置语言模式、时间戳粒度、上下文提示
- 翻译模式：
  - 不翻译
  - `Mistral API`
  - `OpenAI 兼容 Chat API`
- 输出格式：
  - `.srt`
  - `.lrc`
  - `.txt`
  - `.json`
- 支持双语 SRT、原文 SRT 保留、输出目录切换、批量停止/取消

## 转写后端说明

### Mistral

- 使用 `mistralai` SDK
- 支持 `diarize`
- 启用时间戳粒度时，Mistral 会忽略显式 `language`

### Whisper(OpenAI 兼容)

- 使用 OpenAI-compatible `audio/transcriptions` 接口
- 需要配置：
  - `base_url`
  - `api_key`
  - `model`
- `context_bias` 会映射为 `prompt`
- 若第三方实现不支持 `timestamp_granularities`，程序会自动重试一次不带该字段的请求
- `diarize` 不适用于 Whisper

## Silero VAD

- 默认关闭
- 开启后会先将音频标准化为 `16k / 单声道 / wav`
- 使用本地 `ONNX` 模型，不依赖运行时联网下载
- 内部默认参数：
  - `min_speech_ms=250`
  - `min_silence_ms=400`
  - `speech_pad_ms=200`
  - 单切片上限 `15 分钟`

## 运行要求

- Python 3.10+
- `PATH` 中可用的 `ffmpeg`，或设置 `FFMPEG_BINARY`
- 对应后端所需的 API Key

## 安装

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 运行

```bash
python mistral_subtitle_gui.py
```

## 环境变量

- `MISTRAL_API_KEY`
- `OPENAI_API_KEY`
- `FFMPEG_BINARY`

## 打包

```powershell
./scripts/build_exe.ps1
```

PyInstaller 会从根入口脚本启动，并将 `Silero VAD` 模型一起打包。
