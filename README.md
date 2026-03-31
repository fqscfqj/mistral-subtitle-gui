# AI-Subtitle-Studio

基于 `PySide6` 的桌面字幕工具，面向本地批量转写、字幕翻译和 Windows 打包分发。

## 核心能力

- 拖拽导入文件或文件夹，支持批量任务
- 可递归扫描目录中的媒体文件
- 支持直接导入并翻译字幕文件：`.srt`、`.vtt`、`.txt`
- 转写后端支持：
  - `Mistral`
  - `Whisper (OpenAI 兼容接口)`，可对接第三方服务
- 可选本地 `Silero VAD` 预切分，减少长音频请求压力
- 可配置语言模式、时间戳粒度、上下文提示词、线程数
- 翻译模式支持：
  - 不翻译
  - `Mistral API`
  - `OpenAI 兼容 Chat API`
- 输出格式支持：
  - `.srt`
  - `.lrc`
  - `.txt`
  - `.json`
- 支持双语 SRT、保留原文 SRT、自定义输出目录、批量停止/取消

## 项目结构

```text
main.py                    桌面应用入口
subtitle_studio/           核心业务与界面代码
scripts/build_exe.py       Windows 打包脚本
tests/                     自动化测试
assets/                    图标与本地模型资源
```

## 环境要求

- Python `3.10+`
- Windows（用于构建 EXE）
- 打包版会自动附带 `ffmpeg`；源码运行如果要处理视频或启用 VAD，仍可通过 `FFMPEG_BINARY` 或系统 `PATH` 提供 `ffmpeg`
- 至少一种可用的 API 凭证：
  - `MISTRAL_API_KEY`
  - `OPENAI_API_KEY`

## 安装

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 启动

```powershell
python main.py
```

应用会在当前工作目录生成设置文件 `.ai_subtitle_studio_settings.json`，用于保存界面配置和默认参数。

## 转写后端说明

### Mistral

- 使用 `mistralai` SDK
- 支持 `diarize`
- 启用时间戳粒度时，Mistral 侧可能忽略显式语言设置

### Whisper（OpenAI 兼容）

- 使用兼容 `audio/transcriptions` 的接口
- 需要配置 `base_url`、`api_key`、`model`
- `context_bias` 会映射为请求里的 `prompt`
- 如果服务端不支持 `timestamp_granularities`，程序会自动降级重试
- `diarize` 不适用于该后端

## Silero VAD

- 默认关闭
- 启用后先将音频标准化为 `16kHz / 单声道 / wav`
- 使用仓库内打包的本地 `ONNX` 模型，不依赖运行时下载
- 默认参数：
  - `min_speech_ms = 250`
  - `min_silence_ms = 400`
  - `speech_pad_ms = 200`
  - `max_segment_seconds = 900`

## 环境变量

| 变量名 | 用途 |
| --- | --- |
| `MISTRAL_API_KEY` | Mistral 转写或翻译认证 |
| `OPENAI_API_KEY` | Whisper/OpenAI 兼容接口认证 |
| `FFMPEG_BINARY` | 自定义 `ffmpeg` 可执行文件路径 |

## 打包 EXE

```powershell
python ./scripts/build_exe.py
```

也可以在资源管理器中直接双击 `scripts/build_exe.py`。脚本在双击启动时会在构建完成后自动打开产物位置，并保留窗口直到手动关闭。

构建脚本会：

1. 在虚拟环境中安装项目依赖与 `PyInstaller`
2. 下载并临时准备 `ffmpeg.exe` 与 `ffprobe.exe`
3. 使用 `ai_subtitle_studio.spec` 打包 `main.py`
4. 输出 `dist/AI-Subtitle-Studio.exe`

如果你是在终端中执行且不想要结束时的暂停或自动打开产物目录，可以使用：

```powershell
python ./scripts/build_exe.py --no-pause --no-open-dist
```

## 测试

```powershell
pytest
```

## 说明

- 默认输出目录为当前目录下的 `subtitles/`
- 设置页不再提供 `ffmpeg` 路径选择；打包版会自动使用附带的 `ffmpeg.exe`
- 源码运行时，如果当前目录下没有打包资源，程序仍会回退到环境变量或系统 `PATH` 中的 `ffmpeg`
- 项目当前主要面向 Windows 桌面使用场景
