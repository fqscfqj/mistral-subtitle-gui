# Mistral Subtitle Studio（桌面 GUI）

这是一个基于 PySide6 的桌面应用，使用 Mistral 音频转写 API 从视频/音频文件生成字幕。

## 功能

- 支持拖拽文件/文件夹
- 可添加单个文件，或递归扫描整个文件夹
- 支持导入字幕文件并直接翻译（`.srt` / `.vtt` / `.txt`）
- 可在设置中开启/关闭“允许导入字幕文件并翻译”
- 独立页面：`任务` 和 `设置`
- 多线程任务执行（线程数可配置）
- 字幕导入翻译支持按分块并行请求（使用“字幕翻译线程数”）
- 支持单任务状态与进度 + 全局进度
- 可配置的 Mistral 选项：
  - `model`
  - 语言模式：自动检测或手动指定语言代码
  - `timestamp_granularities`（`none`、`segment`、`word`）：选择 `segment` 或 `word` 可获得时间戳；`none` 将生成不带时间戳的转写文本。
  - `diarize`（说话人分离）：让模型尝试识别不同说话者并在输出中添加 `speaker` 标签，生成的 SRT 会在每行前用 `[Speaker X]` 前缀显示。启用后如果音频中有多个人的声音，可在字幕里区分发言者。
  - `context_bias`
- 可选字幕翻译：
  - 不翻译 / Mistral API / OpenAI 兼容 API
  - OpenAI 兼容 `base_url` + `api_key`（支持第三方服务商）
  - 目标语言代码
  - 双语 SRT 输出（原文 + 译文）
- 输出目录模式：
  - 使用源文件所在目录
  - 使用自定义目录
- 输出文件名支持语言后缀（示例：`abc.zh.srt`）
- 输出格式：
  - `.srt`
  - `.lrc`（纯音频任务可选）
  - `.txt`
  - `.json`
- 支持批量任务停止/取消（包括队列中的任务）

## 运行要求

- Python 3.10+
- `PATH` 中可用的 `ffmpeg`（处理视频文件时必需）
- Mistral API 密钥（处理音视频转录，或使用 Mistral 翻译模式时必需）

## 安装

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

## 运行

```bash
python mistral_subtitle_gui.py
```

## API 行为说明

- 根据 Mistral 文档，`language` 不能与 `timestamp_granularities` 同时使用。
- 当时间戳粒度为 `segment` 或 `word` 时，本应用会忽略 `language`。
- `context_bias` 会进行规范化处理，并最多保留 100 个唯一词条。

## 环境变量

- `MISTRAL_API_KEY`：启动时默认加载的 API 密钥
- `OPENAI_API_KEY`：用于 OpenAI 兼容翻译的可选默认密钥（字幕导入翻译可仅使用该项）
- `FFMPEG_BINARY`：可选，`ffmpeg` 可执行文件的绝对路径

## 输出

输出文件命名规则为：源文件名（不含扩展名）+ 语言代码后缀：

- `input_video.zh.srt`
- `input_audio.zh.lrc`（纯音频任务且启用 `.lrc` 输出时）
- `input_video.zh.txt`
- `input_video.zh.json`

启用翻译后：

- 原文转写：`input_video.en.srt`（示例）
- 转录后翻译字幕会基于转写文件追加目标语言后缀：`input_video.en.srt` -> `input_video.en.zh.srt`
- 导入已有字幕翻译时，会额外生成目标语言文件（例如：`input_video.srt` -> `input_video.zh.srt`）；若文件名已带目标语言后缀，则会使用 `.translated` 避免覆盖（例如：`input_video.zh.srt` -> `input_video.zh.translated.srt`）
