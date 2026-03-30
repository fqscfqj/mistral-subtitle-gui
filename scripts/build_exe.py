from __future__ import annotations

import argparse
import ctypes
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Sequence
from urllib.request import urlopen


FFMPEG_URL = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build AI-Subtitle-Studio Windows EXE.")
    parser.add_argument(
        "--no-pause",
        action="store_true",
        help="Do not wait for Enter when launched from Explorer.",
    )
    parser.add_argument(
        "--no-open-dist",
        action="store_true",
        help="Do not open Explorer with the built EXE selected after success.",
    )
    return parser.parse_args()


def is_explorer_launch() -> bool:
    if sys.platform != "win32":
        return False
    try:
        kernel32 = ctypes.windll.kernel32
        snapshot = kernel32.CreateToolhelp32Snapshot(0x00000002, 0)
        if snapshot == ctypes.c_void_p(-1).value:
            return False

        class PROCESSENTRY32W(ctypes.Structure):
            _fields_ = [
                ("dwSize", ctypes.c_uint32),
                ("cntUsage", ctypes.c_uint32),
                ("th32ProcessID", ctypes.c_uint32),
                ("th32DefaultHeapID", ctypes.c_void_p),
                ("th32ModuleID", ctypes.c_uint32),
                ("cntThreads", ctypes.c_uint32),
                ("th32ParentProcessID", ctypes.c_uint32),
                ("pcPriClassBase", ctypes.c_long),
                ("dwFlags", ctypes.c_uint32),
                ("szExeFile", ctypes.c_wchar * 260),
            ]

        entry = PROCESSENTRY32W()
        entry.dwSize = ctypes.sizeof(PROCESSENTRY32W)
        current_pid = kernel32.GetCurrentProcessId()
        parent_pid = None

        if kernel32.Process32FirstW(snapshot, ctypes.byref(entry)):
            while True:
                if entry.th32ProcessID == current_pid:
                    parent_pid = entry.th32ParentProcessID
                    break
                if not kernel32.Process32NextW(snapshot, ctypes.byref(entry)):
                    break

        if not parent_pid:
            return False

        entry = PROCESSENTRY32W()
        entry.dwSize = ctypes.sizeof(PROCESSENTRY32W)
        if kernel32.Process32FirstW(snapshot, ctypes.byref(entry)):
            while True:
                if entry.th32ProcessID == parent_pid:
                    return entry.szExeFile.lower() == "explorer.exe"
                if not kernel32.Process32NextW(snapshot, ctypes.byref(entry)):
                    break
        return False
    except Exception:
        return False
    finally:
        try:
            kernel32.CloseHandle(snapshot)
        except Exception:
            pass


def wait_for_exit_if_needed(*, no_pause: bool, explorer_launch: bool, message: str) -> None:
    if no_pause or not explorer_launch:
        return
    print()
    input(message)


def run(cmd: Sequence[str], *, cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def download_file(url: str, target: Path) -> None:
    with urlopen(url) as response, target.open("wb") as output:
        shutil.copyfileobj(response, output)


def find_binary(root: Path, name: str) -> Path:
    preferred = [path for path in root.rglob(name) if path.as_posix().endswith(f"/bin/{name}")]
    if preferred:
        return preferred[0]

    for path in root.rglob(name):
        return path
    raise FileNotFoundError(f"下载包中未找到 {name}，可能是下载源结构已变化：{FFMPEG_URL}")


def main() -> int:
    args = parse_args()
    explorer_launch = is_explorer_launch()

    repo_root = Path(__file__).resolve().parent.parent
    python_exe = repo_root / ".venv" / "Scripts" / "python.exe"
    spec_path = repo_root / "ai_subtitle_studio.spec"
    requirements_path = repo_root / "requirements.txt"
    dist_exe_path = repo_root / "dist" / "AI-Subtitle-Studio.exe"

    temp_root = repo_root / "build_tmp"
    zip_path = temp_root / "ffmpeg.zip"
    extract_dir = temp_root / "ffmpeg_extract"
    build_assets_dir = repo_root / "build_assets"
    bundled_ffmpeg_path = build_assets_dir / "ffmpeg.exe"
    bundled_ffprobe_path = build_assets_dir / "ffprobe.exe"

    if not python_exe.exists():
        raise FileNotFoundError(f"未找到虚拟环境 Python：{python_exe}")
    if not spec_path.exists():
        raise FileNotFoundError(f"未找到 PyInstaller spec 文件：{spec_path}")
    if not requirements_path.exists():
        raise FileNotFoundError(f"未找到依赖文件：{requirements_path}")

    build_succeeded = False

    try:
        print("[1/4] 安装构建依赖...")
        run([str(python_exe), "-m", "pip", "install", "--upgrade", "pip"])
        run([str(python_exe), "-m", "pip", "install", "-r", str(requirements_path), "pyinstaller"])

        print("[2/4] 下载并准备 ffmpeg / ffprobe...")
        shutil.rmtree(temp_root, ignore_errors=True)
        shutil.rmtree(build_assets_dir, ignore_errors=True)
        extract_dir.mkdir(parents=True, exist_ok=True)
        build_assets_dir.mkdir(parents=True, exist_ok=True)

        download_file(FFMPEG_URL, zip_path)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(extract_dir)

        shutil.copy2(find_binary(extract_dir, "ffmpeg.exe"), bundled_ffmpeg_path)
        shutil.copy2(find_binary(extract_dir, "ffprobe.exe"), bundled_ffprobe_path)

        print("[3/4] 运行 PyInstaller 构建...")
        run([str(python_exe), "-m", "PyInstaller", "--noconfirm", "--clean", str(spec_path)], cwd=repo_root)

        if not dist_exe_path.exists():
            raise FileNotFoundError(f"构建完成但未找到目标产物：{dist_exe_path}")

        build_succeeded = True
        print(f"[4/4] Build complete: {dist_exe_path}")
        return 0
    except Exception as exc:
        print(exc, file=sys.stderr)
        return 1
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
        shutil.rmtree(build_assets_dir, ignore_errors=True)

        if build_succeeded and explorer_launch and not args.no_open_dist and sys.platform == "win32":
            subprocess.Popen(["explorer.exe", f'/select,"{dist_exe_path}"'])

        if build_succeeded:
            wait_for_exit_if_needed(
                no_pause=args.no_pause,
                explorer_launch=explorer_launch,
                message="构建完成，按 Enter 键关闭此窗口",
            )
        else:
            wait_for_exit_if_needed(
                no_pause=args.no_pause,
                explorer_launch=explorer_launch,
                message="构建失败，按 Enter 键关闭此窗口",
            )


if __name__ == "__main__":
    raise SystemExit(main())
