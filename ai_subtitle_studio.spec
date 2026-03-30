# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_submodules

mistral_hiddenimports = collect_submodules("mistralai")

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=[],
    datas=[
        ("assets/logo.ico", "assets"),
        ("assets/silero_vad.onnx", "assets"),
        ("build_assets/ffmpeg.exe", "."),
        ("build_assets/ffprobe.exe", "."),
    ],
    hiddenimports=[
        "onnxruntime",
        "onnxruntime.capi.onnxruntime_pybind11_state",
    ]
    + mistral_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="AI-Subtitle-Studio",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=["assets/logo.ico"],
)
