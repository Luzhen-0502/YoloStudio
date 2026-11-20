# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['../screen_yolo_detect_onnx.py'],
    pathex=[],
    binaries=[],
    datas=[('C:/Users/39846/yolov8n.onnx', '.')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['torch', 'torchvision', 'torchaudio', 'ultralytics', 'matplotlib', 'pandas', 'scipy', 'IPython', 'jupyter', 'notebook', 'tensorboard', 'pytest', 'PyQt5', 'PySide6'],
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
    name='ScreenYOLO',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
