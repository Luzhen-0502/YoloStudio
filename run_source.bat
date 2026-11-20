@echo off
chcp 936 >nul
title ScreenYOLO 源码运行
cd /d "%~dp0"

echo ============================================
echo   ScreenYOLO - 从源码运行
echo ============================================
echo.

where python >nul 2>&1
if errorlevel 1 (
  echo [错误] 找不到 python，请先装 Python 并勾选 Add to PATH
  pause
  exit /b 1
)

python -c "import onnxruntime, cv2, mss, numpy" >nul 2>&1
if errorlevel 1 (
  echo [提示] 缺依赖，正在安装...
  python -m pip install onnxruntime opencv-python mss numpy
)

echo 启动中... 按 q 或 ESC 退出
echo.
python screen_yolo_detect_onnx.py
echo.
echo 程序已退出
pause
