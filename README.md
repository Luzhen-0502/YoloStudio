# ScreenYOLO

**屏幕实时 YOLOv8 人物检测 —— 单文件 EXE，双击即用，内置模型，无需联网**

一个用 YOLOv8 做屏幕人物检测的小工具。它不抓摄像头，而是直接抓你整个屏幕，认出画面里所有人，画绿框标出来。

---

## 快速开始

### 方式一：一键安装（推荐）

**复制下面这一整行**，粘贴到 PowerShell（Win 键 → 输入 `powershell` → 回车），再回车执行：

```powershell
$u='https://github.com/Luzhen-0502/YoloStudio/raw/main/ScreenYOLO.exe'; $d="$env:USERPROFILE\Desktop\ScreenYOLO.exe"; Invoke-WebRequest $u -OutFile $d; $w=New-Object -ComObject WScript.Shell; $s=$w.CreateShortcut("$env:USERPROFILE\Desktop\ScreenYOLO.lnk"); $s.TargetPath=$d; $s.WorkingDirectory=(Split-Path $d); $s.IconLocation="$d,0"; $s.Save(); Write-Host "装好了！桌面双击 ScreenYOLO 就能用" -ForegroundColor Green
```

它会做两件事：把 85 MB 的 exe 下载到桌面 + 在桌面建一个快捷方式。
（85 MB 视网速大概 1~5 分钟，没有进度条，别以为卡住了。）

### 方式二：手动下载

1. 点上面的 **`ScreenYOLO.exe`**（85 MB）下载
2. 双击运行
3. 按 **`q`** 或 **`ESC`** 退出

不用装 Python，不用装 CUDA，不用联网下模型——模型（`yolov8n.onnx`，12.8 MB）已经打包进 exe 里了。

> 首次启动会花几秒钟解包，之后启动会快一些。

---

## 功能

- **全屏实时抓屏**（mss），不是摄像头
- **YOLOv8n 推理**（ONNX Runtime），只检测 `person` 这一类
- 检测到人物画**绿色框**，框上标注 `person` + 置信度
- OpenCV 实时窗口显示
- 模型内嵌：首次运行时自动从 exe 里释放到 `%LOCALAPPDATA%\ScreenYOLO\`，之后直接读缓存

---

## 命令行参数

```bat
ScreenYOLO.exe                          :: 正常实时检测，按 q 退出
ScreenYOLO.exe --max-frames 60 --save-snapshot snap.png   :: 跑完 60 帧自动退出并存图
```

| 参数 | 说明 |
|---|---|
| `--max-frames N` | 最多处理 N 帧后自动退出（用于自动化测试） |
| `--save-snapshot PATH` | 保存最后一帧截图 |
| `--model PATH` | 手动指定模型文件路径 |

---

## 技术细节

| 项 | 值 |
|---|---|
| 模型 | YOLOv8n（COCO，只取 person 类） |
| 推理后端 | ONNX Runtime（不用 PyTorch，体积小一半） |
| 抓屏 | mss |
| 图像处理 | OpenCV |
| 打包 | PyInstaller 单文件（`--onefile`），内嵌模型 |
| 平台 | Windows x64 |
| exe 体积 | 约 85 MB（解包后运行时约 250 MB 临时目录） |

### 为什么用 ONNX 而不是 PyTorch

PyTorch 版打包出来有 321 MB，启动时还要解压 350 MB 的 torch DLL，冷启动很慢。
换成 ONNX Runtime 之后，exe 直接砍到 85 MB，启动也快得多。

---

## 从源码运行

```bash
pip install onnxruntime opencv-python mss numpy
python screen_yolo_detect_onnx.py
```

### 重新打包成 exe

```bash
pip install pyinstaller
pyinstaller --clean --noconfirm ScreenYOLO.spec
```

`ScreenYOLO.spec` 里已经写好了 `datas=[("yolov8n.onnx", ".")]`，模型会自动打进去。

---

## 常见问题

**Q：杀毒软件报毒？**
PyInstaller 打包的 exe 经常被误报（因为它的自解压行为），这是行业老问题了。源码就在仓库里，可以自己打包。

**Q：能检测别的东西吗？**
改 `PERSON_CLASS_ID` 或换别的 YOLO 权重即可。当前只认 person。

**Q：屏幕太大跑不动？**
可以在源码里加缩放（`cv2.resize`），或者用 `yolov8n` 换成更小的输入尺寸。

---

## 许可

仅供学习交流使用。
YOLOv8 权重来自 [Ultralytics](https://github.com/ultralytics/ultralytics)（AGPL-3.0）。
