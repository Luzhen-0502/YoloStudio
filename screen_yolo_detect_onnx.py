# -*- coding: utf-8 -*-
"""屏幕实时 YOLOv8 人物检测（ONNX Runtime 轻量版，计算机视觉学习实验）

与 screen_yolo_detect.py 功能一致，但推理后端从 PyTorch 换成 ONNX Runtime：
  - exe 体积从 ~321MB 降到 ~1/2 以下，启动不再需要解包 350MB 的 torch DLL
  - 不依赖 torch / ultralytics，只需要 onnxruntime + opencv + numpy + mss

功能：
  1. mss 全屏实时抓屏（不是摄像头）
  2. YOLOv8n 只检测 person（COCO 类别 0）
  3. 检测到人物画绿色框，框上标注 "person 置信度"
  4. 可视化窗口实时显示，按 q（或 ESC）退出
  5. 模型缺失时自动联网下载 yolov8n.onnx（本地有就直接用）

运行：
  python screen_yolo_detect_onnx.py
  python screen_yolo_detect_onnx.py --max-frames 60 --save-snapshot snap.png   # 自动化验证
"""

import argparse
import os
import shutil
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

PERSON_CLASS_ID = 0
MODEL_FILENAME = "yolov8n.onnx"
GREEN = (0, 255, 0)
ESC_KEY = 27
# 退出键：q / Q / ESC。实测不同平台/版本的 cv2.waitKey 返回的键值不一样：
# 有的返回 ASCII（q=113），有的返回 Windows 虚拟键码（Q=81），这里都收。
QUIT_KEYS = {ord("q"), ord("Q"), ESC_KEY}

MODEL_URLS = [
    f"https://github.com/ultralytics/assets/releases/download/v8.3.0/{MODEL_FILENAME}",
    f"https://ghfast.top/https://github.com/ultralytics/assets/releases/download/v8.3.0/{MODEL_FILENAME}",
    f"https://gh-proxy.com/https://github.com/ultralytics/assets/releases/download/v8.3.0/{MODEL_FILENAME}",
]


def log(msg):
    print(msg, flush=True)


def is_quit_key(key):
    """兼容 ASCII(113) 与虚拟键码(81) 两种键值。"""
    return key != 255 and key in QUIT_KEYS


def bundle_dir():
    """打包成 exe 后，PyInstaller 会把内嵌资源解压到 sys._MEIPASS 指向的临时目录。"""
    base = getattr(sys, "_MEIPASS", None)
    return Path(base) if base else None


def app_data_dir():
    """可写缓存目录：打包后模型从这里加载，避免临时目录只读/被清理的问题。"""
    if os.name == "nt":
        root = Path(os.environ.get("LOCALAPPDATA") or Path.home())
    else:
        root = Path.home() / ".cache"
    d = root / "ScreenYOLO"
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    return d


def ensure_model_from_bundle():
    """exe 内嵌了模型时，把它缓存到本地可写目录并返回路径（只复制一次）。"""
    bd = bundle_dir()
    if bd is None:
        return None
    bundled = bd / MODEL_FILENAME
    if not bundled.is_file() or bundled.stat().st_size < 1_000_000:
        return None
    cache = app_data_dir()
    if cache is None:
        return bundled
    target = cache / MODEL_FILENAME
    try:
        if not (target.is_file() and target.stat().st_size > 1_000_000):
            shutil.copyfile(bundled, target)
            log(f"[模型] 已从程序内嵌资源释放到 {target}")
        return target
    except OSError:
        return bundled


def find_local_model(path_arg=None):
    """在常见位置查找本地 onnx 权重，找到返回 Path，否则 None。"""
    candidates = []
    if path_arg:
        candidates.append(Path(path_arg).expanduser())
    candidates.append(Path.cwd() / MODEL_FILENAME)
    try:
        candidates.append(Path(__file__).resolve().parent / MODEL_FILENAME)
    except NameError:
        pass
    candidates.append(Path.home() / MODEL_FILENAME)
    bd = bundle_dir()
    if bd is not None:
        candidates.append(bd / MODEL_FILENAME)

    for c in candidates:
        try:
            if c.is_file() and c.stat().st_size > 1_000_000:
                return c
        except OSError:
            continue
    return None


def download_model(dest, timeout=20):
    """自动联网下载权重；逐个尝试镜像并设超时，成功返回路径，全部失败返回 None。"""
    dest = Path(dest)
    tmp = dest.with_name(dest.name + ".part")
    for url in MODEL_URLS:
        try:
            log(f"[下载] 尝试 {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as fh:
                shutil.copyfileobj(resp, fh)
            size = tmp.stat().st_size
            if size > 1_000_000:
                tmp.replace(dest)
                log(f"[下载] 成功 {dest} ({size / 1e6:.1f} MB)")
                return dest
            log(f"[下载] 文件异常偏小({size} 字节)，换下一个源")
            tmp.unlink(missing_ok=True)
        except Exception as exc:
            log(f"[下载] 失败 {type(exc).__name__}: {exc}")
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
    return None


def resolve_model(path_arg=None, download_timeout=20):
    """返回模型路径字符串。优先级：exe 内嵌 > 本地已有 > 自动下载 > 返回文件名交给 ORT 报错。"""
    if getattr(sys, "frozen", False) and not path_arg:
        bundled = ensure_model_from_bundle()
        if bundled is not None:
            log(f"[模型] 使用程序内嵌权重: {bundled}")
            return str(bundled)

    local = find_local_model(path_arg)
    if local is not None:
        log(f"[模型] 使用本地权重: {local}")
        return str(local)

    dest = Path.cwd() / MODEL_FILENAME
    log(f"[模型] 本地未找到 {MODEL_FILENAME}，开始自动联网下载 ...")
    got = download_model(dest, timeout=download_timeout)
    if got is not None:
        return str(got)
    log("[模型] 所有下载源均不可用，尝试直接用默认文件名交由 onnxruntime 处理 ...")
    return MODEL_FILENAME


def window_state(title):
    """返回 (可见, 置顶) 状态，用于日志取证。"""
    if os.name != "nt":
        return None
    try:
        import ctypes

        user32 = ctypes.windll.user32
        hwnd = user32.FindWindowW(None, title)
        if not hwnd:
            return None
        return bool(user32.IsWindowVisible(hwnd)), bool(user32.GetWindowLongW(hwnd, -20) & 0x8)
    except Exception:
        return None


def make_topmost(title):
    """把指定标题的窗口设为置顶（Windows）。失败不影响主流程。

    坑：SetWindowPos 的第二个参数是 HWND（64 位指针），不能直接传 Python 的 -1，
    否则会被当成 32 位整数，变成无效句柄 0xFFFFFFFF，函数静默返回 0（表现为“置顶一直失败”）。
    必须用 wintypes.HWND(-1) 生成真正的指针值。
    """
    if os.name != "nt":
        return False
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        user32.FindWindowW.restype = wintypes.HWND
        user32.FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
        user32.SetWindowPos.restype = wintypes.BOOL
        user32.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int,
                                        ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                        ctypes.c_uint]
        hwnd = user32.FindWindowW(None, title)
        if not hwnd:
            return False
        HWND_TOPMOST = wintypes.HWND(-1)
        SWP_NOMOVE, SWP_NOSIZE = 0x0002, 0x0001
        return bool(user32.SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0,
                                        SWP_NOMOVE | SWP_NOSIZE))
    except Exception as exc:
        log(f"[警告] 置顶失败: {exc}")
        return False


def imwrite_unicode(path, image):
    """cv2.imwrite 不支持中文路径，用 imencode + 二进制写盘绕过。"""
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        raise IOError(f"图片编码失败: {path}")
    Path(path).write_bytes(buf.tobytes())


# ---------------------------------------------------------------- 推理部分

class Detector:
    """YOLOv8 ONNX 推理封装：letterbox 预处理 + 只取 person 类 + 自写 NMS。"""

    def __init__(self, model_path, imgsz=640, conf=0.35, iou=0.45):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.log_severity_level = 3          # 静音 ORT 常规日志
        opts.intra_op_num_threads = 0        # 让 ORT 自己决定线程数
        self.sess = ort.InferenceSession(
            model_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.sess.get_inputs()[0].name
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        log(f"[模型] onnxruntime {ort.__version__} 就绪，输入 {self.input_name} "
            f"{self.sess.get_inputs()[0].shape}")

    def _letterbox(self, img):
        """等比缩放 + 灰边填充到 imgsz x imgsz，返回 (blob, 缩放比, 左填充, 上填充)。"""
        h, w = img.shape[:2]
        r = min(self.imgsz / h, self.imgsz / w)
        nw, nh = int(round(w * r)), int(round(h * r))
        if (w, h) != (nw, nh):
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        dw, dh = (self.imgsz - nw) / 2, (self.imgsz - nh) / 2
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        padded = cv2.copyMakeBorder(img, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT, value=(114, 114, 114))
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        blob = rgb.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None]      # 1,3,H,W
        return np.ascontiguousarray(blob), r, left, top

    @staticmethod
    def _nms(boxes, scores, iou_thres):
        """标准 NMS，输入 xyxy 框与分数，返回保留索引。"""
        if len(boxes) == 0:
            return []
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(int(i))
            if order.size == 1:
                break
            rest = order[1:]
            xx1 = np.maximum(x1[i], x1[rest])
            yy1 = np.maximum(y1[i], y1[rest])
            xx2 = np.minimum(x2[i], x2[rest])
            yy2 = np.minimum(y2[i], y2[rest])
            inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
            iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
            order = rest[iou <= iou_thres]
        return keep

    def detect(self, frame):
        """返回 [(x1,y1,x2,y2,score), ...]，坐标是原图坐标系，只含 person。"""
        blob, r, pad_left, pad_top = self._letterbox(frame)
        out = self.sess.run(None, {self.input_name: blob})[0]   # (1, 84, 8400)
        pred = out[0]                                           # (84, 8400)

        scores = pred[4 + PERSON_CLASS_ID]                      # 只要 person 这一类的分数
        mask = scores > self.conf
        if not np.any(mask):
            return []

        xywh = pred[:4, mask].T
        confs = scores[mask]
        boxes = np.empty_like(xywh)
        boxes[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
        boxes[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
        boxes[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
        boxes[:, 3] = xywh[:, 1] + xywh[:, 3] / 2

        keep = self._nms(boxes, confs, self.iou)

        h, w = frame.shape[:2]
        result = []
        for i in keep:
            x1, y1, x2, y2 = boxes[i]
            # 去 letterbox 填充和缩放，还原到原图坐标
            x1 = (x1 - pad_left) / r
            x2 = (x2 - pad_left) / r
            y1 = (y1 - pad_top) / r
            y2 = (y2 - pad_top) / r
            x1 = int(max(0, min(w - 1, x1)))
            y1 = int(max(0, min(h - 1, y1)))
            x2 = int(max(0, min(w - 1, x2)))
            y2 = int(max(0, min(h - 1, y2)))
            if x2 > x1 and y2 > y1:
                result.append((x1, y1, x2, y2, float(confs[i])))
        return result


# ---------------------------------------------------------------- 绘制部分

def draw(frame, boxes):
    for (x1, y1, x2, y2, score) in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, 2)
        label = f"person {score:.2f}"
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        top = max(0, y1 - th - base - 4)
        cv2.rectangle(frame, (x1, top), (x1 + tw + 8, y1), GREEN, -1)
        cv2.putText(frame, label, (x1 + 4, y1 - base - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    return frame


def hud(frame, fps, n_person):
    text = f"FPS {fps:5.1f} | person {n_person} | q = quit"
    cv2.rectangle(frame, (0, 0), (430, 30), (0, 0, 0), -1)
    cv2.putText(frame, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    return frame


# ---------------------------------------------------------------- 主流程

def main():
    parser = argparse.ArgumentParser(description="屏幕实时 YOLOv8 人物检测（ONNX 轻量版）")
    parser.add_argument("--model", default=None, help="onnx 模型路径；默认自动查找/下载")
    parser.add_argument("--conf", type=float, default=0.35, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU 阈值")
    parser.add_argument("--imgsz", type=int, default=640, help="推理输入尺寸")
    parser.add_argument("--display-width", type=int, default=1100, help="显示窗口最大宽度")
    parser.add_argument("--max-frames", type=int, default=0, help=">0 时跑完指定帧数自动退出")
    parser.add_argument("--save-snapshot", default=None, help="把检测结果最好的一帧保存到该路径")
    parser.add_argument("--monitor", type=int, default=1, help="mss 显示器编号，1 = 主屏")
    parser.add_argument("--window", default="Screen YOLOv8 - Person Detection", help="窗口标题")
    parser.add_argument("--no-window", action="store_true", help="不弹窗（纯批处理验证）")
    parser.add_argument("--download-timeout", type=float, default=20, help="单次模型下载的超时秒数")
    parser.add_argument("--topmost", dest="topmost", action="store_true", default=True,
                        help="可视化窗口置顶显示（默认开，双击运行时不会被其他窗口盖住）")
    parser.add_argument("--no-topmost", dest="topmost", action="store_false", help="取消置顶")
    parser.add_argument("--log-keys", action="store_true", help="打印窗口收到的每个按键")
    args = parser.parse_args()

    import mss

    model_path = resolve_model(args.model, args.download_timeout)
    detector = Detector(model_path, imgsz=args.imgsz, conf=args.conf, iou=args.iou)

    show_window = not args.no_window
    if show_window:
        try:
            cv2.namedWindow(args.window, cv2.WINDOW_AUTOSIZE)
        except cv2.error as exc:
            log(f"[警告] 无法创建窗口（{exc}）；改用 --no-window 模式继续")
            show_window = False
    if show_window and args.topmost:
        log("[运行] 尝试将可视化窗口置顶（窗口就绪后生效）")

    frames = 0
    topmost_done = not args.topmost
    person_frames = 0
    total_persons = 0
    best = None
    best_n = -1
    frame = None
    t_start = time.time()
    t_last = t_start
    fps = 0.0

    log("[运行] 开始全屏实时人物检测，按 q 退出 ...")
    MSS = getattr(mss, "MSS", None) or mss.mss
    with MSS() as sct:
        monitor = sct.monitors[args.monitor]
        log(f"[运行] 抓屏区域: {monitor['width']}x{monitor['height']} @ ({monitor['left']},{monitor['top']})")

        while True:
            shot = sct.grab(monitor)
            frame = np.ascontiguousarray(np.asarray(shot)[:, :, :3])

            boxes = detector.detect(frame)
            draw(frame, boxes)
            n = len(boxes)

            frames += 1
            total_persons += n
            if n > 0:
                person_frames += 1
            if n > best_n:
                best_n, best = n, frame.copy()

            now = time.time()
            if now - t_last >= 0.5:
                fps = frames / max(now - t_start, 1e-6)
                t_last = now
            hud(frame, fps, n)

            if args.save_snapshot and frames % 10 == 0:
                imwrite_unicode(args.save_snapshot, best if best is not None else frame)

            if show_window:
                disp = frame
                if disp.shape[1] > args.display_width:
                    scale = args.display_width / disp.shape[1]
                    disp = cv2.resize(frame, (args.display_width, int(disp.shape[0] * scale)),
                                      interpolation=cv2.INTER_AREA)
                cv2.imshow(args.window, disp)
                if frames == 2:
                    log(f"[运行] 可视化窗口状态 可见/置顶 = {window_state(args.window)}")
                if not topmost_done:
                    topmost_done = make_topmost(args.window) or frames >= 40
                key = cv2.waitKey(1) & 0xFF
                if args.log_keys and key != 255:
                    log(f"[按键] 收到键值 {key} ({chr(key) if 32 <= key < 127 else '?'})")
                if is_quit_key(key):
                    log(f"[运行] 收到 {'ESC' if key == ESC_KEY else 'q'} 键，退出。")
                    break

            if args.max_frames and frames >= args.max_frames:
                log(f"[运行] 已达到 --max-frames {args.max_frames}，自动退出。")
                break

    if show_window:
        cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    if args.save_snapshot and frame is not None:
        imwrite_unicode(args.save_snapshot, best if best is not None else frame)
        log(f"[输出] 标注截图已保存: {args.save_snapshot}")

    log(f"[统计] 处理帧数={frames} 耗时={elapsed:.1f}s 平均FPS={frames / max(elapsed,1e-6):.2f}")
    log(f"[统计] 检出人物的帧数={person_frames} 帧内人物总数={total_persons} 单帧最多={max(best_n,0)}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except BaseException:
        import traceback

        traceback.print_exc()
        # 打包成 exe 双击运行时，出错不能让窗口一闪而过，停住让人看清原因
        if getattr(sys, "frozen", False):
            try:
                input("\n[错误] 程序异常结束，按回车键关闭窗口 ...")
            except (EOFError, KeyboardInterrupt):
                pass
        sys.exit(1)
