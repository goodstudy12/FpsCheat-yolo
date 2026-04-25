# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, r"D:\python_packages")
import threading
import time
import numpy as np
import torch
from models.common import DetectMultiBackend
from utils.general import (cv2, non_max_suppression, scale_boxes, xyxy2xywh)
from utils.plots import Annotator
from utils.torch_utils import smart_inference_mode

from SendInput import mouse_xy, smooth_move, recoil_compensate
import SendInput as _si
from ScreenShot import screenshot
from OverlayWindow import OverlayWindow
from GUI import ConfigGUI, SharedState

from pynput import keyboard

# TensorRT-YOLO 后端（可选）
try:
    from trtyolo import TRTYOLO
    HAS_TRTYOLO = True
except ImportError:
    HAS_TRTYOLO = False

# ─── 全局共享状态 ────────────────────────────────────────────
state = SharedState()

# 配置面板（全局，供键盘监听器调用）
gui = ConfigGUI()

# 选区覆盖层（置顶透明框，可拖动定位）
overlay: OverlayWindow | None = None


# ─── 键盘监听 ────────────────────────────────────────────────

def on_key_press(key):
    """F1 切换自瞄，F2 切换压枪，END 退出"""
    if key == keyboard.Key.f1:
        active = state.toggle_aim()
        if overlay:
            overlay.update_status(active)
        print(f"[F1] 自瞄已{'开启' if active else '暂停'}")

    elif key == keyboard.Key.f2:
        active = state.toggle_recoil()
        if overlay:
            overlay.update_recoil(active)
        print(f"[F2] 压枪已{'开启' if active else '暂停'}")

    elif key == keyboard.Key.end:
        print("[END] 退出")
        state.running = False
        sys.exit(0)


def keyboard_listener():
    with keyboard.Listener(on_press=on_key_press) as listener:
        listener.join()


# ─── 推理主循环 ──────────────────────────────────────────────

@smart_inference_mode()
def run():
    """推理主循环，由 GUI START 按钮触发，在子线程中运行"""
    global overlay

    # 从 GUI 读取硬件配置（仅在启动时读取一次）
    cfg = gui.get_config()
    fov_size   = cfg.get("fov_size", 640)
    model_path = cfg.get("model", "./weights/Valorant.pt")
    use_fp16   = cfg.get("fp16", True)
    backend    = cfg.get("backend", "auto")

    # 安全解析 CUDA 设备 ID（格式："CUDA:0 (设备名)"）
    try:
        gpu_str = cfg.get("gpu", "CUDA:0")
        gpu_id  = int(gpu_str.split("(")[0].split(":")[1].strip())
    except Exception:
        gpu_id = 0
    if gpu_id >= torch.cuda.device_count():
        print(f"[警告] CUDA:{gpu_id} 不存在，回退到 CUDA:0")
        gpu_id = 0

    device = torch.device(f"cuda:{gpu_id}")

    # 启动/调整覆盖层尺寸
    if overlay is None:
        overlay = OverlayWindow(size=fov_size)
        overlay.start()
        print(f"[启动] 选区窗口已就绪（{fov_size}×{fov_size}），拖动绿色边框调整位置")
    else:
        overlay.size = fov_size

    # ── 选择推理后端 ──────────────────────────────────────────
    use_trtyolo = False
    if backend == "trtyolo" or (backend == "auto" and model_path.endswith(".engine")):
        if HAS_TRTYOLO:
            use_trtyolo = True
        else:
            print("[警告] trtyolo 未安装，回退到默认后端")

    if use_trtyolo:
        model = TRTYOLO(model_path, task="detect", swap_rb=True)
        print(f"[后端] TensorRT-YOLO（{model_path}）")
    else:
        model = DetectMultiBackend(
            weights=model_path, device=device,
            dnn=False, data=False, fp16=use_fp16
        )
        print(f"[后端] DetectMultiBackend（{model_path}）")

    half_size   = overlay.size // 2
    frame_count = 0
    fps_timer   = time.time()
    fps_count   = 0

    if not use_trtyolo:
        img_size  = overlay.size
        dtype     = torch.float16 if model.fp16 else torch.float32
        input_buf = torch.zeros((1, 3, img_size, img_size), dtype=dtype, device=device)

    print("[推理] 开始……")
    while state.running:
        # ── 帧率统计 ──────────────────────────────────────────
        fps_count += 1
        now = time.time()
        if now - fps_timer >= 1.0:
            fps = fps_count / (now - fps_timer)
            state.set_fps(fps)
            overlay.update_fps(fps)
            fps_count = 0
            fps_timer = now

        # ── 每帧读取运行时配置（允许 GUI 实时调参）────────────
        cfg = gui.get_config()
        conf_thres = cfg.get("conf_thres", 0.6)
        _si.LOCK_SMOOTH  = cfg.get("lock_smooth", 2.5)
        _si.EMA_ALPHA    = cfg.get("ema_alpha", 0.4)
        _si.DEAD_ZONE    = cfg.get("dead_zone", 3.0)
        _si.RECOIL_STRENGTH_Y = cfg.get("recoil_strength_y", 3.0)
        _si.PREDICT_STRENGTH  = cfg.get("predict_strength", 0.5)
        import math
        _si._K = 4.07 * (1.0 / max(_si.LOCK_SMOOTH, 0.1))

        is_active = state.is_active
        is_recoil = state.is_recoil

        # ── 截图 ──────────────────────────────────────────────
        region = overlay.region
        im0    = screenshot(region)

        # ── 推理 + NMS ────────────────────────────────────────
        # 两个后端统一输出: xyxy_np [N,4], conf_np [N], cls_np [N]
        if use_trtyolo:
            result = model.predict(im0)
            if result is not None and len(result) > 0:
                xyxy_np = result.xyxy
                conf_np = result.confidence
                cls_np  = result.class_id
                mask = (cls_np == 0) & (conf_np >= conf_thres)
                xyxy_np = xyxy_np[mask]
                conf_np = conf_np[mask]
                cls_np  = cls_np[mask]
                n_det = len(xyxy_np)
            else:
                n_det = 0
        else:
            im = im0[:, :, ::-1].transpose((2, 0, 1))
            input_buf[0] = torch.from_numpy(np.ascontiguousarray(im)).to(dtype=dtype)
            input_buf.div_(255.0)
            pred = model(input_buf, augment=False, visualize=False)
            pred = non_max_suppression(pred, conf_thres=conf_thres,
                                       iou_thres=0.45, classes=0, max_det=1000)
            det = pred[0] if pred else torch.zeros((0, 6))
            if len(det):
                det[:, :4] = scale_boxes(
                    input_buf.shape[2:], det[:, :4], im0.shape).round()
                xyxy_np = det[:, :4].cpu().numpy()
                conf_np = det[:, 4].cpu().numpy()
                cls_np  = det[:, 5].cpu().numpy()
                n_det = len(xyxy_np)
            else:
                n_det = 0

        # ── 压枪补偿（每帧执行，不依赖目标检测）──────────────
        if is_recoil:
            recoil_compensate()

        # ── 检测结果处理（统一 numpy 格式）─────────────────────
        if n_det > 0:
            cx = (xyxy_np[:, 0] + xyxy_np[:, 2]) / 2.0
            cy = (xyxy_np[:, 1] + xyxy_np[:, 3]) / 2.0
            dx_all = cx - half_size
            dy_all = cy - half_size
            distances = dx_all ** 2 + dy_all ** 2
            min_idx = int(np.argmin(distances))

            if is_active:
                dx    = int(round(dx_all[min_idx]))
                dy    = int(round(dy_all[min_idx]))
                box_h = int(round(xyxy_np[min_idx, 3] - xyxy_np[min_idx, 1]))
                smooth_move(dx, dy, box_h)

            frame_count += 1
            if not is_active or frame_count % 5 == 0:
                dist_display = np.sqrt(distances.astype(np.float32))
                overlay_dets = []
                for idx in range(n_det):
                    x1, y1, x2, y2 = xyxy_np[idx]
                    conf_val = float(conf_np[idx])
                    dist     = float(dist_display[idx])
                    is_tgt   = (idx == min_idx)
                    overlay_dets.append({
                        "xyxy"     : [int(x1), int(y1), int(x2), int(y2)],
                        "label"    : f"D:{dist:.0f} C:{conf_val:.2f}",
                        "color"    : "#FF4444" if is_tgt else "#00FF00",
                        "is_target": is_tgt,
                    })
                if overlay_dets:
                    overlay.draw_detections(overlay_dets)

                if not is_active:
                    annotator = Annotator(im0, line_width=2)
                    for idx in range(n_det):
                        x1, y1, x2, y2 = xyxy_np[idx]
                        xyxy_t = torch.tensor([x1, y1, x2, y2])
                        dist   = float(dist_display[idx])
                        cls    = int(cls_np[idx])
                        annotator.box_label(xyxy_t,
                                            label=f'[{cls}D:{dist:.0f}]',
                                            color=(34, 139, 34),
                                            txt_color=(0, 191, 255))
                    im0 = annotator.result()
                    cv2.imshow("window", im0)
                    cv2.waitKey(1)
        else:
            overlay.clear_detections()
            if not is_active:
                cv2.waitKey(1)

    print("[推理] 已停止")


# ─── 程序入口 ────────────────────────────────────────────────

if __name__ == "__main__":
    # 启动键盘监听（后台线程）
    threading.Thread(target=keyboard_listener, daemon=True).start()

    # 打开配置面板（阻塞主线程，直到窗口关闭）
    # START 按钮触发 run()，STOP 按钮设置 state.running = False
    def _start_infer():
        state.running = True
        run()

    def _stop_infer():
        state.running = False

    print("[启动] 配置面板已打开，点击 START 开始推理")
    gui.start(state, on_start=_start_infer, on_stop=_stop_infer)
