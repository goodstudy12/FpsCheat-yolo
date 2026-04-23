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
    fov_size  = cfg.get("fov_size", 640)
    model_path = cfg.get("model", "./weights/Valorant.pt")
    use_fp16  = cfg.get("fp16", True)
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

    model = DetectMultiBackend(
        weights=model_path, device=device,
        dnn=False, data=False, fp16=use_fp16
    )

    half_size  = overlay.size // 2
    frame_count = 0
    fps_timer  = time.time()
    fps_count  = 0

    img_size = overlay.size
    dtype    = torch.float16 if model.fp16 else torch.float32
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
        # 将 GUI 滑块值同步到 SendInput 全局参数
        _si.LOCK_SMOOTH  = cfg.get("lock_smooth", 2.5)
        _si.EMA_ALPHA    = cfg.get("ema_alpha", 0.4)
        _si.DEAD_ZONE    = cfg.get("dead_zone", 3.0)
        _si.RECOIL_STRENGTH_Y = cfg.get("recoil_strength_y", 3.0)
        _si.PREDICT_STRENGTH  = cfg.get("predict_strength", 0.5)
        # 重新计算 _K（atan 平滑系数依赖 LOCK_SMOOTH）
        import math
        _si._K = 4.07 * (1.0 / max(_si.LOCK_SMOOTH, 0.1))

        is_active  = state.is_active
        is_recoil  = state.is_recoil

        # ── 截图 + 预处理 ──────────────────────────────────────
        region = overlay.region
        im0    = screenshot(region)

        im = im0[:, :, ::-1].transpose((2, 0, 1))
        input_buf[0] = torch.from_numpy(np.ascontiguousarray(im)).to(dtype=dtype)
        input_buf.div_(255.0)

        # ── 推理 ───────────────────────────────────────────────
        pred = model(input_buf, augment=False, visualize=False)
        pred = non_max_suppression(pred, conf_thres=conf_thres,
                                   iou_thres=0.45, classes=0, max_det=1000)

        # ── 压枪补偿（每帧执行，不依赖目标检测）──────────────
        if is_recoil:
            recoil_compensate()

        # ── 检测结果处理 ───────────────────────────────────────
        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = scale_boxes(
                    input_buf.shape[2:], det[:, :4], im0.shape).round()

                xywh_all  = xyxy2xywh(det[:, :4])
                dx_all    = xywh_all[:, 0] - half_size
                dy_all    = xywh_all[:, 1] - half_size
                distances = dx_all ** 2 + dy_all ** 2
                min_idx   = torch.argmin(distances).item()

                if is_active:
                    dx    = int(round(dx_all[min_idx].item()))
                    dy    = int(round(dy_all[min_idx].item()))
                    box_h = int(round((det[min_idx, 3] - det[min_idx, 1]).item()))
                    smooth_move(dx, dy, box_h)

                # 可视化（每 5 帧绘制一次，节省开销）
                frame_count += 1
                if not is_active or frame_count % 5 == 0:
                    dist_display = torch.sqrt(distances.float())
                    overlay_dets = []
                    for idx in range(det.shape[0]):
                        xyxy     = det[idx, :4]
                        conf_val = det[idx, 4].item()
                        dist     = dist_display[idx].item()
                        is_tgt   = (idx == min_idx)
                        overlay_dets.append({
                            "xyxy"     : [int(v.item()) for v in xyxy],
                            "label"    : f"D:{dist:.0f} C:{conf_val:.2f}",
                            "color"    : "#FF4444" if is_tgt else "#00FF00",
                            "is_target": is_tgt,
                        })
                    if overlay_dets:
                        overlay.draw_detections(overlay_dets)

                    if not is_active:
                        annotator = Annotator(im0, line_width=2)
                        for idx in range(det.shape[0]):
                            xyxy  = det[idx, :4]
                            dist  = dist_display[idx].item()
                            cls   = det[idx, 5].item()
                            annotator.box_label(xyxy,
                                                label=f'[{int(cls)}D:{dist:.0f}]',
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
