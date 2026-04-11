# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, r"D:\python_packages")
import math
import threading
import time
import numpy as np
import torch
from utils.augmentations import letterbox
from models.common import DetectMultiBackend
from utils.general import (cv2, non_max_suppression, scale_boxes, xyxy2xywh)
from utils.plots import Annotator
from utils.torch_utils import smart_inference_mode

from SendInput import mouse_xy, smooth_move
from ScreenShot import screenshot
from OverlayWindow import OverlayWindow

from pynput import keyboard

# F1 切换开关：按一次开启，再按一次暂停
is_active = False

# 选区窗口（640x640 置顶透明框，可拖动定位）
overlay = OverlayWindow(size=640)


def on_key_press(key):
    global is_active
    if key == keyboard.Key.f1:
        is_active = not is_active
        overlay.update_status(is_active)
        print(f"[F1] 自瞄已{'开启' if is_active else '暂停'}")


def keyboard_listener():
    with keyboard.Listener(on_press=on_key_press) as listener:
        listener.join()


@smart_inference_mode()
def run():
    global is_active
    # 加载模型
    device = torch.device('cuda:0')
    model = DetectMultiBackend(weights='./weights/Valorant.pt', device=device, dnn=False, data=False, fp16=True)
    # device = torch.device('cpu')
    # model = DetectMultiBackend(weights='./weights/Valorant.pt', device=device, dnn=False, data=False, fp16=False)

    half_size = overlay.size // 2  # 320
    frame_count = 0  # 帧计数器，用于控制可视化频率
    # 帧率统计
    fps_timer = time.time()
    fps_count = 0

    # 读取图片
    while True:
        # 帧率计算：每秒更新一次
        fps_count += 1
        now = time.time()
        if now - fps_timer >= 1.0:
            overlay.update_fps(fps_count / (now - fps_timer))
            fps_count = 0
            fps_timer = now

        # 从选区窗口的当前位置截图
        region = overlay.region
        im = screenshot(region)

        im0 = im

        # 处理图片
        im = letterbox(im, (640, 640), stride=32, auto=True)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous
        im = torch.from_numpy(im).to(model.device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # 推理
        pred = model(im, augment=False, visualize=False)
        # 非极大值抑制  classes=0 只检测人
        pred = non_max_suppression(pred, conf_thres=0.6, iou_thres=0.45, classes=0, max_det=1000)

        # 处理推理内容
        for i, det in enumerate(pred):
            if len(det):
                # 将转换后的图片画框结果转换成原图上的结果
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # 快速计算所有目标到中心的距离，找到最近目标
                xywh_all = xyxy2xywh(det[:, :4])  # 批量转换，避免逐个循环
                dx_all = xywh_all[:, 0] - half_size
                dy_all = xywh_all[:, 1] - half_size
                distances = torch.sqrt(dx_all ** 2 + dy_all ** 2)
                min_idx = torch.argmin(distances).item()

                # 自瞄激活时，使用平滑移动（动态速度 + 头部偏移 + 跳变检测）
                if is_active:
                    dx = int(round(dx_all[min_idx].item()))
                    dy = int(round(dy_all[min_idx].item()))
                    # 获取检测框高度，用于头部偏移计算
                    box_h = int(round((det[min_idx, 3] - det[min_idx, 1]).item()))
                    smooth_move(dx, dy, box_h)

                # 可视化部分：自瞄激活时每 5 帧才绘制一次，减少开销
                frame_count += 1
                if not is_active or frame_count % 5 == 0:
                    annotator = Annotator(im0, line_width=2)
                    overlay_dets = []
                    for idx in range(det.shape[0]):
                        xyxy = det[idx, :4]
                        conf_val = det[idx, 4].item()
                        cls = det[idx, 5].item()
                        dist = distances[idx].item()
                        is_target = (idx == min_idx)
                        annotator.box_label(xyxy,
                                            label=f'[{int(cls)}D:{dist:.0f}]',
                                            color=(34, 139, 34),
                                            txt_color=(0, 191, 255))
                        xyxy_int = [int(v.item()) for v in xyxy]
                        overlay_dets.append({
                            "xyxy": xyxy_int,
                            "label": f"D:{dist:.0f} C:{conf_val:.2f}",
                            "color": "#FF4444" if is_target else "#00FF00",
                            "is_target": is_target,
                        })
                    if overlay_dets:
                        overlay.draw_detections(overlay_dets)
                    im0 = annotator.result()
                    cv2.imshow('window', im0)
                    cv2.waitKey(1)
                else:
                    # 自瞄激活时跳过绘制，只做最小化 UI 刷新
                    cv2.waitKey(1)
            else:
                overlay.clear_detections()
                cv2.waitKey(1)


if __name__ == "__main__":
    # 启动置顶选区窗口（可拖动）
    overlay.start()
    print(f"[启动] 选区窗口已就绪，拖动绿色边框调整监控位置")
    threading.Thread(target=keyboard_listener, daemon=True).start()
    run()
