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

from SendInput import mouse_xy
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

    # 读取图片
    while True:
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
            # 画框
            annotator = Annotator(im0, line_width=2)
            if len(det):
                distance_list = []  # 距离列表
                target_list = []  # 敌人列表
                # 将转换后的图片画框结果转换成原图上的结果
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()

                    # 鼠标移动值（相对选区中心的偏移）
                    X = xywh[0] - half_size
                    Y = xywh[1] - half_size

                    distance = math.sqrt(X ** 2 + Y ** 2)
                    xywh.append(distance)
                    annotator.box_label(xyxy, label=f'[{int(cls)}Distance:{round(distance, 2)}]',
                                        color=(34, 139, 34),
                                        txt_color=(0, 191, 255))

                    distance_list.append(distance)
                    target_list.append(xywh)

                # 获取距离最小的目标
                target_info = target_list[distance_list.index(min(distance_list))]
                print(f"目标信息：{target_info}")

                if is_active:
                    # 目标相对选区中心的偏移量，即鼠标需要的相对位移
                    dx = int(target_info[0]) - half_size
                    dy = int(target_info[1]) - half_size
                    # 灵敏度系数：>1 加速，<1 减速，根据实际效果调整
                    sensitivity = 4.0
                    mouse_xy(int(dx * sensitivity), int(dy * sensitivity))

            im0 = annotator.result()
            cv2.imshow('window', im0)
            cv2.waitKey(1)


if __name__ == "__main__":
    # 启动置顶选区窗口（可拖动）
    overlay.start()
    print(f"[启动] 选区窗口已就绪，拖动绿色边框调整监控位置")
    threading.Thread(target=keyboard_listener, daemon=True).start()
    run()
