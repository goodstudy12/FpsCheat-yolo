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
    # ============ 模型后端选择 ============
    # PyTorch (.pt)  ：兼容性好，速度一般
    # ONNX (.onnx)   ：比 .pt 快 ~30%，需安装 onnxruntime-gpu
    # TensorRT (.engine)：最快，比 .pt 快 2-4x，需安装 tensorrt
    #
    # TensorRT 导出命令（在项目根目录执行）：
    #   python export.py --weights ./weights/Valorant.pt --include engine --device 0 --half --imgsz 640
    # 导出后将生成 ./weights/Valorant.engine，修改下方路径即可使用
    device = torch.device('cuda:0')
    # PyTorch fp16 后端：当前环境下最快（203 FPS 纯推理）
    model = DetectMultiBackend(weights='./weights/Valorant.pt', device=device, dnn=False, data=False, fp16=True)
    # ONNX 后端（备用，需安装 onnxruntime-gpu，当前因 cpu→numpy 转换开销略慢）
    # model = DetectMultiBackend(weights='./weights/Valorant.onnx', device=device, dnn=False, data=False, fp16=False)

    half_size = overlay.size // 2  # 320
    frame_count = 0  # 帧计数器，用于控制可视化频率
    # 帧率统计
    fps_timer = time.time()
    fps_count = 0

    # 预分配 GPU tensor 缓冲区，避免每帧重复分配内存
    # 截图固定 640x640，无需 letterbox resize
    img_size = overlay.size  # 640
    dtype = torch.float16 if model.fp16 else torch.float32
    input_buffer = torch.zeros((1, 3, img_size, img_size), dtype=dtype, device=device)

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
        im0 = screenshot(region)

        # 图像预处理：截图已是 640x640，跳过 letterbox resize
        # mss 截图为 BGR 格式（已在 ScreenShot.py 中去掉 alpha），直接 BGR→RGB + HWC→CHW
        im = im0[:, :, ::-1].transpose((2, 0, 1))  # BGR→RGB, HWC→CHW
        # 写入预分配的 GPU 缓冲区，避免每帧 torch.from_numpy + .to(device) 的分配开销
        input_buffer[0] = torch.from_numpy(np.ascontiguousarray(im)).to(dtype=dtype)
        input_buffer.div_(255.0)  # 归一化到 [0, 1]（原地操作）

        # 推理
        pred = model(input_buffer, augment=False, visualize=False)
        # 非极大值抑制  classes=0 只检测人
        pred = non_max_suppression(pred, conf_thres=0.6, iou_thres=0.45, classes=0, max_det=1000)

        # 处理推理内容
        for i, det in enumerate(pred):
            if len(det):
                # 截图与推理尺寸一致（640x640），scale_boxes 实际无缩放，但保留以兼容 padding
                det[:, :4] = scale_boxes(input_buffer.shape[2:], det[:, :4], im0.shape).round()

                # 快速计算所有目标到中心的距离，找到最近目标
                xywh_all = xyxy2xywh(det[:, :4])  # 批量转换，避免逐个循环
                dx_all = xywh_all[:, 0] - half_size
                dy_all = xywh_all[:, 1] - half_size
                distances = dx_all ** 2 + dy_all ** 2  # 省略 sqrt，比较距离不需要开根号
                min_idx = torch.argmin(distances).item()

                # 自瞄激活时，使用平滑移动（动态速度 + 头部偏移 + 跳变检测）
                if is_active:
                    dx = int(round(dx_all[min_idx].item()))
                    dy = int(round(dy_all[min_idx].item()))
                    # 获取检测框高度，用于头部偏移计算
                    box_h = int(round((det[min_idx, 3] - det[min_idx, 1]).item()))
                    smooth_move(dx, dy, box_h)

                # 可视化部分：自瞄激活时每 5 帧才绘制 overlay，且完全跳过 cv2.imshow
                frame_count += 1
                if not is_active or frame_count % 5 == 0:
                    # 对距离开根号仅在需要显示时计算
                    dist_display = torch.sqrt(distances.float())
                    overlay_dets = []
                    for idx in range(det.shape[0]):
                        xyxy = det[idx, :4]
                        conf_val = det[idx, 4].item()
                        dist = dist_display[idx].item()
                        is_target = (idx == min_idx)
                        xyxy_int = [int(v.item()) for v in xyxy]
                        overlay_dets.append({
                            "xyxy": xyxy_int,
                            "label": f"D:{dist:.0f} C:{conf_val:.2f}",
                            "color": "#FF4444" if is_target else "#00FF00",
                            "is_target": is_target,
                        })
                    if overlay_dets:
                        overlay.draw_detections(overlay_dets)

                    # 非自瞄模式下才显示 OpenCV 调试窗口（自瞄时跳过以节省帧率）
                    if not is_active:
                        annotator = Annotator(im0, line_width=2)
                        for idx in range(det.shape[0]):
                            xyxy = det[idx, :4]
                            dist = dist_display[idx].item()
                            cls = det[idx, 5].item()
                            annotator.box_label(xyxy,
                                                label=f'[{int(cls)}D:{dist:.0f}]',
                                                color=(34, 139, 34),
                                                txt_color=(0, 191, 255))
                        im0 = annotator.result()
                        cv2.imshow('window', im0)
                        cv2.waitKey(1)
            else:
                overlay.clear_detections()
                if not is_active:
                    cv2.waitKey(1)


if __name__ == "__main__":
    # 启动置顶选区窗口（可拖动）
    overlay.start()
    print(f"[启动] 选区窗口已就绪，拖动绿色边框调整监控位置")
    threading.Thread(target=keyboard_listener, daemon=True).start()
    run()
