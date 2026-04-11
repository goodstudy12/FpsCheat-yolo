# -*- coding: utf-8 -*-
"""截图模块 —— 支持动态区域截图"""

import cv2
import numpy as np
from mss import mss

# 实例化 mss
Screenshot_value = mss()


def screenshot(region):
    """
    截取指定区域的屏幕图像。
    :param region: (left, top, right, bottom) 四元组
    :return: BGR 格式的 numpy 数组
    """
    left, top, right, bottom = region
    monitor = {
        "left": left,
        "top": top,
        "width": right - left,
        "height": bottom - top,
    }
    img = Screenshot_value.grab(monitor)
    # 直接丢弃 alpha 通道，避免 cvtColor 全图转换开销
    img = np.ascontiguousarray(np.array(img, dtype=np.uint8)[:, :, :3])
    return img
