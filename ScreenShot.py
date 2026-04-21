# -*- coding: utf-8 -*-
"""截图模块 —— 支持动态区域截图"""

import cv2
import numpy as np
from mss import mss

import threading

# 每个线程持有自己的 mss 实例（mss 内部使用 thread-local 句柄，不能跨线程共享）
_local = threading.local()


def _get_mss():
    """返回当前线程的 mss 实例，首次调用时自动创建"""
    if not hasattr(_local, "sct"):
        _local.sct = mss()
    return _local.sct


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
    img = _get_mss().grab(monitor)
    # 直接丢弃 alpha 通道，避免 cvtColor 全图转换开销
    img = np.ascontiguousarray(np.array(img, dtype=np.uint8)[:, :, :3])
    return img
