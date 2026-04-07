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
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img
