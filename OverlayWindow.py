# -*- coding: utf-8 -*-
"""可拖动的置顶透明选区窗口，用户自行选择监控位置"""

import tkinter as tk
import threading


class OverlayWindow:
    """置顶半透明选区框，可自由拖动定位，支持绘制检测框"""

    def __init__(self, size=640, border_width=2, border_color="lime"):
        self.size = size
        self.border_width = border_width
        self.border_color = border_color
        # 窗口左上角坐标（线程安全读取）
        self._x = 0
        self._y = 0
        self._root = None
        self._canvas = None
        self._ready = threading.Event()
        # 动态绘制的对象 id 列表，每帧清除重绘
        self._draw_ids = []
        # 自瞄状态文字 id
        self._status_id = None
        # 帧率显示文字 id
        self._fps_id = None

    @property
    def region(self):
        """返回当前选区的 (left, top, right, bottom)"""
        return (self._x, self._y, self._x + self.size, self._y + self.size)

    def start(self):
        """在后台线程启动 tkinter 主循环"""
        t = threading.Thread(target=self._run, daemon=True)
        t.start()
        # 等待窗口就绪
        self._ready.wait()

    def _run(self):
        root = tk.Tk()
        self._root = root
        root.title("选区")
        root.overrideredirect(True)  # 无边框
        root.attributes("-topmost", True)  # 置顶
        root.attributes("-transparentcolor", "black")  # 黑色区域完全透明（点击穿透）

        # 窗口初始位置：屏幕中心
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        init_x = (screen_w - self.size) // 2
        init_y = (screen_h - self.size) // 2
        self._x = init_x
        self._y = init_y

        # 窗口大小 = 选区 + 边框
        win_size = self.size + self.border_width * 2
        root.geometry(f"{win_size}x{win_size}+{init_x}+{init_y}")

        # 画布：黑色背景（透明穿透），只画边框线
        canvas = tk.Canvas(root, width=win_size, height=win_size,
                           bg="black", highlightthickness=0)
        canvas.pack()

        # 画四条边框线
        bw = self.border_width
        color = self.border_color
        # 上
        canvas.create_rectangle(0, 0, win_size, bw, fill=color, outline="")
        # 下
        canvas.create_rectangle(0, win_size - bw, win_size, win_size, fill=color, outline="")
        # 左
        canvas.create_rectangle(0, 0, bw, win_size, fill=color, outline="")
        # 右
        canvas.create_rectangle(win_size - bw, 0, win_size, win_size, fill=color, outline="")

        # 中心准心（绿色十字线）
        center = win_size // 2
        cross_len = 10  # 准心半长（像素）
        cross_w = 2     # 准心线宽
        # 水平线
        canvas.create_rectangle(center - cross_len, center - cross_w // 2,
                                center + cross_len, center + cross_w // 2,
                                fill=color, outline="")
        # 垂直线
        canvas.create_rectangle(center - cross_w // 2, center - cross_len,
                                center + cross_w // 2, center + cross_len,
                                fill=color, outline="")

        # 四个角落画小方块便于拖动（增大可点击区域）
        corner = 20
        for cx, cy in [(0, 0), (win_size - corner, 0),
                        (0, win_size - corner), (win_size - corner, win_size - corner)]:
            canvas.create_rectangle(cx, cy, cx + corner, cy + corner,
                                    fill=color, outline="")

        # 拖动逻辑：在边框/角落区域按下拖动
        self._drag_data = {"x": 0, "y": 0}

        def on_press(event):
            self._drag_data["x"] = event.x
            self._drag_data["y"] = event.y

        def on_drag(event):
            dx = event.x - self._drag_data["x"]
            dy = event.y - self._drag_data["y"]
            new_x = root.winfo_x() + dx
            new_y = root.winfo_y() + dy
            root.geometry(f"+{new_x}+{new_y}")
            self._x = new_x + self.border_width
            self._y = new_y + self.border_width

        canvas.bind("<ButtonPress-1>", on_press)
        canvas.bind("<B1-Motion>", on_drag)

        # 初始状态文字（左上角，边框内侧）
        self._status_id = canvas.create_text(
            bw + 22, bw + 2, anchor="nw",
            text="自瞄：关", fill="#FF4444",
            font=("Microsoft YaHei", 10, "bold")
        )

        # 帧率显示（右上角，边框内侧）
        self._fps_id = canvas.create_text(
            win_size - bw - 22, bw + 2, anchor="ne",
            text="FPS: --", fill="#FFFF00",
            font=("Consolas", 10, "bold")
        )

        self._canvas = canvas
        self._ready.set()
        root.mainloop()

    def draw_detections(self, detections):
        """线程安全地在覆盖窗口上绘制检测框。

        参数:
            detections: 列表，每个元素为 dict:
                {
                    "xyxy": (x1, y1, x2, y2),  # 像素坐标（相对截图区域）
                    "label": str,                # 标签文字
                    "color": str,                # 框颜色，如 "red", "#FF0000"
                    "is_target": bool,           # 是否为当前锁定目标（加粗高亮）
                }
        """
        if self._root and self._canvas:
            self._root.after(0, self._do_draw, detections)

    def _do_draw(self, detections):
        """在 tkinter 主线程中执行实际绘制"""
        canvas = self._canvas
        bw = self.border_width

        # 清除上一帧的动态绘制
        for item_id in self._draw_ids:
            canvas.delete(item_id)
        self._draw_ids.clear()

        for det in detections:
            x1, y1, x2, y2 = det["xyxy"]
            label = det.get("label", "")
            color = det.get("color", "red")
            is_target = det.get("is_target", False)
            line_w = 3 if is_target else 1

            # 坐标偏移：加上边框宽度（canvas 坐标 = 截图坐标 + border_width）
            cx1, cy1 = x1 + bw, y1 + bw
            cx2, cy2 = x2 + bw, y2 + bw

            # 画矩形框
            rect_id = canvas.create_rectangle(
                cx1, cy1, cx2, cy2,
                outline=color, width=line_w
            )
            self._draw_ids.append(rect_id)

            # 画标签文字（框顶部上方）
            if label:
                text_id = canvas.create_text(
                    cx1, cy1 - 2, anchor="sw",
                    text=label, fill=color,
                    font=("Consolas", 10, "bold") if is_target else ("Consolas", 9)
                )
                self._draw_ids.append(text_id)

    def update_status(self, active: bool):
        """线程安全地更新自瞄状态显示"""
        if self._root and self._canvas:
            self._root.after(0, self._do_update_status, active)

    def _do_update_status(self, active: bool):
        """在 tkinter 主线程中更新状态文字"""
        if self._status_id:
            text = "自瞄：开" if active else "自瞄：关"
            color = "#00FF00" if active else "#FF4444"
            self._canvas.itemconfig(self._status_id, text=text, fill=color)

    def update_fps(self, fps: float):
        """线程安全地更新帧率显示"""
        if self._root and self._canvas:
            self._root.after(0, self._do_update_fps, fps)

    def _do_update_fps(self, fps: float):
        """在 tkinter 主线程中更新帧率文字"""
        if self._fps_id:
            self._canvas.itemconfig(self._fps_id, text=f"FPS: {fps:.0f}")

    def clear_detections(self):
        """清除所有检测框绘制"""
        if self._root and self._canvas:
            self._root.after(0, self._do_clear)

    def _do_clear(self):
        for item_id in self._draw_ids:
            self._canvas.delete(item_id)
        self._draw_ids.clear()
