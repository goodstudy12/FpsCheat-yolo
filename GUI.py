# -*- coding: utf-8 -*-
"""
YOLOv8 通用瞄准框架 - 配置面板
仿照截图风格，深色主题 + 红色高亮
"""

import tkinter as tk
from tkinter import ttk
import threading
import torch

# ─── 颜色主题 ───────────────────────────────────────────────
BG_MAIN    = "#1c1c2c"   # 主背景
BG_PANEL   = "#252535"   # 左侧面板
BG_SECTION = "#2e2e42"   # 区块头背景
BG_ENTRY   = "#1a1a28"   # 输入框背景
RED        = "#c0392b"   # 红色高亮（标题栏/激活状态）
ORANGE     = "#e67e22"   # 橙色（START 按钮）
YELLOW     = "#f1c40f"   # 黄色（数值显示）
GREEN      = "#27ae60"   # 绿色（开启状态）
WHITE      = "#ecf0f1"   # 主文字
GRAY       = "#7f8c8d"   # 次要文字
TROUGH     = "#e74c3c"   # 滑块槽颜色


def _style_combobox(root: tk.Tk):
    """全局设置 ttk.Combobox 深色主题"""
    style = ttk.Style(root)
    style.theme_use("default")
    style.configure("Dark.TCombobox",
                    fieldbackground=BG_ENTRY,
                    background=BG_SECTION,
                    foreground=YELLOW,
                    selectbackground=BG_SECTION,
                    selectforeground=YELLOW,
                    arrowcolor=WHITE)


# ─── 共享状态（线程安全读写，由 main 和 GUI 共同使用）────────
class SharedState:
    """全局运行时状态，供推理线程与 GUI 读写"""
    def __init__(self):
        self.is_active  = False   # 自瞄开关
        self.is_recoil  = False   # 压枪开关
        self.fps        = 0.0     # 当前推理帧率
        self.running    = False   # 推理线程是否在跑
        self._lock      = threading.Lock()

    def toggle_aim(self) -> bool:
        with self._lock:
            self.is_active = not self.is_active
            return self.is_active

    def toggle_recoil(self) -> bool:
        with self._lock:
            self.is_recoil = not self.is_recoil
            return self.is_recoil

    def set_fps(self, fps: float):
        with self._lock:
            self.fps = fps


# ─── 主 GUI 类 ───────────────────────────────────────────────
class ConfigGUI:
    """
    配置面板主窗口。
    调用 start(state, on_start, on_stop) 后阻塞，直到窗口关闭。
    """

    def __init__(self):
        self._root: tk.Tk | None = None
        self._state: SharedState | None = None
        self._on_start = None   # 回调：启动推理
        self._on_stop  = None   # 回调：停止推理

        # tkinter 变量（在 _init_vars 中初始化，必须在 Tk() 之后）
        self.conf_var       = None  # 置信度
        self.smooth_var     = None  # 平滑系数
        self.sen_var        = None  # 灵敏度
        self.offset_y_var   = None  # Y轴头部偏移
        self.fov_x_var      = None  # FOV X
        self.fov_y_var      = None  # FOV Y
        self.recoil_y_var   = None  # 压枪垂直强度
        self.ema_alpha_var  = None  # EMA 系数
        self.dead_zone_var  = None  # 死区像素
        self.aim_range_var  = None  # 瞄准范围
        self.algo_var       = None  # 算法选择
        self.gpu_var        = None  # GPU 选择
        self.aim_key_var    = None  # 瞄准键
        self.auto_trig_var  = None  # 自动扳机
        self.trig_delay_var = None  # 扳机延迟
        self.fov_size_var   = None  # 截图范围
        self.model_var      = None  # 模型路径
        self.fp16_var       = None  # FP16 精度

        # 状态标签引用（用于实时刷新）
        self._fps_lbl      = None
        self._aim_lbl      = None
        self._recoil_lbl   = None
        self._start_btn    = None

    # ── 公开接口 ───────────────────────────────────────────────

    def start(self, state: SharedState, on_start=None, on_stop=None):
        """
        启动 GUI（阻塞直到窗口关闭）。
        :param state:    SharedState 共享状态对象
        :param on_start: 点击 START 时的回调（在新线程中执行）
        :param on_stop:  点击 STOP 时的回调
        """
        self._state    = state
        self._on_start = on_start
        self._on_stop  = on_stop
        self._run()

    def get_config(self) -> dict:
        """返回当前配置字典（供推理线程读取）"""
        if not self._root:
            return {}
        try:
            return {
                "conf_thres":       float(self.conf_var.get()),
                "lock_smooth":      float(self.smooth_var.get()),
                "lock_sen":         float(self.sen_var.get()),
                "offset_y":         float(self.offset_y_var.get()),
                "fov_x":            int(self.fov_x_var.get()),
                "fov_y":            int(self.fov_y_var.get()),
                "recoil_strength_y":float(self.recoil_y_var.get()),
                "ema_alpha":        float(self.ema_alpha_var.get()),
                "dead_zone":        float(self.dead_zone_var.get()),
                "aim_range":        int(float(self.aim_range_var.get())),
                "algo":             self.algo_var.get(),
                "gpu":              self.gpu_var.get(),
                "aim_key":          self.aim_key_var.get(),
                "auto_trigger":     bool(self.auto_trig_var.get()),
                "trigger_delay_ms": int(self.trig_delay_var.get()),
                "fov_size":         int(self.fov_size_var.get()),
                "model":            self.model_var.get(),
                "fp16":             bool(self.fp16_var.get()),
            }
        except Exception:
            return {}

    # ── 内部构建 ───────────────────────────────────────────────

    def _run(self):
        root = tk.Tk()
        self._root = root
        self._init_vars()
        _style_combobox(root)

        root.title("YOLOv8 AI 通用瞄准框架")
        root.configure(bg=BG_MAIN)
        root.resizable(False, False)

        win_w, win_h = 820, 560
        sw = root.winfo_screenwidth()
        sh = root.winfo_screenheight()
        root.geometry(f"{win_w}x{win_h}+{(sw - win_w) // 2}+{(sh - win_h) // 2}")

        self._build_title(root)

        content = tk.Frame(root, bg=BG_MAIN)
        content.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))

        self._build_left(content)
        self._build_right(content)

        # 启动状态轮询（每 200ms 刷新一次实时信息）
        self._poll_state()

        root.protocol("WM_DELETE_WINDOW", self._on_close)
        root.mainloop()

    def _init_vars(self):
        """在 Tk() 创建后初始化所有 tkinter 变量"""
        self.conf_var       = tk.DoubleVar(value=0.60)
        self.smooth_var     = tk.DoubleVar(value=2.5)
        self.sen_var        = tk.DoubleVar(value=1.0)
        self.offset_y_var   = tk.DoubleVar(value=0.75)
        self.fov_x_var      = tk.StringVar(value="4800")
        self.fov_y_var      = tk.StringVar(value="1500")
        self.recoil_y_var   = tk.DoubleVar(value=3.0)
        self.ema_alpha_var  = tk.DoubleVar(value=0.44)
        self.dead_zone_var  = tk.DoubleVar(value=3.0)
        self.aim_range_var  = tk.DoubleVar(value=141)
        self.algo_var       = tk.StringVar(value="hard")
        # 动态枚举 CUDA 设备，避免与任务管理器编号混淆
        self._cuda_devices = []
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            self._cuda_devices.append(f"CUDA:{i} ({name})")
        if not self._cuda_devices:
            self._cuda_devices = ["CPU（无CUDA设备）"]
        self.gpu_var        = tk.StringVar(value=self._cuda_devices[0])
        self.aim_key_var    = tk.StringVar(value="鼠标右键")
        self.auto_trig_var  = tk.BooleanVar(value=False)
        self.trig_delay_var = tk.StringVar(value="100")
        self.fov_size_var   = tk.IntVar(value=640)
        self.model_var      = tk.StringVar(value="./weights/Valorant.pt")
        self.fp16_var       = tk.BooleanVar(value=True)

    def _poll_state(self):
        """每 200ms 轮询共享状态并刷新 GUI 显示"""
        if self._state and self._root:
            # 帧率
            fps = self._state.fps
            if self._fps_lbl:
                self._fps_lbl.config(text=f"FPS: {fps:.0f}")
            # 自瞄状态
            if self._aim_lbl:
                on = self._state.is_active
                self._aim_lbl.config(
                    text="开" if on else "关",
                    fg=GREEN if on else RED
                )
            # 压枪状态
            if self._recoil_lbl:
                on = self._state.is_recoil
                self._recoil_lbl.config(
                    text="开" if on else "关",
                    fg=GREEN if on else RED
                )
            self._root.after(200, self._poll_state)

    def _on_close(self):
        """窗口关闭：停止推理再退出"""
        if self._state and self._state.running and self._on_stop:
            self._on_stop()
        self._root.destroy()

    # ── 标题栏 ─────────────────────────────────────────────────

    def _build_title(self, root):
        bar = tk.Frame(root, bg=RED, height=44)
        bar.pack(fill=tk.X)
        bar.pack_propagate(False)

        # Logo
        tk.Label(bar, text="YOLOv8", bg=RED, fg=WHITE,
                 font=("Arial Black", 14, "bold")).pack(side=tk.LEFT, padx=(12, 0), pady=6)
        tk.Label(bar, text=" AI", bg="#f39c12", fg=WHITE,
                 font=("Arial Black", 14, "bold")).pack(side=tk.LEFT, pady=6)
        tk.Label(bar, text=" 通用瞄准框架", bg=RED, fg=WHITE,
                 font=("Microsoft YaHei", 11)).pack(side=tk.LEFT, padx=6, pady=6)

        # 副标题（灰色）
        tk.Label(bar, text="课程配套教学素材", bg=RED, fg="#ffaaaa",
                 font=("Microsoft YaHei", 8)).pack(side=tk.LEFT, pady=(18, 0))

        # 右侧关闭
        tk.Button(bar, text="✕", bg=RED, fg=WHITE,
                  relief=tk.FLAT, font=("Arial", 13, "bold"),
                  activebackground="#a93226", activeforeground=WHITE,
                  command=self._on_close).pack(side=tk.RIGHT, padx=6)

        tk.Label(bar, text="需要购买YOLO训练数据集？点这里！",
                 bg=RED, fg=YELLOW,
                 font=("Microsoft YaHei", 9)).pack(side=tk.RIGHT, padx=10)

    # ── 左侧实时信息面板 ───────────────────────────────────────

    def _build_left(self, parent):
        left = tk.Frame(parent, bg=BG_PANEL, width=185)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 4))
        left.pack_propagate(False)

        # 标题
        hdr = tk.Frame(left, bg=RED, height=26)
        hdr.pack(fill=tk.X)
        hdr.pack_propagate(False)
        tk.Label(hdr, text="▶ 实时信息", bg=RED, fg=WHITE,
                 font=("Microsoft YaHei", 9, "bold")).pack(side=tk.LEFT, padx=8)

        info = tk.Frame(left, bg=BG_PANEL)
        info.pack(fill=tk.X, padx=8, pady=6)

        tk.Label(info, text="出检范围: 800,380,320,320",
                 bg=BG_PANEL, fg=GRAY, font=("Consolas", 8)).pack(anchor="w")

        self._fps_lbl = tk.Label(info, text="FPS: --",
                                 bg=BG_PANEL, fg=GREEN,
                                 font=("Consolas", 9, "bold"))
        self._fps_lbl.pack(anchor="w")

        tk.Label(info, text="键状态", bg=BG_PANEL, fg=GRAY,
                 font=("Microsoft YaHei", 8)).pack(anchor="w", pady=(6, 0))

        # 自瞄
        row = tk.Frame(info, bg=BG_PANEL)
        row.pack(fill=tk.X, pady=1)
        tk.Label(row, text="自瞄：", bg=BG_PANEL, fg=WHITE,
                 font=("Microsoft YaHei", 8)).pack(side=tk.LEFT)
        self._aim_lbl = tk.Label(row, text="关", bg=BG_PANEL, fg=RED,
                                 font=("Microsoft YaHei", 8, "bold"))
        self._aim_lbl.pack(side=tk.LEFT)

        # 压枪
        row2 = tk.Frame(info, bg=BG_PANEL)
        row2.pack(fill=tk.X, pady=1)
        tk.Label(row2, text="压枪：", bg=BG_PANEL, fg=WHITE,
                 font=("Microsoft YaHei", 8)).pack(side=tk.LEFT)
        self._recoil_lbl = tk.Label(row2, text="关", bg=BG_PANEL, fg=RED,
                                    font=("Microsoft YaHei", 8, "bold"))
        self._recoil_lbl.pack(side=tk.LEFT)

        tk.Frame(left, bg=BG_SECTION, height=1).pack(fill=tk.X, padx=6, pady=6)

        # 瞄准位置
        tk.Label(left, text="▶ 瞄准位置", bg=BG_PANEL, fg=WHITE,
                 font=("Microsoft YaHei", 9, "bold")).pack(anchor="w", padx=8)

        pos = tk.Frame(left, bg=BG_PANEL)
        pos.pack(fill=tk.X, padx=8, pady=4)

        tk.Label(pos, text="锁X轴", bg=BG_PANEL, fg=WHITE,
                 font=("Microsoft YaHei", 8)).pack(anchor="w")
        xrow = tk.Frame(pos, bg=BG_PANEL)
        xrow.pack(fill=tk.X)
        tk.Label(xrow, text="位置:", bg=BG_PANEL, fg=GRAY, font=("Consolas", 8)).pack(side=tk.LEFT)
        self._x_lbl = tk.Label(xrow, text="0.500", bg=BG_PANEL, fg=YELLOW, font=("Consolas", 8, "bold"))
        self._x_lbl.pack(side=tk.LEFT)

        tk.Label(pos, text="Y轴偏移", bg=BG_PANEL, fg=WHITE,
                 font=("Microsoft YaHei", 8)).pack(anchor="w", pady=(4, 0))
        yrow = tk.Frame(pos, bg=BG_PANEL)
        yrow.pack(fill=tk.X)
        tk.Label(yrow, text="位置:", bg=BG_PANEL, fg=GRAY, font=("Consolas", 8)).pack(side=tk.LEFT)
        self._y_lbl = tk.Label(yrow, text="0.430", bg=BG_PANEL, fg=YELLOW, font=("Consolas", 8, "bold"))
        self._y_lbl.pack(side=tk.LEFT)

        # 人体示意图
        body_cv = tk.Canvas(left, bg=BG_PANEL, width=165, height=180,
                            highlightthickness=0)
        body_cv.pack(padx=10, pady=6)
        self._draw_body(body_cv)

        # 快捷键说明
        tk.Frame(left, bg=BG_SECTION, height=1).pack(fill=tk.X, padx=6, pady=4)
        tk.Label(left, text="F1 自瞄  F2 压枪  END 退出",
                 bg=BG_PANEL, fg=GRAY, font=("Consolas", 8)).pack(padx=8)

    def _draw_body(self, cv: tk.Canvas):
        """绘制人体瞄准示意图（简化线条风格）"""
        cx = 82
        col = "#00cc66"
        # 头
        cv.create_oval(cx - 16, 8, cx + 16, 40, outline=col, width=2)
        # 颈
        cv.create_line(cx, 40, cx, 55, fill=col, width=2)
        # 躯干
        cv.create_rectangle(cx - 22, 55, cx + 22, 110, outline=col, width=2)
        # 左臂
        cv.create_line(cx - 22, 62, cx - 38, 92, fill=col, width=2)
        # 右臂
        cv.create_line(cx + 22, 62, cx + 38, 92, fill=col, width=2)
        # 左腿
        cv.create_line(cx - 10, 110, cx - 16, 158, fill=col, width=2)
        # 右腿
        cv.create_line(cx + 10, 110, cx + 16, 158, fill=col, width=2)
        # 瞄准圈（头部）
        cv.create_oval(cx - 24, 4, cx + 24, 44, outline="#ff4444", width=1, dash=(4, 2))
        # 准心
        cv.create_line(cx - 10, 24, cx + 10, 24, fill="#ff4444", width=1)
        cv.create_line(cx, 14, cx, 34, fill="#ff4444", width=1)
        # 标注
        cv.create_text(cx, 170, text="IDS键切换骨骼", fill=GRAY, font=("Microsoft YaHei", 7))

    # ── 右侧设置面板 ───────────────────────────────────────────

    def _build_right(self, parent):
        right = tk.Frame(parent, bg=BG_MAIN)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 顶部：自动扳机栏
        self._build_trigger_bar(right)

        # Tab 按钮行
        tab_btn_row = tk.Frame(right, bg=BG_MAIN)
        tab_btn_row.pack(fill=tk.X, pady=(4, 0))

        self._tab_content = tk.Frame(right, bg=BG_PANEL)
        self._tab_content.pack(fill=tk.BOTH, expand=True)

        self._tab_frames: dict[str, tk.Frame] = {}
        self._tab_btns: dict[str, tk.Button] = {}

        for name in ["参数设置", "硬件设置", "免费声明"]:
            btn = tk.Button(
                tab_btn_row, text=f"▶ {name}",
                bg=BG_SECTION, fg=WHITE,
                relief=tk.FLAT, bd=0,
                font=("Microsoft YaHei", 9),
                padx=10, pady=4,
                activebackground=RED, activeforeground=WHITE,
                command=lambda n=name: self._show_tab(n)
            )
            btn.pack(side=tk.LEFT, padx=(0, 2))
            self._tab_btns[name] = btn

        # 构建各页
        self._build_params_tab()
        self._build_hardware_tab()
        self._build_about_tab()
        self._show_tab("参数设置")

        # 底部控制栏
        self._build_bottom(right)

    def _build_trigger_bar(self, parent):
        bar = tk.Frame(parent, bg=BG_SECTION, height=30)
        bar.pack(fill=tk.X)
        bar.pack_propagate(False)

        # 自动扳机开关
        tk.Checkbutton(bar, text="自动扳机", variable=self.auto_trig_var,
                       bg=BG_SECTION, fg=WHITE, selectcolor=BG_MAIN,
                       font=("Microsoft YaHei", 8)).pack(side=tk.LEFT, padx=10)

        tk.Label(bar, text="持续时间", bg=BG_SECTION, fg=WHITE,
                 font=("Microsoft YaHei", 8)).pack(side=tk.LEFT)
        tk.Entry(bar, textvariable=self.trig_delay_var, width=5,
                 bg=BG_ENTRY, fg=YELLOW, font=("Consolas", 9),
                 relief=tk.FLAT, insertbackground=WHITE).pack(side=tk.LEFT, padx=4)
        tk.Label(bar, text="ms", bg=BG_SECTION, fg=GRAY,
                 font=("Microsoft YaHei", 8)).pack(side=tk.LEFT)

        # 右侧：扳机键选择（占位）
        tk.Label(bar, text="扳机键  鼠标键▼", bg=BG_SECTION, fg=GRAY,
                 font=("Microsoft YaHei", 8)).pack(side=tk.RIGHT, padx=10)

    def _show_tab(self, name: str):
        for n, f in self._tab_frames.items():
            f.pack_forget()
        self._tab_frames[name].pack(fill=tk.BOTH, expand=True)
        for n, btn in self._tab_btns.items():
            btn.config(bg=RED if n == name else BG_SECTION)

    # ── 参数设置标签页 ─────────────────────────────────────────

    def _build_params_tab(self):
        frame = tk.Frame(self._tab_content, bg=BG_PANEL)
        self._tab_frames["参数设置"] = frame

        # ─ 综合设置 ─
        self._section_hdr(frame, "综合设置  SETTINGS")

        sg = tk.Frame(frame, bg=BG_PANEL)
        sg.pack(fill=tk.X, padx=8, pady=4)

        # 行1：瞄准键1 + GPU
        r1 = tk.Frame(sg, bg=BG_PANEL)
        r1.pack(fill=tk.X, pady=2)

        tk.Label(r1, text="瞄准键1", bg=BG_PANEL, fg=WHITE,
                 font=("Microsoft YaHei", 8), width=7, anchor="w").pack(side=tk.LEFT)
        ttk.Combobox(r1, textvariable=self.aim_key_var, style="Dark.TCombobox",
                     values=["鼠标右键", "鼠标左键", "鼠标中键", "Alt键"],
                     width=9, state="readonly").pack(side=tk.LEFT, padx=4)

        tk.Checkbutton(r1, text="持续", bg=BG_PANEL, fg=WHITE,
                       selectcolor=BG_MAIN, font=("Microsoft YaHei", 8),
                       variable=tk.BooleanVar()).pack(side=tk.LEFT, padx=6)

        tk.Label(r1, text="显卡GPU选择", bg=BG_PANEL, fg=WHITE,
                 font=("Microsoft YaHei", 8)).pack(side=tk.LEFT, padx=(20, 4))
        ttk.Combobox(r1, textvariable=self.gpu_var, style="Dark.TCombobox",
                     values=self._cuda_devices,
                     width=28, state="readonly").pack(side=tk.LEFT)

        # 行2：瞄准键2 + 置信度
        r2 = tk.Frame(sg, bg=BG_PANEL)
        r2.pack(fill=tk.X, pady=2)

        tk.Label(r2, text="瞄准键2", bg=BG_PANEL, fg=WHITE,
                 font=("Microsoft YaHei", 8), width=7, anchor="w").pack(side=tk.LEFT)
        _key2 = tk.StringVar(value="鼠标左键")
        ttk.Combobox(r2, textvariable=_key2, style="Dark.TCombobox",
                     values=["鼠标右键", "鼠标左键", "鼠标中键"],
                     width=9, state="readonly").pack(side=tk.LEFT, padx=4)

        tk.Checkbutton(r2, text="持续", bg=BG_PANEL, fg=WHITE,
                       selectcolor=BG_MAIN, font=("Microsoft YaHei", 8),
                       variable=tk.BooleanVar()).pack(side=tk.LEFT, padx=6)

        tk.Label(r2, text="置信度", bg=BG_PANEL, fg=WHITE,
                 font=("Microsoft YaHei", 8)).pack(side=tk.LEFT, padx=(20, 4))

        _conf_lbl = tk.Label(r2, text="0.60", bg=BG_PANEL, fg=YELLOW,
                             font=("Consolas", 8), width=4)
        _conf_lbl.pack(side=tk.RIGHT, padx=4)
        tk.Scale(r2, variable=self.conf_var, from_=0.1, to=1.0, resolution=0.01,
                 orient=tk.HORIZONTAL, length=130,
                 bg=BG_PANEL, fg=WHITE, troughcolor=TROUGH,
                 highlightthickness=0, bd=0, showvalue=False,
                 command=lambda v: _conf_lbl.config(text=f"{float(v):.2f}")
                 ).pack(side=tk.RIGHT)

        # ─ 丝滑拉枪算法 ─
        self._section_hdr(frame, "丝滑拉枪算法")

        sf = tk.Frame(frame, bg=BG_PANEL)
        sf.pack(fill=tk.X, padx=8, pady=3)

        self._smooth_on = tk.BooleanVar(value=True)
        tk.Checkbutton(sf, text="启用", variable=self._smooth_on,
                       bg=BG_PANEL, fg=WHITE, selectcolor=BG_MAIN,
                       font=("Microsoft YaHei", 8)).pack(side=tk.LEFT)

        tk.Label(sf, text="分段式算法", bg=BG_PANEL, fg=WHITE,
                 font=("Microsoft YaHei", 8)).pack(side=tk.LEFT, padx=(12, 4))
        _seg_v = tk.StringVar(value="10")
        tk.Entry(sf, textvariable=_seg_v, width=4,
                 bg=BG_ENTRY, fg=YELLOW, font=("Consolas", 9),
                 relief=tk.FLAT, insertbackground=WHITE).pack(side=tk.LEFT)

        tk.Label(sf, text="FOV算法  X", bg=BG_PANEL, fg=WHITE,
                 font=("Microsoft YaHei", 8)).pack(side=tk.LEFT, padx=(14, 2))
        tk.Entry(sf, textvariable=self.fov_x_var, width=6,
                 bg=BG_ENTRY, fg=YELLOW, font=("Consolas", 9),
                 relief=tk.FLAT, insertbackground=WHITE).pack(side=tk.LEFT)
        tk.Label(sf, text="Y", bg=BG_PANEL, fg=WHITE,
                 font=("Microsoft YaHei", 8)).pack(side=tk.LEFT, padx=(6, 2))
        tk.Entry(sf, textvariable=self.fov_y_var, width=6,
                 bg=BG_ENTRY, fg=YELLOW, font=("Consolas", 9),
                 relief=tk.FLAT, insertbackground=WHITE).pack(side=tk.LEFT)

        # ─ 曲线或者硬锁 ─
        self._section_hdr(frame, "曲线或者硬锁")

        af = tk.Frame(frame, bg=BG_PANEL)
        af.pack(fill=tk.X, padx=8, pady=3)

        tk.Radiobutton(af, text="贝塞尔曲线", variable=self.algo_var, value="bezier",
                       bg=BG_PANEL, fg=WHITE, selectcolor=BG_MAIN,
                       font=("Microsoft YaHei", 8)).pack(side=tk.LEFT)
        _bz_v = tk.StringVar(value="1.5266")
        tk.Entry(af, textvariable=_bz_v, width=7,
                 bg=BG_ENTRY, fg=YELLOW, font=("Consolas", 9),
                 relief=tk.FLAT, insertbackground=WHITE).pack(side=tk.LEFT, padx=6)

        tk.Radiobutton(af, text="硬锁算法（速度最快）", variable=self.algo_var, value="hard",
                       bg=BG_PANEL, fg=WHITE, selectcolor=RED,
                       font=("Microsoft YaHei", 8)).pack(side=tk.LEFT, padx=(8, 0))
        _hd_v = tk.StringVar(value="1.04")
        tk.Entry(af, textvariable=_hd_v, width=5,
                 bg=BG_ENTRY, fg=YELLOW, font=("Consolas", 9),
                 relief=tk.FLAT, insertbackground=WHITE).pack(side=tk.LEFT, padx=4)

        # ─ 参数滑块区 ─
        sliders_outer = tk.Frame(frame, bg=BG_PANEL)
        sliders_outer.pack(fill=tk.X, padx=8, pady=4)

        slider_cfgs = [
            ("鼠标移动",  self.smooth_var,    0.5,  8.0,  0.01, "2.50", True),
            ("瞄准1范围", self.aim_range_var, 30,   800,  1,    "141",  False),
            ("瞄准2范围", tk.DoubleVar(value=141), 30, 800, 1, "141",  False),
            ("预测算法",  self.ema_alpha_var,  0.0,  1.0,  0.01, "0.44", True),
            ("压枪力度",  self.recoil_y_var,   0.0, 10.0,  0.1,  "3.0",  True),
        ]

        for label, var, from_, to, res, default, is_float in slider_cfgs:
            row = tk.Frame(sliders_outer, bg=BG_PANEL)
            row.pack(fill=tk.X, pady=1)

            tk.Label(row, text=label, bg=BG_PANEL, fg=WHITE,
                     font=("Microsoft YaHei", 8), width=8, anchor="w").pack(side=tk.LEFT)

            val_lbl = tk.Label(row, text=default, bg=BG_PANEL, fg=YELLOW,
                               font=("Consolas", 8), width=5, anchor="e")
            val_lbl.pack(side=tk.RIGHT, padx=4)

            fmt = (lambda v, lbl=val_lbl, f=is_float:
                   lbl.config(text=f"{float(v):.2f}" if f else str(int(float(v)))))

            tk.Scale(row, variable=var, from_=from_, to=to, resolution=res,
                     orient=tk.HORIZONTAL,
                     bg=BG_PANEL, fg=WHITE, troughcolor=TROUGH,
                     highlightthickness=0, bd=0, showvalue=False,
                     command=fmt).pack(side=tk.LEFT, fill=tk.X, expand=True)

    # ── 硬件设置标签页 ─────────────────────────────────────────

    def _build_hardware_tab(self):
        frame = tk.Frame(self._tab_content, bg=BG_PANEL)
        self._tab_frames["硬件设置"] = frame

        self._section_hdr(frame, "硬件设置")

        hw = tk.Frame(frame, bg=BG_PANEL)
        hw.pack(fill=tk.X, padx=16, pady=12)

        rows = [
            ("推理模型",   "entry",    self.model_var,  38),
            ("推理精度",   "check",    self.fp16_var,   "FP16 半精度（更快）"),
        ]

        for i, row_cfg in enumerate(rows):
            label = row_cfg[0]
            kind  = row_cfg[1]
            tk.Label(hw, text=label, bg=BG_PANEL, fg=WHITE,
                     font=("Microsoft YaHei", 9), width=10, anchor="w"
                     ).grid(row=i, column=0, sticky="w", pady=6)
            if kind == "entry":
                var, width = row_cfg[2], row_cfg[3]
                tk.Entry(hw, textvariable=var, width=width,
                         bg=BG_ENTRY, fg=YELLOW, font=("Consolas", 9),
                         relief=tk.FLAT, insertbackground=WHITE
                         ).grid(row=i, column=1, padx=8, pady=6, sticky="w")
            elif kind == "check":
                var, text = row_cfg[2], row_cfg[3]
                tk.Checkbutton(hw, text=text, variable=var,
                               bg=BG_PANEL, fg=WHITE, selectcolor=BG_MAIN,
                               font=("Microsoft YaHei", 9)
                               ).grid(row=i, column=1, padx=8, sticky="w")

        # 截图范围
        tk.Label(hw, text="截图范围", bg=BG_PANEL, fg=WHITE,
                 font=("Microsoft YaHei", 9), width=10, anchor="w"
                 ).grid(row=2, column=0, sticky="w", pady=6)
        sz = tk.Frame(hw, bg=BG_PANEL)
        sz.grid(row=2, column=1, padx=8, sticky="w")
        tk.Radiobutton(sz, text="320（低配）", variable=self.fov_size_var, value=320,
                       bg=BG_PANEL, fg=WHITE, selectcolor=BG_MAIN,
                       font=("Microsoft YaHei", 9)).pack(side=tk.LEFT)
        tk.Radiobutton(sz, text="640（高配）", variable=self.fov_size_var, value=640,
                       bg=BG_PANEL, fg=WHITE, selectcolor=BG_MAIN,
                       font=("Microsoft YaHei", 9)).pack(side=tk.LEFT, padx=10)

        # 死区 / EMA 细参
        self._section_hdr(frame, "高级参数")
        adv = tk.Frame(frame, bg=BG_PANEL)
        adv.pack(fill=tk.X, padx=16, pady=6)

        adv_sliders = [
            ("EMA 系数",   self.ema_alpha_var,  0.0, 1.0, 0.01, True),
            ("死区像素",   self.dead_zone_var,  0.0, 20.0, 0.5,  False),
            ("Y轴头部偏移",tk.DoubleVar(value=0.75), 0.0, 1.0, 0.01, True),
        ]
        for label, var, from_, to, res, is_float in adv_sliders:
            row = tk.Frame(adv, bg=BG_PANEL)
            row.pack(fill=tk.X, pady=1)
            tk.Label(row, text=label, bg=BG_PANEL, fg=WHITE,
                     font=("Microsoft YaHei", 8), width=12, anchor="w").pack(side=tk.LEFT)
            val_lbl = tk.Label(row, text=f"{var.get():.2f}", bg=BG_PANEL, fg=YELLOW,
                               font=("Consolas", 8), width=5, anchor="e")
            val_lbl.pack(side=tk.RIGHT, padx=4)
            fmt = lambda v, lbl=val_lbl, f=is_float: lbl.config(
                text=f"{float(v):.2f}" if f else str(int(float(v))))
            tk.Scale(row, variable=var, from_=from_, to=to, resolution=res,
                     orient=tk.HORIZONTAL, bg=BG_PANEL, fg=WHITE, troughcolor=TROUGH,
                     highlightthickness=0, bd=0, showvalue=False,
                     command=fmt).pack(side=tk.LEFT, fill=tk.X, expand=True)

    # ── 免费声明标签页 ─────────────────────────────────────────

    def _build_about_tab(self):
        frame = tk.Frame(self._tab_content, bg=BG_PANEL)
        self._tab_frames["免费声明"] = frame

        self._section_hdr(frame, "免费声明")
        tk.Label(frame,
                 text=(
                     "本软件完全免费，仅供学习研究 YOLOv8 目标检测技术使用。\n\n"
                     "请勿将本软件用于任何商业用途或违反游戏服务条款的行为。\n\n"
                     "使用本软件产生的任何后果由使用者自行承担。\n\n"
                     "作者不对任何滥用行为负责。\n\n"
                     "项目基于 YOLOv8 + PyTorch + ONNX Runtime 构建。"
                 ),
                 bg=BG_PANEL, fg=GRAY,
                 font=("Microsoft YaHei", 10),
                 justify=tk.LEFT, wraplength=520
                 ).pack(padx=24, pady=24, anchor="w")

    # ── 底部控制栏 ─────────────────────────────────────────────

    def _build_bottom(self, parent):
        bottom = tk.Frame(parent, bg=BG_MAIN)
        bottom.pack(fill=tk.X, pady=4)

        # 配置范围选择
        left_btm = tk.Frame(bottom, bg=BG_MAIN)
        left_btm.pack(side=tk.LEFT, padx=10)

        tk.Radiobutton(left_btm, text="低配置320范围", variable=self.fov_size_var, value=320,
                       bg=BG_MAIN, fg=WHITE, selectcolor=BG_MAIN,
                       font=("Microsoft YaHei", 9)).pack(side=tk.LEFT)
        tk.Radiobutton(left_btm, text="高配置640范围", variable=self.fov_size_var, value=640,
                       bg=BG_MAIN, fg=WHITE, selectcolor=BG_MAIN,
                       font=("Microsoft YaHei", 9)).pack(side=tk.LEFT, padx=8)

        tk.Button(left_btm, text="快捷键说明",
                  bg=BG_SECTION, fg=WHITE, relief=tk.FLAT,
                  font=("Microsoft YaHei", 8), padx=8, pady=2,
                  activebackground=RED, activeforeground=WHITE,
                  command=self._show_hotkeys).pack(side=tk.LEFT, padx=6)

        # START / STOP 按钮
        self._start_btn = tk.Button(
            bottom,
            text="▶ 启动推理\n    START",
            bg=ORANGE, fg=WHITE,
            relief=tk.FLAT, bd=0,
            font=("Microsoft YaHei", 11, "bold"),
            width=11, height=2,
            activebackground="#d35400", activeforeground=WHITE,
            command=self._toggle_infer
        )
        self._start_btn.pack(side=tk.RIGHT, padx=12)

    # ── 辅助方法 ───────────────────────────────────────────────

    def _section_hdr(self, parent, text: str):
        hdr = tk.Frame(parent, bg=BG_SECTION, height=26)
        hdr.pack(fill=tk.X, pady=(6, 2))
        hdr.pack_propagate(False)
        tk.Frame(hdr, bg=RED, width=4).pack(side=tk.LEFT, fill=tk.Y)
        tk.Label(hdr, text=f"  {text}", bg=BG_SECTION, fg=WHITE,
                 font=("Microsoft YaHei", 9, "bold")).pack(side=tk.LEFT)

    def _toggle_infer(self):
        if self._state and self._state.running:
            self._state.running = False
            self._start_btn.config(text="▶ 启动推理\n    START", bg=ORANGE)
            if self._on_stop:
                self._on_stop()
        else:
            if self._state:
                self._state.running = True
            self._start_btn.config(text="■ 停止推理\n    STOP", bg="#c0392b")
            if self._on_start:
                t = threading.Thread(target=self._on_start, daemon=True)
                t.start()

    def _show_hotkeys(self):
        win = tk.Toplevel(self._root)
        win.title("快捷键说明")
        win.configure(bg=BG_MAIN)
        win.geometry("280x180+{}+{}".format(
            self._root.winfo_x() + 200, self._root.winfo_y() + 180))
        win.grab_set()
        win.resizable(False, False)

        for key, desc in [("F1", "切换自瞄 开/关"),
                           ("F2", "切换压枪 开/关"),
                           ("F3", "切换自动扳机"),
                           ("END", "紧急退出程序")]:
            row = tk.Frame(win, bg=BG_MAIN)
            row.pack(fill=tk.X, padx=20, pady=5)
            tk.Label(row, text=f" {key} ", bg=RED, fg=WHITE,
                     font=("Consolas", 10, "bold")).pack(side=tk.LEFT)
            tk.Label(row, text=f"  {desc}", bg=BG_MAIN, fg=WHITE,
                     font=("Microsoft YaHei", 9)).pack(side=tk.LEFT)
