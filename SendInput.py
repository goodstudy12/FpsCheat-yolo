# 鼠标移动
import math
import time
from collections import deque
from ctypes import windll, c_long, c_ulong, Structure, Union, c_int, POINTER, sizeof

LONG = c_long
DWORD = c_ulong
ULONG_PTR = POINTER(DWORD)


# https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-sendinput

class MOUSEINPUT(Structure):
    _fields_ = (('dx', LONG),
                ('dy', LONG),
                ('mouseData', DWORD),
                ('dwFlags', DWORD),
                ('time', DWORD),
                ('dwExtraInfo', ULONG_PTR))


class _INPUTunion(Union):
    _fields_ = (('mi', MOUSEINPUT), ('mi', MOUSEINPUT))


class INPUT(Structure):
    _fields_ = (('type', DWORD),
                ('union', _INPUTunion))


def SendInput(*inputs):
    nInputs = len(inputs)
    LPINPUT = INPUT * nInputs
    pInputs = LPINPUT(*inputs)
    cbSize = c_int(sizeof(INPUT))
    return windll.user32.SendInput(nInputs, pInputs, cbSize)


def Input(structure):
    return INPUT(0, _INPUTunion(mi=structure))


def MouseInput(flags, x, y, data):
    return MOUSEINPUT(x, y, data, flags, 0, None)


def Mouse(flags, x=0, y=0, data=0):
    return Input(MouseInput(flags, x, y, data))


# 移动鼠标
def mouse_xy(x, y):
    return SendInput(Mouse(0x0001, x, y))


# 按下鼠标(1为左键/2为右键)
def mouse_down(key=1):
    if key == 1:
        return SendInput(Mouse(0x0002))
    elif key == 2:
        return SendInput(Mouse(0x0008))


# 松开鼠标(1为左键/2为右键)
def mouse_up(key=1):
    if key == 1:
        return SendInput(Mouse(0x0004))
    elif key == 2:
        return SendInput(Mouse(0x0010))


# ============ 平滑自瞄配置 ============
LOCK_SMOOTH = 2.5            # 平滑系数，越大越平滑（最低1.0）
LOCK_SEN = 1.0               # 游戏灵敏度，桌面测试用1.0，游戏中按实际灵敏度填
OFFSET_Y = 0.75              # Y轴头部偏移比例（0=中心，1=顶部）
JUMP_THRESHOLD = 100         # 跳变检测阈值（像素），超过则判定目标切换
DEAD_ZONE = 3.0              # 死区半径（像素），在此范围内不移动
EMA_ALPHA = 0.4              # EMA 平滑系数（0~1），越小越平滑、越大越跟手
CLOSE_RANGE = 50.0           # 近距离阈值（像素），低于此值启用二次衰减防过冲

# ── 运动预测配置 ──
PREDICT_STRENGTH = 0.5       # 预测强度（0=关闭，1.0=全预测），越大越超前
PREDICT_HISTORY  = 8         # 历史缓冲帧数，用于速度估计

# atan 平滑系数
_K = 4.07 * (1.0 / LOCK_SMOOTH)

# 内部状态
_last_distance = 0.0         # 跳变检测用
_ema_dx = 0.0                # EMA 滤波后的目标偏移 X
_ema_dy = 0.0                # EMA 滤波后的目标偏移 Y
_ema_initialized = False     # EMA 是否已初始化
_residual_x = 0.0            # 亚像素残差累积 X
_residual_y = 0.0            # 亚像素残差累积 Y

# 运动预测内部状态
_pos_history = deque(maxlen=PREDICT_HISTORY)   # 存储 (timestamp, dx, dy)
_pred_vx = 0.0               # 估计速度 X（像素/秒）
_pred_vy = 0.0               # 估计速度 Y（像素/秒）


def _estimate_velocity():
    """
    基于加权最小二乘估计目标速度。
    近帧权重指数递增，抑制历史帧的噪声干扰。
    返回 (vx, vy) 单位：像素/秒。
    """
    global _pred_vx, _pred_vy

    n = len(_pos_history)
    if n < 3:
        _pred_vx = 0.0
        _pred_vy = 0.0
        return

    entries = list(_pos_history)
    t0 = entries[0][0]

    # 加权最小二乘：w_i = exp(decay * i)，近帧权重更高
    decay = 0.5
    sum_w = 0.0
    sum_wt = 0.0
    sum_wt2 = 0.0
    sum_wx = 0.0
    sum_wy = 0.0
    sum_wtx = 0.0
    sum_wty = 0.0

    for i, (ts, px, py) in enumerate(entries):
        t = ts - t0
        w = math.exp(decay * i)
        sum_w   += w
        sum_wt  += w * t
        sum_wt2 += w * t * t
        sum_wx  += w * px
        sum_wy  += w * py
        sum_wtx += w * t * px
        sum_wty += w * t * py

    denom = sum_w * sum_wt2 - sum_wt * sum_wt
    if abs(denom) < 1e-9:
        _pred_vx = 0.0
        _pred_vy = 0.0
        return

    # 斜率即速度
    _pred_vx = (sum_w * sum_wtx - sum_wt * sum_wx) / denom
    _pred_vy = (sum_w * sum_wty - sum_wt * sum_wy) / denom


def _predict_position(dx, dy):
    """
    根据当前偏移和估计速度，预测未来位置。
    预测量 = 速度 × 帧间隔 × 预测强度。
    """
    if PREDICT_STRENGTH <= 0.0 or len(_pos_history) < 3:
        return dx, dy

    # 用最近两帧间隔估计帧率
    entries = list(_pos_history)
    dt = entries[-1][0] - entries[-2][0]
    if dt <= 0:
        dt = 0.016  # 默认 ~60fps

    # 外推：预测 1 帧后的位置
    pred_dx = dx + _pred_vx * dt * PREDICT_STRENGTH
    pred_dy = dy + _pred_vy * dt * PREDICT_STRENGTH

    return pred_dx, pred_dy


def _reset_prediction():
    """目标切换时重置预测状态"""
    global _pred_vx, _pred_vy
    _pos_history.clear()
    _pred_vx = 0.0
    _pred_vy = 0.0


def smooth_move(dx, dy, box_h=0):
    """
    平滑自瞄移动：EMA 滤波 + atan 非线性 + 距离自适应增益 + 亚像素累积。

    三层防抖机制：
    1. EMA 滤波：平滑 YOLO 检测框的帧间抖动噪声
    2. 距离自适应增益：接近目标时二次衰减速度，防止过冲震荡
    3. 亚像素残差累积：浮点累积移动量，消除 int 截断导致的微抖

    参数:
        dx: 目标中心相对屏幕中心的 X 偏移（像素）
        dy: 目标中心相对屏幕中心的 Y 偏移（像素）
        box_h: 检测框高度（像素），用于计算头部偏移
    """
    global _last_distance, _ema_dx, _ema_dy, _ema_initialized
    global _residual_x, _residual_y

    # 1. 头部偏移：瞄准点从检测框中心向上偏移（瞄头）
    if box_h > 0:
        dy = dy - box_h * OFFSET_Y * 0.5

    # 2. 跳变检测
    distance = math.sqrt(dx * dx + dy * dy)
    if distance > _last_distance + JUMP_THRESHOLD and _last_distance > 0:
        # 目标切换，重置 EMA 和预测状态以快速锁定新目标
        _ema_initialized = False
        _residual_x = 0.0
        _residual_y = 0.0
        _reset_prediction()
        _last_distance = distance
        return
    _last_distance = distance

    # 3. 记录位置历史并估计速度
    now = time.time()
    _pos_history.append((now, float(dx), float(dy)))
    _estimate_velocity()

    # 4. 运动预测：将当前位置外推到未来
    pred_dx, pred_dy = _predict_position(float(dx), float(dy))

    # 5. EMA 滤波：对预测后的坐标平滑检测框噪声
    if not _ema_initialized:
        _ema_dx = pred_dx
        _ema_dy = pred_dy
        _ema_initialized = True
    else:
        _ema_dx = EMA_ALPHA * pred_dx + (1.0 - EMA_ALPHA) * _ema_dx
        _ema_dy = EMA_ALPHA * pred_dy + (1.0 - EMA_ALPHA) * _ema_dy

    # 使用滤波后的坐标
    sdx = _ema_dx
    sdy = _ema_dy
    smooth_dist = math.sqrt(sdx * sdx + sdy * sdy)

    # 6. 死区：已对准时不移动
    if smooth_dist < DEAD_ZONE:
        _residual_x = 0.0
        _residual_y = 0.0
        return

    # 7. atan 非线性平滑
    move_x_f = _K / LOCK_SEN * math.atan(sdx / 640) * 640
    move_y_f = _K / LOCK_SEN * math.atan(sdy / 640) * 640

    # 8. 距离自适应增益：接近目标时二次衰减，防止过冲震荡
    if smooth_dist < CLOSE_RANGE:
        # ratio 从 0→1，gain 从 ~0→1 平滑过渡
        ratio = smooth_dist / CLOSE_RANGE
        gain = ratio * ratio  # 二次衰减：越近速度降得越狠
        gain = max(gain, 0.1)  # 保底 10%，避免完全停滞
        move_x_f *= gain
        move_y_f *= gain

    # 9. 亚像素残差累积：浮点精度移动，消除 int 截断抖动
    move_x_f += _residual_x
    move_y_f += _residual_y
    move_x = int(round(move_x_f))
    move_y = int(round(move_y_f))
    _residual_x = move_x_f - move_x
    _residual_y = move_y_f - move_y

    if move_x == 0 and move_y == 0:
        return

    mouse_xy(move_x, move_y)


# ============ 压枪（反后坐力补偿）配置 ============
RECOIL_STRENGTH_Y = 3.0       # 基础垂直补偿强度（像素/帧），越大压枪力度越强
RECOIL_STRENGTH_X = 0.0       # 水平补偿强度（像素/帧），部分武器有水平后坐力时调整
RECOIL_RAMP_TIME = 0.8        # 后坐力爬升时间（秒），从 0 线性增长到满强度
RECOIL_MAX_MULT = 2.0         # 最大倍率：持续射击时后坐力从 1x 增长到此倍率
RECOIL_SMOOTH = 0.6           # 压枪平滑系数（0~1），越小越平滑

# 压枪内部状态
_recoil_start_time = 0.0      # 本轮射击开始时间
_recoil_active = False         # 上一帧是否在射击
_recoil_residual_x = 0.0      # 亚像素残差 X
_recoil_residual_y = 0.0      # 亚像素残差 Y


def is_left_button_down():
    """检测鼠标左键是否按下（通过 GetAsyncKeyState）"""
    # 0x01 = VK_LBUTTON，返回值最高位为 1 表示当前按下
    return (windll.user32.GetAsyncKeyState(0x01) & 0x8000) != 0


def recoil_compensate():
    """
    压枪补偿：在鼠标左键按住（射击）时，每帧自动向下移动鼠标抵消后坐力。

    特性：
    1. 线性爬升：射击前期补偿较小，持续射击后逐渐增强（模拟后坐力递增）
    2. 亚像素累积：浮点精度计算，避免 int 截断导致补偿不均匀
    3. 松开左键后自动重置状态，下次射击重新开始爬升
    """
    global _recoil_start_time, _recoil_active
    global _recoil_residual_x, _recoil_residual_y

    shooting = is_left_button_down()

    if not shooting:
        # 松开左键，重置状态
        if _recoil_active:
            _recoil_active = False
            _recoil_residual_x = 0.0
            _recoil_residual_y = 0.0
        return

    # 刚开始射击，记录起始时间
    if not _recoil_active:
        _recoil_active = True
        _recoil_start_time = time.time()
        _recoil_residual_x = 0.0
        _recoil_residual_y = 0.0

    # 计算射击持续时间与爬升倍率
    elapsed = time.time() - _recoil_start_time
    if RECOIL_RAMP_TIME > 0:
        # 线性爬升：0s → 1x，RECOIL_RAMP_TIME → RECOIL_MAX_MULT
        ramp = 1.0 + (RECOIL_MAX_MULT - 1.0) * min(elapsed / RECOIL_RAMP_TIME, 1.0)
    else:
        ramp = RECOIL_MAX_MULT

    # 计算本帧补偿量（向下为正 Y）
    comp_y = RECOIL_STRENGTH_Y * ramp * RECOIL_SMOOTH
    comp_x = RECOIL_STRENGTH_X * ramp * RECOIL_SMOOTH

    # 亚像素累积
    comp_x += _recoil_residual_x
    comp_y += _recoil_residual_y
    move_x = int(round(comp_x))
    move_y = int(round(comp_y))
    _recoil_residual_x = comp_x - move_x
    _recoil_residual_y = comp_y - move_y

    if move_x == 0 and move_y == 0:
        return

    mouse_xy(move_x, move_y)
