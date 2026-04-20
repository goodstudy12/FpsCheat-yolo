# 鼠标移动
import math
import time
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

# atan 平滑系数
_K = 4.07 * (1.0 / LOCK_SMOOTH)

# 内部状态
_last_distance = 0.0         # 跳变检测用
_ema_dx = 0.0                # EMA 滤波后的目标偏移 X
_ema_dy = 0.0                # EMA 滤波后的目标偏移 Y
_ema_initialized = False     # EMA 是否已初始化
_residual_x = 0.0            # 亚像素残差累积 X
_residual_y = 0.0            # 亚像素残差累积 Y


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
        # 目标切换，重置 EMA 状态以快速锁定新目标
        _ema_initialized = False
        _residual_x = 0.0
        _residual_y = 0.0
        _last_distance = distance
        return
    _last_distance = distance

    # 3. EMA 滤波：平滑检测框噪声
    if not _ema_initialized:
        _ema_dx = float(dx)
        _ema_dy = float(dy)
        _ema_initialized = True
    else:
        _ema_dx = EMA_ALPHA * dx + (1.0 - EMA_ALPHA) * _ema_dx
        _ema_dy = EMA_ALPHA * dy + (1.0 - EMA_ALPHA) * _ema_dy

    # 使用滤波后的坐标
    sdx = _ema_dx
    sdy = _ema_dy
    smooth_dist = math.sqrt(sdx * sdx + sdy * sdy)

    # 4. 死区：已对准时不移动
    if smooth_dist < DEAD_ZONE:
        _residual_x = 0.0
        _residual_y = 0.0
        return

    # 5. atan 非线性平滑
    move_x_f = _K / LOCK_SEN * math.atan(sdx / 640) * 640
    move_y_f = _K / LOCK_SEN * math.atan(sdy / 640) * 640

    # 6. 距离自适应增益：接近目标时二次衰减，防止过冲震荡
    if smooth_dist < CLOSE_RANGE:
        # ratio 从 0→1，gain 从 ~0→1 平滑过渡
        ratio = smooth_dist / CLOSE_RANGE
        gain = ratio * ratio  # 二次衰减：越近速度降得越狠
        gain = max(gain, 0.1)  # 保底 10%，避免完全停滞
        move_x_f *= gain
        move_y_f *= gain

    # 7. 亚像素残差累积：浮点精度移动，消除 int 截断抖动
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
