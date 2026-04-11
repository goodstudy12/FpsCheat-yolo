# 鼠标移动
import math
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
# 参考 aim_lock_pi.py 的 atan 非线性平滑方案
LOCK_SMOOTH = 1.7            # 平滑系数，越大越平滑（最低1.0）
LOCK_SEN = 1.0               # 游戏灵敏度，桌面测试用1.0，游戏中按实际灵敏度填
OFFSET_Y = 0.75              # Y轴头部偏移比例（0=中心，1=顶部）
JUMP_THRESHOLD = 80          # 跳变检测阈值（像素），超过则判定目标切换

# atan 平滑系数：4.07 是经验常数，将 atan 输出映射到合理的鼠标移动量
_K = 4.07 * (1.0 / LOCK_SMOOTH)

# 跳变检测状态
_last_distance = 0.0


def smooth_move(dx, dy, box_h=0):
    """
    平滑自瞄移动。使用 atan 非线性平滑（参考 aim_lock_pi.py）。

    atan 天然形成 S 曲线：
    - 小偏移 → 几乎线性响应，精确微调
    - 大偏移 → 自动压缩，防止准心飞跳

    参数:
        dx: 目标中心相对屏幕中心的 X 偏移（像素）
        dy: 目标中心相对屏幕中心的 Y 偏移（像素）
        box_h: 检测框高度（像素），用于计算头部偏移
    """
    global _last_distance

    # 1. 头部偏移：瞄准点从检测框中心向上偏移（瞄头）
    if box_h > 0:
        dy = dy - box_h * OFFSET_Y * 0.5

    # 2. 计算到目标的距离（用于跳变检测和死区判断）
    distance = math.sqrt(dx * dx + dy * dy)

    # 3. 跳变检测：距离突增超过阈值，判定目标切换，跳过本帧
    if distance > _last_distance + JUMP_THRESHOLD and _last_distance > 0:
        _last_distance = distance
        return
    _last_distance = distance

    # 4. 死区：已对准时不移动
    if distance < 2:
        return

    # 5. atan 非线性平滑移动
    #    公式: move = k / sen * atan(offset / 640) * 640
    #    - offset / 640 归一化偏移量
    #    - atan() 将大偏移压缩到 [-π/2, π/2]
    #    - * 640 还原到像素尺度
    #    - k / sen 控制最终速度
    move_x = int(_K / LOCK_SEN * math.atan(dx / 640) * 640)
    move_y = int(_K / LOCK_SEN * math.atan(dy / 640) * 640)

    if move_x == 0 and move_y == 0:
        return

    mouse_xy(move_x, move_y)
