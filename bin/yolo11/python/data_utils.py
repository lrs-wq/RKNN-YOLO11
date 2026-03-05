# billiard_utils.py
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Union

# ================= 数据类定义 =================

@dataclass
class DetectionBox:
    """存储边界框信息"""
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    # 兼容字典访问方式 obj['box']['x1']
    def __getitem__(self, item):
        if item == 'width': return self.width
        if item == 'height': return self.height
        return getattr(self, item)

@dataclass
class MotionState:
    """存储运动学信息 (速度和加速度)"""
    vx: float = 0.0
    vy: float = 0.0
    ax: float = 0.0
    ay: float = 0.0

    # 兼容字典访问方式 obj['speed']['vx']
    def __getitem__(self, item):
        return getattr(self, item)

@dataclass  
class SafeDetection:
    """
    用于安全存储从 C++ 提取的数据，防止内存非法访问。
    使用 __slots__ 减少内存占用并提高访问速度。
    """
    __slots__ = ['id', 'prob', 'x1', 'y1', 'x2', 'y2']

    def __init__(self, id, prob, x1, y1, x2, y2):
        self.id = int(id)
        self.prob = float(prob)
        self.x1 = float(x1)
        self.y1 = float(y1)
        self.x2 = float(x2)
        self.y2 = float(y2)

@dataclass
class DataObject:
    """主要的对象数据类，替代原来的 obj 字典"""
    frame_id: int
    timestamp: float
    class_id: int
    class_name: str
    confidence: float
    box: DetectionBox
    center: Tuple[float, float]
    speed: MotionState  # 为了保持兼容性，变量名仍叫 speed，实际包含加速度
    fps: float = 0.0

    # 兼容字典访问方式 obj['key']
    def __getitem__(self, item):
        return getattr(self, item)
    
    # 允许 .get() 方法，兼容 dict.get()
    def get(self, item, default=None):
        return getattr(self, item, default)

# ================= 辅助计算函数 =================

def get_center(box):
    # 这里为了兼容，box 可能是 dict 也可能是 DetectionBox 对象
    if isinstance(box, dict):
        x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
    else:
        x1, y1, x2, y2 = box.x1, box.y1, box.x2, box.y2
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def calculate_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def calculate_velocity(curr_id: int, x: float, y: float, prev_obj: Optional[DataObject]):
    """计算速度 vx, vy"""
    if prev_obj is None:
        return 0.0, 0.0
    
    # 统一使用对象属性访问
    prev_id = prev_obj.frame_id
    
    dt = curr_id - prev_id
    if dt < 0: dt += 10000 
    if dt == 0: return 0.0, 0.0

    if dt > 10:
        # 使用对象属性访问 speed
        if prev_obj.speed.vx != 0 or prev_obj.speed.vy != 0:
            return 0.0, 0.0

    dx = x - prev_obj.center[0]
    dy = y - prev_obj.center[1]

    vx = dx / dt
    vy = dy / dt

    return vx, vy

def calculate_accelerated(curr_id: int, vx: float, vy: float, prev_obj: Optional[DataObject]):
    """计算加速度 ax, ay"""
    if prev_obj is None:
        return 0.0, 0.0
    
    prev_id = prev_obj.frame_id
    dt = curr_id - prev_id
    if dt < 0: dt += 10000 
    if dt == 0: return 0.0, 0.0

    if dt > 10:
        if prev_obj.speed.ax != 0 or prev_obj.speed.ay != 0:
            return 0.0, 0.0
            
    dvx = vx - prev_obj.speed.vx
    dvy = vy - prev_obj.speed.vy
    
    ax = dvx / dt
    ay = dvy / dt
    return ax, ay

def rect_to_points(rect: List):
    """将矩形坐标 [xmin, ymin, xmax, ymax] 转换为四个点的数组"""
    xmin, ymin, xmax, ymax = rect
    # 左上、右上、右下、左下
    return np.array([
        [xmin, ymin],    # 左上
        [xmax, ymin],    # 右上
        [xmax, ymax],    # 右下
        [xmin, ymax]     # 左下
    ], dtype=np.float32)

def calculate_iou(box1, box2):
    """
    计算两个矩形框的交并比 (Intersection over Union)
    box 输入格式可以是字典 {'x1':..., 'y1':...} 或对象
    """
    # 统一获取坐标
    def get_coords(b):
        if isinstance(b, dict):
            return b['x1'], b['y1'], b['x2'], b['y2']
        else:
            return b.x1, b.y1, b.x2, b.y2

    b1_x1, b1_y1, b1_x2, b1_y2 = get_coords(box1)
    b2_x1, b2_y1, b2_x2, b2_y2 = get_coords(box2)

    # 计算交集区域坐标
    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)

    # 计算交集面积
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # 计算两个框各自的面积
    area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    # 计算并集面积
    union_area = area1 + area2 - inter_area

    if union_area == 0:
        return 0
    return inter_area / union_area