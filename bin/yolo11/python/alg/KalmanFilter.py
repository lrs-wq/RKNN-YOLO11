import numpy as np
from typing import List, Tuple, Dict, Optional
import sys 
sys.path.append("..") 
from data_utils import DataObject, MotionState, DetectionBox

class KalmanFilter:
    """
    针对台球运动优化的卡尔曼滤波器。
    (保持不变)
    """

    def __init__(self, initial_pos: Tuple[float, float], timestamp: float):
        # [px, py, vx, vy, ax, ay]
        self.x = np.zeros((6, 1))
        self.x[0] = initial_pos[0]
        self.x[1] = initial_pos[1]
        
        self.last_timestamp = timestamp
        
        # 状态转移矩阵 F
        self.F = np.eye(6)
        
        # 观测矩阵 H (只观测 x, y)
        self.H = np.zeros((2, 6))
        self.H[0, 0] = 1
        self.H[1, 1] = 1
        
        # 状态协方差 P
        self.P = np.eye(6)
        self.P[0:2, 0:2] *= 1.0     
        self.P[2:4, 2:4] *= 100.0   
        self.P[4:6, 4:6] *= 1000.0  

        # 测量噪声 R (检测误差)
        self.R = np.eye(2) * 1.0 
        
        # 过程噪声 Q (物理突变)
        self.Q = np.eye(6)
        self.process_noise_scale = 0.3

    def predict(self, timestamp: float) -> None:
        dt = timestamp - self.last_timestamp
        self.last_timestamp = timestamp
        
        if dt <= 0:
            return

        # 更新 F
        dt2_half = 0.5 * dt**2
        self.F[0, 2] = dt; self.F[0, 4] = dt2_half
        self.F[1, 3] = dt; self.F[1, 5] = dt2_half
        self.F[2, 4] = dt
        self.F[3, 5] = dt

        # 更新 Q
        q_pos = self.process_noise_scale * (dt**3) / 3.0
        q_vel = self.process_noise_scale * (dt**2) / 2.0
        q_acc = self.process_noise_scale * dt
        np.fill_diagonal(self.Q, [q_pos, q_pos, q_vel, q_vel, q_acc, q_acc])

        # 预测
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, measurement: Tuple[float, float]) -> None:
        z = np.array([[measurement[0]], [measurement[1]]])
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.x.shape[0])
        self.P = np.dot((I - np.dot(K, self.H)), self.P)

    def get_state(self) -> Tuple[Tuple[float, float], MotionState]:
        center = (float(self.x[0, 0]), float(self.x[1, 0]))
        motion = MotionState(
            vx=float(self.x[2, 0]),
            vy=float(self.x[3, 0]),
            ax=float(self.x[4, 0]),
            ay=float(self.x[5, 0])
        )
        return center, motion

class Tracker:
    """
    多目标台球跟踪与平滑器。
    修改后：
    1. 输入列表可以包含 None。
    2. 严格只处理索引 0 到 15 (共16个球)。
    3. 如果输入是 None，输出也是 None。
    """
    def __init__(self, max_time_gap: float = 0.5):
        self.filters: Dict[int, KalmanFilter] = {}
        self.max_time_gap = max_time_gap

    def process(self, detections: List[Optional[DataObject]]) -> List[Optional[DataObject]]:
        """
        处理一帧中的检测对象。
        Args:
            detections: 长度通常为16的列表，索引对应球号(0-15)，元素可能是None。
        Returns:
            results: 固定长度为16的列表。
        """
        results = []
        
        # 强制遍历前 16 个类别 (0-15)
        for i in range(16):
            # 安全检查：防止输入列表长度不足16
            if i >= len(detections):
                results.append(None)
                continue

            det = detections[i]

            # --- 逻辑修改点：处理 None ---
            if det is None:
                # 如果当前帧没有检测到该球，直接返回 None
                # 注意：这里我们不进行“纯预测”填充，而是严格遵循输入为None输出为None
                # 滤波器状态保持不变，下次出现时会计算 dt 进行更新
                results.append(None)
                continue
            
            # --- 以下为正常的滤波逻辑 ---
            
            # 这里的 track_key 直接使用索引 i (即 class_id)
            track_key = i 
            obs_center = det.center
            timestamp = det.timestamp

            if track_key not in self.filters:
                # 新出现的球：初始化
                self.filters[track_key] = KalmanFilter(obs_center, timestamp)
                results.append(det)
                continue

            kf = self.filters[track_key]

            # 检查时间间隔
            dt = timestamp - kf.last_timestamp
            if dt > self.max_time_gap:
                # 丢失太久，重置
                self.filters[track_key] = KalmanFilter(obs_center, timestamp)
                results.append(det)
                continue

            # 滤波 (Predict -> Update)
            kf.predict(timestamp)
            kf.update(obs_center)

            # 获取结果
            smoothed_center, motion_state = kf.get_state()

            # 修正 Box
            dx = smoothed_center[0] - obs_center[0]
            dy = smoothed_center[1] - obs_center[1]

            new_box = DetectionBox(
                x1=det.box.x1 + dx,
                y1=det.box.y1 + dy,
                x2=det.box.x2 + dx,
                y2=det.box.y2 + dy
            )

            processed_obj = DataObject(
                frame_id=det.frame_id,
                timestamp=det.timestamp,
                class_id=det.class_id,
                class_name=det.class_name,
                confidence=det.confidence,
                box=new_box,
                center=smoothed_center,
                speed=motion_state,
                fps=det.fps
            )
            
            results.append(processed_obj)

        return results

# --- 测试代码 ---
if __name__ == "__main__":
    tracker = Tracker()
    
    # 模拟输入列表，长度必须支持到索引15，中间包含 None
    # 假设我们只关心 0号球(白球) 和 1号球(黄球)，其他位置为 None
    print(f"{'Frame':<5} | {'Idx':<3} | {'Status':<10} | {'Obs X':<10} | {'Smooth X':<10}")
    print("-" * 60)

    for i in range(5):
        t = i * 0.033
        
        # 构造一个长度为 16 的列表，默认全为 None
        current_frame_list: List[Optional[DataObject]] = [None] * 16

        # 1. 设置 0号球 (存在)
        w_x = 100 + 100 * t
        white_ball = DataObject(
            frame_id=i, timestamp=t, class_id=0, class_name="white", confidence=0.9,
            box=DetectionBox(0,0,0,0), center=(w_x, 100), speed=MotionState()
        )
        current_frame_list[0] = white_ball

        # 2. 设置 1号球 (模拟闪烁：第2帧丢失 -> i==2时为None)
        if i != 2:
            y_x = 200 + 50 * t
            yellow_ball = DataObject(
                frame_id=i, timestamp=t, class_id=1, class_name="yellow", confidence=0.9,
                box=DetectionBox(0,0,0,0), center=(y_x, 200), speed=MotionState()
            )
            current_frame_list[1] = yellow_ball
        else:
            # 显式保持为 None
            current_frame_list[1] = None

        # --- 调用处理 ---
        results = tracker.process(current_frame_list)

        # 验证结果长度
        assert len(results) == 16, "Result length must be 16"

        # 打印部分结果
        # 0号球
        res_0 = results[0]
        print(f"{i:<5} | 0   | {'Tracked':<10} | {white_ball.center[0]:<10.1f} | {res_0.center[0]:<10.1f}")
        
        # 1号球
        res_1 = results[1]
        status_1 = "None" if res_1 is None else "Tracked"
        obs_1_val = current_frame_list[1].center[0] if current_frame_list[1] else 0.0
        smooth_1_val = res_1.center[0] if res_1 else 0.0
        print(f"{i:<5} | 1   | {status_1:<10} | {obs_1_val:<10.1f} | {smooth_1_val:<10.1f}")
        
        # 验证其他位置 (比如索引 15) 是否为 None
        assert results[15] is None
    
    print("-" * 60)