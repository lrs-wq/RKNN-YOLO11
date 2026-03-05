import numpy as np
from scipy.optimize import linear_sum_assignment

class KalmanFilter:
    """
    简单的卡尔曼滤波器，用于跟踪边界框状态。
    状态空间: [x, y, a, h, vx, vy, va, vh]
    x, y: 框中心坐标
    a: 宽高比 (w/h)
    h: 高度
    """
    def __init__(self):
        ndim, dt = 4, 1.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement):
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T
        )) + motion_cov
        return mean, covariance

    def project(self, mean, covariance):
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]
        ]
        innovation_cov = np.diag(np.square(std))
        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T
        )) + innovation_cov
        return mean, covariance

    def update(self, mean, covariance, measurement):
        projected_mean, projected_cov = self.project(mean, covariance)
        
        # 计算卡尔曼增益 K = P * H.T * S^-1
        # 为了数值稳定性，我们计算 K.T = S^-1 * H * P
        # projected_cov 是 S (4x4)
        # self._update_mat 是 H (4x8)
        # covariance 是 P (8x8)
        
        # 1. 计算 H * P (结果为 4x8)
        chol_factor = np.linalg.cholesky(projected_cov) # Optional: still compute for reuse if needed, or just use solve
        hp = self._update_mat @ covariance.T
        
        # 2. 求解 S * K.T = H * P
        # 使用 solve(S, hp) 得到 K.T (4x8)
        # 这里直接使用 np.linalg.solve，它会自动处理线性方程组
        kt = np.linalg.solve(projected_cov, hp)
        
        # 3. 转置得到 K (8x4)
        kalman_gain = kt.T
        # --- 修正结束 ---

        innovation = measurement - projected_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T
        ))
        return new_mean, new_covariance

class STrack:
    """单个轨迹对象"""
    def __init__(self, tlwh, score, class_id):
        # tlwh: top_left_x, top_left_y, width, height
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.kalman_filter = None
        self.mean = None
        self.covariance = None
        self.is_activated = False
        self.score = score
        self.class_id = int(class_id) # 台球ID是非常重要的强特征
        self.tracklet_len = 0
        self.state = 1 # 1: Tracked, 2: Lost, 3: Removed
        self.frame_id = 0

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != 1:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    def activate(self, kalman_filter, frame_id):
        self.kalman_filter = kalman_filter
        self.frame_id = frame_id
        self.state = 1
        self.tracklet_len = 0
        # xyah format
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.state = 1
        self.tracklet_len = 0
        self.frame_id = frame_id
        self.score = new_track.score

    def update(self, new_track, frame_id):
        self.frame_id = frame_id
        self.tracklet_len += 1
        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh)
        )
        self.state = 1
        self._tlwh = new_tlwh
        self.score = new_track.score

    @property
    def tlwh(self):
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def center(self):
        # 返回中心点 (x, y)
        tlwh = self.tlwh
        return (tlwh[0] + tlwh[2]/2.0, tlwh[1] + tlwh[3]/2.0)

    @property
    def box_dict(self):
        t = self.tlwh
        return {
            'x1': t[0], 
            'y1': t[1], 
            'x2': t[0] + t[2], 
            'y2': t[1] + t[3]
        }

    @staticmethod
    def tlwh_to_xyah(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

def iou_batch(bboxes1, bboxes2):
    """
    计算IoU矩阵
    """
    # 1. 显式转换为 numpy 数组
    bboxes1 = np.array(bboxes1)
    bboxes2 = np.array(bboxes2)

    # 2. 如果任一输入为空，直接返回空矩阵 (N, M)
    if bboxes1.shape[0] == 0 or bboxes2.shape[0] == 0:
        return np.zeros((bboxes1.shape[0], bboxes2.shape[0]))

    # 3. 维度扩展，准备广播计算
    bboxes2 = np.expand_dims(bboxes2, 0)
    bboxes1 = np.expand_dims(bboxes1, 1)
    
    # 4. 计算重叠区域 (Intersection)
    xx1 = np.maximum(bboxes1[..., 0], bboxes2[..., 0])
    yy1 = np.maximum(bboxes1[..., 1], bboxes2[..., 1])
    xx2 = np.minimum(bboxes1[..., 0] + bboxes1[..., 2], bboxes2[..., 0] + bboxes2[..., 2])
    yy2 = np.minimum(bboxes1[..., 1] + bboxes1[..., 3], bboxes2[..., 1] + bboxes2[..., 3])
    
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    
    # 5. 计算 IoU
    # area = width * height
    area1 = bboxes1[..., 2] * bboxes1[..., 3]
    area2 = bboxes2[..., 2] * bboxes2[..., 3]
    
    iou = wh / (area1 + area2 - wh)
    return iou

class BilliardByteTracker:
    def __init__(self, track_thresh=0.5, track_buffer=30, match_thresh=0.8):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.kalman_filter = KalmanFilter()
        
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []     # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        self.frame_id = 0

    def update(self, output_results):
        """
        output_results: list of dicts {'box':{'x1'...}, 'conf':float, 'id':int}
        """
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        # 1. 转换输入数据并分离高低分检测
        high_scores = []
        low_scores = []
        
        # 将输入转换为 STrack 对象
        strack_pool = []
        for det in output_results:
            x1 = det['box']['x1']
            y1 = det['box']['y1']
            w = det['box']['x2'] - x1
            h = det['box']['y2'] - y1
            # box格式转为 tlwh
            s = STrack([x1, y1, w, h], det['conf'], det['id'])
            strack_pool.append(s)

        # 分离
        remain_inds = []
        inds_low = []
        for i, s in enumerate(strack_pool):
            if s.score > self.track_thresh:
                high_scores.append(s)
            elif s.score > 0.1: # 最低容忍阈值
                low_scores.append(s)

        # 2. 预测当前轨迹位置
        strack_pool_exist = [st for st in self.tracked_stracks if st.state == 1]
        strack_pool_exist = strack_pool_exist + self.lost_stracks # 包含丢失的轨迹尝试找回
        
        for st in strack_pool_exist:
            st.predict()

        # 3. 第一次匹配：高分检测框 vs 现有轨迹
        # 注意：台球中，类别ID (球号) 是绝对特征。我们强制要求类别一致。
        dists = 1 - iou_batch([st.tlwh for st in strack_pool_exist], [d.tlwh for d in high_scores])
        dists = self._apply_class_penalty(dists, strack_pool_exist, high_scores)
        
        # 使用匈牙利算法匹配
        matches, u_track, u_detection = self._linear_assignment(dists, thresh=self.match_thresh)

        for itracked, idet in matches:
            track = strack_pool_exist[itracked]
            det = high_scores[idet]
            if track.state == 1:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id)
                refind_stracks.append(track)

        # 4. 第二次匹配 (BYTE核心)：未匹配的轨迹 vs 低分检测框
        # 这一步能把快速移动模糊的球（低置信度）找回来
        r_tracked_stracks = [strack_pool_exist[i] for i in u_track if strack_pool_exist[i].state == 1]
        dists = 1 - iou_batch([st.tlwh for st in r_tracked_stracks], [d.tlwh for d in low_scores])
        dists = self._apply_class_penalty(dists, r_tracked_stracks, low_scores)
        
        matches, u_track_2, u_detection_2 = self._linear_assignment(dists, thresh=0.5) # 低分匹配阈值放宽

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = low_scores[idet]
            if track.state == 1:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id)
                refind_stracks.append(track)

        # 5. 处理未匹配的轨迹
        for it in u_track_2:
            track = r_tracked_stracks[it]
            if not track.state == 2:
                track.state = 2 # 标记为丢失
                lost_stracks.append(track)

        # 6. 处理未匹配的高分检测框 -> 初始化为新轨迹
        for inew in u_detection:
            track = high_scores[inew]
            if track.score < self.track_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)

        # 7. 状态清理
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == 1]
        self.tracked_stracks = list(set(self.tracked_stracks + activated_stracks + refind_stracks)) # 简单去重
        
        # 从 lost_stracks 中移除超时的
        self.lost_stracks = [t for t in self.lost_stracks if t.state != 3] 
        # 将本次丢失的加入
        for track in lost_stracks:
            if self.frame_id - track.frame_id > self.track_buffer:
                track.state = 3 # Removed
                self.removed_stracks.append(track)
            else:
                self.lost_stracks.append(track)
        
        # 合并返回结果
        return [t for t in self.tracked_stracks if t.state == 1]

    def _apply_class_penalty(self, dists, tracks, detections):
        """
        强制类别约束：如果 Track.class_id != Det.class_id，距离设为无穷大
        """
        if len(tracks) == 0 or len(detections) == 0:
            return dists
            
        for t_i, track in enumerate(tracks):
            for d_i, det in enumerate(detections):
                if track.class_id != det.class_id:
                    dists[t_i, d_i] = 1e5 # Penalty
        return dists

    def _linear_assignment(self, cost_matrix, thresh):
        if cost_matrix.size == 0:
            return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
        
        matches, unmatched_a, unmatched_b = [], [], []
        # 使用 scipy 进行匈牙利算法匹配
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] > thresh:
                unmatched_a.append(r)
                unmatched_b.append(c)
            else:
                matches.append((r, c))

        # 找出未匹配的索引
        matched_indices_a = set(r for r, c in matches)
        matched_indices_b = set(c for r, c in matches)
        
        for i in range(cost_matrix.shape[0]):
            if i not in matched_indices_a:
                unmatched_a.append(i)
        for i in range(cost_matrix.shape[1]):
            if i not in matched_indices_b:
                unmatched_b.append(i)
                
        return matches, unmatched_a, unmatched_b