import sys
import os
import time
from collections import deque

# ================= 工具库导入 =================
from data_utils import DataObject, MotionState, DetectionBox
from alg import KalmanFilter

# 动态加载 yolo_rtsp (C++绑定)
def _import_yolo_rtsp(lib_dir):
    sys.path.insert(0, lib_dir)
    try:
        import yolo_rtsp
    except ImportError as e:
        print(f"加载 yolo_rtsp 失败, 请检查 lib_dir 路径: {e}")
        sys.exit(1)
    finally:
        sys.path.pop(0)
    return yolo_rtsp

# ================= 全局状态 =================
COCO_CLASSES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", 
                "10", "11", "12", "13", "14", "15", "16", "17"]
last_time = time.time()
frame_count = 0

# 初始化台球跟踪的卡尔曼滤波器
kalman = KalmanFilter.Tracker()

def on_frame_detected(frame_id, detections):
    """
    RTSP 数据流抓取到的检测帧回调函数。
    由 C++ 层 (yolo_rtsp.start_stream) 触发并推送帧数据。
    """
    global last_time, frame_count, kalman
    current_time = time.time()
    
    # 1. 简单的获取 FPS
    frame_count += 1
    fps = 0.0
    if current_time - last_time > 1.0:
        fps = frame_count / (current_time - last_time)
        frame_count = 0
        last_time = current_time

    # 2. 从 C++ 返回的特征对象向 Python 的数据结构进行转换 
    ordered_results = [None] * len(COCO_CLASSES)
    
    if detections:
        for det in detections:
            if det is None:
                continue
            
            d_id = int(det.id)
            d_prob = float(det.prob)
            
            # 使用目标框获取中心点
            cx = (float(det.x1) + float(det.x2)) / 2.0
            cy = (float(det.y1) + float(det.y2)) / 2.0
            
            # 过滤不可用 ID
            if d_id >= len(COCO_CLASSES):
                continue
                
            # 使用 billiard_utils 中定义的结构，以方便进入 kalman.process
            obj = DataObject(
                frame_id=frame_id,
                timestamp=current_time,
                class_id=d_id,
                class_name=COCO_CLASSES[d_id],
                confidence=round(d_prob, 4),
                box=DetectionBox(x1=det.x1, y1=det.y1, x2=det.x2, y2=det.y2),
                center=(cx, cy),
                speed=MotionState(), 
                fps=fps
            )
            
            # 置入有序结果对象
            ordered_results[d_id] = obj
            
            # --------------------------------
            # 输出: 原始检测结果 
            # --------------------------------
            print(f"[Frame {frame_id} - {fps:.1f}fps] 原始检测 -> ID: {d_id} | 置信度: {d_prob:.2f} | 坐标: ({cx:.1f}, {cy:.1f})")

    # 3. 将有序对象送入 Kalman 滤波处理：生成预测数据 (主要针对台球的高速和遮挡轨迹平滑)
    pre_data = kalman.process(ordered_results)
    
    # --------------------------------
    # 输出: 卡尔曼滤波平滑预测结果 
    # --------------------------------
    if pre_data:
        for pd in pre_data:
            if pd is not None:
                print(f"[Frame {frame_id}] Kalman预测 -> ID: {pd.class_id} | 新坐标: ({pd.center[0]:.1f}, {pd.center[1]:.1f})")
    
    print("-" * 60)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='YOLO11 C++ Binding with Kalman Filter')
    # 模糊化了具体路径
    parser.add_argument('--rtsp-url', type=str, default="rtsp://<your_ip>:<port>/stream", help='输入RTSP视频流URL')
    parser.add_argument('--model', type=str, default="/path/to/your/model.rknn", help='RKNN 模型文件路径')
    parser.add_argument('--lib-dir', type=str, default="/path/to/your/lib_dir", help='动态链接库(yolo_rtsp.so)所在目录')
    
    args = parser.parse_args()

    yolo_rtsp = _import_yolo_rtsp(args.lib_dir)

    print(f"启动 C++ RTSP 检测流水线 (主程序)...")
    print(f"-> RTSP URL: {args.rtsp_url}")
    print(f"-> 模型路径: {args.model}")

    try:
        # C++接口启动：阻塞接管子线程直至异常抛出或流中断
        yolo_rtsp.start_stream(args.rtsp_url, args.model, on_frame_detected)
    except KeyboardInterrupt:
        print("用户主动停止运行")

if __name__ == "__main__":
    main()
