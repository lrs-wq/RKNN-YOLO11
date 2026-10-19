# RKNN YOLO11 · 演示说明

![逐帧回调示意](callback-preview.gif)

## 不需要板卡的交互示意

克隆后直接打开 [callback-demo.html](callback-demo.html)，或从仓库根目录启动 `python3 -m http.server 8080 --bind 127.0.0.1`，访问 `/docs/showcase/callback-demo.html`。页面完全离线，没有第三方脚本或网络请求。

“播放示意”“下一帧”“重置”会更新合成台球画面、框坐标和回调参数。坐标与分数由文档脚本生成，不是模型推理；GIF 是该交互页的操作节选，不是摄像头录屏。`workflow.svg` 根据当前绑定源码绘制。

## 真实板端调用

真实推理需要兼容的 Rockchip 板卡、RKNN Runtime、模型文件、FFmpeg 解码依赖，以及与扩展一致的 Python ABI。按主 README 编译后，将生成的 `yolo_rtsp` 扩展所在目录加入 Python 模块路径。模型与芯片必须匹配，已有 aarch64 二进制不能直接在 macOS 上加载。

下面展示当前 [C++ 绑定](../../bin/yolo11/cpp/rtsp_python_binding.cc) 的实际函数签名；请在自己的环境设置 `RTSP_URL` 与 `RKNN_MODEL`，不要将带密码的摄像头地址提交到仓库。

```python
import os
import yolo_rtsp

def on_frame(frame_id, detections):
    for detection in detections:
        print(frame_id, detection.id, detection.prob,
              detection.x1, detection.y1, detection.x2, detection.y2)

yolo_rtsp.start_stream(
    os.environ["RTSP_URL"],
    os.environ["RKNN_MODEL"],
    on_frame,
)
```

Python 接收的是 `Detection` 对象列表，不是界面里显示的 JSON；JSON 只是为了方便理解而整理的示意。当前源码会尝试 `hevc_rkmpp` / `h264_rkmpp` 解码器，并在不可用时回退到对应普通解码器。本次没有板端硬件，未验证实际推理性能。
