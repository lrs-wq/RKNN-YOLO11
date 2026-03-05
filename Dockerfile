# -------------------------------------------------------------------
# 使用原生 ARM64 镜像 (M4 Mac 会自动拉取 linux/arm64)
# -------------------------------------------------------------------
FROM python:3.10-slim-bookworm

# 1. 软件源配置
# 注意：系统检测你在新加坡。如果你没有连接国内 VPN，
# 建议注释掉下面这两行换源命令，直接使用 Debian 官方源速度可能更快。
# 如果你在新加坡连接国内网络较慢，请保留默认源（删除下面这行 SED）。
# RUN sed -i 's/deb.debian.org/mirrors.ustc.edu.cn/g' /etc/apt/sources.list.d/debian.sources

# 2. 安装编译工具链
# 这里的依赖保持不变，M4 编译这些非常快
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    patchelf \
    ccache \
    libpython3-dev \
    build-essential \
    libffi-dev \
    git \
    wget \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 3. 配置 pip 源
# 同上，如果在新加坡且未使用 VPN，建议注释掉下一行清华源，使用默认 PyPI。
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 4. 安装 Nuitka 和优化库
RUN pip install nuitka ordered-set

# 设置工作目录
WORKDIR /app

# 将依赖复制进镜像
COPY requirements.txt .

# 安装项目依赖
RUN pip install --no-cache-dir -r requirements.txt

CMD ["/bin/bash"]


