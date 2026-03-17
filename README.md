# split_qwen_cpp

跨机器 Pipeline Parallel 推理实验（C++ 版）：将 Qwen2.5-3B-Instruct 的 TensorRT engine 分布在两台机器上，通过 MPI + NCCL 实现跨机器推理。

## 架构
```
Machine A: Ubuntu naked (192.168.0.106) - RTX 5060 Ti 16GB
  rank0: 前半段 layers + 生成控制

        │  NCCL over TCP (eth0)
        ▼

Machine B: Ubuntu notebook (192.168.0.101) - RTX 4070 Laptop 8GB
  rank1: 后半段 layers
```

TensorRT-LLM 的 Pipeline Parallel 自动处理层切分和跨机器通信，每台机器加载对应的 `rank0.engine` 或 `rank1.engine`。

## 文件说明

| 文件 | 用途 |
|---|---|
| `src/infer.cpp` | C++ 推理主程序，MPI 多机启动，输出性能指标 |
| `src/encode.py` | 将自然语言 prompt 编码为 token ids |
| `src/decode.py` | 将 token ids 解码回文字 |
| `CMakeLists.txt` | 构建配置 |

## 环境要求
```
两台机器同一局域网，网卡统一命名为 eth0
TensorRT-LLM Docker 镜像：nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc6.post3
CUDA 13.x
Open MPI 4.1.x
```

## 构建流程

### 1. 准备模型和 engine
```bash
# convert checkpoint（两边各自执行）
python3 /app/tensorrt_llm/examples/models/core/qwen/convert_checkpoint.py \
    --model_dir /workspace/models/Qwen2.5-3B-Instruct \
    --output_dir /workspace/checkpoints/qwen25_3b_fp16_pp2 \
    --dtype float16 \
    --pp_size 2

# build engine（两边各自执行，GPU 架构不同不能共用）
trtllm-build \
    --checkpoint_dir /workspace/checkpoints/qwen25_3b_fp16_pp2 \
    --output_dir /workspace/engines/qwen25_3b_pp2 \
    --max_batch_size 4 \
    --max_input_len 1024 \
    --max_seq_len 1280 \
    --gpt_attention_plugin float16 \
    --gemm_plugin float16
```

### 2. 编译 C++ 程序
```bash
# 两边各自执行
mkdir -p build && cd build
cmake .. && make -j$(nproc)
```

### 3. 配置 SSH 免密
```bash
# 两边容器里都安装 sshd，监听 2222 端口
apt-get install -y openssh-server
mkdir -p /run/sshd
echo "Port 2222" >> /etc/ssh/sshd_config
echo "PermitRootLogin yes" >> /etc/ssh/sshd_config
/usr/sbin/sshd

# naked 容器里推公钥
ssh-keygen -t rsa -N "" -f ~/.ssh/id_rsa
ssh-copy-id -p 2222 root@192.168.0.101
ssh-copy-id -p 2222 root@localhost
```

## 使用

### encode prompt
```bash
python3 src/encode.py
# 输出 token ids，复制到 infer.cpp 的 input_ids
```

### 跨机器推理
```bash
# naked 容器里执行
mpirun -n 2 \
    --allow-run-as-root \
    -H naked:1,notebook:1 \
    -mca plm_rsh_args "-p 2222 -o StrictHostKeyChecking=no" \
    -mca pml ob1 \
    -mca btl tcp,self \
    -mca coll ^ucc \
    -mca oob_tcp_if_include 192.168.0.0/24 \
    -mca btl_tcp_if_include 192.168.0.0/24 \
    -x NCCL_SOCKET_IFNAME=eth0 \
    -x NCCL_IB_DISABLE=1 \
    -x UCX_TLS=tcp \
    -x UCX_NET_DEVICES=eth0 \
    -x LD_LIBRARY_PATH \
    -x PATH \
    /workspace/split_qwen_cpp/build/infer \
    /workspace/split_qwen_cpp/engines/qwen25_3b_pp2
```

### decode 输出
```bash
# 把 infer 输出的 token ids 粘贴到 decode.py 的 txt 变量
python3 src/decode.py
```

## 性能

测试模型：Qwen2.5-3B-Instruct，fp16，pp_size=2，Wi-Fi 局域网。

| 指标 | 数值 |
|---|---|
| Throughput | 26.1 tokens/s |
| TTFT | 609 ms |
| TPOT | 35.4 ms/token |
| Total tokens | 200 |

## 注意事项

- 两边 GPU 架构不同（5060 Ti vs 4070 Laptop），engine 必须各自 build，不能复用
- 网卡统一命名为 `eth0` 是必须的，否则 NCCL 无法正确选择网卡
- 跨机器环境下 `MPI_Barrier` 会触发 hpcx UCX/UCC 的除零 bug，用点对点 `MPI_Send/Recv` 替代
- Docker 容器使用 `--network host` 模式，宿主机需要提前配置好 SSH 免密
- naked 机器上有 Docker，NCCL 会错误选择 Docker 网桥（172.x.x.x），必须用 `NCCL_SOCKET_IFNAME=eth0` 指定物理网卡