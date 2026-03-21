# split_qwen_cpp

跨机器 Pipeline Parallel 推理实验（C++ 版）：将 Qwen2.5-3B-Instruct 的 TensorRT engine 分布在两台机器上，通过 MPI + NCCL 实现跨机器推理。

## 架构

```
Machine A: Ubuntu desktop (192.168.0.106) - RTX 5060 Ti 16GB
  rank0 / stage0: 前半段 layers + 生成控制

        │  NCCL over TCP (eth0)
        ▼

Machine B: Ubuntu notebook (192.168.0.101) - RTX 4070 Laptop 8GB
  rank1 / stage1: 后半段 layers
```

TensorRT-LLM 的 Pipeline Parallel 自动处理层切分和跨机器通信，每台机器加载对应的 `rank0.engine` 或 `rank1.engine`。

## 文件说明

| 文件 | 用途 |
|---|---|
| `src/infer.cpp` | 单请求推理程序，验证跨机推理正确性，输出性能指标 |
| `src/bench.cpp` | 并发压测程序，支持 in-flight batching，输出 TTFT/TPOT/Throughput |
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
    --output_dir /workspace/engines/qwen25_3b_pp2_b64 \
    --max_batch_size 64 \
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

# desktop 容器里推公钥
ssh-keygen -t rsa -N "" -f ~/.ssh/id_rsa
ssh-copy-id -p 2222 root@192.168.0.101
ssh-copy-id -p 2222 root@localhost
```

## 使用

### 单请求推理

```bash
python3 src/encode.py          # 编码 prompt，得到 token ids
# 将 token ids 填入 src/infer.cpp 的 input_ids

mpirun -n 2 \
    --allow-run-as-root \
    -H desktop:1,notebook:1 \
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
    /workspace/split_qwen_cpp/engines/qwen25_3b_pp2_b64

python3 src/decode.py          # 解码输出 token ids 为文字
```

### 并发压测

```bash
mpirun -n 2 \
    --allow-run-as-root \
    -H desktop:1,notebook:1 \
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
    /workspace/split_qwen_cpp/build/bench \
    /workspace/split_qwen_cpp/engines/qwen25_3b_pp2_b64 \
    --num_requests 100 \
    --concurrency  8 \
    --input_len    512 \
    --output_len   200 \
    --warmup       5

# 开启 prefix cache 复用（所有请求共享同一 prompt prefix）
# 在末尾加 --reuse_kv
```

bench 参数说明：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--num_requests` | 100 | 压测请求总数（不含预热） |
| `--concurrency` | 8 | 最大并发（飞行中）请求数 |
| `--input_len` | 512 | 每个请求的 prompt token 数 |
| `--output_len` | 200 | 每个请求的生成 token 数 |
| `--warmup` | 5 | 预热请求数，不计入统计 |
| `--reuse_kv` | off | 开启后所有请求使用相同 prompt，触发 prefix cache 命中 |

## 性能

测试模型：Qwen2.5-3B-Instruct，fp16，pp_size=2，有线局域网（1Gbps）。  
测试配置：100 requests，input=512 tokens，output=200 tokens，warmup=5。

| 场景 | Concurrency | Throughput | TTFT avg | TTFT p99 | TPOT avg |
|---|---|---|---|---|---|
| 单请求（baseline） | 1 | 26.1 tokens/s | 609 ms | - | 35.4 ms/token |
| 无 prefix cache | 8 | 153.8 tokens/s | 2040 ms | 2316 ms | 37.6 ms/token |
| 无 prefix cache | 16 | 215.5 tokens/s | 4250 ms | 4696 ms | 46.0 ms/token |
| 有 prefix cache | 8 | 221.3 tokens/s | 73 ms | 821 ms | 32.1 ms/token |
| 有 prefix cache | 16 | **367.6 tokens/s** | 135 ms | 772 ms | 38.3 ms/token |
| 有 prefix cache | 32 | OOM | - | - | - |

**关键结论：**

- prefix cache 收益显著：concurrency=8 下 TTFT 降低 96%（2040ms → 73ms），吞吐提升 44%（153 → 221 tokens/s）
- 有 prefix cache 时并发扩展性更好：concurrency 从 8 翻倍到 16，吞吐涨 66%（221 → 367 tokens/s）；无 cache 时同样翻倍只涨 40%（153 → 215 tokens/s），原因是大量并发 prefill 抢占 GPU 时间拖慢了 decode
- concurrency=32 时 notebook 显存（KV cache 可用约 3.5GB）耗尽 OOM，**16 是当前配置的并发上限**
- 网络带宽实测峰值约 10 MB/s，远低于 1Gbps 上限，每个 decode step 传输耗时约 0.5ms，占 TPOT 不到 2%，**网络不是瓶颈**
- 瓶颈在 notebook RTX 4070 Laptop 的 8GB 显存，限制了最大并发数

## 注意事项

- 两边 GPU 架构不同（5060 Ti vs 4070 Laptop），engine 必须各自 build，不能复用
- 网卡统一命名为 `eth0` 是必须的，否则 NCCL 无法正确选择网卡
- 跨机器环境下 `MPI_Barrier` 会触发 hpcx UCX/UCC 的除零 bug，用点对点 `MPI_Send/Recv` 替代
- Docker 容器使用 `--network host` 模式，宿主机需要提前配置好 SSH 免密
- desktop 机器上有 Docker，NCCL 会错误选择 Docker 网桥（172.x.x.x），必须用 `NCCL_SOCKET_IFNAME=eth0` 指定物理网卡
- notebook 显存仅 8GB，KV cache 可用约 3.5GB，高并发时注意 `setFreeGpuMemoryFraction` 不宜过高
