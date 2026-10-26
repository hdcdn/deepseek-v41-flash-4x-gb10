# 四机 DGX Spark (GB10) 跑 DeepSeek-V4.1-Flash MXFP4 · 完整实测与部署

4 台 NVIDIA DGX Spark（GB10）组成纯 TP4 集群，跑 DeepSeek-V4.1-Flash-1017 MXFP4（510.3 GB）的生产级实测：硬件规格、拓扑、模型加载、并发吞吐、预填充、长上下文、模型质量，以及从零开始的部署步骤。

**交流群**：QQ 群 **1128010665**（加群备注：GitHub）

---

- 测试时间：2026-10 上旬至中旬
- 硬件：4 × NVIDIA DGX Spark（GB10 Grace Blackwell）
- 部署形态：纯 TP4，无 PD 分离、无独立 Prefill 节点
- 模型：`DeepSeek-V4.1-Flash-1017-MXFP4-uncensored`（510.3 GB）
- 所有数字均为本机实测，口径见第 6 节

---

## 1. 硬件规格（NVIDIA 官方文档）

| 组件 | 规格 |
|---|---|
| GPU | Blackwell 架构，第 5 代 Tensor Core，第 4 代 RT Core |
| CPU | 20 核 Arm（10× Cortex-X925 + 10× Cortex-A725） |
| 内存 | **128 GB LPDDR5x 统一内存**，256-bit，4266 MHz，**273 GB/s** |
| 存储 | 1 TB 或 4 TB NVMe M.2（自加密） |
| 网络 | 1× RJ-45 10GbE、**ConnectX-7 智能网卡（2× QSFP）**、Wi-Fi 7、BT 5.4 |
| 接口 | 4× USB Type-C（其一供电）、1× HDMI 2.1a |
| 编解码 | 1× NVENC、1× NVDEC |
| 尺寸/重量 | 150 × 150 × 50.5 mm / 1.2 kg |
| 工作环境 | 5–30 °C，10–90% RH 无凝露 |
| 官方标称 | 最高 1 PFLOP FP4（稀疏）；单机 ≤200B 模型，双机 ≤405B |

出处：`docs.nvidia.com/dgx/dgx-spark/hardware.html`

**四机合计**：512 GB 统一内存 / 8× QSFP 200GbE 端口 / 最高 4 PFLOP FP4（稀疏）。

---

## 2. 集群拓扑

```
   Node A (.1) ────── Node B (.2) ────── Node C (.3) ────── Node D (.4)
   rank 0 (head)      rank 1            rank 2              rank 3
              200GbE RoCE · 无交换机 · 点对点环
```

| 项 | 配置 |
|---|---|
| 互联 | 200GbE RoCE（QSFP56），无交换机，环拓扑 |
| 张量并行 | TP4（每机 1 rank） |
| 通信库 | NCCL 2.30.7（含 switchless-ring 补丁） |
| NCCL 环境 | `NCCL_NET=IB`、`NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1`、`NCCL_P2P_DISABLE=1`、`NCCL_SHM_DISABLE=1`、`NCCL_PROTO=^LL128`、`NCCL_MAX_NCHANNELS=8`、`NCCL_BUFFSIZE=1048576` |
| 引导网络 | Gloo/NCCL bootstrap 走管理口 `enP7s7` |
| 系统盘 | NVMe，/ 分区 1.8 TB |

---

## 3. 模型规格

| 项 | 数值 |
|---|---|
| 权重仓库 | `Infatoshi/DeepSeek-V4.1-Flash-1017-MXFP4-uncensored` |
| 大小 | **510.3 GB**（落盘 476 GiB） |
| 分片 | 48 个 safetensors |
| 张量数 | 96,085 |
| 层数 | 40 层 + MTP |
| 专家 | 384 专家 / 每 token 激活 6 |
| 量化 | 路由专家 E2M1 block32 + `float8_e8m0fnu` scale；注意力/共享专家 FP8 |
| 上下文 | 1,048,576（1M） |
| 特殊结构 | engram 2 层（第 1、14 层），各 91.55 GiB |
| 许可 | MIT |

---

## 4. 部署参数（SGLang 实际启动命令）

```bash
python3 -m sglang.launch_server \
  --model-path /models/DeepSeek-V4.1-Flash \
  --served-model-name deepseek-v4.1-flash \
  --trust-remote-code --load-format safetensors \
  --tp 4 --ep-size 1 \
  --attention-backend dsv4 \
  --moe-runner-backend flashinfer_mxfp4 \
  --fp8-gemm-backend flashinfer_cutlass \
  --mem-fraction-static 0.90 \
  --chunked-prefill-size 4096 \
  --context-length 1048576 \
  --max-total-tokens 1000000 \
  --max-running-requests 32 \
  --cuda-graph-max-bs-decode 32 \
  --page-size 128 \
  --enable-cache-report \
  --speculative-algorithm DSPARK \
  --speculative-dspark-block-size 5 \
  --enable-decoder-swa-bounded-replay \
  --enable-deepseek-v4-fp4-indexer \
  --nnodes 4 --node-rank <0-3> --dist-init-addr <head>:20000 \
  --tool-call-parser deepseekv41 --reasoning-parser deepseek-v41 \
  --watchdog-timeout 1800 --sleep-on-idle \
  --host 0.0.0.0 --port 8888
```

| 参数 | 作用 | 调优备注 |
|---|---|---|
| `--mem-fraction-static 0.90` | 静态显存占比 | 0.90 已贴上限（0.91 需 109.5 GB > 可用 108.93 GB） |
| `--max-total-tokens 1000000` | KV 池预算 | 与 32 并发互斥：32 并发时每请求平均仅 3.1 万 token |
| `--chunked-prefill-size 4096` | 分块预填 | 让 12.5 万 token 输入也能稳定 5,000 tok/s |
| `--speculative-algorithm DSPARK --speculative-dspark-block-size 5` | 投机解码 | 每步提 5 token |
| `--page-size 128` | KV 分页粒度 | 影响 prefix cache 命中粒度（128 token 对齐） |
| `--enable-cache-report` | 返回缓存统计 | 未命中时 `prompt_tokens_details` 为 null，命中才出现 `cached_tokens` |

---

## 5. 实测数据

### 5.1 解码吞吐（代码提示词 / 512 tok / 3 轮 / 剥离 TTFT）

| 并发 | 第 1 次 | 第 2 次 | 第 3 次 | **中位数** | 单请求范围 |
|---:|---:|---:|---:|---:|---|
| **C1** | 106.1 | 106.7 | 107.0 | **106.7** | 110.6–113.7（单发） |
| **C4** | 221.5 | 225.7 | 222.8 | **222.8** | 57.7–61.2 |
| **C8** | 314.3 | 317.0 | 319.0 | **317.0** | 40.1–44.1 |
| **C16** | 459.9 | 433.9 | 439.9 | **439.9** | 27.9–32.0 |

单位 tok/s。扩展效率：

| 并发 | 实测聚合 | 线性理想 | 效率 |
|---:|---:|---:|---:|
| 1 | 106.7 | 106.7 | 100% |
| 4 | 222.8 | 426.8 | **52.2%** |
| 8 | 317.0 | 853.6 | **37.1%** |
| 16 | 439.9 | 1,707.2 | **25.8%** |

### 5.2 预填充（随机 nonce 前缀 + 清缓存 + 断言 `cached_tokens=0`）

| 输入 token | TTFT | 预填速度 | 校验 |
|---:|---:|---:|---|
| 13,696 | 2.81 s | **4,875 tok/s** | cached=0 ✓ |
| 53,189 | 10.45 s | **5,090 tok/s** | cached=0 ✓ |
| 125,351 | 24.79 s | **5,057 tok/s** | cached=0 ✓ |

结论：1.3 万–12.5 万 token 区间预填恒速 ≈5,000 tok/s，分块预填无衰减。

### 5.3 长上下文

| 项目 | 结果 |
|---|---|
| 73 万 token 文档埋点检索 | ✅ 全部命中 |
| 标称上下文 | 1,048,576 token |
| 实际交互上限 | ≈300K（超过后首 token 延迟显著上升） |
| 多跳推理（1M 上下文） | ❌ 多跳相加、10 级变量追踪失败 |
| 检索类任务 | ✅ 满血 |

### 5.4 模型质量（历史测试）

| 项目 | 结果 | 备注 |
|---|---|---|
| GSM8K | **97%** | 数学应用题 |
| MATH-500 | **84%** | 严格判分；早期版本因 `\boxed{}` 嵌套解析错误曾报 77% |
| 指令遵循 | 9/10 | 10 项严格格式测试 |
| 代码任务 | 全部通过 | 可执行且结果正确 |
| JSON 输出 | ⚠️ 需强制约束 | 不加提示易附解释文字 |

### 5.5 稳定性

| 项目 | 结果 |
|---|---|
| 连续运行 | 2 小时无崩溃，容器 healthy |
| C8 压测 | 全部成功，零失败 |
| 显存 | 无 OOM、无 NVRM Xid |

---

## 6. 测试方法与口径

1. **解码**：`stream=true`、`stream_options.include_usage=true`、`temperature=0`、固定代码提示词、`max_tokens=512`；TTFT 从首个 token 起算并剔除；每档 3 轮取中位数。
2. **预填**：`max_tokens=1` 非流式；prompt 前置 `[req <uuid>]` 破坏 prefix cache；同时读 `usage.prompt_tokens` 与 `prompt_tokens_details.cached_tokens`，**cached 必须为 0** 才计入。
3. **长上下文**：长文档中埋 5 个唯一标记，检验召回率。
4. **质量**：GSM8K 500 题子集、MATH-500 全量、代码题人工执行验证。

> 为什么预填要这么测：不做 nonce 前缀、不查 cached_tokens，很容易把"缓存命中"测成 2 万 tok/s 的假数据。

---

## 7. 已知问题与局限

| 问题 | 说明 |
|---|---|
| 4×TP 各自拉全量 KV | MLA 单 KV head 无法按 head 切分，KV 冗余 4.28× |
| 1M 下组合推理退化 | 多跳相加、10 级变量追踪失败；检索满血 |
| `max_completion_tokens` 不兼容 | 代理忽略该字段，必须用 `max_tokens` |
| 长 prompt 首字延迟高 | 125K 输入 TTFT 24.8 s，无独立 Prefill 节点 |
| 并发与长度互斥 | KV 池 100 万 token，32 并发时每请求仅 3.1 万 |

---

## 8. 复现步骤

```bash
# 1. 四台同路径放权重（/models/DeepSeek-V4.1-Flash），逐文件校验大小
# 2. 确认 RoCE 互通：ping 10.10.10.x + ib_write_bw
# 3. rank 0 起 head，rank 1-3 起 worker，--dist-init-addr 指向 rank 0
# 4. 等 /health 返回 200（首次加载 6–10 分钟）
# 5. 先跑带 nonce 的预填脚本，确认 cached_tokens=0 再采信数字
```

---

## 9. 数据来源

| 内容 | 来源 |
|---|---|
| 硬件规格 | NVIDIA 官方 `docs.nvidia.com/dgx/dgx-spark/hardware.html` |
| 模型规格 | 权重仓库文件清单 + `config.json` |
| 其余全部数字 | 本机实测（四台 DGX Spark） |

*报告生成：2026-10-26　交流群：QQ 1128010665*

---

## 附录 A · 部署方法（从零到服务可用）

### A.1 前置准备

| 项 | 要求 |
|---|---|
| 机器 | 4 × DGX Spark（GB10），同型号同内存，DGX OS 已初始化 |
| 互联 | 每台 2× QSFP 200GbE，用 4 根 DAC/AOC 组成环网（A-B-C-D-A） |
| 系统盘 | ≥1.8 TB NVMe（权重 510 GB + 容器 40 GB + 余量） |
| 系统依赖 | docker 或 containerd（GB10 上用 docker 29 即可）、NCCL 2.30.7 |

### A.2 准备权重（四台同路径）

```bash
# 只在一台上下载，再内网分发（200GbE 下 10 GB/分钟级）
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1        # 关键：否则大分片会 302 到 Xet CDN 卡死
python3 - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    "Infatoshi/DeepSeek-V4.1-Flash-1017-MXFP4-uncensored",
    local_dir="/models/DeepSeek-V4.1-Flash",
    max_workers=8,
)
PY

# 逐文件校验（必须做）：49 个文件总 252.4 GB / 本模型 48 分片 510.3 GB
```

**分发**（源机 → 其余三台）：

```bash
rsync -a --ignore-existing -e "ssh -i ~/.ssh/id_ed25519" \
  /models/DeepSeek-V4.1-Flash/ user@10.10.10.2:/models/DeepSeek-V4.1-Flash/
```

> 坑：不要在文件还在下载时 rsync，会把半成品复制过去且 `--ignore-existing` 不再更新。必须等源端落盘 + 字节数校验后再同步。

### A.3 网络与 NCCL 配置

```bash
# 1. 确认 RoCE 链路
ibdev2netdev                      # 应看到 2 个 rocep 口
ping -c2 10.10.10.2               # 环内互通

# 2. NCCL 环境（四台一致）
export NCCL_NET=IB
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_PROTO=^LL128
export NCCL_MAX_NCHANNELS=8
export NCCL_BUFFSIZE=1048576
export NCCL_SOCKET_IFNAME=enP7s7      # 管理口做 bootstrap
export GLOO_SOCKET_IFNAME=enP7s7
```

**为什么 `NCCL_SHM_DISABLE=1`**：GB10 是统一内存，共享内存段会占用 GPU 可分配内存；关掉后通信缓冲区直接走 pinned host 内存。

### A.4 启动四机服务

```bash
# rank 0（head）
VLLM_HOST_IP=10.10.10.1 python3 -m sglang.launch_server \
  --model-path /models/DeepSeek-V4.1-Flash \
  --tp 4 --nnodes 4 --node-rank 0 --dist-init-addr 10.10.10.1:20000 \
  --mem-fraction-static 0.90 --context-length 1048576 \
  --max-total-tokens 1000000 --chunked-prefill-size 4096 \
  --moe-runner-backend flashinfer_mxfp4 \
  --attention-backend dsv4 \
  --speculative-algorithm DSPARK --speculative-dspark-block-size 5 \
  --host 0.0.0.0 --port 8888

# rank 1-3：同上，改 --node-rank 1/2/3，去掉 --host/--port
```

启动顺序：**先起 rank 1–3，再起 rank 0**（head 负责 rendezvous，先起会等不到 worker 超时）。

### A.5 验证

```bash
# 1. 健康检查（首次加载 6–10 分钟）
curl -s -o /dev/null -w '%{http_code}\n' http://10.0.0.1:8888/health   # 期望 200

# 2. 模型列表
curl -s http://10.0.0.1:8888/v1/models | jq '.data[].id'

# 3. 真实推理（含思考开关）
curl -s http://10.0.0.1:8888/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model":"deepseek-v4.1-flash",
  "messages":[{"role":"user","content":"17*23=?"}],
  "max_tokens":32
}'

# 4. 预填必须校验缓存
curl -s ... -d '{"max_tokens":1, "messages":[{"role":"user","content":"[req 4f2a...] 长文本"}]}' \
  | jq '.usage | {prompt_tokens, cached: .prompt_tokens_details.cached_tokens}'
# cached_tokens 必须为 0，否则测出来的是缓存速度
```

### A.6 常见故障

| 现象 | 原因 | 处理 |
|---|---|---|
| 启动即退出 `mem-fraction-static` | 0.90 已贴上限 | 降到 0.88 或减 `--max-total-tokens` |
| `/health` 一直 000 | rank 未全部就绪 | 查 `--dist-init-addr` 连通性、NCCL 报错 |
| 首 token 极慢（>60s） | 输入太长 + 无 P 节点 | 分批预填，或加 6000D 做 P |
| 吞吐远低于表中 | 命中缓存被当性能 | 用 nonce 前缀复测 |
| KV 不足报错 | 并发 × 长度 > 100 万 | 降 `--max-running-requests` 或 `--max-total-tokens` |

---

*文档生成：2026-10-26　数据口径与来源见各节*
