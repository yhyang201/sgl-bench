# CUDA IPC Transport 性能测试报告

**日期**: 2026-04-05  
**仓库**: [yhyang201/sgl-bench](https://github.com/yhyang201/sgl-bench)  
**原始数据**: [records/20260405/](https://github.com/yhyang201/sgl-bench/tree/main/records/20260405)

---

## 1. 测试目的

评估 SGLang 的 `SGLANG_USE_CUDA_IPC_TRANSPORT` 特性对多模态推理性能的影响，涵盖：

- **吞吐量与延迟**：对比 IPC 开/关在不同 TP 并行度和并发数下的表现
- **显存占用**：探测 IPC 开/关对最大可处理图片数量的影响（即显存开销差异）

---

## 2. 测试环境

| 项目 | 值 |
|------|-----|
| GPU | 8× NVIDIA H200 (139.80 GiB each) |
| CUDA | 12.9 |
| Driver | 580.126.09 |
| SGLang | v0.5.10rc0 (commit `34d5765e`) |
| 模型 | `Qwen/Qwen3-VL-8B-Instruct` |
| Attention Backend | FA3 |
| Sampling Backend | FlashInfer |
| Chunked Prefill Size | 8192 |
| Max Prefill Tokens | 16384 |
| Schedule Policy | FCFS |

---

## 3. 测试配置

### 3.1 变量空间

共 **2 × 2 × 2 = 8** 组基准测试 + **3 组**探测测试：

| 维度 | 取值 |
|------|------|
| CUDA IPC | `off` (SGLANG_USE_CUDA_IPC_TRANSPORT=0) / `on` (=1) |
| Tensor Parallelism | tp1 (单卡) / tp4 (4卡) |
| 最大并发数 | c16 / c32 |

### 3.2 基准测试参数（固定）

```
--backend sglang
--dataset-name image
--num-prompts 128
--request-rate 1.0
--random-input-len 512
--random-output-len 256
--random-range-ratio 1.0
--image-count 3
--image-resolution 1080p
```

### 3.3 服务端配置差异

唯一差异为环境变量 `SGLANG_USE_CUDA_IPC_TRANSPORT` 和 `--tp-size`：

```toml
# ipc-on 示例
[server]
model_path = "Qwen/Qwen3-VL-8B-Instruct"
extra_args = "--port 30000 --tp-size 4 --enable-multimodal"
[server.env]
SGLANG_USE_CUDA_IPC_TRANSPORT = "1"
```

---

## 4. 基准测试结果

### 4.1 吞吐量对比

| 配置 | IPC | TP | 并发 | 耗时 (s) | 请求吞吐 (req/s) | 输入吞吐 (tok/s) | 输出吞吐 (tok/s) |
|------|-----|-----|------|----------|------------------|------------------|------------------|
| [tp1-off-c16] | off | 1 | 16 | 136.4 | 0.939 | 6,264 | 240 |
| [tp1-off-c32] | off | 1 | 32 | 117.6 | 1.088 | 7,261 | 279 |
| [tp1-on-c16] | **on** | 1 | 16 | 109.3 | **1.171** | **7,816** | **300** |
| [tp1-on-c32] | **on** | 1 | 32 | 129.1 | 0.992 | 6,618 | 254 |
| [tp4-off-c16] | off | 4 | 16 | 125.6 | 1.019 | 6,800 | 261 |
| [tp4-off-c32] | off | 4 | 32 | 139.8 | 0.915 | 6,107 | 234 |
| [tp4-on-c16] | **on** | 4 | 16 | 110.0 | **1.164** | **7,764** | **298** |
| [tp4-on-c32] | **on** | 4 | 32 | 118.6 | **1.079** | **7,203** | **276** |

[tp1-off-c16]: https://github.com/yhyang201/sgl-bench/tree/main/records/20260405/20260405_113759_cuda-ipc_tp1_ipc-off_c16_Qwen3-VL-8B_3x1
[tp1-off-c32]: https://github.com/yhyang201/sgl-bench/tree/main/records/20260405/20260405_114252_cuda-ipc_tp1_ipc-off_c32_Qwen3-VL-8B_3x1
[tp1-on-c16]: https://github.com/yhyang201/sgl-bench/tree/main/records/20260405/20260405_114726_cuda-ipc_tp1_ipc-on_c16_Qwen3-VL-8B_3x10
[tp1-on-c32]: https://github.com/yhyang201/sgl-bench/tree/main/records/20260405/20260405_115149_cuda-ipc_tp1_ipc-on_c32_Qwen3-VL-8B_3x10
[tp4-off-c16]: https://github.com/yhyang201/sgl-bench/tree/main/records/20260405/20260405_115634_cuda-ipc_tp4_ipc-off_c16_Qwen3-VL-8B_3x1
[tp4-off-c32]: https://github.com/yhyang201/sgl-bench/tree/main/records/20260405/20260405_120126_cuda-ipc_tp4_ipc-off_c32_Qwen3-VL-8B_3x1
[tp4-on-c16]: https://github.com/yhyang201/sgl-bench/tree/main/records/20260405/20260405_120630_cuda-ipc_tp4_ipc-on_c16_Qwen3-VL-8B_3x10
[tp4-on-c32]: https://github.com/yhyang201/sgl-bench/tree/main/records/20260405/20260405_121101_cuda-ipc_tp4_ipc-on_c32_Qwen3-VL-8B_3x10

**IPC 开启后吞吐提升幅度**：

| TP × 并发 | IPC-off (req/s) | IPC-on (req/s) | 提升 |
|-----------|-----------------|----------------|------|
| tp1 × c16 | 0.939 | 1.171 | **+24.7%** |
| tp1 × c32 | 1.088 | 0.992 | −8.8% |
| tp4 × c16 | 1.019 | 1.164 | **+14.2%** |
| tp4 × c32 | 0.915 | 1.079 | **+17.9%** |

### 4.2 延迟对比 — TTFT（首 Token 延迟）

| 配置 | IPC | TP | 并发 | Mean (ms) | Median (ms) | P99 (ms) |
|------|-----|-----|------|-----------|-------------|----------|
| tp1-off-c16 | off | 1 | 16 | 956 | 851 | 2,643 |
| tp1-off-c32 | off | 1 | 32 | 1,172 | 928 | 3,432 |
| tp1-on-c16 | **on** | 1 | 16 | **438** | **369** | **1,043** |
| tp1-on-c32 | **on** | 1 | 32 | **438** | **366** | **834** |
| tp4-off-c16 | off | 4 | 16 | 806 | 738 | 1,509 |
| tp4-off-c32 | off | 4 | 32 | 797 | 646 | 1,985 |
| tp4-on-c16 | **on** | 4 | 16 | **233** | **210** | **474** |
| tp4-on-c32 | **on** | 4 | 32 | **224** | **208** | **395** |

**IPC 开启后 TTFT 降低幅度**：

| TP × 并发 | IPC-off Mean | IPC-on Mean | 降低 |
|-----------|-------------|-------------|------|
| tp1 × c16 | 956 ms | 438 ms | **−54.2%** |
| tp1 × c32 | 1,172 ms | 438 ms | **−62.6%** |
| tp4 × c16 | 806 ms | 233 ms | **−71.1%** |
| tp4 × c32 | 797 ms | 224 ms | **−71.9%** |

### 4.3 延迟对比 — TPOT（Token 间延迟）

| 配置 | IPC | TP | 并发 | Mean (ms) | Median (ms) | P99 (ms) |
|------|-----|-----|------|-----------|-------------|----------|
| tp1-off-c16 | off | 1 | 16 | 12.53 | 11.07 | 31.51 |
| tp1-off-c32 | off | 1 | 32 | 18.40 | 13.65 | 59.56 |
| tp1-on-c16 | **on** | 1 | 16 | **9.76** | **9.49** | **16.10** |
| tp1-on-c32 | **on** | 1 | 32 | **9.68** | **8.84** | **19.18** |
| tp4-off-c16 | off | 4 | 16 | 5.57 | 4.68 | 14.92 |
| tp4-off-c32 | off | 4 | 32 | 5.47 | 4.74 | 14.15 |
| tp4-on-c16 | **on** | 4 | 16 | **3.84** | **3.67** | **7.38** |
| tp4-on-c32 | **on** | 4 | 32 | **3.76** | **3.65** | **6.67** |

### 4.4 延迟对比 — 端到端（E2E）

| 配置 | IPC | TP | 并发 | Mean (ms) | Median (ms) | P99 (ms) |
|------|-----|-----|------|-----------|-------------|----------|
| tp1-off-c16 | off | 1 | 16 | 4,150 | 3,637 | 8,950 |
| tp1-off-c32 | off | 1 | 32 | 5,865 | 4,543 | 16,458 |
| tp1-on-c16 | **on** | 1 | 16 | **2,926** | **2,800** | **4,986** |
| tp1-on-c32 | **on** | 1 | 32 | **2,905** | **2,715** | **5,241** |
| tp4-off-c16 | off | 4 | 16 | 2,226 | 2,040 | 4,725 |
| tp4-off-c32 | off | 4 | 32 | 2,192 | 2,108 | 4,754 |
| tp4-on-c16 | **on** | 4 | 16 | **1,212** | **1,156** | **2,189** |
| tp4-on-c32 | **on** | 4 | 32 | **1,182** | **1,142** | **1,972** |

**IPC 开启后 E2E 延迟降低幅度**：

| TP × 并发 | IPC-off Mean | IPC-on Mean | 降低 |
|-----------|-------------|-------------|------|
| tp1 × c16 | 4,150 ms | 2,926 ms | **−29.5%** |
| tp1 × c32 | 5,865 ms | 2,905 ms | **−50.5%** |
| tp4 × c16 | 2,226 ms | 1,212 ms | **−45.5%** |
| tp4 × c32 | 2,192 ms | 1,182 ms | **−46.1%** |

---

## 5. 显存影响 — 最大图片数探测

探测实验在 **tp1** 下进行，逐步增加单请求图片数量直到 OOM，以衡量 IPC 对显存的额外开销。

- **探测参数**: input_len=256, output_len=32, timeout=300s, 每步增加 1 张图片
- **原始数据**: [ipc-off](https://github.com/yhyang201/sgl-bench/tree/main/records/20260405/20260405_132418_probe_max_images_tp1_ipc-off_Qwen3-VL-8B) | [ipc-on](https://github.com/yhyang201/sgl-bench/tree/main/records/20260405/20260405_140137_probe_max_images_tp1_ipc-on_Qwen3-VL-8B)

### 5.1 最大图片数对比

| 分辨率 | IPC-off 最大图片数 | IPC-on 最大图片数 | 差异 | 减少比例 |
|--------|-------------------|------------------|------|---------|
| 720p (1280×720) | 64 (未触顶) | 64 (未触顶) | — | — |
| 1080p (1920×1080) | 64 (未触顶) | 61 (62 OOM) | **−3** | **−4.7%** |
| 1440p (2560×1440) | 37 (38 OOM) | 33 (34 OOM) | **−4** | **−10.8%** |

> **注**: 720p 下两者均达到了探测上限 64 张而未 OOM，实际上限可能更高。

### 5.2 OOM 时的显存状态

| 场景 | OOM 图片数 | 尝试分配 | GPU 剩余显存 |
|------|-----------|---------|-------------|
| ipc-off, 1440p | 38 | 3.13 GiB | 3.05 GiB |
| ipc-on, 1080p | 62 | 2.90 GiB | 2.79 GiB |
| ipc-on, 1440p | 34 | 2.80 GiB | 1.65 GiB |

### 5.3 TTFT 随图片数增长曲线（对比 IPC 开/关）

#### 720p

| 图片数 | IPC-off TTFT (ms) | IPC-on TTFT (ms) | IPC-on 加速比 |
|--------|-------------------|------------------|--------------|
| 1 | 111 | 63 | 1.76× |
| 8 | 793 | 374 | 2.12× |
| 16 | 1,535 | 783 | 1.96× |
| 32 | 3,221 | 1,771 | 1.82× |
| 48 | 5,251 | 3,087 | 1.70× |
| 64 | 7,378 | 4,555 | 1.62× |

#### 1080p

| 图片数 | IPC-off TTFT (ms) | IPC-on TTFT (ms) | IPC-on 加速比 |
|--------|-------------------|------------------|--------------|
| 1 | 243 | 120 | 2.03× |
| 8 | 1,977 | 1,013 | 1.95× |
| 16 | 4,352 | 2,354 | 1.85× |
| 32 | 9,619 | 5,720 | 1.68× |
| 48 | 16,121 | 10,220 | 1.58× |
| 61 | 22,649* | 14,609 | 1.55× |
| 64 | 23,472 | OOM at 62 | — |

> *ipc-off 在 61 张图片时 TTFT 取自原始数据

#### 1440p

| 图片数 | IPC-off TTFT (ms) | IPC-on TTFT (ms) | IPC-on 加速比 |
|--------|-------------------|------------------|--------------|
| 1 | 444 | 223 | 1.99× |
| 8 | 4,171 | 2,453 | 1.70× |
| 16 | 9,334 | 5,902 | 1.58× |
| 32 | 21,839 | 15,053 | 1.45× |
| 33 | 23,289 | 15,617 | 1.49× |
| 37 | 26,425 | OOM at 34 | — |

---

## 6. 分析与结论

### 6.1 性能收益

CUDA IPC Transport 在所有延迟指标上都带来了**显著提升**：

- **TTFT 降低 54%–72%**：这是最大的收益点。IPC 传输消除了多模态图像数据在 CPU↔GPU 间的冗余拷贝，使 prefill 阶段的图像编码延迟大幅缩短。
- **TPOT 降低 22%–47%**（tp1 场景改善更显著），decode 阶段也因减少了内存搬运开销而获益。
- **E2E 延迟降低 30%–50%**，其中高并发 (c32) 场景受益更大，因为 IPC 减少了并发请求间的内存带宽争用。
- **吞吐量提升 14%–25%**（c16 并发场景），c32 场景下 tp1 出现小幅回退（−8.8%），可能与显存压力增大导致调度效率下降有关。

### 6.2 显存代价

CUDA IPC Transport 需要在 GPU 上额外分配共享内存缓冲区，导致：

- **720p**: 影响不可见（64 张图均未 OOM）
- **1080p**: 最大图片数从 64+ 降至 **61**（−4.7%），额外占用约 **3 张 1080p 图像的等效显存**
- **1440p**: 最大图片数从 37 降至 **33**（−10.8%），额外占用约 **4 张 1440p 图像的等效显存**

高分辨率场景下显存代价更明显——因为每张图像本身占用的显存更大，IPC 缓冲区的固定开销相对基线可用空间的占比就越高。

### 6.3 性能 vs 显存 Trade-off

| 分辨率 | TTFT 加速（单图/多图） | 最大图片数损失 | 建议 |
|--------|----------------------|---------------|------|
| 720p | 1.6×–2.1× | 无可见影响 | **强烈建议开启** |
| 1080p | 1.5×–2.0× | −4.7% | **建议开启**（收益远大于代价） |
| 1440p | 1.4×–2.0× | −10.8% | **按需开启**（如图片数接近上限则需权衡） |

### 6.4 关键发现

1. **IPC 对 prefill 的加速效果随图片数量增加而递减**：从单图 ~2× 降至极限数量时的 ~1.5×，说明在大量图片场景下瓶颈可能逐渐转移到计算本身。
2. **TP4 + IPC-on 是最佳配置**：在 c16 并发下，tp4+ipc-on 实现了 E2E 1,212ms 的最低延迟和最高吞吐。
3. **IPC 使延迟对并发数更不敏感**：ipc-on 场景下 c16→c32 的延迟增长远小于 ipc-off（如 tp1 E2E: +0.7% vs +41.3%），说明 IPC 传输有效缓解了高并发时的资源争抢。

---

## 7. 原始数据索引

| Session | 配置 | 数据链接 |
|---------|------|---------|
| 20260405_113759 | tp1, ipc-off, c16 | [查看](https://github.com/yhyang201/sgl-bench/tree/main/records/20260405/20260405_113759_cuda-ipc_tp1_ipc-off_c16_Qwen3-VL-8B_3x1) |
| 20260405_114252 | tp1, ipc-off, c32 | [查看](https://github.com/yhyang201/sgl-bench/tree/main/records/20260405/20260405_114252_cuda-ipc_tp1_ipc-off_c32_Qwen3-VL-8B_3x1) |
| 20260405_114726 | tp1, ipc-on, c16 | [查看](https://github.com/yhyang201/sgl-bench/tree/main/records/20260405/20260405_114726_cuda-ipc_tp1_ipc-on_c16_Qwen3-VL-8B_3x10) |
| 20260405_115149 | tp1, ipc-on, c32 | [查看](https://github.com/yhyang201/sgl-bench/tree/main/records/20260405/20260405_115149_cuda-ipc_tp1_ipc-on_c32_Qwen3-VL-8B_3x10) |
| 20260405_115634 | tp4, ipc-off, c16 | [查看](https://github.com/yhyang201/sgl-bench/tree/main/records/20260405/20260405_115634_cuda-ipc_tp4_ipc-off_c16_Qwen3-VL-8B_3x1) |
| 20260405_120126 | tp4, ipc-off, c32 | [查看](https://github.com/yhyang201/sgl-bench/tree/main/records/20260405/20260405_120126_cuda-ipc_tp4_ipc-off_c32_Qwen3-VL-8B_3x1) |
| 20260405_120630 | tp4, ipc-on, c16 | [查看](https://github.com/yhyang201/sgl-bench/tree/main/records/20260405/20260405_120630_cuda-ipc_tp4_ipc-on_c16_Qwen3-VL-8B_3x10) |
| 20260405_121101 | tp4, ipc-on, c32 | [查看](https://github.com/yhyang201/sgl-bench/tree/main/records/20260405/20260405_121101_cuda-ipc_tp4_ipc-on_c32_Qwen3-VL-8B_3x10) |
| 20260405_132418 | probe, tp1, ipc-off | [查看](https://github.com/yhyang201/sgl-bench/tree/main/records/20260405/20260405_132418_probe_max_images_tp1_ipc-off_Qwen3-VL-8B) |
| 20260405_140137 | probe, tp1, ipc-on | [查看](https://github.com/yhyang201/sgl-bench/tree/main/records/20260405/20260405_140137_probe_max_images_tp1_ipc-on_Qwen3-VL-8B) |
