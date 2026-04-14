# CUDA IPC Per-request Batch vs Pool 测试指南

## 前置准备

```bash
cd /sgl-workspace/sgl-bench
pip install -e .
```

---

## 一、性能对比 (perf)

`-b` 指向目录 `presets/perf/ipc_test/`，自动跑目录下所有 bench config (n1/n5/n10)。

### Qwen3-VL-235B

```bash
# IPC 开启 (per-request batch, 新方案)
sgl-bench run \
  -s presets/server/qwen3_vl_235b/tp8_cuda_ipc.toml \
  -b presets/perf/ipc_test \
  -d "qwen3-vl-235b-ipc-on" \
  -g 0,1,2,3,4,5,6,7

# IPC 关闭 (CPU fallback, baseline)
sgl-bench run \
  -s presets/server/qwen3_vl_235b/tp8.toml \
  -b presets/perf/ipc_test \
  -d "qwen3-vl-235b-ipc-off" \
  -g 0,1,2,3,4,5,6,7
```

### Kimi-K2.5

```bash
# IPC 开启
sgl-bench run \
  -s presets/server/kimi_k2_5/tp8_cuda_ipc.toml \
  -b presets/perf/ipc_test \
  -d "kimi-k2.5-ipc-on" \
  -g 0,1,2,3,4,5,6,7

# IPC 关闭
sgl-bench run \
  -s presets/server/kimi_k2_5/tp8.toml \
  -b presets/perf/ipc_test \
  -d "kimi-k2.5-ipc-off" \
  -g 0,1,2,3,4,5,6,7
```

### 对比结果

```bash
# 查看所有实验
ls records/

# 对比 (替换为实际路径)
sgl-bench compare records/<日期>/qwen3-vl-235b-ipc-on/<experiment> \
                  records/<日期>/qwen3-vl-235b-ipc-off/<experiment>
```

**关注指标**：
- `mean_ttft_ms` / `p99_ttft_ms` — 首 token 延迟 (IPC 开销在这里)
- `request_throughput` — 吞吐量
- `output_throughput` — 输出 token/s

---

## 二、压力测试 (stress)

### 5 分钟冒烟测试

```bash
# Qwen3-VL-235B
sgl-bench stress \
  -s presets/server/qwen3_vl_235b/tp8_cuda_ipc.toml \
  -b presets/stress/soak_5m_smoke.toml \
  -d "qwen3-vl-235b-stress-smoke" \
  -g 0,1,2,3,4,5,6,7

# Kimi-K2.5
sgl-bench stress \
  -s presets/server/kimi_k2_5/tp8_cuda_ipc.toml \
  -b presets/stress/soak_5m_smoke.toml \
  -d "kimi-k2.5-stress-smoke" \
  -g 0,1,2,3,4,5,6,7
```

### 1 小时长跑 (检测显存泄漏)

```bash
# Qwen3-VL-235B
sgl-bench stress \
  -s presets/server/qwen3_vl_235b/tp8_cuda_ipc.toml \
  -b presets/stress/soak_1h.toml \
  -d "qwen3-vl-235b-stress-1h" \
  -g 0,1,2,3,4,5,6,7

# Kimi-K2.5
sgl-bench stress \
  -s presets/server/kimi_k2_5/tp8_cuda_ipc.toml \
  -b presets/stress/soak_1h.toml \
  -d "kimi-k2.5-stress-1h" \
  -g 0,1,2,3,4,5,6,7
```

**同时监控** (另开终端)：

```bash
# GPU 显存 — 不应持续上涨
watch -n 2 nvidia-smi

# ShmSyncBuffer 文件数 — 应稳定不增长
watch -n 5 'ls /dev/shm/psm_* 2>/dev/null | wc -l'
```

**判定标准**：
- stress 报告所有窗口 `health_ok = true`
- 无 `abort_reason`
- GPU 显存稳定波动，不持续上涨
- `/dev/shm/psm_*` 文件数稳定

---

## 三、测试矩阵总结

| 模型 | 测试类型 | 配置 | 预计耗时 |
|------|---------|------|---------|
| Qwen3-VL-235B | perf (IPC on) | n1/n5/n10 x c32 | ~15min |
| Qwen3-VL-235B | perf (IPC off) | n1/n5/n10 x c32 | ~15min |
| Qwen3-VL-235B | stress smoke | 5min soak | ~8min |
| Qwen3-VL-235B | stress long | 1h soak | ~65min |
| Kimi-K2.5 | perf (IPC on) | n1/n5/n10 x c32 | ~15min |
| Kimi-K2.5 | perf (IPC off) | n1/n5/n10 x c32 | ~15min |
| Kimi-K2.5 | stress smoke | 5min soak | ~8min |
| Kimi-K2.5 | stress long | 1h soak | ~65min |

建议顺序：先跑 stress smoke 确认基本稳定，再跑 perf 对比，最后跑 stress 1h 长跑。
