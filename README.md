# 四机 DGX Spark (GB10) 跑 DeepSeek-V4.1-Flash MXFP4

4 台 DGX Spark（GB10，128GB 统一内存）纯 TP4、无 PD 分离，跑 DeepSeek-V4.1-Flash-1017 MXFP4（510.3GB 权重）的生产实测。

**交流群**：QQ 群 **1128010665**（加群备注：GitHub）

## 关键数字（全部本机实测）

| 指标 | 数值 |
|---|---|
| 解码 C1 | 106.7 tok/s（单流可达 110.6–113.7） |
| 解码 C4 | 222.8 tok/s |
| 解码 C8 | 317.0 tok/s |
| 解码 C16 | 439.9 tok/s |
| 干净预填（13.7K / 53.2K / 125.4K 输入） | 4,875 / 5,090 / 5,057 tok/s |
| 上下文 | 标称 1M，>300K 后交互体验明显下降 |

> 口径：流式请求、temperature=0、剥离 TTFT、三次取中位数；预填用随机 nonce 前缀并校验 `cached_tokens=0`。

## 文档

- [REPORT.md](REPORT.md)：完整测试报告（硬件、部署参数、逐项数据、踩坑记录）

## 拓扑

- 4 × DGX Spark GB10（sm_121），200GbE RoCE，无交换机环形
- TP4（每机一 rank），驱动 DGX OS 580 / CUDA 13.0
- 权重：`Infatoshi/DeepSeek-V4.1-Flash-1017-MXFP4-uncensored`（48 分片 / 96,085 张量）
