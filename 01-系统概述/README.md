# 系统概述

| 文档 | 内容 |
|---|---|
| `FINAL_REPORT.md` | **主报告**。系统架构、量化方案、算子实现、性能数据、结论 |
| `ACCEPTANCE.md` | 验收摘要：功能项、性能项、硬件项逐条判定 |
| `HARDWARE_LIMITS.md` | 硬件上限分析：内存带宽、NEON 发射、DDR3 仲裁、PL 通路 |
| `CPU_PROFILE_RESULTS.md` | 逐 token 性能剖析：各算子耗时分解、瓶颈定位 |
| `FINAL_BW_ALIGNMENT.md` | 带宽标定：实测值 vs 理论峰值，多路测量交叉验证 |
| `DEMO_GUIDE.md` | 演示指南：启动步骤、可提问示例、交互命令 |
| `ATTN_C_KERNEL.md` | 注意力 C 内核设计说明 |
| `GDN_REC_OPT_RESULTS.md` | GDN 递归状态优化（布局转置消除跨步访问）结果 |

**建议阅读顺序**：`FINAL_REPORT.md` → `ACCEPTANCE.md` → `CPU_PROFILE_RESULTS.md` → `HARDWARE_LIMITS.md`
