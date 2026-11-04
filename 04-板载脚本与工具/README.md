# 板载脚本与工具

在 ALINX AX7020 的 Debian 8 / Python 3.4.2 环境上运行的完整推理栈。

## NEON 算子库（C）

| 文件 | 说明 |
|---|---|
| `qwen_kernels_opt.c` | GEMV、RMSNorm、SiLU、卷积、GDN 等核心 NEON 内核 |
| `qwen_attn.c` | 注意力：QK-L2 归一化、RoPE、分页 KV cache、门控 |
| `qwen_sampling.c` | 采样：top-k / top-p / 温度，argmax 快路径 |
| `tokenizer.c` | BPE 分词器（C 实现，避免 Python 开销） |
| `pl_apply.c` | PL 输出后处理 |

编译产物为同名 `.so`，由 Python 通过 ctypes 调用。

## 推理引擎（Python）

| 文件 | 说明 |
|---|---|
| `qwen_real_engine.py` | 推理引擎：权重 mmap、24 层前向、分页注意力、KV 管理 |
| `qwen_tokenizer.py` | 分词器 Python 封装 |
| `chat_v3.py` | 交互式对话前端：多轮上下文、`/clear`、`/think`、`/stats` |

## 启动

```bash
cd /mnt/qwen && ./real_chat_v3
```

`real_chat_v3` 会自动挂载 `/mnt/qwen`、设置环境变量并拉起 `chat_v3.py`。

## 交互命令

| 命令 | 作用 |
|---|---|
| `/clear` `/new` `/reset` | 清空上下文 |
| `/think on` / `/think off` | 开启 / 关闭思考模式 |
| `/think show` / `/think hide` | 显示 / 隐藏思考内容 |
| `/stats` | 显示当前性能与上下文统计 |
| `/help` `/quit` | 帮助 / 退出 |

## 构建

`build_*.sh` 为板上编译脚本（gcc 4.9.2 + NEON），`accept.sh` 为自动化验收脚本。
