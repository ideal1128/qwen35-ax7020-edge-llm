# 板载脚本与工具

在 ALINX AX7020 的 Debian 8 / Python 3.4.2 环境上运行的全部代码。

## 运行入口

| 文件 | 说明 |
|---|---|
| `real_chat_v3` | 启动脚本：挂载分区、设置环境变量、拉起控制台 |
| `chat_v3.py` | 交互控制台：多轮上下文、`/clear`、`/think`、`/stats` |
| `qwen_real_engine.py` | 推理引擎：权重 mmap、24 层前向、分页 KV Cache、RoPE |
| `qwen_tokenizer.py` | 分词器 Python 封装 |

## 算子源码（C + NEON）

| 文件 | 说明 |
|---|---|
| `qwen_kernels_opt.c` | GEMV、RMSNorm、SiLU、GDN 等核心 NEON 内核 |
| `qwen_attn.c` | 全注意力：QK-L2 归一化、RoPE、分页 KV、输出门控 |
| `qwen_sampling.c` | 采样：top-k / top-p / 温度，argmax 快路径 |
| `libqwen_kernels.so` | 算子库编译产物 |
| `libqwen_attn.so` | 注意力库编译产物 |
| `libqwen_sampling.so` | 采样库编译产物 |
| `libtokenizer.so` | 分词器库（源码见 `06-模型与权重/tokenizer.c`） |

## 验收与测试

| 文件 | 说明 |
|---|---|
| `accept.sh` | 一键验收：内存、权重完整性、推理、输出正确性 |
| `accept_prompts.sh` | 固定 prompt 的回归测试集 |
| `expected.md5` | 权重前 1 MB 的 md5 基准（`accept.sh` 用它校验权重未被篡改） |
| `attn_post_verify.py` | 注意力后处理正确性验证 |
| `multiturn_test.py` | 多轮上下文测试 |
| `real_text_test.py` | 真实文本生成测试 |
| `repl_test.sh` | 交互命令回归测试 |

## 开机自启

| 文件 | 说明 |
|---|---|
| `startup.sh` | 启动脚本本体 |
| `install_startup.sh` | 安装到 init 系统，实现掉电重启后自动运行 |

## 编译

```bash
gcc -std=c99 -O3 -mcpu=cortex-a9 -mfpu=neon -mfloat-abi=hard -DUSE_NEON \
    -shared -fPIC -o libqwen_kernels.so qwen_kernels_opt.c -lm -fopenmp
```

其余两个库的编译命令见各源文件头部注释。
