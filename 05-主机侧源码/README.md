# 主机侧源码

在开发主机上运行、用于准备板载运行环境的工具链。

| 文件 | 说明 |
|---|---|
| `prune_vocab_64k.py` | 将原始词表切分至 64K 子集，生成新旧 ID 双向映射 |
| `slice_embeddings_64k.py` | 按新词表切分 embedding 矩阵 |
| `qwen_kernels.c` | 板载 NEON 算子库的主机侧源码 |
| `qwen_tokenizer.py` | 分词器参考实现，用于与板载 C 版本逐 token 校验 |
| `membw.c` | 内存带宽标定工具，用于测出 DDR3 读墙 |
| `profile_kernels.py` | 算子级性能剖析脚本 |

权重 BFP8 量化导出脚本、逐张量参数清单与校验工具见
[`06-模型与权重/README.md`](../06-模型与权重/README.md)。
