# 模型与权重

模型元数据、词表、分词器实现，以及权重的逐层参数清单。

## 模型配置

| 文件 | 说明 |
|---|---|
| `config.json` | Qwen3.5-0.8B 原始配置 |
| `chat_template.jinja` | ChatML 对话模板 |
| `model.safetensors.index.json` | 原始权重索引 |

## 本作品使用的结构参数

| 项目 | 值 |
|---|---|
| 层数 | 24（18 层 Gated DeltaNet + 6 层全注意力，间隔 4） |
| hidden / FFN | 1024 / 3584 |
| 注意力 | 8 个 Q 头、2 个 KV 头（GQA 4:1），head_dim 256 |
| GDN | key/value head_dim 128，各 16 头，卷积核 4 |
| RoPE | theta 1e7，partial_rotary_factor 0.25 |
| 词表 | 65,536（从 151,936 切分） |

## 词表与分词器

| 文件 | 说明 |
|---|---|
| `tokenizer_64k.bin` | 64K 词表二进制（运行时加载） |
| `new_to_old_64k.json` | 新 ID → 原 ID 映射（导出脚本的输入） |
| `tokenizer.c` / `tokenizer.h` | BPE 分词器 C 实现 |
| `tb_tokenizer.c` | 分词器测试，与 Python 参考逐 token 比对 |

## 权重清单

`qwen35_weights_64k_manifest.json` 记录了 **320 个张量**的逐层名称、形状、dtype 与
量化后偏移，是权重文件的完整索引。

## 权重文件

量化后的二进制（643 MB）与 embedding（134 MB）超出仓库容量，以 Release 附件形式提供。
配合 `05-主机侧源码/` 的导出脚本与上面的清单，可以从原始 safetensors 完整重建。
