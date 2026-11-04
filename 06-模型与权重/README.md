# 模型与权重

本目录提供重建量化权重所需的全部元数据与工具。

## 文件

| 文件 | 大小 | 说明 |
|---|---|---|
| `tokenizer_64k.bin` | 735 KB | 切分至 64K 词表的 BPE 分词器二进制 |
| `tokenizer.c` / `tokenizer.h` | 8 KB | 板载 C 分词器实现 |
| `tb_tokenizer.c` | 2 KB | 分词器 testbench |
| `old_to_new_64k.json` | 922 KB | 原始词表 ID → 切分后 ID 映射 |
| `new_to_old_64k.json` | 408 KB | 反向映射 |
| `qwen35_weights_64k_manifest.json` | 98 KB | **逐张量参数清单**：320 个张量的名称、形状、数据类型、文件偏移、Physical 地址、字节数 |
| `model.safetensors.index.json` | 51 KB | 原始模型分片索引 |
| `export_qwen35_weights_bfp8_64k.py` | 11 KB | **BFP8 量化导出脚本**（尾数/指数分离布局） |
| `bfp8_packer.py` | 4 KB | BFP8 分组打包实现 |
| `verify_exported_weights.py` | 4 KB | 导出结果逐张量浮点校验 |
| `config.json` | 3 KB | 模型结构配置 |
| `chat_template.jinja` | 8 KB | ChatML 对话模板 |

## 量化布局

每个张量在权重文件中占据一段连续区间：

```
[ M×K 个 int8 尾数 ][ M×K/64 个 uint8 指数 ]
```

- 每 64 个元素共享一个 `uint8` 指数，实际缩放为 `2^(exp - 134)`
- 尾数为 `clip(round(w × 2^-(exp-134)), -127, 127)` 的 int8
- 有效位宽 **8.125 bit/元素**，全模型约 643 MB

## 重建权重

权重二进制因体积超出仓库容量，通过发布附件单独分发。如需自行重建：

```bash
python export_qwen35_weights_bfp8_64k.py     # 先按需修改脚本顶部的路径常量
python verify_exported_weights.py            # 逐张量浮点校验
```

`qwen35_weights_64k_manifest.json` 提供逐张量的偏移与字节数，可据此直接定位任意张量。
