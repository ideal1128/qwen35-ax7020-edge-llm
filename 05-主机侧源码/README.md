# 主机侧源码

在开发主机上运行、用于准备板载运行环境的工具链。

## 模型转换

| 文件 | 说明 |
|---|---|
| `prune_vocab_64k.py` | 将 151,936 词表切分至 64K，生成新旧 ID 双向映射 |
| `slice_embeddings_64k.py` | 按新词表切分 embedding 矩阵 |
| `export_qwen35_weights_bfp8_64k.py` | 权重 BFP8 分块量化导出（尾数/指数分离布局） |
| `bfp8_packer.py` | BFP8 分组打包实现 |
| `verify_exported_weights.py` | 导出结果逐张量浮点校验 |

转换流程：

```bash
python3 prune_vocab_64k.py
python3 slice_embeddings_64k.py
python3 export_qwen35_weights_bfp8_64k.py
python3 verify_exported_weights.py
```

> 注意：导出脚本的文件路径是硬编码的，运行前需按本地目录修改文件头部的常量。
> 依赖 `numpy` 与 `safetensors`。

## 性能测量

| 文件 | 说明 |
|---|---|
| `membw.c` | 内存带宽标定工具，用于测出 DDR3 读墙 1270.8 MB/s |
| `profile_kernels.py` | 算子级性能剖析，产出逐算子耗时分解 |

`membw.c` 必须带 `-DPF` 编译才会启用预取；不加会低报约 36%：

```bash
gcc -O3 -mcpu=cortex-a9 -mfpu=neon -mfloat-abi=hard -DPF -o membw membw.c -lm
```
