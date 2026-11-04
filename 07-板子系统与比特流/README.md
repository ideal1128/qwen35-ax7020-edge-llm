# 板子系统镜像与比特流

| 文件 | 大小 | 说明 |
|---|---|---|
| `BOOT.BIN` | 2.30 MB | Zynq 启动镜像（FSBL + PL 比特流 + U-Boot），由 `ps7_init` 设定 PL 时钟 100 MHz |
| `image.ub` | 4.05 MB | Linux 内核 + 设备树 + 根文件系统镜像（FIT 格式） |
| `gemv_sys.bit` | 4.05 MB | PL 加速系统比特流（完整含 PS 配置） |
| `gemv_sys.bit.bin` | 4.05 MB | 同上，raw 二进制形式，可通过 `/dev/xdevcfg` 动态加载 |

## 加载 PL 比特流

```bash
cat /mnt/qwen/gemv_sys.bit.bin > /dev/xdevcfg
```

加载前建议先确认时钟配置未被改动（PL 时钟由 BOOT.BIN 中的 `ps7_init` 固定为 100 MHz）。
