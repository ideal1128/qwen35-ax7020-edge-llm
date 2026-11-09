# 板子系统与比特流

可直接烧录/加载的板级系统文件。

| 文件 | 大小 | 说明 |
|---|---|---|
| `BOOT.BIN` | 2.2 MB | 启动镜像：FSBL + bitstream + U-Boot，内含 `ps7_init`（设定 FCLK0 = 100 MHz） |
| `image.ub` | 3.9 MB | Linux 内核 + 设备树 + initramfs |
| `gemv_sys.bit.bin` | 3.9 MB | PL 比特流，Linux 下可直接加载的二进制格式 |

## 烧录

`BOOT.BIN` 与 `image.ub` 放入 SD 卡第一个 FAT 分区，从 SD 卡启动。

## 加载 PL 比特流

```bash
cat gemv_sys.bit.bin > /dev/xdevcfg
```

或通过 fpga_manager：

```bash
echo 1 > /sys/class/fpga_manager/fpga0/flags
cat gemv_sys.bit.bin > /sys/class/fpga_manager/fpga0/firmware
```

> 演示与推理默认使用纯 CPU 路径（`real_chat_v3` 不加载比特流），
> 因此不加载 PL 也能完整运行。
