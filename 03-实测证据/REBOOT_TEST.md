# 掉电重启自恢复实测

**日期**：2026-11-04 10:28（主机时间）
**目的**：证明板子**不需要人工干预**就能从一次完整重启回到可推理状态。这是答辩现场"不会翻车"的关键证据 —— 也是唯一一项此前只有设计、没有实测的验收项。
**结论**：**PASS**。

---

## 1. 为什么必须测这一项

板子上的推理栈由四段拼起来，任何一段没接上，演示当场就是黑屏：

| 环节 | 依赖 |
|---|---|
| 大文件 | `/dev/mmcblk0p3`（13 GB FAT32）挂到 `/mnt/qwen`，643 MB 权重在里面 |
| 启动脚本 | `/dev/mmcblk0p1` 挂到 `/sd`，`/etc/rc.local` 调 `/sd/startup.sh` |
| 健康检查 | `startup.sh` 里跑 `accept.sh`，结果写 `/mnt/qwen/boot.log` |
| 提示 | 检查通过后往串口打印一段横幅 |

此前只验证过"手工敲命令能跑通"，没验证过"上电 40 秒后自己就好了"。

## 2. 方法

```powershell
# 05-主机侧源码\reboot_test.ps1 —— 打开 COM4 / 115200，全程捕获串口流
powershell -NoProfile -ExecutionPolicy Bypass -File .\reboot_test.ps1
```

脚本做的事：打开串口 → 排空缓冲 → 发 `Ctrl-C` → `sync` → `reboot` → 把整段串口输出写入 `03-实测证据\reboot_capture.log`（23 049 字符）→ 检测横幅与 `login:` → 自动以 `root`/`root` 登录 → 打印判定并返回退出码（0 = PASS）。

## 3. 完整启动链（摘自 `reboot_capture.log`）

```
U-Boot 2017.01 (Jun 22 2017 - 06:00:00)
DRAM:  1 GiB
*** Warning - bad CRC, using default environment
...
Linux 4.9.0-xilinx ...
[  OK  ] Started Login Service.
[  OK  ] Started Network Manager.

==============================================================
  Qwen3.5-0.8B on ALINX AX7020
  readiness: PASS   (log: /mnt/qwen/boot.log)

  start a chat with:   cd /mnt/qwen && ./real_chat_v3
  health check with:   cd /mnt/qwen && ./accept.sh
==============================================================
[  OK  ] Started /etc/rc.local Compatibility.
[  OK  ] Started Serial Getty on ttyPS0.
[  OK  ] Reached target Multi-User System.

Debian GNU/Linux 8 zynq ttyPS0

zynq login:
```

横幅是 `startup.sh` 在 `accept.sh` 返回 PASS **之后**才打印的，所以横幅出现本身就等价于"推理栈自检通过"。

## 4. 二次确认（登录后立即执行）

| 检查 | 结果 |
|---|---|
| `uptime` | `up 0 min` |
| `/proc/uptime` | `25.00` 秒 |
| `/mnt/qwen/boot.log` | 行数 13 → **16**，新增三行：`[1970-01-03 17:10:00] startup.sh begin` / `17:10:05 accept.sh -> PASS` / `status=PASS` |
| 空闲内存 | 854 MB / 1006 MB |
| `/mnt/qwen` 占用 | 767 MB / 13 GB |
| 权重可见 | 是 |

（板上没有 RTC，所以日志时间戳是 1970；判读启动时间要看 `uptime`，不要看 `boot.log` 的时间。）

## 5. 两个已知的无害现象

1. **`*** Warning - bad CRC, using default environment`** —— U-Boot 环境变量区的 CRC 失效，回落到编译默认值。默认值里的 `bootcmd` 正好指向正确的 QSPI 启动镜像，所以不影响启动。**不要"修"它**：一旦执行 `saveenv`，写入的环境变量如果不对，反而会启动失败。
2. **重启后 PL 里的设计被冲掉**，回到 ALINX 出厂的 HDMI 例程。所有 PL 实验（`gemv_sys.bit.bin`）每次重启后都必须重新加载：
   ```
   cat /mnt/qwen/gemv_sys.bit.bin > /dev/xdevcfg
   python3 /mnt/qwen/pl_clk.py 0x00100600
   ```
   纯 CPU 推理路径（`./real_chat_v3`）不受影响。

## 6. 尚未做的验证

**物理拔电（冷启动）没有做。** 软件 `reboot` 会完整走一遍内核重载和 `rc.local`，但它**不重新走 QSPI 上的 FSBL → bitstream → U-Boot** 这一段（那一段只在冷启动时发生）。建议交接后补一次"拔电源 30 秒再上电"的测试，预期行为与本次完全一致，但这是唯一还留着的空白。

## 7. 复现

```powershell
cd E:\ds_workplace
powershell -NoProfile -ExecutionPolicy Bypass -File .\scratch\reboot_test.ps1
# 退出码 0 = PASS；完整串口流自动存到 E:\ds_workplace\logs\reboot_<时间戳>.log
```
