# 演示手册 —— Qwen3.5-0.8B on ALINX AX7020

> 现场照着做即可。全程只用一根 USB 串口线：双击 `START_DEMO.cmd` 就有终端（见 §0）。

---

## 0. 开始之前（5 分钟）

### 0.1 先拿到一个能打字的终端

硬件：板子已上电；USB-UART 线插在板子的 **DEBUG/UART** 口（不是 USB-OTG 口）。
Windows「设备管理器 → 端口 (COM 和 LPT)」里应该看到一个 `COMx`（本机是 `COM4`，CP210x）。

**不需要装 PuTTY / Tera Term。** 双击包根目录的 `START_DEMO.cmd`，它执行的就是：

```
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0demo_console.ps1" -Port COM4 -Baud 115200 -Auto
```

带 `-Auto` 的意思是：连上后它自己 `cd /mnt/qwen && ./real_chat_v3` 把对话程序启动好，
等到 `you>` 提示符再把键盘交给你。**你不需要手动敲任何启动命令。**

整个文件夹可以拷到 U 盘带走，双击即用。若板子枚举成别的端口（以设备管理器为准），
用记事本改 `START_DEMO.cmd` 里的 `-Port COM4` —— 只有这一处。

手动方式：任意串口终端，**115200 8N1、无流控**，再用 **`root` / `root`** 登录
（密码不回显，输完直接回车）。

怎么算连上了：看到 `root@zynq:~#` 提示符（进了 `/mnt/qwen` 之后是 `root@zynq:/mnt/qwen#`）。
如果只看到 `zynq login:`，依次输入 `root`、回车、`root`、回车。

开机成功的标志是这段横幅（`startup.sh` 原文；连晚了就往上翻）：

```
==============================================================
  Qwen3.5-0.8B on ALINX AX7020
  readiness: PASS   (log: /mnt/qwen/boot.log)

  start a chat with:   cd /mnt/qwen && ./real_chat_v3
  health check with:   cd /mnt/qwen && ./accept.sh
==============================================================
```

控制台里的按键（2026-11-04 改过，注意和旧版不同）：

| 按键 | 作用 |
|---|---|
| `Enter` | 发送这一行 |
| `Ctrl-C` | **往板子发一个真正的 `^C`**，打断正在生成的长回答；**窗口不关**，回到 `you>` |
| `Ctrl-X` | 关闭这个控制台窗口（板子和 shell 继续跑，随后窗口停在 `pause`） |
| `Ctrl-D` 或在板子上敲 `/quit` | 退出对话程序，回到 `root@zynq:/mnt/qwen#` |

> 旧版是「`Ctrl-C` 直接关窗口」，那样只断了本地读循环，板子上的 Python 还在算几分钟，
> 下一次提问看起来就像被截断了。现在 `Ctrl-C` 是转发给板子的，`Ctrl-X` 才是关窗口。

### 0.2 演示前 5 分钟：预热（**必须做**）

板子冷启动后第一次推理要把 643 MB 权重从 SD 卡读进页缓存，**第一个 token 总共要等约 42–47 秒**
（prefill 本身约 13 秒 + 一次性 28–34 秒冷页缓存），现场演示时看起来像卡死。
**提前跑一次预热**，之后就只要 0.67–0.85 秒/token。

> 实测（2026-11-04，重启/清缓存后立刻 vs 紧接着再问一次）：
> - 26 token 提问：冷 `[prefill] 45975.7 ms` → 热 `[prefill] 17940.4 ms`，差 **28.0 秒**
> - 28 token 提问：冷 `[prefill] 47291.7 ms`（1689.0 ms/tok）→ 热 `[prefill] 22638.0 ms`（686.0 ms/tok）

**最省事的预热**：双击 `START_DEMO.cmd`（带 `-Auto`），它启动好对话程序后你随便问一句短的
（比如 `你好`），等它答完。那一句慢是正常的，之后每一句都快。

**或者用一条命令预热**（不进入交互）：

```bash
cd /mnt/qwen && ./accept.sh          # 自检 + 顺带把权重读热
```

看到 `[gen] ... avg 850 ms/tok` 左右（默认采样）就说明缓存热了。**这时才开始演示。**

预热后 `free -m` 的 `cached` 应该在 700 MB 左右。

> ⚠️ **不要再用 `--max 2` 预热。** 旧版手册写过 `timeout 200 ./real_chat_v3 --cps 20320 --max 2`，
> 那会先打印一段只有 2 个 token 的答案，现场看起来和「回答被截断」一模一样。已从手册和
> `demo_console.ps1` 里全部删除。

---

## 1. 开场：自检通过

```bash
cd /mnt/qwen && ./accept.sh
```

最后三行应该是：

```
==============================================
  failures: 0   skipped: 1
  RESULT: PASS
```

（`skipped: 1` 是 `cpufreq` 读不到，正常。）可以念一句：「硬件、权重、内存、Python 环境全部自检通过。」

---

## 2. 主演示：中文对话

```bash
cd /mnt/qwen && ./real_chat_v3
```

然后输入：

```
you> 你好
```

预期输出（默认采样，约 12 个 token）：

```
you> [turn] feeding 20 new tokens (1 turns, 0/4096 tok in KV, think=off), reply budget 4068
[prefill] 13588.1 ms for 19 tokens (715.2 ms/tok)
你好！很高兴见到你。有什么我可以帮助你的吗？
[gen] 12 tokens in 11050.0 ms (13 fwd, first 860.0 ms, avg 850.0 ms/tok = 1.176 tok/s)
[gen] unique ids 12/12
[gen] stop: <|im_end|> (natural end of answer, not truncated)
```

`reply budget` 是池子真实剩余容量（4096 − 已用 − 8 slack），**不是人为上限**。

同一句加 `--greedy` 更快、更短（8 个 token、9 次 forward）：

```
[gen] 8 tokens in 6036.3 ms (9 fwd, first 690.0 ms, avg 670.7 ms/tok = 1.491 tok/s)
```

**注意 `[prefill]` 那 13.6 秒**：模型要把你的问题逐 token 过一遍 24 层网络，
每 token 715 ms，**19 个 token** 就是 13.6 秒。**这是正常的**，念一句「它正在读你的问题」即可。

再输入：

```
you> 你是谁
```

预期（只喂 12 个增量 token，约 8 秒）：

```
you> [turn] feeding 12 new tokens (2 turns, 32/4096 tok in KV, think=off), reply budget 4044
[prefill] 8053.0 ms for 11 tokens (732.1 ms/tok)
我是 Qwen3.5，由阿里巴巴通义实验室自主研发。
```

**对比价值**：第二轮只喂了 12 个新 token（KV 里已有 32 个），
如果重新喂全部历史就是 44 个 token / 约 31 秒。这是增量 KV 复用的效果。

**要讲的三个数字**（贪心口径；现场默认采样是 850.0 ms/token，见 §3.3）：
- **670.7 ms/token**
- **1.491 tok/s**
- 相对未优化基线 **11,889 ms / 0.0841 tok/s**，**17.7 倍**

---

## 3. 加分项（可选）

### 3.1 证明中文是真算出来的，不是预置的

```bash
./real_chat_v3 --cps 20320,22909 --greedy
```

看 `[gen] ascii-safe = '...'` 这一行，里面是 Unicode 转义，
现场可以用手机/计算器核对 `\u4f60` = 你、`\u597d` = 好。

再补一句更有说服力的：最后会打印

```
[gen] stop: <|im_end|> (natural end of answer, not truncated)
```

说明是**模型自己输出结束符停下的**，不是程序截断的。

### 3.2 证明多轮上下文

连续问两轮，第二轮的回答会引用第一轮 —— 上面 §2 已经覆盖。

### 3.3 解码方式说明

**现在默认就是采样**（T=0.7 / top_k=40 / top_p=0.9），因为贪心解码在长回答上会重复
（实测 48 个 token 里有 11 个重复）。启动横幅会打印：

```
  ChatML | 4096 ctx | no hard cap (stops at <|im_end|>)
  reasoning: off
  decoding: sampling T=0.70 top_k=40 top_p=0.90
```

想切回贪心（更快、但可能重复）：

```bash
./real_chat_v3 --greedy
```

| 模式 | 速度 | 特点 |
|---|---|---|
| 采样（默认） | 850.0 ms/token（1.176 tok/s） | 不重复，适合长回答 |
| `--greedy` | 670.7 ms/token（1.491 tok/s） | 比采样省 21.1% 时间（吞吐 +26.8%），长回答可能复读 |

**回答长度不再有硬上限**：模型输出 `<|im_end|>` 就停，只有上下文快满时才截断
（会打印 `[gen] stop: … budget reached` 提示）。正常结束会打印
`[gen] stop: <|im_end|> (natural end of answer, not truncated)`。

### 3.4 思考模式（`/think`）—— 展示模型原始能力

Qwen3.5 本身支持「先思考再回答」。演示时可以在 `you>` 里当场开关：

```
you> /think            # 查看当前状态
you> /think on         # 下一轮开启思考
you> /think off        # 关掉（默认）
you> /think show       # 把思考过程打印出来（默认就是 show）
you> /think hide       # 只显示最终答案，思考期间打点
```

开启后横幅和 `/stats` 都会显示 `think=on`，生成时先出现 `[think]` 段、再出现 `[answer]` 段，
最后统计会打印 `[gen] reasoning: N tok, X s before the first answer token`。

> ⚠️ **现场别真的开。** 板子上思考模式下一条回答要 **1.5–4 分钟**（思考动辄 100+ token，
> 每 token 0.87 秒）。答辩时建议只演示 `/think on` → `/think` 看状态 → `/think off`，
> 或者提前录屏。要演示真实思考过程就预留 3 分钟。

---

## 4. 如果被问到"为什么不用 FPGA 加速"

这是最可能被问的问题，答案是有实测证据的，不是"没做完"：

| 层次 | 证据 |
|---|---|
| 做了 | 写了 dequant GEMV 核心，4 个算术 bug 全修，独立浮点参考 10/10 过，误差 2.35e-5 |
| 时序够 | 166.67 MHz 下 WNS **+0.362 ns** 余量 |
| 但上板坏了 | 同一输入四次运行给出四个结果（corr 0.744 / 0.708 / 0.655 / 0.678），降到 100 MHz 仍错 |
| 就算完美也超不过 | 回读修好、再叠第二轮引擎，PL 最好 ≈ **672 ms**（1.49 tok/s），仍不优于 CPU 的 **670.7 ms**（1.491 tok/s） |

**一句话结论**：同样搬 681.1 MB，PL 的 1178 MB/s 要 **578 ms**，CPU 缓存的 1270.8 MB/s 只要 **536 ms**；
再算上 PL 必须多付的回读与 PS 侧开销，**PL 结构性赢不了 CPU**。

---

## 5. 已知限制（被问到就直说）

| 限制 | 说明 |
|---|---|
| 速度 | 0.67 秒/token（贪心）/ 0.85–0.87（采样），一段 30 token 的回答要 20–26 秒 |
| 上下文 | 默认 4096 token（KV 池 96.0 MB）；`--ctx 8192` 可开到 8192（池 192 MB，会挤掉权重页缓存，反而变慢） |
| 词表 | 65536（从原始 151k 裁剪），**emoji / 生僻字 / 韩文输入会被静默丢弃** |
| 采样 | **默认开**（T=0.7/top_k=40/top_p=0.9）；`--greedy` 切回贪心（更快，长回答可能重复） |
| 思考模式 | 支持，但一条回答 1.5–4 分钟；默认关闭，`/think on` 打开 |
| 生成长度 | **无硬上限**，模型输出 `<|im_end|>` 自然停 |
| 时钟 | 板子无 RTC 电池，`date` 显示 1970 年，不影响推理 |

---

## 6. 应急

| 症状 | 处理 |
|---|---|
| 第一次提问后迟迟不出字 | 正常，冷页缓存下首个 token 要 ~29 秒；先按 §0.2 预热 |
| 输出乱码 | 多半是波特率不对（必须 115200）；再确认终端编码是 UTF-8 |
| 输入中文没反应 | 同上，或改用 `--cps 码点` 方式 |
| `RESULT: FAIL` | 看 `./accept.sh` 哪一项 FAIL；常见是内存不足，重启板子 |
| 进程卡死 | `Ctrl-C`；再不行**断电重启**（复位按钮无效） |

---

## 7. 命令速查

```bash
cd /mnt/qwen

./accept.sh                            # 自检（顺带把权重读进页缓存）
./real_chat_v3                         # 交互对话（默认采样、4096 ctx、思考关）
./real_chat_v3 --greedy                # 贪心解码（快，可能复读）
./real_chat_v3 --ctx 8192              # 开大 KV 池到 8192 token
./real_chat_v3 --think                 # 启动时就开思考模式
./real_chat_v3 --cps 20320,22909       # 单次「你好」
./real_chat_v3 --max 32                # 显式限制生成长度
./real_chat_v3 --sample 0.5,50,0.95    # 自定义采样参数
./install_startup.sh                   # 重装开机自启

# 预热：直接问一句短的就行，不要用 --max 2
./real_chat_v3 --cps 20320,22909
```

交互模式内（在 `you>` 提示符下敲，**不是 DSH 的命令**）：

```
/clear      清空历史和 KV 缓存（/new、/reset 等价；打印 [clear] conversation and KV cache cleared）
/think      查看思考状态（[think] reasoning is off, text is shown）
/think on   下一轮开启思考        /think off   关闭
/think show 显示思考文本          /think hide  隐藏
/stats      轮次 + KV 占用 + 思考状态（[stats] N turns, M/4096 tok in KV, think=off）
/help       命令列表
/quit       退出（/exit、/q、Ctrl-D 等价）
```

> 旧版只有 `/reset`、`/stats`、`/quit`，且 `/reset` 不清 KV。现在 `/clear` 是完整的
> 「历史 + KV 缓存」清空，`/stats` 也会显示真实池容量（4096）而不是轮次数。

---

## 8. 开机自启说明

已装好（`/sd/startup.sh`），开机后串口会打印：

```
==============================================================
  Qwen3.5-0.8B on ALINX AX7020
  readiness: PASS   (log: /mnt/qwen/boot.log)

  start a chat with:   cd /mnt/qwen && ./real_chat_v3
  health check with:   cd /mnt/qwen && ./accept.sh
==============================================================
```

**注意：自启不会自动开对话**，只做自检和提示。这是故意的 ——
把交互程序绑到 `/dev/ttyPS0` 会和登录 shell 抢终端，
一旦卡住板子就彻底失联（串口是唯一通道，`eth0` 是 NO-CARRIER）。

> 自启**尚未做过断电重启验证**。如果时间允许，答辩前做一次：
> 断电 → 上电 → 串口看横幅是否出现 → 跑 `./accept.sh`。

---

## 9. 故障排查（现场最常遇到的三个）

| 症状 | 原因 | 处理 |
|---|---|---|
| 双击 `START_DEMO.cmd` 毫无反应 | PowerShell 执行策略，或缺 `System.IO.Ports` | 开一个 `cmd` 窗口，`cd` 到包目录后运行 `START_DEMO.cmd`，报错会留在窗口里 |
| 板子明明已登录，控制台却是乱码 | 波特率不对 | 必须是 **115200 8N1、无流控**（`START_DEMO.cmd` 里的 `-Baud` 也是 115200） |
| 第一个 token 要等很久 | 预热后 ~14–23 秒（19–33 个 prompt token 的 prefill，正常）；**冷启动后第一次 42–50 秒** | 别冷启动：先跑 `cd /mnt/qwen && ./accept.sh` 把权重读热，再开始演示（**不要用 `--max 2` 预热**） |
| 回答中途自己停了、像被截断 | 看 `[gen] stop:` 那一行 | `<|im_end|> (natural end of answer)` = 模型正常说完；`budget reached` = 池子满了，用 `/clear` 或 `--ctx 8192` |
| 敲了 `/clear` 或 `/think` 却被当成聊天内容 | 用的是旧版 `chat_v3.py` | 确认板子上 `/mnt/qwen/chat_v3.py` 是 28105 B / md5 `d7bc2096a0603581881f0cd3e710f279`；不对就用 `04-板载脚本与工具\chat_v3.py` 重新推一次 |
| 按 `Ctrl-C` 窗口直接关了，板子还在算 | 旧版 `demo_console.ps1` | 新版 `Ctrl-C` 转发给板子、`Ctrl-X` 才关窗口；用包里这份覆盖 |

---

> **口径更正（2026-11-04）**：每 token 实际搬运 **681.1 MB**（不是 777.6 MB）。纯读墙 **1270.8 MB/s**。
> 各算子按自身上限加总的复合下限 **592 ms = 1.69 tok/s**；各阶段取历史最佳加总 **653.3 ms = 1.53 tok/s**。
> 当前实测 **670.7 ms = 1.491 tok/s**，即复合下限的 88.3%、历史最佳加总的 97.4%。
> 详见 `HARDWARE_LIMITS.md` §1/§2 与 `MEASURED_CORRECTIONS.md` §7。
