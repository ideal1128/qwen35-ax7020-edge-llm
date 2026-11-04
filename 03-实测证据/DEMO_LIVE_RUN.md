# 板上实时生成实测（演示证据）

> 记录时间：2026-11-04 17:1x（板内时钟，无 RTC，显示 1970-01-03）
> 板子状态：**刚做过掉电重启自恢复测试之后**（见 `REBOOT_TEST.md`），`up 11 min`，
> `./accept.sh` → `failures: 0   skipped: 1   RESULT: PASS`（唯一 skip 是 `cpufreq not readable`），
> `free -m` 可用 854 MB，`/mnt/qwen/*.bit.bin` 三个位流都在，`/dev/xdevcfg` 存在。
>
> 目的：给"演示功能可用"留一份**当场的、可核对的**原始输出，同时顺手把
> **冷/热页缓存对首 token 的影响**量化出来（这个数字以前文档里没有）。
>
> ⚠️ **先读这一条，否则会误读下面的输出**：这一份里的两次运行都**故意**加了
> `--max 28` / `--max 20`，因为目的是**数 token、测 ms/tok**，不是让模型把话说完。
> 所以它们都以 `[gen] stop: context budget reached` 结束——这是**测试参数**，不是能力上限。
> 引擎默认 `--max 0` = **不设上限**，模型自己吐 `<|im_end|>` 才停；
> 自然停止的真实对话在 `01-交接文档/DEMO_GUIDE.md` §2（`reply budget 1940` →
> `[gen] stop: <|im_end|> (natural end of answer, not truncated)`）。
> 演示时按 `START_DEMO.cmd` 走，交互那句是裸 `./real_chat_v3`，**没有** `--max`。

---

## 1. 用的命令

```bash
cd /mnt/qwen
./real_chat_v3 --cps 20320,22909,65292,35831,29992,19968,21477,35805,20171,32461,20320,33258,24049,12290 --max 28
```

`--cps` 是"按码点喂问题"，绕开串口输入法问题。这 14 个码点就是

> **你好，请问用一句话介绍你自己。**

引擎加上 ChatML 模板后是 27 个新 token（`[turn] feeding 27 new tokens`），其中 26 个进 prefill。

---

## 2. 第一次（冷页缓存，重启后第一次推理）

（这次只摘了关键几行，`[turn]` 那行与第二次相同：`feeding 27 new tokens, reply budget 28`）

```
[prefill] 45975.7 ms for 26 tokens (1768.3 ms/tok)
[gen] 28 tokens in 24380.3 ms (28 fwd, first 875.7 ms, avg 870.7 ms/tok = 1.148 tok/s)
[gen] unique ids 26/28
[gen] stop: context budget reached (28 tokens) -- answer may be cut
```

## 3. 第二次（热页缓存，紧接着同一句再问一次）

```
[init] engine ready in 0.224 s
[turn] feeding 27 new tokens (1 turns, 0 tok in KV), reply budget 20
[prefill] 17940.4 ms for 26 tokens (690.0 ms/tok)
[gen] 20 tokens in 17444.8 ms (20 fwd, first 863.1 ms, avg 872.2 ms/tok = 1.146 tok/s)
[gen] unique ids 20/20
[gen] stop: context budget reached (20 tokens) -- answer may be cut
[gen] ids = [48536, 6093, 43001, 1166, 16356, 18, 13, 20, 3698, 49889, 35429, 35740,
             43188, 56622, 35323, 35926, 44295, 37571, 43002, 1706]
[gen] utf8 = E4 BD A0 E5 A5 BD EF BC 81 E6 88 91 E6 98 AF 20 51 77 65 6E 33 2E 35
             EF BC 8C E9 98 BF E9 87 8C E5 B7 B4 E5 B7 B4 E9 80 9A E4 B9 89 E5 AE 9E
             E9 AA 8C E5 AE A4 E8 87 AA E4 B8 BB E7 A0 94 E5 8F 91 E7 9A 84 E8 B6 85
             E5 A4 A7 E8 A7 84 E6 A8 A1 E8 AF AD E8 A8 80 E6 A8 A1 E5 9E 8B E3 80 82
```

那 95 个 UTF-8 字节解出来是（37 个字符，以句号结尾）：

> **你好！我是 Qwen3.5，阿里巴巴通义实验室自主研发的超大规模语言模型。**

两次运行都因为 `--max` 预算被截断（`stop: context budget reached`），不是模型的问题；
自然停止（`stop: <|im_end|>`）另有样例，见 `01-交接文档/DEMO_GUIDE.md` §2。

---

## 4. 从这两次运行里读出来的四件事

### 4.1 冷页缓存一次性多花 28.0 秒（这是唯一的新数字）

| | prefill | 每 token |
|---|---|---|
| 冷（重启后第一次） | 45 975.7 ms / 26 tok | 1 768.3 ms |
| 热（紧接着第二次） | 17 940.4 ms / 26 tok | 690.0 ms |
| **差值** | **28 035.3 ms** | — |

643 MB 权重不在页缓存里时要从 SD 卡读，约 **23 MB/s**。引擎自己会打印
`[note] ... cold page cache can take ~29 s`，与实测的 28.0 秒一致。

**演示含义**：重启后第一次提问要等约 **42 秒**（13 秒 prefill + 28 秒读盘），
现场看起来像卡死。**必须先按 `DEMO_GUIDE.md` §0.2 预热。**

### 4.2 prefill 没有批处理，每个 prompt token 都是一次完整前向

热的 prefill 是 **690.0 ms/token**，和逐 token 生成的 670.7 ms 是同一个量级。
也就是说这个引擎**没有 batched prefill**：问 26 个 token 就要跑 26 遍 24 层网络、
读 26 遍全部权重（26 × 681 MB）。这既是"长 prompt 很贵"的原因，也是
`HARDWARE_LIMITS.md` 里所有下界都按"每 token 一次全量权重流"计算的原因。
若要提速，**batched prefill 是唯一还能砍掉一大块的方向**（不在本轮范围内）。

### 4.3 默认采样器比贪心慢约 200 ms/token

| 模式 | ms/token | tok/s |
|---|---|---|
| 贪心（`--greedy`，权威值） | 670.7 | 1.491 |
| 默认采样（本次两次） | 870.7 / 872.2 | 1.148 / 1.146 |
| 默认采样（早先那次，文档引用值） | 850.0 | 1.176 |

采样开销来自 `lm_head` 之后对 65 536 个 logit 做 top-k/top-p（走 float32 路径，
不能复用贪心的 int 归约）。**850.0 是较早一次运行的权威值**，本次 870.7/872.2
说明这个数在 **850–872 ms/tok** 之间随上下文长度浮动，文档里引用 850.0 时
不要把它当成精确常数。

### 4.4 中文是对的，而且是连贯的

37 个字符、95 个字节、以 `。` 结尾，模型自报身份与名字完全正确（Qwen3.5 / 通义实验室）。
这不是"能跑"，是"能用"。

---

## 5. 复现

```bash
# 1) 板子健康
cd /mnt/qwen && ./accept.sh          # 期望 RESULT: PASS

# 2) 冷页缓存对照（想复现 28 秒那次就 reboot 之后立刻跑）
./real_chat_v3 --cps 20320,22909,65292,35831,29992,19968,21477,35805,20171,32461,20320,33258,24049,12290 --max 20

# 3) 再跑一次（此时应看到 prefill ≈ 690 ms/tok）
```

> 抓串口输出时注意：`scratch/bx.ps1` 用的是 .NET `SerialPort` 默认的 ASCII 编码，
> 中文会被写成 `?`；要原样看到中文请用 **`scratch/bx8.ps1`**（已把
> `SerialPort.Encoding` 和 `[Console]::OutputEncoding` 都设成 UTF-8），
> 或者直接用 `demo/START_DEMO.cmd`（`demo_console.ps1` 本身也是按 UTF-8 收发和解码的）。
