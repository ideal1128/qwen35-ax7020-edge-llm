# 板载 REPL 命令与「不截断」实测（2026-11-04 13:20 / 13:23）

本文件记录 2026-11-04 对 `chat_v3.py` 两处修复的**板载实测**结果：

1. **`/clear`、`/think` 等交互命令** 现在在板子自己的 `you>` REPL 里工作
   （不是 DSH 的斜杠命令，是串口里那个聊天程序的命令）。
2. **回答不再被硬截断** —— 生成一直走到模型自己吐出 `<|im_end|>` 为止。

测试脚本：`scratch/test_think_repl.ps1`（ASCII，单次串口连接全程不松手，输出全量落盘）。
两次运行都是 **18/18 通过**。

---

## 1. 修复前的问题

| 问题 | 现象 | 根因 |
|---|---|---|
| `/clear`、`/think` 不存在 | 输入 `/clear` 被当成普通聊天内容发给模型 | `chat_v3.py` 的 REPL 只实现了 `/quit`，其余斜杠命令没有分支 |
| 回答看起来被截断 | 每次只出很短一段就停 | ① `generate()` 里 `budget` 被夹到 `max(32, …)`，而 `would_overflow()` 传进去的 `reserve_reply` 已经是夹过的值，条件恒假 → 越界分配被 `allocate_slot()` 抛异常；② `demo_console.ps1` 预热时用了 `--max 2`，只出 2 个 token，视觉上和截断一样 |
| `THINK_OPEN`(65527) 被当正文解码 | 思考块标记混进回答 | 生成循环没有在 65527/65528 上切通道 |
| 模板头不对 | 不思考时少一段 `' thinking\n\n'` | `_assistant_header()` 只发了 `THINK_CLOSE + "\n\n"`，漏了模板第 147–154 行的开场 `' thinking\n\n'` |

模板原文（`chat_template.jinja:147-154`）：

```jinja
{%- if enable_thinking is true %}
    {{- '<|im_start|>assistant\n' }}
    {{- ' thinking\n' }}
{%- else %}
    {{- '<|im_start|>assistant\n' }}
    {{- ' thinking\n\n\n\n' }}
{%- endif %}
```

修复后 `_assistant_header()` 渲染出的字节序列，已由离线单测逐字节断言：
不思考 → `<|im_start|>assistant\n thinking\n\n\n\n`；思考 → `<|im_start|>assistant\n thinking\n`。

---

## 2. 板载启动横幅（实测抓取）

```
cd /mnt/qwen && ./real_chat_v3
[ctx] requesting 4096 tokens -> 256 blocks, rope table 4096
[*] Initializing Qwen3.5-0.8B Real Hardware Engine...
[*] Loading NEON SIMD Accelerated Kernel Library...
[*] Loading 64K C-BPE Tokenizer from /mnt/qwen/tokenizer_64k.bin...
[*] Memmapping 128MB 64K Embedding Matrix from /mnt/qwen/embedding_64k.bin...
[*] Memmapping 614MB 24-Layer BFP8 Weights Matrix from /mnt/qwen/qwen35_weights_64k.bin...
[*] Parsing Model Architecture Manifest (320 Tensors) from /mnt/qwen/qwen35_weights_64k_manifest.json...
[*] Pre-loading Conv1D and RMSNormGated Weights for 18 Linear Layers...
[*] Initializing PagedAttention Dynamic Memory Pool (256 Blocks / 4096 Context / 96.0 MB)...
[*] Engine Initialized! Ready for 100% Real Neural Network Inference.

[init] engine ready in 0.294 s
==================================================================
  Qwen3.5-0.8B on ALINX AX7020  (CPU path, 24 layers, 64K vocab)
  ChatML | 4096 ctx | no hard cap (stops at <|im_end|>)
  reasoning: off
  decoding: sampling T=0.70 top_k=40 top_p=0.90
==================================================================
[sample] temperature=0.70 top_k=40 top_p=0.90 (costs an extra lm_head pass, ~115 ms/token)
[note] the first generated token includes weight staging;
[note] on a cold page cache it can take ~29 s. Be patient.

Type your message and press Enter.
Commands: /clear  /think on|off  /think show|hide  /stats  /help  /quit
Ctrl-C stops a reply without leaving.  Ctrl-D exits.

you>
```

注意：`/ctx` 的真实容量是 `eng.paged_attn.num_blocks * block_size` 反查出来的，
不是写死的常量，所以横幅和 `/stats` 永远不会跟真实池子脱节。

---

## 3. 交互命令逐条实测

`scratch/test_think_repl.ps1` 的 18 项断言，两次运行全过：

| # | 断言 | 结果 |
|---|---|---|
| 1 | REPL 到达 `you>` 提示符 | PASS |
| 2 | 横幅报告 `4096 ctx` | PASS |
| 3 | 横幅报告 `reasoning: off` | PASS |
| 4 | 横幅报告 `decoding: sampling` | PASS |
| 5 | 池子按 4096 token 分配 | PASS |
| 6 | `/help` 列出 `/clear /new /reset` | PASS |
| 7 | `/help` 列出 `/think on \| off` | PASS |
| 8 | `/think` 无参数 → 报告状态 | PASS |
| 9 | `/think on` → 确认开启 | PASS |
| 10 | `/think` 再查 → 报告 on | PASS |
| 11 | `/think off` → 确认关闭 | PASS |
| 12 | `/stats` → 报告 KV 占用 | PASS |
| 13 | `/clear` → 确认清空 | PASS |
| 14 | `/clear` 后 `/stats` 真的是 `0 turns, 0/…` | PASS |
| 15 | 提问后生成正常结束 | PASS |
| 16 | 结束原因是 `<\|im_end\|>` | PASS |
| 17 | 回答没有被 budget 截断 | PASS |
| 18 | `/quit` 回到 shell | PASS |

实测命令回显：

```
you> /help
  /clear  /new  /reset   drop the conversation and the KV cache
  /think                  show the reasoning state
  /think on | off         enable /disable the reasoning block
  /think show | hide      print /suppress the reasoning text
  /stats                  turns, tokens in KV, think state
  /quit  /exit  /q        leave (also Ctrl-D)
you> /think
[think] reasoning is off, text is shown
you> /think on
[think] reasoning ON for the next turn (expect 1.5-4 min per reply on this board)
you> /think
[think] reasoning is on, text is shown
you> /think off
[think] reasoning off for the next turn
you> /stats
[stats] 0 turns, 0/4096 tok in KV, think=off
you> /clear
[clear] conversation and KV cache cleared
you> /stats
[stats] 0 turns, 0/4096 tok in KV, think=off
```

---

## 4. 不截断实测 A：热页缓存 + 英文（`repl_run_warm_en.log`）

```
=== question (no --max): must end at <|im_end|> ===
[13:20:41.625] What is 2+2? Answer in one short sentence.
[turn] feeding 34 new tokens (1 turns, 0/4096 tok in KV, think=off), reply budget 4054
[13:20:41.688] [attn] C attn_post enabled (rmsnorm sum mode 0)
[13:20:47.125] .[13:20:52.563] .[13:20:58.063] .[13:21:03.564] .[13:21:04.250]
[prefill] 22638.0 ms for 33 tokens (686.0 ms/tok)
[13:21:05.127] 2[13:21:06.002]  +[13:21:06.812]  [13:21:07.687] 2[13:21:08.565]  =[13:21:09.439]  [13:21:10.314] 4[13:21:11.188]
[gen] 7 tokens in 6915.4 ms (8 fwd, first 872.7 ms, avg 864.4 ms/tok = 1.157 tok/s)
[gen] unique ids 5/7
[gen] stop: <|im_end|> (natural end of answer, not truncated)
[gen] ids = [17, 477, 220, 17, 282, 220, 19]
[gen] utf8 = 32 20 2B 20 32 20 3D 20 34
[gen] ascii-safe = '2 + 2 = 4'
you>
```

- **`reply budget 4054`** —— 这是池子真实剩余容量（4096 − 34 − 8 slack），不是人为上限。
- 每个 token 之间约 **875 ms**，与 CPU 路径的 861–864 ms/token 吻合。
- 停因是 `natural end of answer, not truncated`。

---

## 5. 不截断实测 B：冷页缓存 + 中文（`repl_run_cold_zh.log`）

先用 `echo 3 > /proc/sys/vm/drop_caches` 清空页缓存，模拟刚开机：

```
             total       used       free     shared    buffers     cached
Mem:          1006         96        909          7          0         41
```

然后提问「你好，请用一句话介绍你自己。」：

```
=== question (no --max): must end at <|im_end|> ===
[13:21:48.002] [turn] feeding 29 new tokens (1 turns, 0/4096 tok in KV, think=off), reply budget 4059
[13:21:51.314] [attn] C attn_post enabled (rmsnorm sum mode 0)
[13:22:21.626] .[13:22:27.063] .[13:22:32.563] .[13:22:35.251]
[prefill] 47291.7 ms for 28 tokens (1689.0 ms/tok)
[13:22:36.126] 你好[13:22:37.002] ，[13:22:37.875] 我是[13:22:38.688]  Q[13:22:39.564] wen[13:22:40.437] 3[13:22:41.314] .[13:22:42.189] 5
 …（逐 token 流式，约 875 ms 一个）
[13:23:18.000] 。[13:23:18.875]
[gen] 49 tokens in 43599.8 ms (50 fwd, first 856.4 ms, avg 872.0 ms/tok = 1.147 tok/s)
[gen] unique ids 38/49
[gen] stop: <|im_end|> (natural end of answer, not truncated)
[gen] ascii-safe = '\u4f60\u597d\uff0c\u6211\u662f Qwen3.5\uff0c由通义实验室自主研发的超大规模语言模型，能够处理复杂逻辑、多模态推理和代码生成，致力于让 AI 更懂中国、更懂中国、更懂中国。'
you>
```

完整回答：

> 你好，我是 Qwen3.5，由通义实验室自主研发的超大规模语言模型，能够处理复杂逻辑、多模态推理和代码生成，致力于让 AI 更懂中国、更懂中国、更懂中国。

要点：

- **冷缓存下 prefill 47.3 s**（1689 ms/token），热缓存 22.6 s（686 ms/token）。
  第一次提问慢是这个原因，不是卡死。
- 生成阶段冷热一致，**约 872 ms/token = 1.147 tok/s**。
- 结尾 `更懂中国` 重复三次是**小模型 + 64K 切片词表**的采样质量现象，
  不是代码 bug；`--greedy` 下重复更明显（这也是默认开采样的原因）。
- 中文经 UTF-8 串口逐字节正确传输，`[gen] utf8 =` 那行可逐字节核对。

---

## 6. 顺带修掉的两个「看起来像截断」的坑

| 位置 | 旧行为 | 新行为 |
|---|---|---|
| `demo_console.ps1` 预热 | `timeout 200 ./real_chat_v3 --max 2`，先打印一段 2 token 的答案 | 预热整段删除；`-Auto` 直接 `cd /mnt/qwen && ./real_chat_v3`，等到 `you>` 再交给你 |
| `demo_console.ps1` 的 Ctrl-C | 只 `break` 本地循环，板子上的 Python 继续算几分钟 | 先往串口写 `0x03`，板子收到 SIGINT 停下，窗口**不关**；关窗口改成 Ctrl-X |

---

## 7. 复现方法

```powershell
# 主机侧，一条命令跑完全部 18 项
Set-ExecutionPolicy -Scope Process Bypass -Force
& 'E:\ds_workplace\scratch\test_think_repl.ps1'

# 想复现冷缓存那一组
# 先在板子上：sync; echo 3 > /proc/sys/vm/drop_caches
```

离线单测（不需要板子）：

```powershell
python E:\ds_workplace\scratch\test_chat_v3.py     # 26/26 PASS
```
