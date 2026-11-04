#!/usr/bin/env python3
r"""
chat_qwen35.py - Interactive CLI Chat for Qwen3.5-0.8B
Run directly in your terminal:
    # Mode 1: 64K High-Fidelity Language-Trimmed Vocabulary (default, recommended)
    python D:\fpga2026\work\driver\chat_qwen35.py

    # Mode 2: 32K FPGA Ultra-compact Vocabulary
    python D:\fpga2026\work\driver\chat_qwen35.py --vocab 32k

    # Mode 3: Full 248K original vocabulary
    python D:\fpga2026\work\driver\chat_qwen35.py --vocab 248k

Commands during chat:
    /clear    - Clear conversation history
    /exit     - Exit chat session
    /think    - Toggle thinking / reasoning mode
"""

import os
import sys
import json
import argparse
import warnings
import logging

# Suppress noisy fallback warnings
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import torch
import transformers
transformers.utils.logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)

from transformers import AutoTokenizer, TextStreamer
from transformers.models.qwen3_5 import Qwen3_5ForConditionalGeneration

MODEL_PATH = r"D:\fpga2026\model\qwen3.5_0.8b"
VOCAB_DIR  = r"D:\fpga2026\work\vocab"

# Special token IDs in original 248K vocabulary
ORIG_IM_END_ID     = 248046  # <|im_end|>
ORIG_ENDOFTEXT_ID  = 248044  # <|endoftext|>

class TextStreamerMapped(TextStreamer):
    """Real-time token streamer for pruned vocabulary (64K / 32K) with strict EOS cutoff."""
    def __init__(self, tokenizer, new_to_old, eos_ids, **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.new_to_old = new_to_old
        self.eos_ids = set(eos_ids)
        self.stopped = False

    def put(self, value):
        if self.stopped:
            return

        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("Only batch size 1 supported")
        elif len(value.shape) > 1:
            value = value[0]

        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        tokens = value.tolist()
        filtered_tokens = []
        for t in tokens:
            if t in self.eos_ids:
                self.stopped = True
                break
            filtered_tokens.append(t)

        if not filtered_tokens:
            return

        # Map each pruned token ID back to original vocabulary space
        orig_ids = [self.new_to_old[idx] for idx in filtered_tokens]
        self.token_cache.extend(orig_ids)
        text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)

        if text.endswith("\n"):
            printable_text = text[self.print_len:]
            self.token_cache = []
            self.print_len = 0
        elif len(text) > 0 and self._is_chinese_char(ord(text[-1])):
            printable_text = text[self.print_len:]
            self.print_len += len(printable_text)
        else:
            printable_text = text[self.print_len:text.rfind(" ") + 1]
            self.print_len += len(printable_text)

        self.on_finalized_text(printable_text)

    def end(self):
        text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)
        printable_text = text[self.print_len:]
        self.token_cache = []
        self.print_len = 0
        self.on_finalized_text(printable_text, stream_end=True)

    def reset(self):
        self.stopped = False
        self.token_cache = []
        self.print_len = 0
        self.next_tokens_are_prompt = True

def main():
    parser = argparse.ArgumentParser(description="Qwen3.5-0.8B Interactive Chat")
    parser.add_argument("--vocab", choices=["64k", "32k", "248k"], default="64k",
                        help="Vocabulary mode: '64k' (High-Fidelity pruned, default), '32k' (FPGA 32K), or '248k' (full original)")
    parser.add_argument("--think", action="store_true", default=False,
                        help="Enable Chain-of-Thought (CoT) reasoning mode by default")
    args = parser.parse_args()

    vocab_desc = {
        "64k": "64K (65,536) 高保真语种剪枝版 (保留完整中英数推理与思维链, 净省 356.5MB DDR3)",
        "32k": "32K (32,768) FPGA 极小剪枝版 (节省 224MB DDR3)",
        "248k": "官方 248K (248,077) 原版全量词表"
    }[args.vocab]

    print("=" * 70, flush=True)
    print("  Qwen3.5-0.8B 实时交互控制台", flush=True)
    print("  底层架构: Hybrid Gated DeltaNet + GQA + SwiGLU", flush=True)
    print(f"  当前词表: {args.vocab.upper()} - {vocab_desc}", flush=True)
    print("=" * 70, flush=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print(f"[*] 运行设备: {device} | 计算精度: {dtype}", flush=True)

    # 1. Load Tokenizer & Mapping
    print("[1/3] 正在加载分词器与词表映射...", end="", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    old_to_new = None
    new_to_old = None

    if args.vocab == "64k":
        with open(os.path.join(VOCAB_DIR, "new_to_old_64k.json"), "r") as f:
            new_to_old = json.load(f)
        with open(os.path.join(VOCAB_DIR, "old_to_new_64k.json"), "r") as f:
            old_to_new = {int(k): v for k, v in json.load(f).items()}
        eos_ids = [old_to_new[ORIG_IM_END_ID], old_to_new[ORIG_ENDOFTEXT_ID]]
        pad_id = old_to_new[ORIG_ENDOFTEXT_ID]
        target_vocab_size = 65536
    elif args.vocab == "32k":
        with open(os.path.join(VOCAB_DIR, "new_to_old.json"), "r") as f:
            new_to_old = json.load(f)
        with open(os.path.join(VOCAB_DIR, "old_to_new.json"), "r") as f:
            old_to_new = {int(k): v for k, v in json.load(f).items()}
        eos_ids = [old_to_new[ORIG_IM_END_ID], old_to_new[ORIG_ENDOFTEXT_ID]]
        pad_id = old_to_new[ORIG_ENDOFTEXT_ID]
        target_vocab_size = 32768
    else:
        eos_ids = [ORIG_IM_END_ID, ORIG_ENDOFTEXT_ID]
        pad_id = ORIG_ENDOFTEXT_ID
        target_vocab_size = len(tokenizer.get_vocab())
    print(" 完成！", flush=True)

    # 2. Load Model & Slice Vocab
    print(f"[2/3] 正在从硬盘加载 1.74GB 模型权重至 {device}（约需 3~5 秒）...", end="", flush=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=dtype,
        device_map=device
    )

    if args.vocab in ["64k", "32k"]:
        sliced_embed = model.model.language_model.embed_tokens.weight.data[new_to_old].clone()
        model.model.language_model.embed_tokens.weight.data = sliced_embed
        model.model.language_model.embed_tokens.num_embeddings = target_vocab_size
        if hasattr(model, "lm_head") and model.lm_head is not None:
            model.lm_head.weight.data = sliced_embed
            model.lm_head.out_features = target_vocab_size

    model.eval()
    print(" 完成！", flush=True)

    # 3. Setup Streamer
    if args.vocab in ["64k", "32k"]:
        streamer = TextStreamerMapped(tokenizer, new_to_old, eos_ids=eos_ids, skip_prompt=True, skip_special_tokens=True)
    else:
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    print("[3/3] 实时流式输出引擎已就绪！", flush=True)
    print("\n" + "-" * 70, flush=True)
    print("  提示: 直接输入内容并按回车即可对话", flush=True)
    print("  指令: 输入 '/clear' 清空上下文 | 输入 '/think' 开关思考链 | 输入 'exit' 退出", flush=True)
    print("-" * 70 + "\n", flush=True)

    system_prompt = "你是 Qwen，是由阿里云通义实验室研发的人工智能语言模型。请简明、亲切且准确地回答用户的问题。"
    messages = [
        {"role": "system", "content": system_prompt}
    ]
    thinking_mode = args.think
    if thinking_mode:
        print("[*] 默认已开启思考链模式 (CoT)\n", flush=True)

    while True:
        try:
            user_input = input("User > ").lstrip("\ufeff").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n退出对话。再见！", flush=True)
            break

        if not user_input:
            continue

        clean_input = user_input.strip("\"'")

        if clean_input.lower() in ["exit", "quit", "q", "/exit"]:
            print("退出对话。再见！", flush=True)
            break

        if clean_input.lower() in ["/clear", "clear"]:
            messages = [
                {"role": "system", "content": system_prompt}
            ]
            print("[*] 对话历史已清空。\n", flush=True)
            continue

        if clean_input.lower() == "/think":
            thinking_mode = not thinking_mode
            status = "开启" if thinking_mode else "关闭"
            print(f"[*] 思考链模式已切换为: {status}\n", flush=True)
            continue

        messages.append({"role": "user", "content": user_input})

        # Apply chat template with thinking mode support
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking_mode
        )

        if args.vocab in ["64k", "32k"]:
            orig_ids = tokenizer.encode(prompt_text)
            ids_pruned = [old_to_new.get(idx, 0) for idx in orig_ids]
            input_tensor = torch.tensor([ids_pruned], dtype=torch.long, device=model.device)
            streamer.reset()
        else:
            inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)
            input_tensor = inputs.input_ids

        prompt_len = input_tensor.shape[1]
        print("Qwen3.5 > ", end="", flush=True)

        with torch.no_grad():
            outputs = model.generate(
                input_tensor,
                max_new_tokens=1024 if thinking_mode else 256,
                do_sample=True,
                temperature=0.6 if thinking_mode else 0.7,
                top_p=0.95 if thinking_mode else 0.80,
                top_k=20,
                repetition_penalty=1.15,
                eos_token_id=eos_ids,
                pad_token_id=pad_id,
                streamer=streamer
            )

        print("\n", flush=True)

        # Truncate at first EOS token to prevent dialogue contamination
        raw_tokens = outputs[0][prompt_len:].tolist()
        clean_tokens = []
        for t in raw_tokens:
            if t in eos_ids:
                break
            clean_tokens.append(t)

        if args.vocab in ["64k", "32k"]:
            gen_orig = [new_to_old[idx] for idx in clean_tokens]
            assistant_reply = tokenizer.decode(gen_orig, skip_special_tokens=True).strip()
        else:
            assistant_reply = tokenizer.decode(clean_tokens, skip_special_tokens=True).strip()

        messages.append({"role": "assistant", "content": assistant_reply})

if __name__ == "__main__":
    main()
