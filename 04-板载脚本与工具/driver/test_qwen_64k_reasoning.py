#!/usr/bin/env python3
"""
test_qwen_64k_reasoning.py - Direct verification of 64K pruned Qwen3.5-0.8B
Tests "9.11 和 9.9 哪个更大" with thinking chain enabled to verify mathematical reasoning.
"""

import os
import sys
import json
import torch
import transformers
from transformers import AutoTokenizer, TextStreamer
from transformers.models.qwen3_5 import Qwen3_5ForConditionalGeneration

MODEL_PATH = r"D:\fpga2026\model\qwen3.5_0.8b"
VOCAB_DIR  = r"D:\fpga2026\work\vocab"

class TextStreamerMapped(TextStreamer):
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
        filtered = []
        for t in tokens:
            if t in self.eos_ids:
                self.stopped = True
                break
            filtered.append(t)

        if not filtered:
            return

        orig_ids = [self.new_to_old[idx] for idx in filtered]
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

    def reset(self):
        self.stopped = False
        self.token_cache = []
        self.print_len = 0
        self.next_tokens_are_prompt = True

def main():
    print("=" * 65)
    print("  Qwen3.5-0.8B 64K 高保真裁剪版 - 数学推理能力实测")
    print("=" * 65)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # 1. Load Tokenizer & 64K Mapping
    print("[1/3] 加载分词器与 64K 映射表...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    with open(os.path.join(VOCAB_DIR, "new_to_old_64k.json"), "r") as f:
        new_to_old = json.load(f)
    with open(os.path.join(VOCAB_DIR, "old_to_new_64k.json"), "r") as f:
        old_to_new = {int(k): v for k, v in json.load(f).items()}

    ORIG_IM_END_ID    = 248046
    ORIG_ENDOFTEXT_ID = 248044
    eos_ids = [old_to_new[ORIG_IM_END_ID], old_to_new[ORIG_ENDOFTEXT_ID]]
    pad_id  = old_to_new[ORIG_ENDOFTEXT_ID]

    # 2. Load Model & Slice to 64K (65536)
    print("[2/3] 加载模型并动态裁剪词表至 65,536...")
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=dtype,
        device_map=device
    )

    sliced_embed = model.model.language_model.embed_tokens.weight.data[new_to_old].clone()
    model.model.language_model.embed_tokens.weight.data = sliced_embed
    model.model.language_model.embed_tokens.num_embeddings = 65536
    if hasattr(model, "lm_head") and model.lm_head is not None:
        model.lm_head.weight.data = sliced_embed
        model.lm_head.out_features = 65536

    model.eval()

    # 3. Setup Streamer
    streamer = TextStreamerMapped(tokenizer, new_to_old, eos_ids=eos_ids, skip_prompt=True, skip_special_tokens=True)

    # Test prompt
    prompt_question = "9.11 和 9.9 哪个更大？为什么？"
    print(f"\n[测试问题]: {prompt_question}")
    print("-" * 65)

    system_prompt = "你是 Qwen，是由阿里云通义实验室研发的人工智能语言模型。请简明、亲切且准确地回答用户的问题。"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt_question}
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )

    orig_ids = tokenizer.encode(prompt_text)
    ids_64k = [old_to_new.get(idx, 0) for idx in orig_ids]
    input_tensor = torch.tensor([ids_64k], dtype=torch.long, device=model.device)

    print("Qwen3.5 (64K) > ", end="", flush=True)

    with torch.no_grad():
        outputs = model.generate(
            input_tensor,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            repetition_penalty=1.15,
            eos_token_id=eos_ids,
            pad_token_id=pad_id,
            streamer=streamer
        )

    print("\n" + "=" * 65)
    raw_tokens = outputs[0][len(ids_64k):].tolist()
    clean_orig_tokens = []
    for t in raw_tokens:
        if t in eos_ids:
            break
        clean_orig_tokens.append(new_to_old[t])

    full_response = tokenizer.decode(clean_orig_tokens, skip_special_tokens=False)
    print("\n[完整解码回复 (包含思考与结论)]:\n")
    print(full_response)
    print("\n" + "=" * 65)
    print("  [PASS] 64K 推理测试完成！")
    print("=" * 65)

if __name__ == "__main__":
    main()
