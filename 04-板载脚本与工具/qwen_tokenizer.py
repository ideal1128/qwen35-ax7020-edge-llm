"""
qwen_tokenizer.py - Production Tokenizer & ChatML Formatter for Qwen3.5-0.8B
Deployable on ALINX AX7020 (Zynq-7020 ARM Cortex-A9 Linux) and Host PC.

Resolves Control Token Fragmentation (<|im_start|>, <|im_end|>, <think>, </think>)
by combining regex-based special token isolation with high-speed C-BPE inference.
"""

import os
import re
import ctypes

# All 33 added/control tokens mapped to the 64K trimmed vocabulary.
# In the 64K vocabulary, the top 33 slots (65503..65535) are explicitly
# reserved for these tokens in contiguous order.
SPECIAL_TOKENS_64K = {
    "<|endoftext|>": 65503,
    "<|im_start|>": 65504,
    "<|im_end|>": 65505,
    "<|object_ref_start|>": 65506,
    "<|object_ref_end|>": 65507,
    "<|box_start|>": 65508,
    "<|box_end|>": 65509,
    "<|quad_start|>": 65510,
    "<|quad_end|>": 65511,
    "<|vision_start|>": 65512,
    "<|vision_end|>": 65513,
    "<|vision_pad|>": 65514,
    "<|image_pad|>": 65515,
    "<|video_pad|>": 65516,
    "<tool_call>": 65517,
    "</tool_call>": 65518,
    "<|fim_prefix|>": 65519,
    "<|fim_middle|>": 65520,
    "<|fim_suffix|>": 65521,
    "<|fim_pad|>": 65522,
    "<|repo_name|>": 65523,
    "<|file_sep|>": 65524,
    "<tool_response>": 65525,
    "</tool_response>": 65526,
    "<think>": 65527,
    "</think>": 65528,
    "<|audio_start|>": 65529,
    "<|audio_end|>": 65530,
    "<tts_pad>": 65531,
    "<tts_text_bos>": 65532,
    "<tts_text_eod>": 65533,
    "<tts_text_bos_single>": 65534,
    "<|audio_pad|>": 65535,
}

# Pre-compiled regex matching any special token (sorted longest first to prevent prefix shadowing)
_SPECIAL_PATTERN = re.compile(
    "(" + "|".join(re.escape(k) for k in sorted(SPECIAL_TOKENS_64K.keys(), key=len, reverse=True)) + ")"
)


class CTokenizer:
    """
    Hybrid C-BPE Tokenizer for Qwen3.5-0.8B on ALINX AX7020.
    
    1. Pre-tokenization: Regex isolates special ChatML and control tokens.
    2. Subword BPE: Passes normal text slices to native C library (libtokenizer.so).
    3. Decoding: Invokes C tokenizer_decode (which contains all 65,536 entries including special tokens).
    """
    SPECIAL_TOKENS = SPECIAL_TOKENS_64K
    SPECIAL_PATTERN = _SPECIAL_PATTERN

    def __init__(self, bin_path="/mnt/qwen/tokenizer_64k.bin", lib_path=None):
        if not os.path.exists(bin_path):
            alt_path = r"D:\fpga2026\work\vocab\tokenizer_64k.bin"
            if os.path.exists(alt_path):
                bin_path = alt_path

        lib_candidates = []
        if lib_path:
            lib_candidates.append(lib_path)
        lib_candidates.extend([
            "/mnt/qwen/libtokenizer.so",
            os.path.join(os.path.dirname(__file__), "libtokenizer.so"),
            os.path.join(os.path.dirname(__file__), "libtokenizer.dll"),
            r"D:\fpga2026\work\driver\libtokenizer.dll",
            r"D:\fpga2026\work\driver\libtokenizer.so",
        ])

        self.lib = None
        for cand in lib_candidates:
            if cand and os.path.exists(cand):
                try:
                    self.lib = ctypes.CDLL(cand)
                    break
                except Exception:
                    pass

        self.tok_ptr = None
        if self.lib and os.path.exists(bin_path):
            self.lib.tokenizer_create.restype = ctypes.c_void_p
            self.lib.tokenizer_create.argtypes = [ctypes.c_char_p]
            self.lib.tokenizer_destroy.argtypes = [ctypes.c_void_p]
            self.lib.tokenizer_encode.argtypes = [
                ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int
            ]
            self.lib.tokenizer_decode.argtypes = [
                ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int
            ]
            self.tok_ptr = self.lib.tokenizer_create(bin_path.encode("utf-8"))

    def _c_encode_chunk(self, text):
        """Invoke underlying C BPE encoder on plain text without special tokens."""
        if not self.tok_ptr or not text:
            return []
        text_bytes = text.encode("utf-8")
        max_tokens = max(1024, len(text_bytes) * 2)
        tokens_arr = (ctypes.c_int * max_tokens)()
        n = self.lib.tokenizer_encode(self.tok_ptr, text_bytes, tokens_arr, max_tokens)
        return [tokens_arr[i] for i in range(n)]

    def encode(self, text, allowed_special=True):
        """
        Encode text into 64K token IDs.
        
        If allowed_special is True, special tokens are recognized as atomic units.
        If allowed_special is False, text is passed directly to C BPE (fragmenting special tokens).
        """
        if not text:
            return []
        if not allowed_special:
            return self._c_encode_chunk(text)

        tokens = []
        parts = self.SPECIAL_PATTERN.split(text)
        for part in parts:
            if not part:
                continue
            if part in self.SPECIAL_TOKENS:
                tokens.append(self.SPECIAL_TOKENS[part])
            else:
                tokens.extend(self._c_encode_chunk(part))
        return tokens

    def decode(self, token_id):
        """Decode single token ID into string piece."""
        if not self.tok_ptr:
            return ""
        buf = ctypes.create_string_buffer(256)
        self.lib.tokenizer_decode(self.tok_ptr, token_id, buf, 256)
        return buf.value.decode("utf-8", errors="replace")

    def decode_batch(self, token_ids):
        """Decode sequence of token IDs into text."""
        return "".join(self.decode(tid) for tid in token_ids)

    @staticmethod
    def apply_chat_template(prompt, enable_thinking=False):
        """
        Apply Qwen3.5 standard ChatML format to raw user prompt.
        If prompt already contains ChatML markers, returns prompt untouched (idempotent).
        """
        if "<|im_start|>" in prompt:
            return prompt
        if enable_thinking:
            return "<|im_start|>user\n{0}<|im_end|>\n<|im_start|>assistant\n".format(prompt)
        else:
            return ("<|im_start|>user\n{0}<|im_end|>\n<|im_start|>assistant\n").format(prompt)

    def __del__(self):
        if getattr(self, "tok_ptr", None) and getattr(self, "lib", None):
            self.lib.tokenizer_destroy(self.tok_ptr)
            self.tok_ptr = None
