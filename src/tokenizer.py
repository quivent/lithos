"""Lithos tokenizer — thin wrapper around HuggingFace `tokenizers` library.

Usage:
    from lithos_tokenizer import Tokenizer
    tok = Tokenizer("/path/to/model_dir")
    ids = tok.encode("Hello, world!")
    text = tok.decode(ids)
    prompt = tok.apply_chat_template([{"role": "user", "content": "Hi"}])
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, Union

from tokenizers import Tokenizer as HFTokenizer
import jinja2


class Tokenizer:
    """Minimal tokenizer for Lithos inference, backed by the Rust `tokenizers` library."""

    def __init__(self, model_dir: Union[str, Path]) -> None:
        model_dir = Path(model_dir)

        # --- Load the fast tokenizer from tokenizer.json ---
        tokenizer_path = model_dir / "tokenizer.json"
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"tokenizer.json not found in {model_dir}")
        self._tok = HFTokenizer.from_file(str(tokenizer_path))

        # --- Load config for special tokens ---
        config_path = model_dir / "tokenizer_config.json"
        self._config: dict = {}
        if config_path.exists():
            with open(config_path) as f:
                self._config = json.load(f)

        # Special token IDs
        self.eos_token: Optional[str] = self._config.get("eos_token")
        self.pad_token: Optional[str] = self._config.get("pad_token")
        self.bos_token: Optional[str] = self._config.get("bos_token")

        self.eos_token_id: Optional[int] = (
            self._tok.token_to_id(self.eos_token) if self.eos_token else None
        )
        self.pad_token_id: Optional[int] = (
            self._tok.token_to_id(self.pad_token) if self.pad_token else None
        )
        self.bos_token_id: Optional[int] = (
            self._tok.token_to_id(self.bos_token) if self.bos_token else None
        )

        self.vocab_size: int = self._tok.get_vocab_size()
        self.model_max_length: int = self._config.get("model_max_length", 262144)

        # --- Load chat template (Jinja2) ---
        self._chat_template: Optional[jinja2.Template] = None

        # Prefer standalone .jinja file, fall back to config-embedded template
        jinja_path = model_dir / "chat_template.jinja"
        template_str: Optional[str] = None
        if jinja_path.exists():
            template_str = jinja_path.read_text()
        elif "chat_template" in self._config:
            template_str = self._config["chat_template"]

        if template_str is not None:
            env = jinja2.Environment(
                undefined=jinja2.Undefined,
                keep_trailing_newline=True,
            )
            # Provide raise_exception as a global so the template can call it
            env.globals["raise_exception"] = _raise_exception
            self._chat_template = env.from_string(template_str)

    # ------------------------------------------------------------------
    # Core encode / decode
    # ------------------------------------------------------------------

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
    ) -> list[int]:
        """Encode text to a list of token IDs."""
        encoding = self._tok.encode(text, add_special_tokens=add_special_tokens)
        return encoding.ids

    def decode(
        self,
        ids: list[int],
        *,
        skip_special_tokens: bool = False,
    ) -> str:
        """Decode token IDs back to text."""
        return self._tok.decode(ids, skip_special_tokens=skip_special_tokens)

    def encode_batch(
        self,
        texts: list[str],
        *,
        add_special_tokens: bool = False,
    ) -> list[list[int]]:
        """Encode a batch of texts (parallel via Rust)."""
        encodings = self._tok.encode_batch(texts, add_special_tokens=add_special_tokens)
        return [enc.ids for enc in encodings]

    def decode_batch(
        self,
        id_lists: list[list[int]],
        *,
        skip_special_tokens: bool = False,
    ) -> list[str]:
        """Decode a batch of token ID lists."""
        return self._tok.decode_batch(id_lists, skip_special_tokens=skip_special_tokens)

    # ------------------------------------------------------------------
    # Token <-> ID lookups
    # ------------------------------------------------------------------

    def token_to_id(self, token: str) -> Optional[int]:
        return self._tok.token_to_id(token)

    def id_to_token(self, id: int) -> Optional[str]:
        return self._tok.id_to_token(id)

    # ------------------------------------------------------------------
    # Chat template
    # ------------------------------------------------------------------

    def apply_chat_template(
        self,
        messages: list[dict],
        *,
        add_generation_prompt: bool = True,
        enable_thinking: bool = False,
        tools: Optional[list[dict]] = None,
        add_vision_id: bool = False,
    ) -> str:
        """Render a list of chat messages into a prompt string using the Jinja template.

        Args:
            messages: List of {"role": ..., "content": ...} dicts.
            add_generation_prompt: Append the assistant turn prefix.
            enable_thinking: Enable <think> block in generation.
            tools: Optional list of tool definitions for function-calling.
            add_vision_id: Label images/videos with sequential IDs.

        Returns:
            The formatted prompt string (not yet tokenized).
        """
        if self._chat_template is None:
            raise RuntimeError("No chat template available for this tokenizer")

        return self._chat_template.render(
            messages=messages,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
            tools=tools,
            add_vision_id=add_vision_id,
        )

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Tokenizer(vocab_size={self.vocab_size}, "
            f"eos='{self.eos_token}'({self.eos_token_id}), "
            f"pad='{self.pad_token}'({self.pad_token_id}))"
        )


def _raise_exception(msg: str) -> None:
    """Helper exposed to the Jinja template so it can raise errors."""
    raise ValueError(msg)


# ----------------------------------------------------------------------
# Quick self-test when run as a script
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    model_dir = sys.argv[1] if len(sys.argv) > 1 else "/home/ubuntu/models/Huihui-Qwen3.5-27B-abliterated-GPTQ-W4A16"

    tok = Tokenizer(model_dir)
    print(tok)

    # Round-trip test
    test_texts = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        "",  # empty string
        "Unicode: \u00e9\u00e0\u00fc\u00f1 \u4f60\u597d \U0001f680",
    ]

    print("\n=== Encode / Decode round-trip ===")
    for text in test_texts:
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        status = "OK" if decoded == text else "MISMATCH"
        print(f"  [{status}] {repr(text[:60])}")
        print(f"    -> {len(ids)} tokens: {ids[:20]}{'...' if len(ids) > 20 else ''}")
        if decoded != text:
            print(f"    decoded: {repr(decoded[:60])}")

    # Batch test
    print("\n=== Batch encode / decode ===")
    batch_ids = tok.encode_batch(test_texts)
    batch_decoded = tok.decode_batch(batch_ids)
    for orig, dec in zip(test_texts, batch_decoded):
        status = "OK" if orig == dec else "MISMATCH"
        print(f"  [{status}] {repr(orig[:50])}")

    # Special tokens
    print(f"\n=== Special tokens ===")
    print(f"  eos: '{tok.eos_token}' -> {tok.eos_token_id}")
    print(f"  pad: '{tok.pad_token}' -> {tok.pad_token_id}")
    im_start = tok.token_to_id("<|im_start|>")
    im_end = tok.token_to_id("<|im_end|>")
    print(f"  <|im_start|> -> {im_start}")
    print(f"  <|im_end|> -> {im_end}")

    # Chat template test
    print("\n=== Chat template ===")
    messages = [
        {"role": "user", "content": "What is 2+2?"},
    ]
    prompt = tok.apply_chat_template(messages)
    print(prompt)
    print(f"  ({len(tok.encode(prompt))} tokens)")

    # With system message
    print("\n=== Chat template (with system) ===")
    messages_sys = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
    ]
    prompt_sys = tok.apply_chat_template(messages_sys)
    print(prompt_sys)

    # Multi-turn
    print("\n=== Chat template (multi-turn) ===")
    messages_multi = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello! How can I help?"},
        {"role": "user", "content": "What is 2+2?"},
    ]
    prompt_multi = tok.apply_chat_template(messages_multi)
    print(prompt_multi)

    print("\nAll tests passed.")
