"""
Lithos Kernel Factory

Reads a model config.json and generates PTX kernels with model-specific
dimensions baked in as compile-time constants (not runtime parameters).
Compiles each PTX to cubin via ptxas, caching by config hash.
"""

import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

KERNEL_DIR = Path(__file__).resolve().parent.parent / "kernels"
DEFAULT_CACHE_ROOT = Path(tempfile.gettempdir()) / "lithos-cache"


class ModelConfig:
    """Parsed model configuration with the fields the factory needs."""

    def __init__(self, config_path: str):
        with open(config_path) as f:
            raw = json.load(f)

        # Qwen3.5 nests text params under text_config; other models may not
        tc = raw.get("text_config", raw)

        self.hidden_dim: int = tc["hidden_size"]
        self.num_heads: int = tc.get("num_attention_heads", tc.get("num_heads"))
        self.num_kv_heads: int = tc.get("num_key_value_heads", self.num_heads)
        self.head_dim: int = tc.get("head_dim", self.hidden_dim // self.num_heads)
        self.intermediate_size: int = tc["intermediate_size"]
        self.num_layers: int = tc.get("num_hidden_layers", tc.get("num_layers"))
        self.vocab_size: int = tc["vocab_size"]
        self.layer_types: List[str] = tc.get("layer_types", ["full_attention"] * self.num_layers)

        # Rotary config
        rope = tc.get("rope_scaling") or tc.get("rope_parameters") or {}
        self.partial_rotary_factor: float = rope.get("partial_rotary_factor", 1.0)

        # Quantization
        self.quantization = raw.get("quantization_config", {})

    def config_hash(self) -> str:
        """Deterministic hash of all dimension parameters that affect kernel code."""
        blob = json.dumps({
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "intermediate_size": self.intermediate_size,
            "num_layers": self.num_layers,
            "vocab_size": self.vocab_size,
            "partial_rotary_factor": self.partial_rotary_factor,
            "quantization": self.quantization,
        }, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    # -- Projection dimension pairs ------------------------------------------

    def projection_pairs(self) -> List[Tuple[int, int]]:
        """Return unique (input_dim, output_dim) pairs for all linear projections."""
        h = self.hidden_dim
        qkv_out = self.num_heads * self.head_dim       # Q projection output
        kv_out = self.num_kv_heads * self.head_dim      # K or V projection output
        ff_up = self.intermediate_size                  # gate / up projection
        pairs = set()

        # Q projection:  hidden_dim -> num_heads * head_dim
        pairs.add((h, qkv_out))
        # K projection:  hidden_dim -> num_kv_heads * head_dim
        pairs.add((h, kv_out))
        # V projection:  hidden_dim -> num_kv_heads * head_dim  (same as K)
        # already added above
        # O projection:  num_heads * head_dim -> hidden_dim
        pairs.add((qkv_out, h))
        # Gate projection: hidden_dim -> intermediate_size
        pairs.add((h, ff_up))
        # Up projection:   hidden_dim -> intermediate_size  (same dims as gate)
        # already added above
        # Down projection:  intermediate_size -> hidden_dim
        pairs.add((ff_up, h))

        return sorted(pairs)


# ---------------------------------------------------------------------------
# PTX specialization helpers
# ---------------------------------------------------------------------------

def _read_template(name: str) -> str:
    path = KERNEL_DIR / f"{name}.ptx"
    return path.read_text()


def _fix_param_commas(ptx: str) -> str:
    """Fix trailing commas in PTX parameter lists after parameter removal.

    Finds the param block between '(' and ')' in each .entry declaration
    and removes any comma that directly precedes the closing ')'.
    """
    # Fix: trailing comma before closing paren (possibly with whitespace/newlines)
    ptx = re.sub(r",(\s*\))", r"\1", ptx)
    return ptx


def _specialize_projection(template: str, M: int, K: int) -> str:
    """Replace M/K runtime params with compile-time constants in projection PTX."""
    entry_name = f"projection_{M}x{K}"

    ptx = template
    # Rename the entry point
    ptx = ptx.replace(
        ".visible .entry projection(",
        f".visible .entry {entry_name}(",
    )

    # Remove the M and K parameter declarations and their loads,
    # replace with immediate constant moves.
    ptx = re.sub(
        r"^\s*\.param\s+\.u32\s+M\s*,?\s*\n",
        "",
        ptx,
        flags=re.MULTILINE,
    )
    ptx = re.sub(
        r"^\s*\.param\s+\.u32\s+K\s*\n",
        "",
        ptx,
        flags=re.MULTILINE,
    )

    # Replace the param loads with immediate moves
    ptx = re.sub(
        r"ld\.param\.u32\s+(%r\d+),\s*\[M\]\s*;",
        rf"mov.u32         \1, {M};    // M baked in",
        ptx,
    )
    ptx = re.sub(
        r"ld\.param\.u32\s+(%r\d+),\s*\[K\]\s*;",
        rf"mov.u32         \1, {K};    // K baked in",
        ptx,
    )

    return ptx


def _specialize_attention_score(template: str, head_dim: int, num_kv_heads: int,
                                 num_heads: int) -> str:
    """Bake head_dim and num_heads into attention_score kernel."""
    ptx = template

    # Remove runtime params for head_dim and num_heads
    ptx = re.sub(
        r"^\s*\.param\s+\.u32\s+head_dim\s*,?\s*\n",
        "",
        ptx,
        flags=re.MULTILINE,
    )
    ptx = re.sub(
        r"^\s*\.param\s+\.u32\s+num_heads\s*\n",
        "",
        ptx,
        flags=re.MULTILINE,
    )

    ptx = re.sub(
        r"ld\.param\.u32\s+(%r\d+),\s*\[head_dim\]\s*;",
        rf"mov.u32         \1, {head_dim};    // head_dim baked in",
        ptx,
    )
    ptx = re.sub(
        r"ld\.param\.u32\s+(%r\d+),\s*\[num_heads\]\s*;",
        rf"mov.u32         \1, {num_heads};    // num_heads baked in",
        ptx,
    )

    # Add GQA ratio as a constant comment for documentation
    if num_kv_heads != num_heads:
        gqa_ratio = num_heads // num_kv_heads
        # Patch GQA line: replace "mov.u32 %r6, %r4" with a division
        ptx = ptx.replace(
            "// GQA: kv_head = head_index (MHA case; for GQA would divide by ratio)\n"
            "    mov.u32         %r6, %r4;",
            f"// GQA: kv_head = head_index / {gqa_ratio} (num_heads={num_heads}, num_kv_heads={num_kv_heads})\n"
            f"    shr.b32         %r6, %r4, {gqa_ratio.bit_length() - 1};    // divide by {gqa_ratio} (power of 2)",
        )

    return ptx


def _specialize_norm(template: str, hidden_dim: int) -> str:
    ptx = template
    ptx = re.sub(
        r"^\s*\.param\s+\.u32\s+hidden_dim\s*,?\s*\n",
        "",
        ptx,
        flags=re.MULTILINE,
    )
    ptx = re.sub(
        r"ld\.param\.u32\s+(%r\d+),\s*\[hidden_dim\]\s*;",
        rf"mov.u32         \1, {hidden_dim};    // hidden_dim baked in",
        ptx,
    )
    return ptx


def _specialize_activate(template: str, intermediate_size: int) -> str:
    ptx = template
    ptx = re.sub(
        r"^\s*\.param\s+\.u32\s+size\s*\n",
        "",
        ptx,
        flags=re.MULTILINE,
    )
    ptx = re.sub(
        r"ld\.param\.u32\s+(%r\d+),\s*\[size\]\s*;",
        rf"mov.u32         \1, {intermediate_size};    // intermediate_size baked in",
        ptx,
    )
    return ptx


def _specialize_rotate(template: str, head_dim: int, partial_rotary_factor: float,
                        num_heads: int) -> str:
    ptx = template

    # Compute the effective rotary dim
    rotary_dim = int(head_dim * partial_rotary_factor)
    # half_dim for partial rotary (pairs to rotate)
    half_rotary = rotary_dim // 2

    ptx = re.sub(
        r"^\s*\.param\s+\.u32\s+head_dim\s*,?\s*\n",
        "",
        ptx,
        flags=re.MULTILINE,
    )
    ptx = re.sub(
        r"^\s*\.param\s+\.u32\s+num_heads\s*\n",
        "",
        ptx,
        flags=re.MULTILINE,
    )
    ptx = re.sub(
        r"ld\.param\.u32\s+(%r\d+),\s*\[head_dim\]\s*;",
        rf"mov.u32         \1, {head_dim};    // head_dim baked in",
        ptx,
    )
    ptx = re.sub(
        r"ld\.param\.u32\s+(%r\d+),\s*\[num_heads\]\s*;",
        rf"mov.u32         \1, {num_heads};    // num_heads baked in",
        ptx,
    )

    # Replace the half_dim computation with a constant if partial rotary
    if partial_rotary_factor < 1.0:
        ptx = ptx.replace(
            "// half_dim = head_dim / 2\n"
            "    shr.b32         %r5, %r1, 1;",
            f"// half_rotary_dim = {half_rotary} (partial_rotary_factor={partial_rotary_factor})\n"
            f"    mov.u32         %r5, {half_rotary};    // rotary pairs baked in",
        )

    return ptx


def _specialize_sample(template: str, vocab_size: int) -> str:
    ptx = template
    ptx = re.sub(
        r"^\s*\.param\s+\.u32\s+vocab_size\s*,?\s*\n",
        "",
        ptx,
        flags=re.MULTILINE,
    )
    ptx = re.sub(
        r"ld\.param\.u32\s+(%r\d+),\s*\[vocab_size\]\s*;",
        rf"mov.u32         \1, {vocab_size};    // vocab_size baked in",
        ptx,
    )
    return ptx


def _specialize_embed(template: str, hidden_dim: int, vocab_size: int) -> str:
    ptx = template
    ptx = re.sub(
        r"^\s*\.param\s+\.u32\s+hidden_dim\s*\n",
        "",
        ptx,
        flags=re.MULTILINE,
    )
    ptx = re.sub(
        r"ld\.param\.u32\s+(%r\d+),\s*\[hidden_dim\]\s*;",
        rf"mov.u32         \1, {hidden_dim};    // hidden_dim baked in",
        ptx,
    )
    return ptx


# ---------------------------------------------------------------------------
# Compiler
# ---------------------------------------------------------------------------

def _compile_ptx(ptx_source: str, output_path: Path, arch: str = "sm_90") -> Path:
    """Write PTX to a temp file, compile to cubin with ptxas."""
    # Fix any trailing commas left after parameter removal
    ptx_source = _fix_param_commas(ptx_source)
    ptx_path = output_path.with_suffix(".ptx")
    ptx_path.write_text(ptx_source)

    cmd = [
        "ptxas",
        f"-arch={arch}",
        "-o", str(output_path),
        str(ptx_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ptxas failed for {ptx_path}:\n{result.stderr}"
        )
    return output_path


# ---------------------------------------------------------------------------
# KernelFactory
# ---------------------------------------------------------------------------

class KernelFactory:
    """
    Reads a model directory's config.json and produces compiled cubin kernels
    with model dimensions baked in as compile-time constants.

    Usage:
        factory = KernelFactory("/path/to/model")
        cubins = factory.build_all()
        # cubins = {'projection_5120x6144': Path('/tmp/lithos-cache/abcd1234/projection_5120x6144.cubin'), ...}
    """

    def __init__(self, model_dir: str, cache_root: Optional[str] = None, arch: str = "sm_90"):
        self.model_dir = Path(model_dir)
        self.config = ModelConfig(str(self.model_dir / "config.json"))
        self.arch = arch

        root = Path(cache_root) if cache_root else DEFAULT_CACHE_ROOT
        self.cache_dir = root / self.config.config_hash()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # -- Individual kernel builders ------------------------------------------

    def _build_projection(self, M: int, K: int) -> Tuple[str, Path]:
        name = f"projection_{M}x{K}"
        out = self.cache_dir / f"{name}.cubin"
        if out.exists():
            return name, out
        template = _read_template("projection")
        ptx = _specialize_projection(template, M, K)
        _compile_ptx(ptx, out, self.arch)
        return name, out

    def _build_attention_score(self) -> Tuple[str, Path]:
        name = "attention_score"
        out = self.cache_dir / f"{name}.cubin"
        if out.exists():
            return name, out
        template = _read_template("attention_score")
        ptx = _specialize_attention_score(
            template,
            self.config.head_dim,
            self.config.num_kv_heads,
            self.config.num_heads,
        )
        _compile_ptx(ptx, out, self.arch)
        return name, out

    def _build_norm(self) -> Tuple[str, Path]:
        name = "norm"
        out = self.cache_dir / f"{name}.cubin"
        if out.exists():
            return name, out
        template = _read_template("norm")
        ptx = _specialize_norm(template, self.config.hidden_dim)
        _compile_ptx(ptx, out, self.arch)
        return name, out

    def _build_activate(self) -> Tuple[str, Path]:
        name = "activate"
        out = self.cache_dir / f"{name}.cubin"
        if out.exists():
            return name, out
        template = _read_template("activate")
        ptx = _specialize_activate(template, self.config.intermediate_size)
        _compile_ptx(ptx, out, self.arch)
        return name, out

    def _build_rotate(self) -> Tuple[str, Path]:
        name = "rotate"
        out = self.cache_dir / f"{name}.cubin"
        if out.exists():
            return name, out
        template = _read_template("rotate")
        ptx = _specialize_rotate(
            template,
            self.config.head_dim,
            self.config.partial_rotary_factor,
            self.config.num_heads,
        )
        _compile_ptx(ptx, out, self.arch)
        return name, out

    def _build_sample(self) -> Tuple[str, Path]:
        name = "sample"
        out = self.cache_dir / f"{name}.cubin"
        if out.exists():
            return name, out
        template = _read_template("sample")
        ptx = _specialize_sample(template, self.config.vocab_size)
        _compile_ptx(ptx, out, self.arch)
        return name, out

    def _build_embed(self) -> Tuple[str, Path]:
        name = "embed"
        out = self.cache_dir / f"{name}.cubin"
        if out.exists():
            return name, out
        template = _read_template("embed")
        ptx = _specialize_embed(
            template,
            self.config.hidden_dim,
            self.config.vocab_size,
        )
        _compile_ptx(ptx, out, self.arch)
        return name, out

    # -- Public API -----------------------------------------------------------

    def build_all(self) -> Dict[str, str]:
        """
        Build all kernel variants for the model. Returns a dict mapping
        kernel name to cubin file path (as string).
        """
        cubins: Dict[str, str] = {}

        # Projection kernels -- one per unique (input_dim, output_dim) pair
        for M, K in self.config.projection_pairs():
            name, path = self._build_projection(M, K)
            cubins[name] = str(path)

        # Single-instance kernels
        for builder in [
            self._build_attention_score,
            self._build_norm,
            self._build_activate,
            self._build_rotate,
            self._build_sample,
            self._build_embed,
        ]:
            name, path = builder()
            cubins[name] = str(path)

        return cubins

    def summary(self) -> str:
        """Return a human-readable summary of the model config."""
        c = self.config
        lines = [
            f"Model directory : {self.model_dir}",
            f"Config hash     : {c.config_hash()}",
            f"Cache directory : {self.cache_dir}",
            f"hidden_dim      : {c.hidden_dim}",
            f"num_heads       : {c.num_heads}",
            f"num_kv_heads    : {c.num_kv_heads}",
            f"head_dim        : {c.head_dim}",
            f"intermediate    : {c.intermediate_size}",
            f"num_layers      : {c.num_layers}",
            f"vocab_size      : {c.vocab_size}",
            f"partial_rotary  : {c.partial_rotary_factor}",
            f"quantization    : {c.quantization}",
            f"projection pairs: {c.projection_pairs()}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    model_dir = sys.argv[1] if len(sys.argv) > 1 else \
        "/home/ubuntu/models/Huihui-Qwen3.5-27B-abliterated-GPTQ-W4A16"

    factory = KernelFactory(model_dir)
    print(factory.summary())
    print()

    cubins = factory.build_all()
    print(f"Built {len(cubins)} kernels:")
    for name, path in sorted(cubins.items()):
        size = os.path.getsize(path)
        print(f"  {name:40s} -> {path}  ({size:,} bytes)")
