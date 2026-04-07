#!/usr/bin/env python3
"""
Convert a merged Hugging Face causal LM into MLX format and quantize it.

Default target:
- mlx-int4 (affine, 4 bits, group size 64)

Why this version exists:
- Replaces the old Optimum-Quanto flow with native MLX-LM conversion.
- Avoids the high-level `mlx_lm.convert.convert(...)` wrapper so we can pass
  tokenizer options directly.
- Works around a tokenizer-loading failure seen with local model directories,
  especially with Transformers 4.57.x, where `fix_mistral_regex` can be
  incorrectly triggered for non-Mistral tokenizers and may crash.
"""

from __future__ import annotations

import argparse
import os
import shutil
import warnings
from pathlib import Path
from typing import Callable, Iterable, Optional

import mlx.core as mx
from mlx.utils import tree_map_with_path

from mlx_lm.utils import load as mlx_load
from mlx_lm.utils import quantize_model, save

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    from packaging.version import Version
    import transformers
except Exception:  # pragma: no cover
    Version = None
    transformers = None


MODEL_CONVERSION_DTYPES = {"float16": mx.float16, "bfloat16": mx.bfloat16, "float32": mx.float32}
TARGET_TO_BITS = {
    "mlx-int4": 4,
    "mlx-int8": 8,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert a merged HF model to MLX and quantize it.")
    p.add_argument(
        "--model_dir",
        type=str,
        default="./merged_qwen_sql",
        help="Path to the merged Hugging Face model directory.",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to write the MLX model directory. Defaults from --target.",
    )
    p.add_argument(
        "--target",
        type=str,
        choices=sorted(TARGET_TO_BITS.keys()),
        default="mlx-int4",
        help="Quantization target. Default is mlx-int4.",
    )
    # Legacy alias retained so older invocations do not explode for sport.
    p.add_argument(
        "--weights",
        type=str,
        choices=["int4", "int8"],
        default=None,
        help="Legacy alias for --target. int4 -> mlx-int4, int8 -> mlx-int8.",
    )
    p.add_argument(
        "--dtype",
        type=str,
        choices=sorted(MODEL_CONVERSION_DTYPES.keys()),
        default=None,
        help="Optional dtype for non-quantized floating parameters in the saved MLX model.",
    )
    p.add_argument(
        "--torch_dtype",
        type=str,
        choices=["auto", *sorted(MODEL_CONVERSION_DTYPES.keys())],
        default="auto",
        help="Legacy alias for --dtype. 'auto' keeps the source model dtype.",
    )
    p.add_argument(
        "--q_mode",
        type=str,
        choices=["affine", "mxfp4", "nvfp4", "mxfp8"],
        default="affine",
        help="MLX quantization mode. Default is affine.",
    )
    p.add_argument(
        "--q_bits",
        type=int,
        default=None,
        help="Override bits per weight. Defaults from --target / --q_mode.",
    )
    p.add_argument(
        "--q_group_size",
        type=int,
        default=None,
        help="Override quantization group size. Defaults from MLX-LM for the selected mode.",
    )
    p.add_argument(
        "--exclude",
        type=str,
        default="lm_head",
        help='Comma-separated module path substrings to exclude from quantization. Default: "lm_head".',
    )
    p.add_argument(
        "--trust_remote_code",
        action="store_true",
        default=True,
        help="Trust remote code when loading tokenizer/model metadata.",
    )
    p.add_argument(
        "--fix_mistral_regex",
        action="store_true",
        default=False,
        help=(
            "Forward fix_mistral_regex=True to AutoTokenizer. Disabled by default because local non-Mistral "
            "directories can spuriously trigger this path in some Transformers versions."
        ),
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the output directory first if it already exists.",
    )
    return p.parse_args()


def resolve_target(args: argparse.Namespace) -> tuple[str, int]:
    target = args.target
    if args.weights is not None:
        target = f"mlx-{args.weights}"
    bits = args.q_bits if args.q_bits is not None else TARGET_TO_BITS[target]
    return target, bits


def resolve_output_dir(args: argparse.Namespace, target: str) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    suffix = target.replace("-", "_")
    return Path(f"./qwen_sql_{suffix}")


def resolve_dtype(args: argparse.Namespace) -> Optional[mx.Dtype]:
    # Prefer explicit --dtype. Fall back to the legacy --torch_dtype alias.
    if args.dtype is not None:
        return MODEL_CONVERSION_DTYPES[args.dtype]
    if args.torch_dtype == "auto":
        return None
    return MODEL_CONVERSION_DTYPES[args.torch_dtype]


def parse_excludes(raw: str) -> list[str]:
    return [s.strip() for s in raw.split(",") if s.strip()]


def build_quant_predicate(exclude_patterns: Iterable[str]) -> Callable[[str, object], bool]:
    patterns = tuple(exclude_patterns)

    def predicate(path: str, module: object) -> bool:
        del module
        return not any(pattern in path for pattern in patterns)

    return predicate


def cast_model_dtype(model, dtype: Optional[mx.Dtype]) -> None:
    if dtype is None:
        return

    cast_predicate = getattr(model, "cast_predicate", lambda _: True)

    def set_dtype(path: str, value):
        if cast_predicate(path) and mx.issubdtype(value.dtype, mx.floating):
            return value.astype(dtype)
        return value

    model.update(tree_map_with_path(set_dtype, model.parameters()))


def warn_about_transformers_version() -> None:
    if Version is None or transformers is None:
        return

    current = Version(transformers.__version__)
    if current < Version("5.0.0"):
        print(
            f"[WARN] Detected transformers=={transformers.__version__}. "
            "mlx-lm 0.31.x officially depends on transformers>=5.0.0. "
            "This script bypasses mlx_lm.convert() to work around tokenizer-loading issues, "
            "but a separate MLX-only environment is still the cleaner long-term setup."
        )


def main() -> None:
    args = parse_args()
    warn_about_transformers_version()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir.resolve()}")

    target, q_bits = resolve_target(args)
    out_dir = resolve_output_dir(args, target)
    dtype = resolve_dtype(args)
    exclude = parse_excludes(args.exclude)

    if out_dir.exists():
        if args.overwrite:
            shutil.rmtree(out_dir)
        else:
            raise ValueError(
                f"Output directory already exists: {out_dir.resolve()}\n"
                "Pass --overwrite to replace it."
            )

    tokenizer_config = {
        "trust_remote_code": args.trust_remote_code,
        # Default False on purpose. See note at top of file.
        "fix_mistral_regex": args.fix_mistral_regex,
    }

    print(f"Loading source model from: {model_dir}")
    model, tokenizer, config = mlx_load(
        str(model_dir),
        tokenizer_config=tokenizer_config,
        lazy=True,
        return_config=True,
    )

    if dtype is not None:
        print(f"Casting floating parameters to: {dtype}")
        cast_model_dtype(model, dtype)

    print(
        f"Converting with MLX (target={target}, q_mode={args.q_mode}, "
        f"q_bits={q_bits}, q_group_size={args.q_group_size or 64 if args.q_mode == 'affine' else args.q_group_size})"
    )
    model, config = quantize_model(
        model,
        config,
        group_size=args.q_group_size,
        bits=q_bits,
        mode=args.q_mode,
        quant_predicate=build_quant_predicate(exclude),
    )

    print(f"Writing MLX model to: {out_dir}")
    save(str(out_dir), str(model_dir), model, tokenizer, config)

    (out_dir / "quantization_meta.txt").write_text(
        "\n".join(
            [
                "backend=mlx-lm",
                f"target={target}",
                f"q_mode={args.q_mode}",
                f"q_bits={q_bits}",
                f"q_group_size={args.q_group_size if args.q_group_size is not None else 'default'}",
                f"exclude={exclude}",
                f"trust_remote_code={args.trust_remote_code}",
                f"fix_mistral_regex={args.fix_mistral_regex}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print("\n✅ MLX quantization complete.")
    print(f"✅ Quantized MLX model dir: {out_dir.resolve()}")
    print("✅ Default target was mlx-int4.\n")


if __name__ == "__main__":
    main()
