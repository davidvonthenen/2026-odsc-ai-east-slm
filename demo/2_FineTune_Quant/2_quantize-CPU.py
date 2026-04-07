#!/usr/bin/env python3
"""
Convert a merged Hugging Face causal LM directory to GGUF and quantize it for CPU inference.

This script is tailored for the output of the provided Qwen fine-tuning workflow:
- the LoRA adapter is merged back into the base model
- the merged model is saved as a standard Hugging Face directory
- weights are stored with safe serialization, typically as sharded `.safetensors`

Expected input directory layout (example: ./merged_qwen_sql):
- config.json
- tokenizer assets (tokenizer.json, tokenizer_config.json, etc.)
- either:
    * model.safetensors
    * model.safetensors.index.json + model-00001-of-0000N.safetensors shards
    * pytorch_model.bin
    * pytorch_model.bin.index.json + pytorch_model-00001-of-0000N.bin shards

The script does not accept adapter-only LoRA exports. It expects a fully merged model
folder that `convert_hf_to_gguf.py` can read directly.

Pipeline
--------
1) Validate the merged Hugging Face model directory.
2) Convert the directory to a high-precision GGUF with llama.cpp's convert_hf_to_gguf.py.
3) Quantize that GGUF with llama-quantize.

Example
-------
python quantize_merged_hf_to_gguf.py \
    --model_dir ./merged_qwen_sql \
    --output_dir ./gguf_out \
    --llama_cpp_dir ./llama.cpp \
    --quant_type Q4_K_M
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Iterable, Sequence


TOKENIZER_MARKERS = (
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
)

ADAPTER_MARKERS = (
    "adapter_config.json",
    "adapter_model.bin",
    "adapter_model.safetensors",
)


@dataclass(frozen=True)
class WeightLayout:
    format: str
    sharded: bool
    index_file: Path | None
    files: tuple[Path, ...]


@dataclass(frozen=True)
class ModelSummary:
    model_dir: Path
    model_name: str
    architecture: tuple[str, ...]
    model_type: str | None
    torch_dtype: str | None
    weight_layout: WeightLayout


@dataclass(frozen=True)
class LlamaCppTools:
    convert_script: Path
    quantize_binary: Path
    workdir: Path


@dataclass(frozen=True)
class OutputPaths:
    intermediate_gguf: Path
    quantized_gguf: Path
    manifest_json: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a merged Hugging Face model directory (for example ./merged_qwen_sql) "
            "to GGUF and quantize it with llama.cpp."
        )
    )

    parser.add_argument(
        "--model_dir",
        "--source",
        dest="model_dir",
        default="./merged_qwen_sql",
        help="Path to the merged Hugging Face model directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./gguf_out",
        help="Directory where GGUF artifacts will be written.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Optional model name override used in output file names and GGUF metadata.",
    )
    parser.add_argument(
        "--quant_type",
        type=str,
        default="Q4_K_M",
        help="Quantization type for llama-quantize, for example: Q4_K_M, Q5_K_M, Q8_0.",
    )
    parser.add_argument(
        "--gguf_outtype",
        type=str,
        choices=["auto", "f16", "bf16", "f32"],
        default="auto",
        help=(
            "Precision for the intermediate GGUF. If 'auto', the value is passed through to "
            "llama.cpp so the converter can infer it from the actual tensor dtypes."
        ),
    )
    parser.add_argument(
        "--llama_cpp_dir",
        type=str,
        default="./llama.cpp",
        help="Path to a local llama.cpp checkout.",
    )
    parser.add_argument(
        "--convert_script",
        type=str,
        default=None,
        help="Explicit path to convert_hf_to_gguf.py. Overrides --llama_cpp_dir for the converter.",
    )
    parser.add_argument(
        "--quantize_binary",
        type=str,
        default=None,
        help="Explicit path to llama-quantize. Overrides auto-discovery.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="Optional thread count passed to llama-quantize. 0 lets llama.cpp decide.",
    )

    # Converter passthroughs.
    parser.add_argument(
        "--use_temp_file",
        action="store_true",
        help="Pass --use-temp-file to convert_hf_to_gguf.py.",
    )
    parser.add_argument(
        "--no_lazy",
        action="store_true",
        help="Pass --no-lazy to convert_hf_to_gguf.py.",
    )
    parser.add_argument(
        "--split_max_tensors",
        type=int,
        default=0,
        help="Optional --split-max-tensors value for convert_hf_to_gguf.py.",
    )
    parser.add_argument(
        "--split_max_size",
        type=str,
        default=None,
        help="Optional --split-max-size value for convert_hf_to_gguf.py, e.g. 45G.",
    )

    # Quantizer passthroughs.
    parser.add_argument(
        "--imatrix",
        type=str,
        default=None,
        help="Optional importance matrix GGUF file passed to llama-quantize.",
    )
    parser.add_argument(
        "--include_weight",
        action="append",
        default=[],
        help="Repeatable tensor pattern passed as --include-weights to llama-quantize.",
    )
    parser.add_argument(
        "--exclude_weight",
        action="append",
        default=[],
        help="Repeatable tensor pattern passed as --exclude-weights to llama-quantize.",
    )
    parser.add_argument(
        "--output_tensor_type",
        type=str,
        default=None,
        help="Optional --output-tensor-type for llama-quantize.",
    )
    parser.add_argument(
        "--token_embedding_type",
        type=str,
        default=None,
        help="Optional --token-embedding-type for llama-quantize.",
    )
    parser.add_argument(
        "--leave_output_tensor",
        action="store_true",
        help="Pass --leave-output-tensor to llama-quantize.",
    )
    parser.add_argument(
        "--pure",
        action="store_true",
        help="Pass --pure to llama-quantize.",
    )
    parser.add_argument(
        "--keep_split",
        action="store_true",
        help="Pass --keep-split to llama-quantize if the input GGUF is split.",
    )

    parser.add_argument(
        "--keep_intermediate",
        action="store_true",
        help="Keep the high-precision intermediate GGUF after quantization.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the resolved commands without executing them.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print extra validation and path resolution details.",
    )

    return parser.parse_args()


def expand_path(value: str | os.PathLike[str]) -> Path:
    # Normalize to an absolute path immediately. This avoids accidental path
    # rebasing when we run subprocesses with cwd set to the llama.cpp tree.
    return Path(value).expanduser().resolve(strict=False)


def ensure(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def quoted(cmd: Sequence[str]) -> str:
    return shlex.join([str(part) for part in cmd])


def maybe_remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        path.unlink(missing_ok=True)


def sanitize_name(name: str) -> str:
    safe = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_", "."):
            safe.append(ch)
        else:
            safe.append("_")
    cleaned = "".join(safe).strip("._")
    return cleaned or "model"


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON file: {path}\n{exc}") from exc


def has_tokenizer_assets(model_dir: Path) -> bool:
    return any((model_dir / marker).exists() for marker in TOKENIZER_MARKERS)


def contains_adapter_markers(model_dir: Path) -> bool:
    return any((model_dir / marker).exists() for marker in ADAPTER_MARKERS)


def find_shards_from_index(index_path: Path) -> tuple[Path, ...]:
    payload = read_json(index_path)
    weight_map = payload.get("weight_map")
    ensure(
        isinstance(weight_map, dict) and weight_map,
        f"Index file does not contain a non-empty 'weight_map': {index_path}",
    )

    shard_names = sorted({str(value) for value in weight_map.values()})
    shard_paths = tuple(index_path.parent / name for name in shard_names)
    missing = [str(path) for path in shard_paths if not path.is_file()]
    ensure(
        not missing,
        "Sharded weight index references missing files:\n- " + "\n- ".join(missing),
    )
    return shard_paths


def detect_weight_layout(model_dir: Path) -> WeightLayout:
    safetensors_index = model_dir / "model.safetensors.index.json"
    pytorch_index = model_dir / "pytorch_model.bin.index.json"

    if safetensors_index.is_file():
        return WeightLayout(
            format="safetensors",
            sharded=True,
            index_file=safetensors_index,
            files=find_shards_from_index(safetensors_index),
        )

    if pytorch_index.is_file():
        return WeightLayout(
            format="pytorch_bin",
            sharded=True,
            index_file=pytorch_index,
            files=find_shards_from_index(pytorch_index),
        )

    single_candidates = [
        (model_dir / "model.safetensors", "safetensors"),
        (model_dir / "pytorch_model.bin", "pytorch_bin"),
    ]
    for candidate, fmt in single_candidates:
        if candidate.is_file():
            return WeightLayout(
                format=fmt,
                sharded=False,
                index_file=None,
                files=(candidate,),
            )

    # Support the common shard naming pattern, but require the matching index.
    stray_st_shards = sorted(model_dir.glob("model-*-of-*.safetensors"))
    stray_bin_shards = sorted(model_dir.glob("pytorch_model-*-of-*.bin"))

    if stray_st_shards:
        raise RuntimeError(
            "Found sharded safetensors files but no model.safetensors.index.json. "
            "The merged Hugging Face directory is incomplete."
        )

    if stray_bin_shards:
        raise RuntimeError(
            "Found sharded PyTorch .bin files but no pytorch_model.bin.index.json. "
            "The merged Hugging Face directory is incomplete."
        )

    raise RuntimeError(
        "No supported model weights were found. Expected one of: model.safetensors, "
        "model.safetensors.index.json + shards, pytorch_model.bin, or pytorch_model.bin.index.json + shards."
    )


def parse_model_summary(model_dir: Path, model_name_override: str | None = None) -> ModelSummary:
    ensure(model_dir.is_dir(), f"Model directory does not exist: {model_dir}")

    config_path = model_dir / "config.json"
    ensure(config_path.is_file(), f"Missing config.json in model directory: {model_dir}")
    ensure(
        has_tokenizer_assets(model_dir),
        "Tokenizer assets are missing. Expected files like tokenizer.json / tokenizer_config.json.",
    )

    config = read_json(config_path)
    architecture_raw = config.get("architectures") or []
    if isinstance(architecture_raw, str):
        architecture = (architecture_raw,)
    elif isinstance(architecture_raw, list):
        architecture = tuple(str(item) for item in architecture_raw)
    else:
        architecture = ()

    model_type = config.get("model_type")
    torch_dtype = config.get("torch_dtype")
    if torch_dtype is not None:
        torch_dtype = str(torch_dtype)

    weight_layout = detect_weight_layout(model_dir)

    # Reject obvious adapter-only folders.
    if contains_adapter_markers(model_dir) and not weight_layout.files:
        raise RuntimeError(
            "The directory looks like an adapter-only export. GGUF conversion requires a fully merged model directory."
        )

    model_name = sanitize_name(model_name_override or model_dir.name)

    return ModelSummary(
        model_dir=model_dir,
        model_name=model_name,
        architecture=architecture,
        model_type=str(model_type) if model_type is not None else None,
        torch_dtype=torch_dtype,
        weight_layout=weight_layout,
    )


def infer_outtype(requested: str, config_torch_dtype: str | None) -> tuple[str, str | None]:
    if requested != "auto":
        return requested, "explicit CLI override"

    # Intentionally trust llama.cpp's own auto detection here. The upstream converter inspects
    # real tensor dtypes and explicitly avoids trusting config.json["torch_dtype"] because some
    # fine-tuned exports lie about it.
    _ = config_torch_dtype
    return "auto", "llama.cpp auto mode (actual tensor dtype detection)"


def resolve_tools(args: argparse.Namespace) -> LlamaCppTools:
    explicit_convert = expand_path(args.convert_script) if args.convert_script else None
    explicit_quantize = expand_path(args.quantize_binary) if args.quantize_binary else None
    llama_cpp_dir = expand_path(args.llama_cpp_dir) if args.llama_cpp_dir else None

    if explicit_convert is not None:
        convert_script = explicit_convert
        workdir = convert_script.parent
    else:
        ensure(llama_cpp_dir is not None, "Either --llama_cpp_dir or --convert_script is required.")
        convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"
        workdir = llama_cpp_dir

    convert_script = convert_script.resolve(strict=False)
    workdir = workdir.resolve(strict=False)

    ensure(
        convert_script.is_file(),
        "Could not find convert_hf_to_gguf.py. Pass --convert_script explicitly or point --llama_cpp_dir at a valid llama.cpp checkout.",
    )

    quantize_candidates: list[Path] = []
    if explicit_quantize is not None:
        quantize_candidates.append(explicit_quantize)

    suffixes = [".exe"] if os.name == "nt" else [""]
    quantize_search_root = llama_cpp_dir if llama_cpp_dir is not None else workdir
    if quantize_search_root is not None:
        for suffix in suffixes:
            for rel in (
                f"build/bin/llama-quantize{suffix}",
                f"build/bin/quantize{suffix}",
                f"build/bin/Release/llama-quantize{suffix}",
                f"build/bin/Release/quantize{suffix}",
                f"llama-quantize{suffix}",
                f"quantize{suffix}",
            ):
                quantize_candidates.append(quantize_search_root / rel)

    for prog in ("llama-quantize", "quantize"):
        found = shutil.which(prog)
        if found:
            quantize_candidates.append(Path(found))

    quantize_binary: Path | None = None
    for candidate in quantize_candidates:
        candidate = candidate.resolve(strict=False)
        if candidate.is_file():
            quantize_binary = candidate
            break

    ensure(
        quantize_binary is not None,
        "Could not find llama-quantize. Pass --quantize_binary explicitly or point --llama_cpp_dir at a built llama.cpp tree.",
    )

    return LlamaCppTools(
        convert_script=convert_script,
        quantize_binary=quantize_binary,
        workdir=workdir,
    )


def build_converter_env(tools: LlamaCppTools) -> dict[str, str]:
    env = os.environ.copy()
    extra_paths: list[str] = []

    root = tools.workdir
    gguf_py = root / "gguf-py"

    if root.exists():
        extra_paths.append(str(root))
    if gguf_py.exists():
        extra_paths.append(str(gguf_py))

    existing = env.get("PYTHONPATH", "")
    pieces = [*extra_paths]
    if existing:
        pieces.append(existing)
    if pieces:
        env["PYTHONPATH"] = os.pathsep.join(pieces)

    return env


def run_command(
    cmd: Sequence[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    dry_run: bool = False,
) -> None:
    print(f"\n$ {quoted(cmd)}")
    if cwd is not None:
        print(f"# cwd: {cwd}")
    if dry_run:
        return

    subprocess.run(
        [str(part) for part in cmd],
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        check=True,
    )


def build_output_paths(
    output_dir: Path,
    model_name: str,
    intermediate_label: str,
    quant_type: str,
) -> OutputPaths:
    intermediate_gguf = output_dir / f"{model_name}.{intermediate_label}.gguf"
    quantized_gguf = output_dir / f"{model_name}.{quant_type}.gguf"
    manifest_json = output_dir / f"{model_name}.manifest.json"
    return OutputPaths(
        intermediate_gguf=intermediate_gguf,
        quantized_gguf=quantized_gguf,
        manifest_json=manifest_json,
    )


def convert_to_gguf(
    *,
    tools: LlamaCppTools,
    model: ModelSummary,
    output_path: Path,
    outtype: str,
    model_name_override: str | None,
    use_temp_file: bool,
    no_lazy: bool,
    split_max_tensors: int,
    split_max_size: str | None,
    verbose: bool,
    dry_run: bool,
) -> None:
    cmd = [
        sys.executable,
        str(tools.convert_script),
        str(model.model_dir),
        "--outfile",
        str(output_path),
    ]

    if outtype != "auto":
        cmd.extend(["--outtype", outtype])
    if model_name_override:
        cmd.extend(["--model-name", model_name_override])
    if use_temp_file:
        cmd.append("--use-temp-file")
    if no_lazy:
        cmd.append("--no-lazy")
    if split_max_tensors and split_max_tensors > 0:
        cmd.extend(["--split-max-tensors", str(split_max_tensors)])
    if split_max_size:
        cmd.extend(["--split-max-size", str(split_max_size)])
    if verbose:
        cmd.append("--verbose")

    env = build_converter_env(tools)
    run_command(cmd, cwd=tools.workdir, env=env, dry_run=dry_run)


def quantize_gguf(
    *,
    tools: LlamaCppTools,
    input_path: Path,
    output_path: Path,
    quant_type: str,
    threads: int,
    imatrix: Path | None,
    include_weights: Iterable[str],
    exclude_weights: Iterable[str],
    output_tensor_type: str | None,
    token_embedding_type: str | None,
    leave_output_tensor: bool,
    pure: bool,
    keep_split: bool,
    dry_run: bool,
) -> None:
    cmd: list[str] = [str(tools.quantize_binary)]

    if imatrix is not None:
        cmd.extend(["--imatrix", str(imatrix)])
    for value in include_weights:
        cmd.extend(["--include-weights", value])
    for value in exclude_weights:
        cmd.extend(["--exclude-weights", value])
    if output_tensor_type:
        cmd.extend(["--output-tensor-type", output_tensor_type])
    if token_embedding_type:
        cmd.extend(["--token-embedding-type", token_embedding_type])
    if leave_output_tensor:
        cmd.append("--leave-output-tensor")
    if pure:
        cmd.append("--pure")
    if keep_split:
        cmd.append("--keep-split")

    cmd.extend([str(input_path), str(output_path), quant_type])
    if threads and threads > 0:
        cmd.append(str(threads))

    run_command(cmd, cwd=tools.workdir, dry_run=dry_run)


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()

    model_dir = expand_path(args.model_dir)
    output_dir = expand_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = parse_model_summary(model_dir, model_name_override=args.model_name)
    tools = resolve_tools(args)

    effective_outtype, outtype_reason = infer_outtype(args.gguf_outtype, model.torch_dtype)
    quant_type = args.quant_type.upper() if args.quant_type.upper() != "COPY" else "COPY"
    intermediate_label = effective_outtype if effective_outtype != "auto" else "unquantized"

    paths = build_output_paths(
        output_dir=output_dir,
        model_name=model.model_name,
        intermediate_label=intermediate_label,
        quant_type=quant_type,
    )

    for path in (paths.intermediate_gguf, paths.quantized_gguf, paths.manifest_json):
        if path.exists():
            if args.overwrite:
                maybe_remove(path)
            else:
                raise FileExistsError(
                    f"Output already exists: {path}\nPass --overwrite to replace it."
                )

    if args.include_weight and args.exclude_weight:
        raise RuntimeError("Use either --include_weight or --exclude_weight, not both.")

    imatrix_path = expand_path(args.imatrix) if args.imatrix else None
    if imatrix_path is not None:
        ensure(imatrix_path.is_file(), f"Importance matrix file not found: {imatrix_path}")

    print("\nResolved input")
    print("--------------")
    print(f"Model dir        : {model.model_dir}")
    print(f"Model name       : {model.model_name}")
    print(f"Architecture     : {', '.join(model.architecture) if model.architecture else 'unknown'}")
    print(f"Model type       : {model.model_type or 'unknown'}")
    print(f"Config dtype     : {model.torch_dtype or 'unknown'}")
    print(
        f"Weights          : {'sharded' if model.weight_layout.sharded else 'single-file'} "
        f"{model.weight_layout.format} ({len(model.weight_layout.files)} file{'s' if len(model.weight_layout.files) != 1 else ''})"
    )
    if model.weight_layout.index_file is not None:
        print(f"Weight index     : {model.weight_layout.index_file.name}")
    print(f"GGUF outtype     : {effective_outtype}")
    if outtype_reason:
        print(f"Outtype source   : {outtype_reason}")
    print(f"Quant type       : {quant_type}")
    print(f"Output dir       : {output_dir}")

    if args.verbose:
        print("\nResolved llama.cpp tools")
        print("------------------------")
        print(f"Convert script   : {tools.convert_script}")
        print(f"Quantize binary  : {tools.quantize_binary}")
        print(f"Workdir          : {tools.workdir}")
        print("\nDetected weight files")
        print("---------------------")
        for file_path in model.weight_layout.files:
            print(f"- {file_path.name}")

    success = False
    try:
        convert_to_gguf(
            tools=tools,
            model=model,
            output_path=paths.intermediate_gguf,
            outtype=effective_outtype,
            model_name_override=args.model_name,
            use_temp_file=bool(args.use_temp_file),
            no_lazy=bool(args.no_lazy),
            split_max_tensors=int(args.split_max_tensors),
            split_max_size=args.split_max_size,
            verbose=bool(args.verbose),
            dry_run=bool(args.dry_run),
        )

        quantize_gguf(
            tools=tools,
            input_path=paths.intermediate_gguf,
            output_path=paths.quantized_gguf,
            quant_type=quant_type,
            threads=int(args.threads),
            imatrix=imatrix_path,
            include_weights=args.include_weight,
            exclude_weights=args.exclude_weight,
            output_tensor_type=args.output_tensor_type,
            token_embedding_type=args.token_embedding_type,
            leave_output_tensor=bool(args.leave_output_tensor),
            pure=bool(args.pure),
            keep_split=bool(args.keep_split),
            dry_run=bool(args.dry_run),
        )

        manifest = {
            "model_dir": str(model.model_dir),
            "model_name": model.model_name,
            "architecture": list(model.architecture),
            "model_type": model.model_type,
            "config_torch_dtype": model.torch_dtype,
            "weight_format": model.weight_layout.format,
            "weight_sharded": model.weight_layout.sharded,
            "weight_index": str(model.weight_layout.index_file) if model.weight_layout.index_file else None,
            "weight_files": [str(path) for path in model.weight_layout.files],
            "convert_script": str(tools.convert_script),
            "quantize_binary": str(tools.quantize_binary),
            "gguf_outtype_requested": args.gguf_outtype,
            "gguf_outtype_effective": effective_outtype,
            "gguf_outtype_reason": outtype_reason,
            "quant_type": quant_type,
            "intermediate_gguf": str(paths.intermediate_gguf),
            "quantized_gguf": str(paths.quantized_gguf),
            "threads": args.threads,
            "use_temp_file": bool(args.use_temp_file),
            "no_lazy": bool(args.no_lazy),
            "split_max_tensors": args.split_max_tensors,
            "split_max_size": args.split_max_size,
            "imatrix": str(imatrix_path) if imatrix_path else None,
            "include_weight": list(args.include_weight),
            "exclude_weight": list(args.exclude_weight),
            "output_tensor_type": args.output_tensor_type,
            "token_embedding_type": args.token_embedding_type,
            "leave_output_tensor": bool(args.leave_output_tensor),
            "pure": bool(args.pure),
            "keep_split": bool(args.keep_split),
        }
        if not args.dry_run:
            write_manifest(paths.manifest_json, manifest)

        success = True

        print("\nDone")
        print("----")
        print(f"Intermediate GGUF : {paths.intermediate_gguf}")
        print(f"Quantized GGUF    : {paths.quantized_gguf}")
        if not args.dry_run:
            print(f"Manifest          : {paths.manifest_json}")

    finally:
        if success and paths.intermediate_gguf.exists() and not args.keep_intermediate and not args.dry_run:
            paths.intermediate_gguf.unlink(missing_ok=True)
            print(f"Removed intermediate GGUF: {paths.intermediate_gguf}")


if __name__ == "__main__":
    main()
