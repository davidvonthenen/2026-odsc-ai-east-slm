#!/usr/bin/env python3
"""
Run local SQL inference against a GGUF model with llama-cpp-python.

This script replaces the older Transformers + Optimum-Quanto SQL runner with a
GGUF-native runtime. It is designed for a Qwen-style SQL model that was
fine-tuned using a chat format like:

  system: "You are a database engineer. Generate valid SQL for the given schema."
  user:   "Schema: <schema>\nQuestion: <question>"

Features
--------
- Accepts either a direct .gguf file or a directory containing GGUF files.
- If a directory is provided, prefers a quantized GGUF over an
  *.unquantized.gguf file.
- Preserves the SQL-specific system prompt and schema/question user message
  format from the prior inference flow.
- Supports bundled examples, one-shot generation, and an interactive mode.
- Defaults to CPU-only inference (n_gpu_layers=0).
- Optionally strips surrounding ```sql fences from the model output.

Install
-------
    pip install llama-cpp-python

If you later want Metal acceleration on Apple Silicon, rebuild/install
llama-cpp-python with the appropriate Metal flags and set --n_gpu_layers > 0.
"""

from __future__ import annotations

import argparse
import inspect
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

try:
    from llama_cpp import Llama
except ImportError as exc:  # pragma: no cover - import failure is runtime-only
    raise SystemExit(
        "llama-cpp-python is required for GGUF inference. "
        "Install it with: pip install llama-cpp-python"
    ) from exc


SYSTEM_PROMPT = "You are a database engineer. Generate valid SQL for the given schema."


EXAMPLES: list[tuple[str, str, str]] = [
    (
        "Example 1",
        "CREATE TABLE users(id INT, name TEXT);",
        "List all user names.",
    ),
    (
        "Example 2",
        "CREATE TABLE employees(id INT, name TEXT, salary INT, department TEXT);",
        "Who is the highest paid employee? Return name and salary.",
    ),
    (
        "Example 3",
        "CREATE TABLE orders(id INT, customer_id INT, total DECIMAL(10,2), order_date DATE);",
        "Show the total revenue by customer_id, highest first.",
    ),
]


def default_threads() -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, cpu_count - 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GGUF inference for a SQL-tuned Qwen model using llama-cpp-python."
    )

    parser.add_argument(
        "--model",
        "--model_path",
        dest="model",
        default="./qwen_sql_gguf",
        help=(
            "Path to a GGUF file or a directory containing GGUF files. "
            "If a directory is given, the script prefers a quantized file over an unquantized one."
        ),
    )
    parser.add_argument(
        "--chat_format",
        type=str,
        default="chatml",
        help=(
            "Chat format passed to llama-cpp-python. Use 'chatml' for Qwen-style chat prompts, "
            "or 'auto' to rely on GGUF metadata/template detection."
        ),
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=SYSTEM_PROMPT,
        help="System prompt prepended to each request.",
    )

    parser.add_argument("--n_ctx", type=int, default=4096, help="Context window to allocate.")
    parser.add_argument(
        "--n_batch",
        type=int,
        default=512,
        help="Prompt processing batch size for llama.cpp.",
    )
    parser.add_argument(
        "--n_threads",
        type=int,
        default=default_threads(),
        help="CPU threads used for token generation.",
    )
    parser.add_argument(
        "--n_threads_batch",
        type=int,
        default=0,
        help="CPU threads for prompt processing. 0 means use n_threads or library defaults.",
    )
    parser.add_argument(
        "--n_gpu_layers",
        type=int,
        default=0,
        help="Layers to offload to GPU. 0 keeps inference CPU-only.",
    )
    parser.add_argument("--seed", type=int, default=3407, help="Sampling seed.")

    parser.add_argument(
        "--max_tokens",
        type=int,
        default=256,
        help="Maximum new tokens to generate per answer.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. 0.0 is deterministic/greedy-like behavior.",
    )
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p nucleus sampling value.")
    parser.add_argument(
        "--repeat_penalty",
        type=float,
        default=1.0,
        help="Repeat penalty passed to llama.cpp.",
    )

    parser.add_argument(
        "--schema",
        type=str,
        default=None,
        help="Schema string for one-shot generation.",
    )
    parser.add_argument(
        "--schema_file",
        type=str,
        default=None,
        help="Read schema text from a file for one-shot generation.",
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Question for one-shot generation.",
    )
    parser.add_argument(
        "--question_file",
        type=str,
        default=None,
        help="Read question text from a file for one-shot generation.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start an interactive SQL generation loop.",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Print raw model output without stripping ```sql fences.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable llama-cpp-python verbose logging.",
    )

    return parser.parse_args()


def _supported_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return only keyword args accepted by the callable when introspection works."""
    filtered = {k: v for k, v in kwargs.items() if v is not None}
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return filtered

    accepted = set(signature.parameters.keys())
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return filtered
    return {k: v for k, v in filtered.items() if k in accepted}


def resolve_model_path(model_arg: str) -> Path:
    path = Path(model_arg).expanduser().resolve()

    if path.is_file():
        if path.suffix.lower() != ".gguf":
            raise ValueError(f"Expected a .gguf file, got: {path}")
        return path

    if not path.exists():
        raise FileNotFoundError(f"Model path not found: {path}")
    if not path.is_dir():
        raise ValueError(f"Model path must be a GGUF file or directory: {path}")

    candidates = sorted([p for p in path.glob("*.gguf") if p.is_file()])
    if not candidates:
        candidates = sorted([p for p in path.rglob("*.gguf") if p.is_file()])
    if not candidates:
        raise FileNotFoundError(f"No GGUF files found under: {path}")

    quantized = [p for p in candidates if "unquantized" not in p.name.lower()]
    pool = quantized or candidates

    def sort_key(p: Path) -> tuple[int, float, str]:
        name = p.name.lower()
        quant_hint = 0 if re.search(r"(?:^|[._-])q\d", name) else 1
        return (quant_hint, -p.stat().st_mtime, name)

    chosen = sorted(pool, key=sort_key)[0]
    return chosen.resolve()


def read_text_file(path_str: str) -> str:
    path = Path(path_str).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def build_llm(args: argparse.Namespace, model_path: Path) -> Llama:
    init_kwargs: dict[str, Any] = {
        "model_path": str(model_path),
        "n_ctx": args.n_ctx,
        "n_batch": args.n_batch,
        "n_threads": args.n_threads,
        "n_threads_batch": args.n_threads_batch if args.n_threads_batch > 0 else None,
        "n_gpu_layers": args.n_gpu_layers,
        "seed": args.seed,
        "verbose": args.verbose,
    }

    if args.chat_format.lower() != "auto":
        init_kwargs["chat_format"] = args.chat_format

    init_kwargs = _supported_kwargs(Llama, init_kwargs)
    return Llama(**init_kwargs)


def build_user_prompt(schema: str, question: str) -> str:
    schema = schema.strip()
    question = question.strip()
    return f"Schema: {schema}\nQuestion: {question}"


def strip_sql_fences(text: str) -> str:
    value = text.strip()
    fence_match = re.match(r"^```(?:sql)?\s*(.*?)\s*```$", value, flags=re.IGNORECASE | re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()
    return value


def generate_sql(
    llm: Llama,
    system_prompt: str,
    schema: str,
    question: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    repeat_penalty: float,
    raw: bool = False,
) -> tuple[str, float]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": build_user_prompt(schema=schema, question=question)},
    ]

    create_kwargs: dict[str, Any] = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "repeat_penalty": repeat_penalty,
    }
    create_kwargs = _supported_kwargs(llm.create_chat_completion, create_kwargs)

    start_time = time.time()
    response = llm.create_chat_completion(**create_kwargs)
    end_time = time.time()

    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected response format from llama-cpp-python: {response!r}") from exc

    text = content.strip()
    if not raw:
        text = strip_sql_fences(text)

    return text, end_time - start_time


def read_multiline_block(label: str, terminator: str = "END") -> str:
    print(f"{label} (finish with a line containing only {terminator}):")
    lines: list[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == terminator:
            break
        lines.append(line)
    return "\n".join(lines).strip()


def run_examples(llm: Llama, args: argparse.Namespace) -> None:
    for title, schema, question in EXAMPLES:
        print("-" * 30)
        print(f"{title}: {question}")
        print()
        sql, elapsed = generate_sql(
            llm=llm,
            system_prompt=args.system_prompt,
            schema=schema,
            question=question,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repeat_penalty=args.repeat_penalty,
            raw=args.raw,
        )
        print(sql)
        print(f"\nGeneration time: {elapsed:.2f} seconds")
        print("-" * 30)
        print()


def run_single(llm: Llama, args: argparse.Namespace, schema: str, question: str) -> None:
    sql, elapsed = generate_sql(
        llm=llm,
        system_prompt=args.system_prompt,
        schema=schema,
        question=question,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repeat_penalty=args.repeat_penalty,
        raw=args.raw,
    )
    print(sql)
    print(f"\nGeneration time: {elapsed:.2f} seconds")


def run_interactive(llm: Llama, args: argparse.Namespace) -> None:
    print("Interactive SQL mode. Submit an empty schema block or press Ctrl-D/Ctrl-C to exit.")
    print()

    while True:
        try:
            schema = read_multiline_block("Schema")
            if not schema:
                print("No schema provided. Exiting.")
                break

            question = input("Question: ").strip()
            if not question:
                print("No question provided. Exiting.")
                break

            print()
            sql, elapsed = generate_sql(
                llm=llm,
                system_prompt=args.system_prompt,
                schema=schema,
                question=question,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                repeat_penalty=args.repeat_penalty,
                raw=args.raw,
            )
            print(sql)
            print(f"\nGeneration time: {elapsed:.2f} seconds")
            print("\n" + "-" * 30 + "\n")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting interactive mode.")
            break


def resolve_single_request(args: argparse.Namespace) -> tuple[str, str] | None:
    schema = args.schema
    question = args.question

    if args.schema_file:
        schema = read_text_file(args.schema_file)
    if args.question_file:
        question = read_text_file(args.question_file)

    if schema is None and question is None:
        return None
    if not schema or not question:
        raise SystemExit(
            "For one-shot generation, provide both --schema and --question, or use --schema_file / --question_file."
        )
    return schema, question


def main() -> None:
    args = parse_args()
    model_path = resolve_model_path(args.model)

    print("Resolved GGUF model")
    print("-------------------")
    print(f"Model path   : {model_path}")
    print(f"Chat format  : {args.chat_format}")
    print(f"System prompt: {args.system_prompt}")
    print(f"Context size : {args.n_ctx}")
    print(f"Batch size   : {args.n_batch}")
    print(f"Threads      : {args.n_threads}")
    print(f"GPU layers   : {args.n_gpu_layers}")
    print(f"Max tokens   : {args.max_tokens}")
    print(f"Temperature  : {args.temperature}")
    print()

    llm = build_llm(args=args, model_path=model_path)

    request = resolve_single_request(args)
    if request is not None:
        schema, question = request
        run_single(llm=llm, args=args, schema=schema, question=question)
        return

    if args.interactive:
        run_interactive(llm=llm, args=args)
        return

    run_examples(llm=llm, args=args)


if __name__ == "__main__":
    main()
