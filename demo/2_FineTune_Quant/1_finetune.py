#!/usr/bin/env python3
"""Fine-tune Qwen2.5 into a Text-to-SQL generator, then merge for quantization.

What this script does:
1) Loads gretelai/synthetic_text_to_sql
2) Formats examples using the Qwen ChatML template:
     system: "You are a database engineer..."
     user:   "Schema: ...\nQuestion: ..."
     assistant: "<SQL>"
3) Tokenizes and masks loss so we *only* train on the assistant completion.
4) Fine-tunes with LoRA (PEFT) to keep it lightweight.
5) Saves the LoRA adapter checkpoints.
6) Reloads the base model + adapter, merges weights, and writes ./merged_qwen_sql
   (ready for 2_quantize.py).

Recommended launch (single node, 1+ GPUs):
  accelerate launch 1_finetune.py \
    --base_model Qwen/Qwen2.5-7B-Instruct \
    --output_dir ./qwen_sql_lora \
    --merged_dir ./merged_qwen_sql
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import warnings

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (  # type: ignore
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


SYSTEM_PROMPT = "You are a database engineer. Generate valid SQL for the given schema."

# -----------------------------
# GPU-aware defaults
# -----------------------------

def _get_gpu_name_and_mem_gb() -> Tuple[str, float]:
    if not torch.cuda.is_available():
        return "cpu", 0.0
    props = torch.cuda.get_device_properties(0)
    return props.name, float(props.total_memory) / (1024**3)


def _auto_hparams(gpu_mem_gb: float) -> Dict[str, int]:
    """Pick conservative defaults.

    Goal: provide a sane starting point that works on H100 80GB/94GB and scales on
    H200 141GB without turning your run into a VRAM bonfire.
    """
    if gpu_mem_gb >= 130:  # H200-class
        return {
            "max_seq_len": 8192,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 4,
        }
    if gpu_mem_gb >= 90:  # H100 94GB / NVL-ish
        return {
            "max_seq_len": 4096,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 4,
        }
    # H100 80GB (and other ~80GB Hopper/Ampere)
    return {
        "max_seq_len": 4096,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
    }


# -----------------------------
# Tokenization + loss masking
# -----------------------------

def _build_user_text(schema: str, question: str) -> str:
    return f"Schema: {schema}\nQuestion: {question}"


def _safe_tokenizer_from_pretrained(model_name_or_path: str, trust_remote_code: bool = True):
    # Keep parity with your quantizer's tokenizer loading quirks.
    try:
        return AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            fix_mistral_regex=True,
        )
    except TypeError:
        # Older/newer transformers may not have fix_mistral_regex.
        return AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )


def _maybe_set_padding(tokenizer) -> None:
    # Some Qwen tokenizers don't define a pad token. We need one for batching.
    if tokenizer.pad_token_id is None:
        # Using EOS as PAD is common for decoder-only LMs.
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"


def _apply_chat_template(tokenizer, messages: List[Dict[str, str]], add_generation_prompt: bool) -> str:
    # Qwen uses ChatML; tokenizer.apply_chat_template handles the exact formatting.
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def _format_pair(tokenizer, schema: str, question: str, answer_sql: str) -> Tuple[str, str]:
    """Returns (prompt_text, full_text).

    prompt_text ends with the assistant generation prefix.
    full_text includes the assistant content.
    """
    user_text = _build_user_text(schema, question)

    msgs_prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    prompt_text = _apply_chat_template(tokenizer, msgs_prompt, add_generation_prompt=True)

    msgs_full = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": answer_sql},
    ]
    full_text = _apply_chat_template(tokenizer, msgs_full, add_generation_prompt=False)

    return prompt_text, full_text


def _tokenize_and_mask_batch(
    batch: Dict[str, List[Any]],
    tokenizer,
    max_seq_len: int,
) -> Dict[str, List[Any]]:
    schemas = batch["sql_context"]
    questions = batch["sql_prompt"]
    answers = batch["sql"]

    prompt_texts: List[str] = []
    full_texts: List[str] = []

    for s, q, a in zip(schemas, questions, answers):
        ptxt, ftxt = _format_pair(tokenizer, s, q, a)
        prompt_texts.append(ptxt)
        full_texts.append(ftxt)

    # Tokenize prompt and full separately, then mask labels up to prompt length.
    full_enc = tokenizer(
        full_texts,
        add_special_tokens=False,
        truncation=True,
        max_length=max_seq_len,
    )
    prompt_enc = tokenizer(
        prompt_texts,
        add_special_tokens=False,
        truncation=True,
        max_length=max_seq_len,
    )

    input_ids_batch = full_enc["input_ids"]
    attn_batch = full_enc["attention_mask"]

    labels_batch: List[List[int]] = []
    prompt_len_batch: List[int] = []
    input_len_batch: List[int] = []

    for input_ids, prompt_ids in zip(input_ids_batch, prompt_enc["input_ids"]):
        prompt_len = min(len(prompt_ids), len(input_ids))
        labels = input_ids.copy()
        labels[:prompt_len] = [-100] * prompt_len

        labels_batch.append(labels)
        prompt_len_batch.append(prompt_len)
        input_len_batch.append(len(input_ids))

    return {
        "input_ids": input_ids_batch,
        "attention_mask": attn_batch,
        "labels": labels_batch,
        "prompt_len": prompt_len_batch,
        "input_len": input_len_batch,
    }


@dataclass
class DataCollatorForCausalLMWithPadding:
    tokenizer: Any
    pad_to_multiple_of: Optional[int] = 8

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            m = self.pad_to_multiple_of
            if max_len % m != 0:
                max_len = ((max_len // m) + 1) * m

        batch_input_ids: List[List[int]] = []
        batch_attention: List[List[int]] = []
        batch_labels: List[List[int]] = []

        for f in features:
            ids = list(f["input_ids"])
            attn = list(f["attention_mask"])
            labels = list(f["labels"])

            pad_len = max_len - len(ids)
            if pad_len > 0:
                ids.extend([pad_id] * pad_len)
                attn.extend([0] * pad_len)
                labels.extend([-100] * pad_len)

            batch_input_ids.append(ids)
            batch_attention.append(attn)
            batch_labels.append(labels)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


# -----------------------------
# Model loading helpers
# -----------------------------

def _pick_attn_impl(requested: str) -> str:
    if requested in {"sdpa", "flash_attention_2", "eager"}:
        return requested

    # auto
    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except Exception:
        return "sdpa"


def _load_base_model(
    model_name_or_path: str,
    torch_dtype: torch.dtype,
    attn_impl: str,
    trust_remote_code: bool = True,
):
    # Try requested attention implementation; fall back to SDPA if needed.
    load_kwargs = dict(
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_cache=False,
    )

    try:
        return AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            attn_implementation=attn_impl,
            **load_kwargs,
        )
    except TypeError:
        # Older transformers: no attn_implementation kwarg.
        return AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            **load_kwargs,
        )
    except Exception:
        # Something about flash-attn availability/build, etc.
        if attn_impl != "sdpa":
            return AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                attn_implementation="sdpa",
                **load_kwargs,
            )
        raise


def _pick_optim(prefer_fused: bool) -> str:
    if not prefer_fused:
        return "adamw_torch"

    # TrainingArguments validates optim names; be defensive across versions.
    try:
        from transformers.training_args import OptimizerNames  # type: ignore

        if hasattr(OptimizerNames, "ADAMW_TORCH_FUSED"):
            return "adamw_torch_fused"
    except Exception:
        pass

    return "adamw_torch"


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune Qwen2.5 into a Text-to-SQL model and merge for quantization.")

    p.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--dataset_name", type=str, default="gretelai/synthetic_text_to_sql")
    p.add_argument("--train_split", type=str, default="train")
    p.add_argument("--eval_split", type=str, default="test")

    p.add_argument("--output_dir", type=str, default="./qwen_sql_lora", help="Where to save LoRA adapters/checkpoints.")
    p.add_argument("--merged_dir", type=str, default="./merged_qwen_sql", help="Where to write merged full-precision model.")

    p.add_argument("--seed", type=int, default=3407)

    # "0" means auto-tune based on GPU memory.
    p.add_argument("--max_seq_len", type=int, default=0)
    p.add_argument("--per_device_train_batch_size", type=int, default=0)
    p.add_argument("--gradient_accumulation_steps", type=int, default=0)

    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--weight_decay", type=float, default=0.0)

    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--save_total_limit", type=int, default=2)

    p.add_argument("--max_train_samples", type=int, default=0, help="0 = use full split")
    p.add_argument("--max_eval_samples", type=int, default=1024)

    p.add_argument("--preprocess_num_proc", type=int, default=4)

    p.add_argument("--gradient_checkpointing", dest="gradient_checkpointing", action="store_true")
    p.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing", action="store_false")
    p.set_defaults(gradient_checkpointing=True)

    p.add_argument("--attn_impl", type=str, default="auto", choices=["auto", "sdpa", "flash_attention_2", "eager"])

    p.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16"])

    p.add_argument("--prefer_fused_optim", dest="prefer_fused_optim", action="store_true")
    p.add_argument("--no_prefer_fused_optim", dest="prefer_fused_optim", action="store_false")
    p.set_defaults(prefer_fused_optim=True)

    # LoRA knobs
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated module names for LoRA injection.",
    )

    return p.parse_args()


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    # Hopper-friendly math settings.
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    gpu_name, gpu_mem_gb = _get_gpu_name_and_mem_gb()
    auto = _auto_hparams(gpu_mem_gb)

    max_seq_len = args.max_seq_len or auto["max_seq_len"]
    per_device_train_batch_size = args.per_device_train_batch_size or auto["per_device_train_batch_size"]
    grad_accum = args.gradient_accumulation_steps or auto["gradient_accumulation_steps"]

    print("\n=== Hardware ===")
    print(f"GPU: {gpu_name}")
    if gpu_mem_gb:
        print(f"VRAM: {gpu_mem_gb:.1f} GB")
    else:
        print("VRAM: n/a (no CUDA detected)")

    if gpu_mem_gb >= 130:
        print("Detected H200-class memory. Auto defaults bias toward longer sequences.")
    elif gpu_mem_gb >= 70:
        print("Detected H100-class memory. Auto defaults are conservative and stable.")

    print("\n=== Training config (resolved) ===")
    print(f"max_seq_len={max_seq_len}")
    print(f"per_device_train_batch_size={per_device_train_batch_size}")
    print(f"gradient_accumulation_steps={grad_accum}")
    print(f"precision={args.precision}")

    out_dir = Path(args.output_dir)
    merged_dir = Path(args.merged_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged_dir.mkdir(parents=True, exist_ok=True)

    print("\nLoading tokenizer...")
    tokenizer = _safe_tokenizer_from_pretrained(args.base_model, trust_remote_code=True)
    _maybe_set_padding(tokenizer)

    print("Loading dataset...")
    ds_train = load_dataset(args.dataset_name, split=args.train_split)
    ds_eval = load_dataset(args.dataset_name, split=args.eval_split)

    required_cols = {"sql_context", "sql_prompt", "sql"}
    for name, ds in [("train", ds_train), ("eval", ds_eval)]:
        missing = required_cols - set(ds.column_names)
        if missing:
            raise KeyError(f"{args.dataset_name}:{name} missing columns: {sorted(missing)}; found={ds.column_names}")

    if args.max_train_samples and args.max_train_samples > 0:
        ds_train = ds_train.select(range(min(args.max_train_samples, len(ds_train))))

    if args.max_eval_samples and args.max_eval_samples > 0:
        ds_eval = ds_eval.select(range(min(args.max_eval_samples, len(ds_eval))))

    print("Tokenizing + masking loss (assistant-only)...")
    map_kwargs_train = dict(
        batched=True,
        num_proc=max(1, int(args.preprocess_num_proc)),
        remove_columns=list(ds_train.column_names),
        desc="tokenize(train)",
    )
    map_kwargs_eval = dict(
        batched=True,
        num_proc=max(1, int(args.preprocess_num_proc)),
        remove_columns=list(ds_eval.column_names),
        desc="tokenize(eval)",
    )

    ds_train_tok = ds_train.map(
        lambda b: _tokenize_and_mask_batch(b, tokenizer=tokenizer, max_seq_len=max_seq_len),
        **map_kwargs_train,
    )
    ds_eval_tok = ds_eval.map(
        lambda b: _tokenize_and_mask_batch(b, tokenizer=tokenizer, max_seq_len=max_seq_len),
        **map_kwargs_eval,
    )

    # Filter out degenerate rows where truncation ate the entire completion.
    def _keep_row(ex: Dict[str, Any]) -> bool:
        return int(ex["input_len"]) > int(ex["prompt_len"]) + 1

    ds_train_tok = ds_train_tok.filter(_keep_row, desc="filter")
    ds_eval_tok = ds_eval_tok.filter(_keep_row, desc="filter")

    # Drop helper columns.
    ds_train_tok = ds_train_tok.remove_columns(["prompt_len", "input_len"])
    ds_eval_tok = ds_eval_tok.remove_columns(["prompt_len", "input_len"])

    torch_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    attn_impl = _pick_attn_impl(args.attn_impl)

    print("\nLoading base model...")
    model = _load_base_model(
        args.base_model,
        torch_dtype=torch_dtype,
        attn_impl=attn_impl,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    # LoRA injection
    target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    optim_name = _pick_optim(args.prefer_fused_optim)

    # TrainingArguments: handle newer/older naming drift.
    ta_kwargs: Dict[str, Any] = dict(
        output_dir=str(out_dir),
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        bf16=(args.precision == "bf16"),
        fp16=(args.precision == "fp16"),
        tf32=True,
        optim=optim_name,
        lr_scheduler_type="cosine",
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=args.gradient_checkpointing,
        ddp_find_unused_parameters=False,
        evaluation_strategy="steps",
        do_eval=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    try:
        training_args = TrainingArguments(**ta_kwargs)
    except TypeError:
        # Newer transformers: eval_strategy instead of evaluation_strategy.
        ta_kwargs["eval_strategy"] = ta_kwargs.pop("evaluation_strategy")
        training_args = TrainingArguments(**ta_kwargs)

    data_collator = DataCollatorForCausalLMWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train_tok,
        eval_dataset=ds_eval_tok,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("\nStarting training...")
    trainer.train()

    print("\nSaving LoRA adapter + tokenizer...")
    trainer.model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    # Pick the checkpoint we want to merge.
    adapter_path = out_dir
    if getattr(trainer.state, "best_model_checkpoint", None):
        adapter_path = Path(trainer.state.best_model_checkpoint)

    print("\nMerging LoRA into base model (full precision) for quantization...")
    # Load base on CPU for a portable merge step.
    base_for_merge = _load_base_model(
        args.base_model,
        torch_dtype=torch_dtype,
        attn_impl="sdpa",
        trust_remote_code=True,
    )
    base_for_merge.config.use_cache = False

    merged = PeftModel.from_pretrained(base_for_merge, str(adapter_path))
    merged = merged.merge_and_unload()

    print(f"Saving merged model to: {merged_dir}")
    merged.save_pretrained(str(merged_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_dir))

    # Breadcrumb for reproducibility.
    (merged_dir / "training_meta.txt").write_text(
        "\n".join(
            [
                f"base_model={args.base_model}",
                f"dataset={args.dataset_name}",
                f"max_seq_len={max_seq_len}",
                f"batch={per_device_train_batch_size}",
                f"grad_accum={grad_accum}",
                f"dtype={args.precision}",
                f"attn_impl={attn_impl}",
                f"lora_r={args.lora_r}",
                f"lora_alpha={args.lora_alpha}",
                f"lora_dropout={args.lora_dropout}",
                f"lora_target_modules={target_modules}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print("\n✅ Done.")
    print(f"✅ LoRA adapters: {out_dir.resolve()}")
    print(f"✅ Merged model (for quantization): {merged_dir.resolve()}\n")


if __name__ == "__main__":
    main()
