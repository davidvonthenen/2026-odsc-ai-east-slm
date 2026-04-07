#!/usr/bin/env python3
# Filename: infer.py

import argparse
import time
import warnings

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def select_device(no_cuda: bool) -> torch.device:
    """Prefer MPS on Apple Silicon, then CUDA, then CPU (unless --no_cuda)."""
    if no_cuda:
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def manual_fill_mask(model, tokenizer, text: str, device: torch.device, top_k: int = 5):
    """Minimal fill-mask implementation that works on CPU/CUDA/MPS."""
    if tokenizer.mask_token not in text:
        raise ValueError(f"Input text must contain the mask token {tokenizer.mask_token!r}")

    inputs = tokenizer(text, return_tensors="pt")
    inputs = move_batch_to_device(inputs, device)

    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits  # (batch, seq_len, vocab)

    mask_positions = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=False)
    # mask_positions: (num_masks, 2) where each row is [batch_idx, seq_pos]

    all_results = []
    for b_idx, seq_pos in mask_positions.tolist():
        mask_logits = logits[b_idx, seq_pos]  # (vocab,)
        probs = torch.softmax(mask_logits, dim=-1)
        top = torch.topk(probs, k=top_k)
        # Build a pipeline-like list of dicts
        results = []
        for score, tok_id in zip(top.values.tolist(), top.indices.tolist()):
            results.append(
                {
                    "token": tok_id,
                    "token_str": tokenizer.decode([tok_id]).strip(),
                    "score": float(score),
                }
            )
        all_results.append(results)

    # If there's exactly one mask, return that list directly for convenience
    return all_results[0] if len(all_results) == 1 else all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=str,
        default="trial_lr0p0003_bs32_wd0p0_dw0p5_cw0p1_ga1_BEST",
        help="Path to trained student model (directory with config.json)",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="The capital of France is [MASK].",
        help="Input text with [MASK] token for fill-mask demo",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Run on CPU even if GPU is available")
    parser.add_argument("--quantize", action="store_true", help="Apply dynamic quantization (CPU only)")
    parser.add_argument("--top_k", type=int, default=5, help="Top-k predictions to display")
    args = parser.parse_args()

    device = select_device(args.no_cuda)

    # Dynamic quantization is CPU-only.
    if args.quantize and device.type != "cpu":
        print(">> --quantize requested: forcing CPU (dynamic quantization is CPU-only).")
        device = torch.device("cpu")

    if device.type == "mps":
        print(">> Using Apple-Silicon MPS")
    elif device.type == "cuda":
        print(">> Using CUDA GPU")
    else:
        print(">> Using CPU")

    print(f"Loading model from {args.model_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(args.model_dir)
    model.eval()

    if args.quantize:
        print(">> Applying dynamic quantization (int8 Linear layers) ...")
        model = torch.quantization.quantize_dynamic(
            model.cpu(),  # quantize on CPU
            {torch.nn.Linear},
            dtype=torch.qint8,
        )
        model.eval()
    else:
        model.to(device)

    print("Model and tokenizer loaded.")

    # 1) Fill-mask demo
    print(f"\nInput: {args.text}")

    # Use HF pipeline on CPU/CUDA, manual on MPS (pipeline device semantics vary by version).
    if device.type in ("cpu", "cuda"):
        pipe_device = 0 if device.type == "cuda" else -1
        nlp = pipeline("fill-mask", model=model, tokenizer=tokenizer, device=pipe_device)
        results = nlp(args.text, top_k=args.top_k)
    else:
        results = manual_fill_mask(model, tokenizer, args.text, device=device, top_k=args.top_k)

    print("Top predictions for [MASK]:")
    for res in results:
        print(f"  {res['token_str']:<15} -> score {res['score']:.4f}")

    # Ensure model is still on the intended device before manual inference
    if not args.quantize:
        model.to(device)

    # 2) Manual inference example
    raw_text = "Distilled models are the [MASK] of large models."
    inputs = tokenizer(raw_text, return_tensors="pt")
    inputs = move_batch_to_device(inputs, device)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits  # (1, seq_len, vocab)
    mask_positions = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=False)

    print(f"\nInput: {raw_text}")
    if mask_positions.numel() == 0:
        print("No [MASK] token found in manual example text.")
        return

    # For the common case of one mask:
    b_idx, seq_pos = mask_positions[0].tolist()
    mask_logits = logits[b_idx, seq_pos]  # (vocab,)
    top = torch.topk(mask_logits, k=args.top_k)

    print(f"Top {args.top_k} predictions for [MASK] (logits):")
    for tok_id, logit in zip(top.indices.tolist(), top.values.tolist()):
        tok_str = tokenizer.decode([tok_id]).strip()
        print(f"  {tok_str:<15} -> {logit:.4f} logit")

    # Optional: quick quantization benchmark if requested
    if args.quantize:
        test_sentence = "This model is distilled and quantized for efficient inference."
        enc = tokenizer(test_sentence, return_tensors="pt")
        t0 = time.time()
        for _ in range(100):
            _ = model(**enc)
        t1 = time.time()
        print(f"\nQuantized model inference time (100 runs, CPU): {t1 - t0:.4f}s")


if __name__ == "__main__":
    main()
