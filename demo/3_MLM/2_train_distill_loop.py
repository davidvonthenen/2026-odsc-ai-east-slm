# Filename: train_distill_loop.py
"""
train_distill_loop.py

Raw-PyTorch (no HF Trainer) teacher→student distillation loop for Masked Language Modeling.

Goal:
  - Distill a 6-layer BERT-style student (~66M params) from a 12-layer BERT-base teacher
    ("bert-base-uncased") using a DistilBERT-style multi-loss objective:

    total_loss =
        mlm_weight     * MLM_CE(student_logits, labels)
      + distill_weight * KL( softmax(teacher/T) || softmax(student/T) ) * T^2
      + cosine_weight  * (1 - cosine_similarity(student_hidden, teacher_hidden))

Key behaviors:
  - Teacher is frozen (requires_grad=False) and always in eval() mode.
  - Teacher forward is inside torch.no_grad().
  - Student is initialized from teacher:
      * copy embeddings
      * copy every other layer (0,2,4,6,8,10 for 12→6)
      * copy MLM head transform + bias
      * student.tie_weights()

Outputs:
  outputs/distill_loop_grid/
    run_config.json
    results.json
    results_summary.csv
    trial_.../
      hparams.json
      metrics.json
      error.json (if failed)
      config.json (from save_pretrained)
      pytorch_model.bin (if safe_serialization=False) or model.safetensors
      tokenizer files
    best_model/ (copy of best trial directory)
"""

from __future__ import annotations

import os
import json
import math
import time
import shutil
import random
import itertools
import traceback
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoConfig,
    DataCollatorForLanguageModeling,
    set_seed,
    get_linear_schedule_with_warmup,
)

# =========================
# User-editable config
# =========================

# Dataset / output paths (override via env vars if desired)
# DATA_DIR: str = os.environ.get("DISTILL_DATA_DIR", "data/wikitext-512")
DATA_DIR: str = os.environ.get("DISTILL_DATA_DIR", "data/processed")
OUTPUT_ROOT: str = os.environ.get("DISTILL_OUTPUT_DIR", "outputs/distill_loop_grid")

# Models
TEACHER_NAME: str = "bert-base-uncased"
STUDENT_NUM_LAYERS: int = 6  # 12 -> 6 (every other layer)

# Training budget per trial (optimizer steps)
MAX_UPDATE_STEPS_PER_TRIAL: int = 500
LOG_EVERY_UPDATES: int = 50
EVAL_MAX_BATCHES: int = 200  # cap eval batches for speed; set None to eval full val set

# Loss / optimization
TEMPERATURE: float = 2.0
MLM_PROBABILITY: float = 0.15
MAX_GRAD_NORM: float = 1.0
WARMUP_RATIO: float = 0.06

# AdamW defaults (tuned for stability with fp16)
ADAM_BETAS: Tuple[float, float] = (0.9, 0.999)
ADAM_EPS: float = 1e-6

# Determinism / reproducibility
GLOBAL_SEED: int = 42
EVAL_SEED: int = 12345  # used to make eval masking deterministic across trials
NUM_WORKERS: int = 0
PIN_MEMORY: bool = True

# Mixed precision (AMP)
USE_AMP: bool = True  # enabled when CUDA is available; ignored on CPU
AMP_DTYPE: torch.dtype = torch.float16  # fp16 AMP (bf16 is also possible on newer GPUs)

# Optional: gradient checkpointing to reduce VRAM (slower). Leave False for simplicity.
ENABLE_GRAD_CHECKPOINTING: bool = False

# Save format: False => pytorch_model.bin (matches your outlined expectation)
SAVE_SAFE_TENSORS: bool = False

# =========================
# Hyperparameter grid
# =========================
# - weight_decay: common regularization knob; include a couple plausible values.
# - distill_weight: relative weight between KD and MLM is important; include a small set.
# - mlm_weight: keep fixed at 1.0; we vary distill_weight to change the ratio.
# - cosine_weight: optional, can help, but costs memory; include 0.0 and a small value.
# - grad_accum_steps: largely redundant with batch_size (it mainly changes effective batch size),
#   so we keep it fixed here to avoid exploding the grid.
param_grid: Dict[str, List[Any]] = {
    "learning_rate": [0.001, 0.0005, 0.0003],
    "batch_size":    [8, 16, 32],
    "weight_decay":     [0.0, 0.01],
    "distill_weight":   [0.5, 1.0],
    "mlm_weight":       [1.0],
    "cosine_weight":    [0.0, 0.1],
    "grad_accum_steps": [1],
}

# =========================
# Utilities
# =========================

def set_global_determinism(seed: int) -> None:
    """Best-effort determinism across Python, NumPy, PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)  # HF helper sets random/np/torch too, but we call explicitly anyway

    # cuDNN determinism flags: improves reproducibility, may reduce throughput.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def json_dump(obj: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def expand_grid(grid: Dict[str, List[Any]]) -> Iterable[Dict[str, Any]]:
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def format_trial_name(hparams: Dict[str, Any]) -> str:
    """Stable, filesystem-friendly trial name."""
    parts = [
        f"lr{hparams['learning_rate']}",
        f"bs{hparams['batch_size']}",
        f"wd{hparams['weight_decay']}",
        f"dw{hparams['distill_weight']}",
        f"cw{hparams['cosine_weight']}",
        f"ga{hparams['grad_accum_steps']}",
    ]
    # Replace '.' to avoid path issues
    return "trial_" + "_".join(parts).replace(".", "p")


def get_optimizer_grouped_parameters(model: nn.Module, weight_decay: float) -> List[Dict[str, Any]]:
    """AdamW param groups with no weight decay for bias and LayerNorm weights (standard Transformer practice)."""
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    decay_params: List[torch.nn.Parameter] = []
    nodecay_params: List[torch.nn.Parameter] = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(nd in n for nd in no_decay):
            nodecay_params.append(p)
        else:
            decay_params.append(p)

    return [
        {"params": decay_params, "weight_decay": float(weight_decay)},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]


def _cpu_clone_state_dict(module: nn.Module) -> Dict[str, torch.Tensor]:
    # Clone to CPU so the init snapshot stays CPU even after moving the teacher to GPU.
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


@dataclass(frozen=True)
class TeacherInitState:
    teacher_name: str
    teacher_num_layers: int
    student_num_layers: int
    layer_map: List[int]  # student layer i ← teacher layer layer_map[i]
    embeddings_sd: Dict[str, torch.Tensor]
    encoder_layers_sd: List[Dict[str, torch.Tensor]]  # aligned to student layers order
    mlm_transform_sd: Dict[str, torch.Tensor]
    mlm_bias: torch.Tensor


def make_layer_map(teacher_layers: int, student_layers: int) -> List[int]:
    """
    DistilBERT-style mapping.
    For 12->6: [0,2,4,6,8,10]
    """
    if teacher_layers == 12 and student_layers == 6:
        return [0, 2, 4, 6, 8, 10]

    # Generic fallback: evenly-spaced indices.
    return [int(i * teacher_layers / student_layers) for i in range(student_layers)]


def extract_teacher_init_state(teacher: nn.Module, student_layers: int) -> TeacherInitState:
    teacher_layers = int(teacher.config.num_hidden_layers)
    layer_map = make_layer_map(teacher_layers, student_layers)

    embeddings_sd = _cpu_clone_state_dict(teacher.base_model.embeddings)

    # Copy only the teacher layers we actually need (saves memory).
    encoder_layers_sd: List[Dict[str, torch.Tensor]] = []
    for t_idx in layer_map:
        encoder_layers_sd.append(_cpu_clone_state_dict(teacher.base_model.encoder.layer[t_idx]))

    # Copy MLM head transform + bias (decoder weight is tied to embeddings).
    mlm_transform_sd = _cpu_clone_state_dict(teacher.cls.predictions.transform)
    mlm_bias = teacher.cls.predictions.bias.detach().cpu().clone()

    return TeacherInitState(
        teacher_name=TEACHER_NAME,
        teacher_num_layers=teacher_layers,
        student_num_layers=student_layers,
        layer_map=layer_map,
        embeddings_sd=embeddings_sd,
        encoder_layers_sd=encoder_layers_sd,
        mlm_transform_sd=mlm_transform_sd,
        mlm_bias=mlm_bias,
    )


def build_student_from_teacher_init(init_state: TeacherInitState) -> nn.Module:
    """
    Build a BERT-like student and initialize from the teacher snapshot:
      - embeddings
      - encoder layers (DistilBERT-style mapping)
      - MLM head transform + bias
      - tie weights
    """
    student_config = AutoConfig.from_pretrained(init_state.teacher_name)
    student_config.num_hidden_layers = int(init_state.student_num_layers)
    student_config.tie_word_embeddings = True
    student_config.use_cache = False

    student = AutoModelForMaskedLM.from_config(student_config)
    student.config.use_cache = False

    # Embeddings
    student.base_model.embeddings.load_state_dict(init_state.embeddings_sd)

    # Encoder layers (already aligned to student order in init_state)
    for i in range(init_state.student_num_layers):
        student.base_model.encoder.layer[i].load_state_dict(init_state.encoder_layers_sd[i])

    # MLM head (transform + bias)
    student.cls.predictions.transform.load_state_dict(init_state.mlm_transform_sd)
    with torch.no_grad():
        student.cls.predictions.bias.copy_(init_state.mlm_bias)

    # Ensure word embeddings are tied (decoder weight ↔ input embeddings)
    student.tie_weights()
    return student


def _autocast_ctx(use_amp: bool, device: torch.device, amp_dtype: torch.dtype):
    from contextlib import nullcontext
    if not use_amp or device.type != "cuda":
        return nullcontext()
    return torch.cuda.amp.autocast(dtype=amp_dtype)


def compute_distillation_losses(
    student: nn.Module,
    teacher: nn.Module,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    temperature: float,
    mlm_weight: float,
    distill_weight: float,
    cosine_weight: float,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute the distillation objective for a batch.

    Implementation detail:
      - To avoid allocating [B,S,V] logits tensors, we compute logits ONLY on masked positions
        (labels != -100). This materially reduces VRAM for larger batches/seq_len.
    """
    input_ids = batch["input_ids"]
    attention_mask = batch.get("attention_mask", None)
    token_type_ids = batch.get("token_type_ids", None)
    labels = batch["labels"]

    mask = (labels != -100)  # [B,S] bool on device

    with _autocast_ctx(use_amp, device, amp_dtype):
        # Teacher forward (frozen, eval) with no gradients
        with torch.no_grad():
            t_base = teacher.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                return_dict=True,
            )
            t_last = t_base.last_hidden_state  # [B,S,H]

        # Student forward (trainable)
        s_base = student.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        s_last = s_base.last_hidden_state  # [B,S,H]

        # MLM + KD only on masked positions
        if mask.any():
            s_masked_logits = student.cls(s_last[mask]).float()  # [N,V]
            t_masked_logits = teacher.cls(t_last[mask]).float()  # [N,V]
            masked_labels = labels[mask]  # [N]

            mlm_loss = F.cross_entropy(s_masked_logits, masked_labels, reduction="mean")

            T = float(temperature)
            s_log_probs = F.log_softmax(s_masked_logits / T, dim=-1)
            t_probs = F.softmax(t_masked_logits / T, dim=-1)

            kd_loss = F.kl_div(s_log_probs, t_probs, reduction="batchmean") * (T * T)
        else:
            # Extremely rare with MLM collator, but handle cleanly.
            mlm_loss = torch.zeros((), device=device, dtype=torch.float32)
            kd_loss = torch.zeros((), device=device, dtype=torch.float32)

        # Optional cosine hidden-state loss (last hidden states)
        if float(cosine_weight) > 0.0:
            if attention_mask is not None:
                active = attention_mask.bool()
                if active.any():
                    cos_sim = F.cosine_similarity(s_last[active].float(), t_last[active].float(), dim=-1)
                    cosine_loss = (1.0 - cos_sim).mean()
                else:
                    cosine_loss = torch.zeros((), device=device, dtype=torch.float32)
            else:
                # No attention_mask: treat all tokens as active.
                cos_sim = F.cosine_similarity(s_last.float(), t_last.float(), dim=-1)  # [B,S]
                cosine_loss = (1.0 - cos_sim).mean()
        else:
            cosine_loss = torch.zeros((), device=device, dtype=torch.float32)

        total = (float(mlm_weight) * mlm_loss) + (float(distill_weight) * kd_loss) + (float(cosine_weight) * cosine_loss)

    comps = {
        "total_loss": total.detach(),
        "mlm_loss": mlm_loss.detach(),
        "kd_loss": kd_loss.detach(),
        "cosine_loss": cosine_loss.detach(),
        "masked_tokens": mask.sum().detach(),
    }
    return total, comps


@torch.no_grad()
def evaluate_mlm_loss(
    model: nn.Module,
    eval_loader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
    eval_seed: int = EVAL_SEED,
) -> Tuple[float, float]:
    """
    Evaluate MLM loss on validation:
      - Uses the collator-provided random masks/labels.
      - For comparability across trials, we set a fixed seed at the start so the mask pattern is deterministic.
      - Computes loss ONLY on masked positions to avoid allocating [B,S,V] logits.
    """
    # Make eval masking deterministic across trials/runs.
    set_global_determinism(eval_seed)

    model.eval()

    total_loss_sum = 0.0
    total_masked = 0

    for b_idx, batch in enumerate(eval_loader):
        if max_batches is not None and b_idx >= max_batches:
            break

        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        labels = batch["labels"]
        mask = (labels != -100)

        if not mask.any():
            continue

        base = model.base_model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask", None),
            token_type_ids=batch.get("token_type_ids", None),
            return_dict=True,
        )
        last = base.last_hidden_state  # [B,S,H]

        masked_logits = model.cls(last[mask]).float()  # [N,V]
        masked_labels = labels[mask]                  # [N]

        loss_sum = F.cross_entropy(masked_logits, masked_labels, reduction="sum").item()

        total_loss_sum += loss_sum
        total_masked += masked_labels.numel()

    if total_masked == 0:
        mean_loss = float("inf")
        ppl = float("inf")
    else:
        mean_loss = total_loss_sum / total_masked
        ppl = math.exp(mean_loss) if mean_loss < 20 else float("inf")

    model.train()
    return mean_loss, ppl


@dataclass
class TrialResult:
    trial_name: str
    hparams: Dict[str, Any]
    status: str  # "ok" | "oom" | "error"
    train_time_sec: float
    final_update_steps: int
    train_loss_mean: float
    train_mlm_loss_mean: float
    train_kd_loss_mean: float
    train_cosine_loss_mean: float
    eval_mlm_loss: Optional[float]
    eval_perplexity: Optional[float]
    peak_cuda_mem_mb: Optional[float]
    error_msg: Optional[str] = None


# =========================
# Save helpers
# =========================

def _weights_main_filename(save_safe_tensors: bool) -> str:
    # HF conventions
    return "model.safetensors" if save_safe_tensors else "pytorch_model.bin"


def _weights_alias_ext(save_safe_tensors: bool) -> str:
    return ".safetensors" if save_safe_tensors else ".bin"


def link_or_copy(src: str, dst: str) -> None:
    """
    Create a lightweight alias to the checkpoint file.
    Prefer hardlink (cheap, fast). Fall back to copy if hardlink fails.
    """
    ensure_dir(os.path.dirname(dst))
    if os.path.exists(dst):
        os.remove(dst)

    try:
        os.link(src, dst)  # hardlink on same filesystem
    except OSError:
        shutil.copy2(src, dst)


def write_results_txt(trial_dir: str, trial_name: str, hparams: Dict[str, Any], metrics: Dict[str, Any]) -> None:
    path = os.path.join(trial_dir, f"{trial_name}_results.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("Hyperparameters:\n")
        for k, v in hparams.items():
            f.write(f"  {k}: {v}\n")

        f.write("\nMetrics:\n")
        for k, v in metrics.items():
            f.write(f"  {k}: {v}\n")


# =========================
# One Trial
# =========================

def run_one_trial(
    trial_name: str,
    hparams: Dict[str, Any],
    teacher: nn.Module,
    teacher_init: TeacherInitState,
    tokenizer: Any,
    train_ds: Any,
    eval_ds: Optional[Any],
    device: torch.device,
) -> TrialResult:
    trial_dir = os.path.join(OUTPUT_ROOT, trial_name)
    ensure_dir(trial_dir)
    json_dump(hparams, os.path.join(trial_dir, "hparams.json"))

    # Reset RNG for cross-trial comparability.
    set_global_determinism(GLOBAL_SEED)

    # Build student fresh for this trial (CPU init snapshot → move to device).
    student = build_student_from_teacher_init(teacher_init)

    if ENABLE_GRAD_CHECKPOINTING:
        try:
            student.base_model.gradient_checkpointing_enable()
        except Exception:
            pass

    student.to(device)
    student.train()

    # Teacher is already frozen + eval + on device (managed in main()).

    # Data collator for dynamic MLM masking
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=MLM_PROBABILITY,
    )

    # Deterministic shuffle independent of other RNG consumers
    train_gen = torch.Generator()
    train_gen.manual_seed(GLOBAL_SEED)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(hparams["batch_size"]),
        shuffle=True,
        collate_fn=data_collator,
        num_workers=NUM_WORKERS,
        generator=train_gen,
        pin_memory=PIN_MEMORY and device.type == "cuda",
        drop_last=True,
    )

    eval_loader = None
    if eval_ds is not None:
        eval_loader = DataLoader(
            eval_ds,
            batch_size=int(hparams["batch_size"]),
            shuffle=False,
            collate_fn=data_collator,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY and device.type == "cuda",
            drop_last=False,
        )

    # Optimizer / scheduler
    lr = float(hparams["learning_rate"])
    wd = float(hparams["weight_decay"])
    optimizer = torch.optim.AdamW(
        get_optimizer_grouped_parameters(student, wd),
        lr=lr,
        betas=ADAM_BETAS,
        eps=ADAM_EPS,
    )

    grad_accum_steps = int(hparams["grad_accum_steps"])
    max_updates = int(MAX_UPDATE_STEPS_PER_TRIAL)
    warmup_steps = max(1, int(WARMUP_RATIO * max_updates))

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_updates,
    )

    # AMP scaler
    use_amp = bool(USE_AMP and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Loss weights
    mlm_weight = float(hparams["mlm_weight"])
    distill_weight = float(hparams["distill_weight"])
    cosine_weight = float(hparams["cosine_weight"])

    # Stats
    update_step = 0
    train_loss_sum = 0.0
    mlm_sum = 0.0
    kd_sum = 0.0
    cos_sum = 0.0
    masked_tok_sum = 0

    peak_mem_mb: Optional[float] = None

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    start_time = time.time()
    train_iter = iter(train_loader)

    try:
        while update_step < max_updates:
            optimizer.zero_grad(set_to_none=True)

            for _ in range(grad_accum_steps):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    batch = next(train_iter)

                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

                total_loss, comps = compute_distillation_losses(
                    student=student,
                    teacher=teacher,
                    batch=batch,
                    device=device,
                    temperature=TEMPERATURE,
                    mlm_weight=mlm_weight,
                    distill_weight=distill_weight,
                    cosine_weight=cosine_weight,
                    use_amp=use_amp,
                    amp_dtype=AMP_DTYPE,
                )

                loss = total_loss / grad_accum_steps

                if use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                train_loss_sum += comps["total_loss"].item()
                mlm_sum += comps["mlm_loss"].item()
                kd_sum += comps["kd_loss"].item()
                cos_sum += comps["cosine_loss"].item()
                masked_tok_sum += int(comps["masked_tokens"].item())

            if use_amp:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), MAX_GRAD_NORM)

            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            scheduler.step()
            update_step += 1

            if update_step % LOG_EVERY_UPDATES == 0 or update_step == 1:
                denom = max(1, update_step * grad_accum_steps)
                avg_masked = masked_tok_sum / denom
                print(
                    f"[{trial_name}] step {update_step:>4}/{max_updates} "
                    f"lr={lr:g} bs={hparams['batch_size']} "
                    f"loss={train_loss_sum/denom:.4f} "
                    f"mlm={mlm_sum/denom:.4f} kd={kd_sum/denom:.4f} cos={cos_sum/denom:.4f} "
                    f"masked_tokens/batch={avg_masked:.1f}"
                )

        train_time = time.time() - start_time

        if device.type == "cuda":
            peak_mem_mb = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))

        eval_loss = None
        eval_ppl = None
        if eval_loader is not None:
            eval_loss, eval_ppl = evaluate_mlm_loss(
                model=student,
                eval_loader=eval_loader,
                device=device,
                max_batches=EVAL_MAX_BATCHES,
                eval_seed=EVAL_SEED,
            )
            print(f"[{trial_name}] eval_mlm_loss={eval_loss:.4f} eval_ppl={eval_ppl:.2f}")

        # Save final model + tokenizer (HF-standard directory checkpoint)
        student.save_pretrained(trial_dir, safe_serialization=SAVE_SAFE_TENSORS)
        tokenizer.save_pretrained(trial_dir)

        # Also write a filename-encoded "checkpoint alias" like your classifier example.
        # This keeps HF reloadability (directory) AND gives you a single file to eyeball.
        main_weights = os.path.join(trial_dir, _weights_main_filename(SAVE_SAFE_TENSORS))
        alias_dir = os.path.join(OUTPUT_ROOT, "checkpoints")
        alias_path = os.path.join(alias_dir, f"{trial_name}{_weights_alias_ext(SAVE_SAFE_TENSORS)}")

        if os.path.exists(main_weights):
            link_or_copy(main_weights, alias_path)
            
        denom = max(1, update_step * grad_accum_steps)
        metrics = {
            "train_time_sec": train_time,
            "final_update_steps": update_step,
            "train_loss_mean": train_loss_sum / denom,
            "train_mlm_loss_mean": mlm_sum / denom,
            "train_kd_loss_mean": kd_sum / denom,
            "train_cosine_loss_mean": cos_sum / denom,
            "avg_masked_tokens_per_microbatch": masked_tok_sum / denom,
            "eval_mlm_loss": eval_loss,
            "eval_perplexity": eval_ppl,
            "peak_cuda_mem_mb": peak_mem_mb,
        }
        json_dump(metrics, os.path.join(trial_dir, "metrics.json"))

        # Human-readable rollup (nice for quick diffing without jq)
        write_results_txt(trial_dir, trial_name, hparams, metrics)

        return TrialResult(
            trial_name=trial_name,
            hparams=hparams,
            status="ok",
            train_time_sec=train_time,
            final_update_steps=update_step,
            train_loss_mean=metrics["train_loss_mean"],
            train_mlm_loss_mean=metrics["train_mlm_loss_mean"],
            train_kd_loss_mean=metrics["train_kd_loss_mean"],
            train_cosine_loss_mean=metrics["train_cosine_loss_mean"],
            eval_mlm_loss=eval_loss,
            eval_perplexity=eval_ppl,
            peak_cuda_mem_mb=peak_mem_mb,
            error_msg=None,
        )

    except RuntimeError as e:
        msg = str(e)
        status = "error"
        if device.type == "cuda" and "out of memory" in msg.lower():
            status = "oom"

        err_payload = {
            "status": status,
            "error": msg,
            "traceback": traceback.format_exc(),
        }
        json_dump(err_payload, os.path.join(trial_dir, "error.json"))
        print(f"[{trial_name}] FAILED ({status}): {msg}")

        # Best-effort cleanup
        try:
            del student
        except Exception:
            pass
        if device.type == "cuda":
            torch.cuda.empty_cache()

        return TrialResult(
            trial_name=trial_name,
            hparams=hparams,
            status=status,
            train_time_sec=time.time() - start_time,
            final_update_steps=update_step,
            train_loss_mean=float("nan"),
            train_mlm_loss_mean=float("nan"),
            train_kd_loss_mean=float("nan"),
            train_cosine_loss_mean=float("nan"),
            eval_mlm_loss=None,
            eval_perplexity=None,
            peak_cuda_mem_mb=None,
            error_msg=msg,
        )

    finally:
        # Free student between trials
        try:
            del student
        except Exception:
            pass
        if device.type == "cuda":
            torch.cuda.empty_cache()


def write_summary(results: List[TrialResult], out_path: str) -> None:
    import csv

    if not results:
        return

    rows: List[Dict[str, Any]] = []
    for r in results:
        row = {
            "trial_name": r.trial_name,
            "status": r.status,
            "train_time_sec": r.train_time_sec,
            "final_update_steps": r.final_update_steps,
            "train_loss_mean": r.train_loss_mean,
            "train_mlm_loss_mean": r.train_mlm_loss_mean,
            "train_kd_loss_mean": r.train_kd_loss_mean,
            "train_cosine_loss_mean": r.train_cosine_loss_mean,
            "eval_mlm_loss": r.eval_mlm_loss,
            "eval_perplexity": r.eval_perplexity,
            "peak_cuda_mem_mb": r.peak_cuda_mem_mb,
            "error_msg": r.error_msg,
        }
        row.update({f"hparam.{k}": v for k, v in r.hparams.items()})
        rows.append(row)

    fieldnames = list(rows[0].keys())
    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _check_dataset_seq_len(train_ds: Any, max_pos: int) -> None:
    # Try to detect obvious mismatch early (e.g., data prepped with seq_length=131072).
    ex = train_ds[0]
    ids = ex["input_ids"]
    seq_len = int(ids.shape[0]) if hasattr(ids, "shape") else len(ids)
    if seq_len > max_pos:
        raise ValueError(
            f"Dataset sequences are length {seq_len}, but model max_position_embeddings is {max_pos}. "
            f"Re-run prepare_data.py with --seq_length <= {max_pos} (typically 512 for bert-base-uncased)."
        )
    

# =========================
# Main
# =========================

def main() -> None:
    set_global_determinism(GLOBAL_SEED)
    ensure_dir(OUTPUT_ROOT)

    print("=== Distillation grid search (raw PyTorch loop) ===")
    print(f"DATA_DIR:          {DATA_DIR}")
    print(f"OUTPUT_ROOT:       {OUTPUT_ROOT}")
    print(f"TEACHER:           {TEACHER_NAME}")
    print(f"STUDENT_LAYERS:    {STUDENT_NUM_LAYERS}")
    print(f"MAX_UPDATES/TRIAL: {MAX_UPDATE_STEPS_PER_TRIAL}")

    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(
            f"Processed dataset not found at {DATA_DIR}. "
            "Run prepare_data.py first, or set DISTILL_DATA_DIR."
        )

    ds = load_from_disk(DATA_DIR)
    if "train" not in ds:
        raise ValueError(f"Dataset at {DATA_DIR} has no 'train' split.")

    train_ds = ds["train"]
    eval_ds = ds["validation"] if "validation" in ds else None

    # Torch-format for speed; keep only existing cols
    keep_cols = [c for c in ["input_ids", "attention_mask", "token_type_ids"] if c in train_ds.column_names]
    train_ds.set_format(type="torch", columns=keep_cols)
    if eval_ds is not None:
        eval_ds.set_format(type="torch", columns=keep_cols)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(TEACHER_NAME, use_fast=True)

    # Load teacher on CPU first (so we can snapshot init weights cheaply)
    teacher = AutoModelForMaskedLM.from_pretrained(TEACHER_NAME)
    teacher.config.use_cache = False
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    _check_dataset_seq_len(train_ds, int(teacher.config.max_position_embeddings))

    teacher_init = extract_teacher_init_state(teacher, STUDENT_NUM_LAYERS)

    # Move teacher to device once, keep there across all trials.
    teacher.to(device)
    teacher.eval()

    trials = list(expand_grid(param_grid))
    print(f"TOTAL_TRIALS:      {len(trials)}")

    run_config = {
        "data_dir": DATA_DIR,
        "output_root": OUTPUT_ROOT,
        "teacher_name": TEACHER_NAME,
        "student_num_layers": STUDENT_NUM_LAYERS,
        "max_update_steps_per_trial": MAX_UPDATE_STEPS_PER_TRIAL,
        "eval_max_batches": EVAL_MAX_BATCHES,
        "temperature": TEMPERATURE,
        "mlm_probability": MLM_PROBABILITY,
        "warmup_ratio": WARMUP_RATIO,
        "max_grad_norm": MAX_GRAD_NORM,
        "use_amp": USE_AMP,
        "amp_dtype": str(AMP_DTYPE),
        "save_safe_tensors": SAVE_SAFE_TENSORS,
        "global_seed": GLOBAL_SEED,
        "eval_seed": EVAL_SEED,
        "param_grid": param_grid,
        "layer_map": teacher_init.layer_map,
    }
    json_dump(run_config, os.path.join(OUTPUT_ROOT, "run_config.json"))

    results: List[TrialResult] = []
    best: Optional[TrialResult] = None

    for idx, hparams in enumerate(trials, start=1):
        trial_name = format_trial_name(hparams)
        print(f"\n--- Running trial {idx}/{len(trials)}: {trial_name} ---")

        res = run_one_trial(
            trial_name=trial_name,
            hparams=hparams,
            teacher=teacher,
            teacher_init=teacher_init,
            tokenizer=tokenizer,
            train_ds=train_ds,
            eval_ds=eval_ds,
            device=device,
        )
        results.append(res)

        # Track best by lowest eval MLM loss among successful trials.
        if res.status == "ok" and res.eval_mlm_loss is not None:
            if best is None or (res.eval_mlm_loss < best.eval_mlm_loss):
                best = res

        json_dump([asdict(r) for r in results], os.path.join(OUTPUT_ROOT, "results.json"))

    summary_csv = os.path.join(OUTPUT_ROOT, "results_summary.csv")
    write_summary(results, summary_csv)
    print(f"\nWrote summary CSV: {summary_csv}")

    if best is not None:
        print(f"\nBEST TRIAL: {best.trial_name}")
        print(f"  eval_mlm_loss: {best.eval_mlm_loss:.4f}")
        print(f"  eval_ppl:      {best.eval_perplexity:.2f}")
        print(f"  hparams:       {best.hparams}")

        # 1) Copy the entire best trial directory to a *_BEST directory
        best_src = os.path.join(OUTPUT_ROOT, best.trial_name)
        best_named_dst = os.path.join(OUTPUT_ROOT, f"{best.trial_name}_BEST")
        if os.path.exists(best_named_dst):
            shutil.rmtree(best_named_dst)
        shutil.copytree(best_src, best_named_dst)

        # 2) Also create a *_BEST single-file alias checkpoint in OUTPUT_ROOT/checkpoints
        best_ckpt_dir = os.path.join(OUTPUT_ROOT, "checkpoints")
        best_alias_src = os.path.join(best_ckpt_dir, f"{best.trial_name}{_weights_alias_ext(SAVE_SAFE_TENSORS)}")
        best_alias_dst = os.path.join(best_ckpt_dir, f"{best.trial_name}_BEST{_weights_alias_ext(SAVE_SAFE_TENSORS)}")

        if os.path.exists(best_alias_src):
            link_or_copy(best_alias_src, best_alias_dst)
    else:
        print("\nNo successful trials produced an eval loss; inspect results.json / per-trial error.json.")


if __name__ == "__main__":
    main()
