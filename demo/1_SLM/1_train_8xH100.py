# torchrun --nproc_per_node=8 2_train_8xH100.py

import os
import math
import time
import inspect
import itertools
import json
import zlib
from dataclasses import dataclass, asdict
from contextlib import nullcontext
from typing import Optional, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import IterableDataset, DataLoader

# External Libraries
import tiktoken
from datasets import load_dataset
import numpy as np

# -----------------------------------------------------------------------------
# 1. Distributed & Hardware Setup
# -----------------------------------------------------------------------------
def ddp_setup():
    """
    Initializes the NCCL backend for distributed training if available.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        
        device = torch.device(f"cuda:{ddp_local_rank}")
        torch.cuda.set_device(device)
        
        dist.init_process_group(backend="nccl", device_id=device)
        
        is_master_process = (ddp_rank == 0)
        is_distributed = True
    else:
        print("WARNING: 'RANK' not found. Running in Single-GPU (Debug) mode.")
        print("To use all 8 GPUs, run: torchrun --nproc_per_node=8 2_train_8xH100.py")
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_master_process = True
        is_distributed = False
        
    return device, ddp_rank, ddp_local_rank, ddp_world_size, is_master_process, is_distributed

# -----------------------------------------------------------------------------
# 2. Configuration & Hyperparameter Grid
# -----------------------------------------------------------------------------
@dataclass
class GPTConfig:
    block_size: int = 2048
    vocab_size: int = 50304       
    n_layer: int = 24
    n_head: int = 32
    n_embd: int = 2048
    dropout: float = 0.0
    bias: bool = False

DATASET_NAME = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG = "sample-10BT"

def get_hyperparameter_grid() -> List[Dict]:
    """
    Generates hyperparameter combinations.
    """
    learning_rates = [3e-4, 6e-4]
    
    # Adjusted for H100 memory limits
    batch_sizes = [4, 8] 
    
    weight_decays = [0.1, 0.01]
    
    grid = list(itertools.product(learning_rates, batch_sizes, weight_decays))
    
    configs = []
    for lr, bs, wd in grid:
        # If BS is 4, we accum 8 steps (Total effective = 4 * 8 * 8GPUs = 256)
        # If BS is 8, we accum 4 steps (Total effective = 8 * 4 * 8GPUs = 256)
        accum_steps = 8 if bs == 4 else 4
        
        configs.append({
            "learning_rate": lr,
            "batch_size": bs,
            "weight_decay": wd,
            "warmup_steps": 500,  
            "max_iters": 2000,
            "gradient_accumulation_steps": accum_steps
        })
    return configs

# -----------------------------------------------------------------------------
# 3. Model Architecture
# -----------------------------------------------------------------------------
class LayerNorm(nn.Module):
    def __init__(self, ndim, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))

    def forward(self, x):
        input_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + 1e-5)
        return (self.weight * x).to(input_dtype)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # WEIGHT TYING
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            return logits, loss
        else:
            logits = self.lm_head(x[:, [-1], :])
            return logits, None

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        use_fused = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        extra_args = dict(fused=True) if use_fused else dict()
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), **extra_args)

# -----------------------------------------------------------------------------
# 4. Data Loading
# -----------------------------------------------------------------------------
class StreamingTokenDataset(IterableDataset):
    def __init__(self, hf_dataset, tokenizer, block_size, rank, world_size):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.rank = rank
        self.world_size = world_size
        self.eot_token = tokenizer.encode_single_token("<|endoftext|>")

    def __iter__(self):
        iterator = iter(self.dataset)
        buffer = []
        for i, sample in enumerate(iterator):
            if i % self.world_size != self.rank:
                continue
            
            text = sample.get('text', '')
            tokens = self.tokenizer.encode_ordinary(text)
            tokens.append(self.eot_token)
            buffer.extend(tokens)
            
            while len(buffer) >= self.block_size + 1:
                x = torch.tensor(buffer[:self.block_size], dtype=torch.long)
                y = torch.tensor(buffer[1:self.block_size+1], dtype=torch.long)
                buffer = buffer[self.block_size:] 
                yield x, y

def get_dataloader(batch_size, split, rank, world_size):
    enc = tiktoken.get_encoding("gpt2")
    ds = load_dataset(DATASET_NAME, name=DATASET_CONFIG, split='train', streaming=True)
    
    def filter_split(sample):
        h = zlib.adler32(sample['text'].encode('utf-8'))
        is_validation = (h % 100) >= 98
        if split == 'train':
            return not is_validation
        elif split == 'validation':
            return is_validation
        return True

    ds = ds.filter(filter_split)
    if split == 'train':
        ds = ds.shuffle(seed=42, buffer_size=10000)
        
    token_ds = StreamingTokenDataset(ds, enc, GPTConfig.block_size, rank, world_size)
    return DataLoader(token_ds, batch_size=batch_size, pin_memory=True, num_workers=0)

# -----------------------------------------------------------------------------
# 5. Benchmarking
# -----------------------------------------------------------------------------
@torch.no_grad()
def estimate_loss(model, dataloader, eval_iters, device):
    model.eval()
    losses = []
    iter_dl = iter(dataloader)
    for k in range(eval_iters):
        try:
            X, Y = next(iter_dl)
        except StopIteration:
            break
        X, Y = X.to(device), Y.to(device)
        _, loss = model(X, Y)
        losses.append(loss.item())
    model.train()
    if not losses: return 0.0
    return torch.tensor(losses).mean().item()

# -----------------------------------------------------------------------------
# 6. Main Training Loop
# -----------------------------------------------------------------------------
def main():
    torch.set_float32_matmul_precision('high')
    
    device, rank, local_rank, world_size, master_process, is_distributed = ddp_setup()
    
    grid_configs = get_hyperparameter_grid()
    
    if master_process:
        print(f"Starting Grid Search.")
        if is_distributed:
            print(f"Distributed Mode: Enabled (World Size: {world_size})")
        print(f"Dataset: {DATASET_NAME} ({DATASET_CONFIG})")
        print(f"Total Configurations to test: {len(grid_configs)}\n")
        results = []

    for run_id, hp in enumerate(grid_configs):
        if is_distributed:
            dist.barrier()
        
        # Explicitly empty cache before every run
        torch.cuda.empty_cache()
        
        if master_process:
            print(f"--- [Run {run_id+1}/{len(grid_configs)}] Starting Config: LR={hp['learning_rate']}, BS={hp['batch_size']}, WD={hp['weight_decay']} ---")

        config = GPTConfig()
        model = GPT(config)
        model.to(device)
        
        # Compile improves speed
        model = torch.compile(model) 
        
        if is_distributed:
            model = DDP(model, device_ids=[local_rank])
            raw_model = model.module
        else:
            raw_model = model
        
        optimizer = raw_model.configure_optimizers(
            weight_decay=hp['weight_decay'], 
            learning_rate=hp['learning_rate'], 
            device_type='cuda'
        )
        
        train_loader = get_dataloader(hp['batch_size'], 'train', rank, world_size)
        val_loader = get_dataloader(hp['batch_size'], 'validation', rank, world_size)
        
        # Scheduler
        def get_lr(it):
            if it < hp['warmup_steps']:
                return hp['learning_rate'] * it / hp['warmup_steps']
            if it > hp['max_iters']:
                return hp['learning_rate'] * 0.1 
            decay_ratio = (it - hp['warmup_steps']) / (hp['max_iters'] - hp['warmup_steps'])
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            return (hp['learning_rate'] * 0.1) + coeff * (hp['learning_rate'] - (hp['learning_rate'] * 0.1))

        ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
        
        model.train()
        iter_data = iter(train_loader)
        final_val_loss = 0.0

        for step in range(hp['max_iters']):
            t0 = time.time()
            
            lr = get_lr(step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                
            loss_accum = 0.0
            for micro_step in range(hp['gradient_accumulation_steps']):
                try:
                    X, Y = next(iter_data)
                except StopIteration:
                    iter_data = iter(train_loader)
                    X, Y = next(iter_data)
                
                X, Y = X.to(device), Y.to(device)
                
                if is_distributed:
                    sync_context = model.no_sync() if micro_step < hp['gradient_accumulation_steps'] - 1 else nullcontext()
                else:
                    sync_context = nullcontext()

                with sync_context:
                    with ctx:
                        _, loss = model(X, Y)
                        loss = loss / hp['gradient_accumulation_steps']
                    
                    loss.backward()
                    loss_accum += loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            if master_process and step % 20 == 0:
                dt = time.time() - t0
                tokens_per_step = hp['batch_size'] * hp['gradient_accumulation_steps'] * GPTConfig.block_size * world_size
                tok_sec = tokens_per_step / (dt + 1e-6)
                print(f"Run {run_id+1} | Step {step} | Loss: {loss_accum:.4f} | LR: {lr:.2e} | {tok_sec:.0f} tok/s")

        # Final Evaluation
        val_loss = estimate_loss(raw_model, val_loader, eval_iters=50, device=device)
        final_val_loss = val_loss
        
        if master_process:
            print(f"--> Run {run_id+1} Finished. Final Val Loss: {final_val_loss:.4f}")
            
            # --- UPDATED: Torch Save with Hyperparams in Filename ---
            ckpt_name = f"model_run_{run_id+1}_lr{hp['learning_rate']}_bs{hp['batch_size']}_wd{hp['weight_decay']}.pt"
            print(f"Saving {ckpt_name}...")
            
            state_dict = raw_model.state_dict()
            # Strip "_orig_mod." prefix added by torch.compile for cleaner loading later
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("_orig_mod."):
                    new_state_dict[k[10:]] = v
                else:
                    new_state_dict[k] = v
            
            torch.save(new_state_dict, ckpt_name)
            
            results.append({
                "run_id": run_id + 1,
                "config": hp,
                "val_loss": final_val_loss,
                "perplexity": math.exp(final_val_loss),
                "checkpoint": ckpt_name
            })
            
        del model, optimizer, train_loader, val_loader
        torch.cuda.empty_cache()

    if is_distributed:
        dist.barrier()
        
    if master_process:
        print("\n================ GRID SEARCH RESULTS ================")
        # Sort by Validation Loss (Low to High)
        results.sort(key=lambda x: x['val_loss'])
        
        for res in results:
            print(f"Run {res['run_id']} | Loss: {res['val_loss']:.4f} | PPL: {res['perplexity']:.2f} | Params: {res['config']}")
        
        # Identify Best Model
        best_run = results[0]
        print(f"\nBest Model Predicted: Run {best_run['run_id']} (Loss: {best_run['val_loss']:.4f})")
        print(f"Best Checkpoint: {best_run['checkpoint']}")

        # Save Grid Search Data
        with open("grid_search_results.json", "w") as f:
            json.dump(results, f, indent=4)
            
        # --- NEW: Save Best Model Info to specific file ---
        with open("best_model_info.txt", "w") as f:
            f.write(f"Best Run ID: {best_run['run_id']}\n")
            f.write(f"Checkpoint Filename: {best_run['checkpoint']}\n")
            f.write(f"Validation Loss: {best_run['val_loss']:.4f}\n")
            f.write(f"Perplexity: {best_run['perplexity']:.2f}\n")
            f.write(f"Config: {json.dumps(best_run['config'])}\n")
        print("Best model information saved to 'best_model_info.txt'.")

    if is_distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()