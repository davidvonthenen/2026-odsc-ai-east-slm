import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tiktoken
import argparse
from dataclasses import dataclass
from datasets import load_dataset
from tqdm import tqdm
from contextlib import nullcontext

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
@dataclass
class GPTConfig:
    block_size: int = 128
    vocab_size: int = 50257  # GPT-2 vocab size
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    bias: bool = True

class TrainConfig:
    learning_rate: float = 1e-4
    max_iters: int = 20000
    warmup_steps: int = 1000
    min_lr: float = 5e-4
    eval_iters: int = 500
    batch_size: int = 32
    gradient_accumulation_steps: int = 32
    # Check for BF16 support
    dtype: str = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'

# Global Device Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[TrainConfig.dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# -----------------------------------------------------------------------------
# Model Architecture
# -----------------------------------------------------------------------------
class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.flash = hasattr(F, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                       .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.flash:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, 
                                               dropout_p=self.attn_dropout.p if self.training else 0.0, 
                                               is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = LayerNorm(config.n_embd, config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln2 = LayerNorm(config.n_embd, config.bias)
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
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
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

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# -----------------------------------------------------------------------------
# Data Processing
# -----------------------------------------------------------------------------
def prepare_data():
    """Downloads TinyStories and tokenizes it into .bin files."""
    print("Downloading dataset...")
    ds = load_dataset("roneneldan/TinyStories")
    enc = tiktoken.get_encoding("gpt2")

    def process(example):
        ids = enc.encode_ordinary(example['text'])
        out = {'ids': ids, 'len': len(ids)}
        return out

    print("Tokenizing dataset...")
    if not os.path.exists("train.bin"):
        tokenized = ds.map(
            process,
            remove_columns=['text'],
            desc="Tokenizing",
            num_proc=8,
        )

        for split, dset in tokenized.items():
            if split not in ['train', 'validation']: continue # Safety check
            
            arr_len = np.sum(dset['len'], dtype=np.uint64)
            filename = f'{split}.bin'
            dtype = np.uint16
            
            print(f"Writing {filename}...")
            arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
            
            # Shard parameters
            total_batches = 1024
            idx = 0
            
            for batch_idx in tqdm(range(total_batches), desc=f'Writing {filename}'):
                # Batch samples for faster write
                batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
                arr_batch = np.concatenate(batch['ids'])
                arr[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            arr.flush()
    print("Data preparation complete.")

def get_batch(split, block_size, batch_size):
    """Fetches a random batch of data from the .bin files."""
    filename = 'train.bin' if split == 'train' else 'validation.bin'
    if not os.path.exists(filename):
        raise FileNotFoundError(f"{filename} not found. Please run with --prepare_data first.")
        
    data = np.memmap(filename, dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

def estimate_loss(model, eval_iters, block_size, batch_size):
    out = {}
    model.eval()
    with torch.no_grad():
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch('train' if split=='train' else 'validation', block_size, batch_size)
                with ctx:
                    logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
    model.train()
    return out

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
def train():
    print(f"Initializing training on device: {device}")
    
    # Initialize Model
    config = GPTConfig()
    model = GPT(config)
    model = model.to(device)
    
    # Optimizer and Schedulers
    optimizer = torch.optim.AdamW(model.parameters(), lr=TrainConfig.learning_rate, betas=(0.9, 0.95), weight_decay=0.1, eps=1e-9)
    scheduler_warmup = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=TrainConfig.warmup_steps)
    scheduler_decay = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TrainConfig.max_iters - TrainConfig.warmup_steps, eta_min=TrainConfig.min_lr)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_decay], milestones=[TrainConfig.warmup_steps])
    
    scaler = torch.amp.GradScaler('cuda', enabled=(TrainConfig.dtype == 'float16'))
    
    best_val_loss = float('inf')
    best_model_path = "best_model_params.pt"

    print("Starting training loop...")
    for iter in tqdm(range(TrainConfig.max_iters)):
        
        # Evaluation phase
        if iter % TrainConfig.eval_iters == 0 and iter != 0:
            losses = estimate_loss(model, TrainConfig.eval_iters, GPTConfig.block_size, TrainConfig.batch_size)
            print(f"\nStep {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                torch.save(model.state_dict(), best_model_path)
                print(f"Saved new best model with val loss: {best_val_loss:.4f}")

        # Training phase
        X, y = get_batch('train', GPTConfig.block_size, TrainConfig.batch_size)
        
        with ctx:
            logits, loss = model(X, y)
            loss = loss / TrainConfig.gradient_accumulation_steps
        
        scaler.scale(loss).backward()
        
        if (iter + 1) % TrainConfig.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

def generate(prompt="Once upon a time"):
    print(f"Loading model for generation on {device}...")
    model = GPT(GPTConfig())
    model.load_state_dict(torch.load("best_model_params.pt", map_location=device))
    model.to(device)
    model.eval()
    
    enc = tiktoken.get_encoding("gpt2")
    context = torch.tensor(enc.encode_ordinary(prompt)).unsqueeze(0).to(device)
    
    print(f"Generating from prompt: '{prompt}'")
    y = model.generate(context, max_new_tokens=200)
    print("--- Generated Text ---")
    print(enc.decode(y.squeeze().tolist()))
    print("----------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vizuara AI Labs SLM App")
    parser.add_argument('--prepare', action='store_true', help='Download and tokenize data')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--generate', type=str, nargs='?', const="Once upon a time", help='Generate text with prompt')
    
    args = parser.parse_args()
    
    if args.prepare:
        prepare_data()
    elif args.train:
        train()
    elif args.generate:
        generate(args.generate)
    else:
        print("Please specify an action: --prepare, --train, or --generate [prompt]")