# python inference.py --checkpoint model_run_3_lr0.0003_bs32.pt

# python inference.py \
#   --checkpoint model_run_3_lr0.0003_bs32.pt \
#   --prompt "The future of artificial intelligence is" \
#   --max_tokens 200 \
#   --temperature 0.9 \
#   --top_k 20

import argparse
from typing import Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import argparse

def select_device(device_str: Optional[str] = None) -> torch.device:
    if device_str:
        d = device_str.lower().strip()
        if d in ("cpu",):
            return torch.device("cpu")
        if d in ("cuda", "gpu"):
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if d in ("mps",):
            return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        return torch.device(device_str)

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# -----------------------------------------------------------------------------
# 1. Configuration (MUST MATCH TRAINING SCRIPT)
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

# -----------------------------------------------------------------------------
# 2. Model Architecture
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
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x[:, [-1], :]) # Only return logits for the last token
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Takes a conditioning sequence (idx) and completes it.
        """
        for _ in range(max_new_tokens):
            # If the sequence context is too long, we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            # Forward the model to get the logits for the index in the sequence
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

# -----------------------------------------------------------------------------
# 3. Main Execution
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="model_run_5_lr0.0006_bs4_wd0.1.pt", help="Path to .pt model file")
    parser.add_argument("--prompt", type=str, default="Once upon a time in Long Beach...", help="Input prompt")
    parser.add_argument("--max_tokens", type=int, default=100, help="How many tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Creativity (0.0 = greedy, 1.0 = random)")
    parser.add_argument("--top_k", type=int, default=50, help="Top-K sampling cutoff")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    args = parser.parse_args()

    # 1. Setup
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = select_device()
    print(f"Using device: {device}")

    # 2. Load Model
    config = GPTConfig()
    model = GPT(config)
    
    print(f"Loading weights from {args.checkpoint}...")
    try:
        # Load state dict via torch.load
        state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
        
        # Clean potential compiler prefixes (redundant if training script already cleaned them, but safe to keep)
        new_sd = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        
        model.load_state_dict(new_sd)
        model.to(device)
        model.eval() 
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 3. Tokenize Prompt
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(args.prompt)
    tokens_tensor = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    print("\n\n")
    print("-" * 50)
    print(f"\nPrompt: '{args.prompt}'")
    print("\n\n")

    # 4. Generate
    generated_indices = model.generate(
        tokens_tensor, 
        max_new_tokens=args.max_tokens, 
        temperature=args.temperature, 
        top_k=args.top_k
    )
    
    # 5. Decode
    decoded_output = enc.decode(generated_indices[0].tolist())
    print("Generated Output:")
    print(decoded_output)
    print("-" * 50)
    print("\n\n")

if __name__ == "__main__":
    main()