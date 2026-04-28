import os
import argparse
import multiprocessing
from datasets import load_dataset, Dataset
import tiktoken
import numpy as np
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# 1B models need ~20-50B tokens. We will use a 10B sample of FineWeb-Edu as a start.
# For a full run, remove the "sample-10BT" config and use the full dataset (requires ~500GB disk).
DATASET_NAME = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG = "sample-10BT" 

def get_tokenizer():
    # consistent with train_production_v2.py
    return tiktoken.get_encoding("gpt2")

def tokenize_chunk(example_batch):
    """
    Tokenizes a batch of text and returns a flat list of token IDs.
    We append the <|endoftext|> token to every document to delineate them.
    """
    enc = get_tokenizer()
    eot = enc.eot_token
    
    all_tokens = []
    for text in example_batch["text"]:
        # fast encoding via tiktoken
        tokens = enc.encode_ordinary(text)
        tokens.append(eot)
        all_tokens.extend(tokens)
        
    return {"input_ids": all_tokens}

def group_chunks(examples, block_size):
    """
    Concatenates all tokens and chunks them into exact block_size.
    This ensures we don't waste compute on padding.
    """
    # Concatenate all input_ids in this batch
    concatenated = sum(examples["input_ids"], [])
    
    total_length = len(concatenated)
    
    # We drop the small remainder at the end of the batch
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    
    # Split into chunks
    result = {
        "input_ids": [concatenated[i : i + block_size] for i in range(0, total_length, block_size)]
    }
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/fineweb_edu_1B", help="Location to save the processed dataset")
    parser.add_argument("--block_size", type=int, default=2049, help="Sequence length (2048 ctx + 1 target)")
    parser.add_argument("--num_proc", type=int, default=os.cpu_count(), help="Number of workers")
    args = parser.parse_args()

    print(f"--- Preparing Data for 1B Model ---")
    print(f"Dataset: {DATASET_NAME} ({DATASET_CONFIG})")
    print(f"Block Size: {args.block_size}")
    
    # 1. Load Dataset (Streaming mode first to peek, but we download for speed if disk allows)
    # For 1B training, downloading to disk is highly recommended over streaming to prevent network bottlenecks.
    print("Downloading dataset (this may take time)...")
    raw_dataset = load_dataset(DATASET_NAME, name=DATASET_CONFIG, split="train", num_proc=args.num_proc)
    
    # 2. Tokenize
    print("Tokenizing (tiktoken)...")
    # We remove 'text' and other metadata columns immediately to save RAM
    tokenized_dataset = raw_dataset.map(
        tokenize_chunk,
        batched=True,
        remove_columns=raw_dataset.column_names,
        num_proc=args.num_proc,
        desc="Tokenizing"
    )
    
    # 3. Pack (Group)
    # This transforms variable length docs into fixed size training blocks
    print(f"Packing into blocks of {args.block_size}...")
    packed_dataset = tokenized_dataset.map(
        lambda x: group_chunks(x, args.block_size),
        batched=True,
        num_proc=args.num_proc,
        desc="Packing"
    )
    
    # 4. Train/Val Split
    # We take 0.5% for validation (FineWeb is huge, so 0.5% is plenty)
    print("Splitting Train/Val...")
    split_dataset = packed_dataset.train_test_split(test_size=0.005, seed=42, shuffle=True)
    
    # 5. Save to Disk
    # This saves in Hugging Face arrow format, which is memory-mapped and incredibly fast to load
    print(f"Saving to {args.output_dir}...")
    split_dataset.save_to_disk(args.output_dir)
    
    print(f"\n--- SUCCESS ---")
    print(f"Train blocks: {len(split_dataset['train'])}")
    print(f"Val blocks:   {len(split_dataset['test'])}")
    print(f"Tokens total: {len(split_dataset['train']) * args.block_size / 1e9:.2f} Billion")
    print(f"To use in training, change dataset_name to '{args.output_dir}' and ensure 'streaming=False' in your loader.")

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()