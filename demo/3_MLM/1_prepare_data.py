# Filename: prepare_data.py
import os
import math
import argparse
from datasets import load_dataset, DatasetDict

from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, choices=["wikipedia", "wikitext"], default="wikipedia",
                        help="Which dataset to use: 'wikipedia' or 'wikitext'.")
    parser.add_argument("--dataset_config", type=str, default="20231101.en",
                        help="Dataset config: for Wikipedia, use <lang> (e.g. '20200501.en'); for wikitext, e.g. 'wikitext-103-raw-v1'.")
    parser.add_argument("--seq_length", type=int, default=512, help="Sequence length for each example after grouping.")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Where to save the processed dataset.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of CPU processes for data processing.")
    parser.add_argument("--seed", type=int, default=47, help="Random seed for splitting data.")
    parser.add_argument("--val_split_percent", type=float, default=5.0,
                        help="Percentage of data to use as validation if no predefined split (for Wikipedia).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load raw dataset
    if args.dataset_name == "wikipedia":
        config = args.dataset_config
        print(f"Loading Wikipedia dataset ({config})...")
        raw_datasets = load_dataset("wikimedia/wikipedia", config, split="train")
        # raw_datasets is a Dataset (single split). We'll split into train/val manually.
        dataset_dict = {}
        if args.val_split_percent > 0:
            val_frac = args.val_split_percent / 100.0
            print(f"Splitting off {args.val_split_percent}% of data for validation...")
            raw_datasets = raw_datasets.shuffle(seed=args.seed)
            split = raw_datasets.train_test_split(test_size=val_frac, seed=args.seed)
            dataset_dict["train"] = split["train"]
            dataset_dict["validation"] = split["test"]
        else:
            dataset_dict["train"] = raw_datasets
            # no validation
        raw_datasets = DatasetDict(dataset_dict)
    else:
        config = args.dataset_config or "wikitext-103-raw-v1"
        print(f"Loading WikiText dataset ({config})...")
        # load_dataset returns a DatasetDict with splits for WikiText
        raw_datasets = load_dataset("wikitext", config)
        # it has 'train', 'validation', 'test' splits by default
        raw_datasets = raw_datasets.remove_columns([col for col in raw_datasets["train"].column_names if col != "text"])
        # keep only the 'text' column in case there are others (e.g., 'title' etc. in certain datasets)
        # We'll use raw_datasets["train"] and raw_datasets["validation"]

    # ---- FIX: Keep only the 'text' column BEFORE tokenization/grouping ----
    # Wikipedia carries metadata columns like 'title', 'url', 'id', etc.
    # Those are strings/ints and will break group_texts() which expects token lists.
    keep_col = "text"
    drop_cols = [c for c in raw_datasets["train"].column_names if c != keep_col]
    if drop_cols:
        raw_datasets = raw_datasets.remove_columns(drop_cols)
        
    # 2. Initialize tokenizer (using BERT's tokenizer for compatibility)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
    tokenizer.model_max_length = args.seq_length  # avoid warnings about tokenizer length
    print("Tokenizer vocab size:", tokenizer.vocab_size)

    # 3. Tokenize text. We concatenate and chunk later, so no truncation to seq_length yet.
    def tokenize_function(examples):
        # Remove empty lines to avoid wasting tokens on them
        texts = [t for t in examples["text"] if t and not t.isspace()]
        # Note: add_special_tokens=True by default, so each text gets [CLS] ... [SEP]
        return tokenizer(texts, return_special_tokens_mask=False)

    print("Tokenizing the dataset...")
    tokenized_ds = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],  # remove original text to save memory
        num_proc=args.num_workers
    )

    # 4. Group tokens into fixed-length sequences
    seq_length = args.seq_length
    def group_texts(examples):
        # Concatenate all token lists in this batch
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_len = len(concatenated["input_ids"])
        # Drop the remainder that doesn't fit into an even multiple of seq_length
        total_len = (total_len // seq_length) * seq_length
        if total_len == 0:
            return {k: [] for k in examples.keys()}  # in case a batch is smaller than one block
        result = {}
        for k, t in concatenated.items():
            # split the list into chunks of length seq_length
            chunks = [t[i: i + seq_length] for i in range(0, total_len, seq_length)]
            result[k] = chunks
        return result

    print(f"Grouping tokens into sequences of length {seq_length}...")
    processed_ds = tokenized_ds.map(
        group_texts,
        batched=True,
        num_proc=args.num_workers
    )
    # 5. Save processed dataset
    print(f"Saving processed dataset to {args.output_dir}...")
    processed_ds.save_to_disk(args.output_dir)
    print("Dataset preparation completed.")

if __name__ == "__main__":
    main()
