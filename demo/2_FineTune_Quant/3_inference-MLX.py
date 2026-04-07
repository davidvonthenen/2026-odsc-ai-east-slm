#!/usr/bin/env python3
"""
Raw Inference Validator for MLX-Quantized Qwen SQL Model.

This script loads the specified MLX directory directly into Apple Silicon's 
Unified Memory and executes a single-shot streaming inference. It calculates 
the critical Token-Per-Second (TPS) metric to validate hardware acceleration.
"""

import argparse
import time
from typing import List, Dict
from xml.parsers.expat import model

# Apple's native MLX framework tools
from mlx_lm import load, stream_generate

# Core Configurations matching your A2A service
SYSTEM_PROMPT = "You are a database engineer. Generate valid SQL for the given schema and request."
DEFAULT_MODEL_DIR = "./qwen_sql_mlx_int4"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-shot MLX inference.")
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR, help="Path to the quantized MLX model.")
    parser.add_argument("--max_tokens", type=int, default=512, help="Maximum tokens to generate.")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"\n>> [INIT] Loading MLX model and tokenizer from: {args.model_dir}")
    print(">> [INIT] Mapping tensors directly to Unified Memory...")
    
    # MLX handles the hardware allocation automatically
    load_start = time.perf_counter()
    model, tokenizer = load(args.model_dir)
    load_time = time.perf_counter() - load_start
    print(f">> [INIT] Model loaded in {load_time:.2f} seconds.\n")

    # Sample Schema and Question
    schema = """
    CREATE TABLE employees (
        emp_id INT PRIMARY KEY,
        first_name VARCHAR(50),
        last_name VARCHAR(50),
        department_id INT,
        salary DECIMAL(10,2)
    );
    CREATE TABLE departments (
        department_id INT PRIMARY KEY,
        department_name VARCHAR(50)
    );
    """
    question = "List the first and last names of all employees who work in the 'Engineering' department, ordered by salary descending."

    # Construct the instruction payload using the tokenizer's chat template
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Schema: {schema}\nQuestion: {question}"}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    print("-" * 60)
    print(">> [QUESTION]")
    print(schema.strip(' \t\n\r'))
    print("\n")
    print(question.strip(' \t\n\r'))
    print("-" * 60)
    print(">> [INFERENCE] Igniting generation stream...\n")
    
    # Generation Tracking
    generated_text = ""
    token_count = 0
    start_time = time.perf_counter()
    first_token_time = 0

    # Stream generation natively through MLX
    # for chunk in stream_generate(model, tokenizer, prompt=prompt, max_tokens=args.max_tokens):
    #     if token_count == 0:
    #         first_token_time = time.perf_counter() - start_time
            
    #     print(chunk, end="", flush=True)
    #     generated_text += chunk
    #     token_count += 1
    for response in stream_generate(model, tokenizer, prompt=prompt, max_tokens=args.max_tokens):
        if token_count == 0:
            first_token_time = time.perf_counter() - start_time
            
        # Extract the string payload from the GenerationResponse object
        text_chunk = response.text
            
        print(text_chunk, end="", flush=True)
        generated_text += text_chunk
        token_count += 1

    total_time = time.perf_counter() - start_time
    
    # Exclude the time-to-first-token for an accurate generation speed metric
    generation_time = total_time - first_token_time
    tps = (token_count - 1) / generation_time if generation_time > 0 else 0

    print("\n" + "-" * 60)
    print(f"\n>> [METRICS] Time to First Token (TTFT): {first_token_time:.4f}s")
    print(f">> [METRICS] Total Generation Time: {total_time:.4f}s")
    print(f">> [METRICS] Tokens Generated: {token_count}")
    print(f">> [METRICS] Generation Speed: {tps:.2f} Tokens/Second\n")


if __name__ == "__main__":
    main()