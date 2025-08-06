# scripts/generate_text.py
import argparse
import torch
import time
import sys
from pathlib import Path

# Add src to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import Qwen3Model
from src.tokenizer import Qwen3Tokenizer
from src.kv_cache import KVCache
from src.utils import QWEN3_CONFIG, load_weights_into_qwen, download_and_load_weights

def generate_stream(model, token_ids, max_new_tokens, use_kv_cache=True):
    model.eval()
    with torch.no_grad():
        if use_kv_cache:
            cache = KVCache(n_layers=model.cfg["n_layers"])
            model.reset_kv_cache()
            # Prime the cache
            logits = model(token_ids, cache=cache)
        else:
            cache = None
            logits = model(token_ids)
        
        for i in range(max_new_tokens):
            next_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True)
            yield next_token
            if use_kv_cache:
                logits = model(next_token, cache=cache)
            else:
                token_ids = torch.cat([token_ids, next_token], dim=1)
                logits = model(token_ids)

def main():
    parser = argparse.ArgumentParser(description="Generate text using a Qwen3 model.")
    parser.add_argument("--prompt", type=str, required=True, help="The input prompt.")
    parser.add_argument("--repo_id", type=str, default="Qwen/Qwen3-Coder-30B-A3B-Instruct", help="Hugging Face model repository ID.")
    parser.add_argument("--model_dir", type=str, default="models", help="Directory where models are stored.")
    parser.add_argument("--max-new-tokens", type=int, default=200, help="Maximum number of new tokens to generate.")
    parser.add_argument("--no-kv-cache", action="store_true", help="Disable the KV Cache for generation.")
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Model Weights ---
    try:
        weights_dict, local_path = download_and_load_weights(args.repo_id, args.model_dir)
    except Exception as e:
        print(f"Could not load or download weights: {e}")
        return

    # --- Initialize Model ---
    with device:
        model = Qwen3Model(QWEN3_CONFIG)
    
    load_weights_into_qwen(model, QWEN3_CONFIG, weights_dict)
    model.to(device)
    del weights_dict # Free up memory

    # --- Initialize Tokenizer ---
    tokenizer_path = local_path / "tokenizer.json"
    tokenizer = Qwen3Tokenizer(
        tokenizer_file_path=tokenizer_path,
        repo_id=args.repo_id,
        add_generation_prompt=True,
        add_thinking=True
    )
    
    # --- Generate Text ---
    input_ids = tokenizer.encode(args.prompt)
    input_ids_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)
    
    print("\nModel Output:")
    print("-" * 20)
    # Print prompt representation
    print(tokenizer.decode(input_ids), end="")

    start_time = time.time()
    for token in generate_stream(model, input_ids_tensor, args.max_new_tokens, use_kv_cache=not args.no_kv_cache):
        if token.item() == tokenizer.eos_token_id:
            break
        print(tokenizer.decode(token.squeeze(0).tolist()), end="", flush=True)

    end_time = time.time()
    print(f"\n\n{'-'*20}\nGeneration finished in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()