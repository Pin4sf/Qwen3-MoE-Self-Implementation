# src/utils.py
import torch
import json
import os
from pathlib import Path
from safetensors.torch import load_file
from huggingface_hub import snapshot_download

# Model configuration
QWEN3_CONFIG = {
    "vocab_size": 151_936,
    "context_length": 262_144,
    "emb_dim": 2048,
    "n_heads": 32,
    "n_layers": 48,
    "head_dim": 128,
    "qk_norm": True,
    "n_kv_groups": 4,
    "rope_base": 10_000_000.0,
    "dtype": torch.bfloat16,
    "num_experts": 128,
    "num_experts_per_tok": 8,
    "moe_intermediate_size": 768,
}

# Function to load model weights
def load_weights_into_qwen(model, param_config, params):
    def assign(left, right):
        if left.shape != right.shape:
            raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
        return torch.nn.Parameter(right.clone().detach())
    
    model.tok_emb.weight = assign(model.tok_emb.weight, params["model.embed_tokens.weight"])
    for l in range(param_config["n_layers"]):
        block = model.trf_blocks[l]
        att = block.att
        att.W_query.weight = assign(att.W_query.weight, params[f"model.layers.{l}.self_attn.q_proj.weight"])
        att.W_key.weight = assign(att.W_key.weight, params[f"model.layers.{l}.self_attn.k_proj.weight"])
        att.W_value.weight = assign(att.W_value.weight, params[f"model.layers.{l}.self_attn.v_proj.weight"])
        att.out_proj.weight = assign(att.out_proj.weight, params[f"model.layers.{l}.self_attn.o_proj.weight"])
        if hasattr(att, "q_norm") and att.q_norm is not None:
            att.q_norm.scale = assign(att.q_norm.scale, params[f"model.layers.{l}.self_attn.q_norm.weight"])
        if hasattr(att, "k_norm") and att.k_norm is not None:
            att.k_norm.scale = assign(att.k_norm.scale, params[f"model.layers.{l}.self_attn.k_norm.weight"])
        block.norm1.scale = assign(block.norm1.scale, params[f"model.layers.{l}.input_layernorm.weight"])
        if "num_experts" in param_config and param_config["num_experts"] > 0:
            block.ff.gate.weight = assign(block.ff.gate.weight, params[f"model.layers.{l}.mlp.gate.weight"])
            for e in range(param_config["num_experts"]):
                prefix = f"model.layers.{l}.mlp.experts.{e}"
                block.ff.fc1[e].weight = assign(block.ff.fc1[e].weight, params[f"{prefix}.gate_proj.weight"])
                block.ff.fc2[e].weight = assign(block.ff.fc2[e].weight, params[f"{prefix}.up_proj.weight"])
                block.ff.fc3[e].weight = assign(block.ff.fc3[e].weight, params[f"{prefix}.down_proj.weight"])
                block.ff.fc1[e], block.ff.fc2[e], block.ff.fc3[e] = block.ff.fc1[e].to("cpu"), block.ff.fc2[e].to("cpu"), block.ff.fc3[e].to("cpu")
        else: # Dense model logic
            block.ff.fc1.weight = assign(block.ff.fc1.weight, params[f"model.layers.{l}.mlp.gate_proj.weight"])
            block.ff.fc2.weight = assign(block.ff.fc2.weight, params[f"model.layers.{l}.mlp.up_proj.weight"])
            block.ff.fc3.weight = assign(block.ff.fc3.weight, params[f"model.layers.{l}.mlp.down_proj.weight"])
        block.norm2.scale = assign(block.norm2.scale, params[f"model.layers.{l}.post_attention_layernorm.weight"])
    model.final_norm.scale = assign(model.final_norm.scale, params["model.norm.weight"])
    model.out_head.weight = assign(model.out_head.weight, params.get("lm_head.weight", params["model.embed_tokens.weight"]))


# Function to download and load weights from Hugging Face
def download_and_load_weights(repo_id, local_dir="models"):
    local_path = Path(local_dir) / Path(repo_id).name
    print(f"Downloading model from {repo_id} to {local_path}...")
    repo_dir = snapshot_download(repo_id=repo_id, local_dir=local_path, local_dir_use_symlinks=False)
    index_path = os.path.join(repo_dir, "model.safetensors.index.json")
    with open(index_path, "r") as f:
        index = json.load(f)
    weights_dict = {}
    for filename in set(index["weight_map"].values()):
        shard_path = os.path.join(repo_dir, filename)
        shard = load_file(shard_path)
        weights_dict.update(shard)
    print("Download and weight loading complete.")
    return weights_dict, local_path