#!/usr/bin/env python3
"""
Convert Wan2.1-1.3B weights from PyTorch .pth/.safetensors to MLX .safetensors format.

Applies all sanitize/key remapping so the Swift app can do a direct load.
Outputs:
  - t5_encoder.safetensors  (~11 GB, bfloat16)
  - dit_1_3b.safetensors    (~2.6 GB, bfloat16)
  - vae_decoder.safetensors (~250 MB, float16 - decoder only)
  - tokenizer.json          (copied as-is)
"""

import argparse
import os
import re
import shutil
import sys

sys.path.insert(0, "/tmp/mlx-examples/video/wan2.1")

import mlx.core as mx
import numpy as np


def convert_t5(repo_id, output_dir):
    """Convert T5 encoder weights to bfloat16 safetensors."""
    from huggingface_hub import hf_hub_download
    import torch

    print("=== Converting T5 encoder ===")
    weight_path = hf_hub_download(repo_id=repo_id, filename="models_t5_umt5-xxl-enc-bf16.pth")

    sd = torch.load(weight_path, map_location="cpu", weights_only=True)

    # Apply T5Encoder.sanitize
    remapped = {}
    for key, value in sd.items():
        new_key = key
        if new_key.startswith("model."):
            new_key = new_key[6:]
        if "ffn.gate.1" in new_key:
            continue
        if "dropout" in new_key:
            continue
        new_key = re.sub(r"ffn\.gate\.0\.", "ffn.gate.", new_key)

        # Convert to numpy then mlx, keeping bfloat16
        if value.dtype == torch.bfloat16:
            np_val = value.half().numpy()
            remapped[new_key] = mx.array(np_val).astype(mx.bfloat16)
        else:
            remapped[new_key] = mx.array(value.float().numpy())

    out_path = os.path.join(output_dir, "t5_encoder.safetensors")
    mx.save_safetensors(out_path, remapped)
    size_gb = os.path.getsize(out_path) / 1e9
    print(f"  Saved {out_path} ({size_gb:.2f} GB, {len(remapped)} tensors)")
    del sd, remapped


def convert_dit(repo_id, output_dir):
    """Convert DiT weights with full sanitize (QKV merge, modulation bake, etc)."""
    from huggingface_hub import hf_hub_download

    print("=== Converting DiT ===")
    weight_path = hf_hub_download(repo_id=repo_id, filename="diffusion_pytorch_model.safetensors")

    weights = mx.load(weight_path)

    # Apply WanModel.sanitize inline
    remapped = {}
    for key, value in weights.items():
        new_key = key

        if "weight_scale" in new_key:
            continue
        if new_key.startswith("model."):
            new_key = new_key[6:]

        # Conv3d transpose: [O,I,kT,kH,kW] -> [O,kT,kH,kW,I]
        if "patch_embedding" in new_key and "weight" in new_key and len(value.shape) == 5:
            value = mx.transpose(value, (0, 2, 3, 4, 1))

        # Sequential key remapping
        new_key = new_key.replace("ffn.0.", "ffn.layers.0.")
        new_key = new_key.replace("ffn.2.", "ffn.layers.2.")
        new_key = new_key.replace("text_embedding.0.", "text_embedding.layers.0.")
        new_key = new_key.replace("text_embedding.2.", "text_embedding.layers.2.")
        new_key = new_key.replace("time_embedding.0.", "time_embedding.layers.0.")
        new_key = new_key.replace("time_embedding.2.", "time_embedding.layers.2.")
        new_key = new_key.replace("time_projection.1.", "time_projection.layers.1.")
        new_key = new_key.replace("head.head.", "head.linear.")

        # I2V img_emb remapping
        new_key = re.sub(r"img_emb\.proj\.0\.(\w+)", r"img_emb_norm1.\1", new_key)
        new_key = re.sub(r"img_emb\.proj\.1\.(\w+)", r"img_emb_linear1.\1", new_key)
        new_key = re.sub(r"img_emb\.proj\.3\.(\w+)", r"img_emb_linear2.\1", new_key)
        new_key = re.sub(r"img_emb\.proj\.4\.(\w+)", r"img_emb_norm2.\1", new_key)

        remapped[new_key] = value

    # Merge Q/K/V weights
    merged = {}
    consumed = set()
    for key in remapped:
        m = re.match(r"(blocks\.\d+\.self_attn)\.(q)\.(weight|bias)$", key)
        if m:
            prefix, _, param = m.groups()
            q_key = f"{prefix}.q.{param}"
            k_key = f"{prefix}.k.{param}"
            v_key = f"{prefix}.v.{param}"
            if q_key in remapped and k_key in remapped and v_key in remapped:
                merged[f"{prefix}.qkv.{param}"] = mx.concatenate(
                    [remapped[q_key], remapped[k_key], remapped[v_key]], axis=0
                )
                consumed.update([q_key, k_key, v_key])
            continue

        m = re.match(r"(blocks\.\d+\.cross_attn)\.(k)\.(weight|bias)$", key)
        if m:
            prefix, _, param = m.groups()
            k_key = f"{prefix}.k.{param}"
            v_key = f"{prefix}.v.{param}"
            if k_key in remapped and v_key in remapped:
                merged[f"{prefix}.kv.{param}"] = mx.concatenate(
                    [remapped[k_key], remapped[v_key]], axis=0
                )
                consumed.update([k_key, v_key])
            continue

    for key, value in remapped.items():
        if key not in consumed:
            merged[key] = value
    remapped = merged

    # Bake modulation: add 1 to scale positions
    for key in list(remapped.keys()):
        if key.endswith(".modulation"):
            v = remapped[key]
            if v.shape[1] == 6:
                remapped[key] = v + mx.array([0, 1, 0, 0, 1, 0])[:, None]
            elif v.shape[1] == 2:
                remapped[key] = v + mx.array([0, 1])[:, None]

    # Cast to bfloat16
    for key in remapped:
        if remapped[key].dtype == mx.float32:
            remapped[key] = remapped[key].astype(mx.bfloat16)

    out_path = os.path.join(output_dir, "dit_1_3b.safetensors")
    mx.save_safetensors(out_path, remapped)
    size_gb = os.path.getsize(out_path) / 1e9
    print(f"  Saved {out_path} ({size_gb:.2f} GB, {len(remapped)} tensors)")
    del remapped


def convert_vae(repo_id, output_dir):
    """Convert VAE weights (decoder only) to float16 safetensors."""
    from huggingface_hub import hf_hub_download
    import torch

    print("=== Converting VAE (decoder only) ===")
    weight_path = hf_hub_download(repo_id=repo_id, filename="Wan2.1_VAE.pth")

    sd = torch.load(weight_path, map_location="cpu", weights_only=True)

    # Filter to decoder + conv2 only (skip encoder)
    remapped = {}
    for key, value in sd.items():
        # Only keep decoder and conv2 (pre-decoder conv)
        if not (key.startswith("decoder.") or key.startswith("conv2.")):
            continue

        new_key = key
        np_val = value.float().numpy()
        mx_val = mx.array(np_val)

        # Transpose convolution weights
        if "weight" in new_key:
            if len(mx_val.shape) == 5:
                mx_val = mx.transpose(mx_val, (0, 2, 3, 4, 1))
            elif len(mx_val.shape) == 4:
                mx_val = mx.transpose(mx_val, (0, 2, 3, 1))

        new_key = new_key.replace(".gamma", ".weight")
        new_key = new_key.replace("decoder.middle.0.", "decoder.middle_res1.")
        new_key = new_key.replace("decoder.middle.1.", "decoder.middle_attn.")
        new_key = new_key.replace("decoder.middle.2.", "decoder.middle_res2.")
        new_key = new_key.replace("decoder.head.0.", "decoder.head_norm.")
        new_key = new_key.replace("decoder.head.2.", "decoder.head_conv.")

        # Remap upsample keys
        if "decoder.upsamples." in new_key:
            m = re.match(r"decoder\.upsamples\.(\d+)\.(.*)", new_key)
            if m:
                layer_idx = int(m.group(1))
                rest = m.group(2)
                num_res_blocks, num_stages = 2, 4
                stage_sizes = [num_res_blocks + 2] * (num_stages - 1) + [num_res_blocks + 1]
                stage = 0
                local_idx = layer_idx
                for s, size in enumerate(stage_sizes):
                    if local_idx < size:
                        stage = s
                        break
                    local_idx -= size
                new_key = f"decoder.upsamples.{stage}.{local_idx}.{rest}"

        # ResidualBlock sub-key remapping
        new_key = re.sub(r"\.residual\.0\.", ".norm1.", new_key)
        new_key = re.sub(r"\.residual\.2\.", ".conv1.", new_key)
        new_key = re.sub(r"\.residual\.3\.", ".norm2.", new_key)
        new_key = re.sub(r"\.residual\.6\.", ".conv2.", new_key)
        new_key = re.sub(r"\.resample\.1\.", ".conv.", new_key)

        # Squeeze 1x1 conv weights to 2D for nn.Linear (to_qkv, proj)
        if ("to_qkv" in new_key or "proj" in new_key) and "weight" in new_key:
            if len(mx_val.shape) == 4 and mx_val.shape[1] == 1 and mx_val.shape[2] == 1:
                mx_val = mx_val.reshape(mx_val.shape[0], mx_val.shape[3])

        # Squeeze norm weights to 1D
        if "norm" in new_key and "weight" in new_key:
            if len(mx_val.shape) > 1:
                mx_val = mx.squeeze(mx_val)

        # Cast to float16
        mx_val = mx_val.astype(mx.float16)
        remapped[new_key] = mx_val

    out_path = os.path.join(output_dir, "vae_decoder.safetensors")
    mx.save_safetensors(out_path, remapped)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"  Saved {out_path} ({size_mb:.1f} MB, {len(remapped)} tensors)")
    del sd, remapped


def copy_tokenizer(repo_id, output_dir):
    """Copy tokenizer.json from HF Hub."""
    from huggingface_hub import hf_hub_download

    print("=== Copying tokenizer ===")
    tok_path = hf_hub_download(repo_id=repo_id, filename="google/umt5-xxl/tokenizer.json")
    out_path = os.path.join(output_dir, "tokenizer.json")
    shutil.copy2(tok_path, out_path)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"  Saved {out_path} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Convert Wan2.1-1.3B weights for iPad")
    parser.add_argument("--output", default="/Users/admin/WanVideoPad/Models",
                        help="Output directory for converted weights")
    parser.add_argument("--repo", default="Wan-AI/Wan2.1-T2V-1.3B",
                        help="HuggingFace repo ID")
    parser.add_argument("--skip-t5", action="store_true", help="Skip T5 conversion")
    parser.add_argument("--skip-dit", action="store_true", help="Skip DiT conversion")
    parser.add_argument("--skip-vae", action="store_true", help="Skip VAE conversion")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if not args.skip_t5:
        convert_t5(args.repo, args.output)
    if not args.skip_dit:
        convert_dit(args.repo, args.output)
    if not args.skip_vae:
        convert_vae(args.repo, args.output)
    copy_tokenizer(args.repo, args.output)

    print("\n=== Done! ===")
    for f in sorted(os.listdir(args.output)):
        path = os.path.join(args.output, f)
        size = os.path.getsize(path)
        if size > 1e9:
            print(f"  {f}: {size/1e9:.2f} GB")
        else:
            print(f"  {f}: {size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
