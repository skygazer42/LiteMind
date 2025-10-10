
import argparse
import io
import json
import os
import sys
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
from PIL import Image
import requests

# Prefer GPU if available (user can pip install onnxruntime-gpu)
try:
    import onnxruntime as ort
except Exception as e:
    print("ERROR: onnxruntime is not installed. Please `pip install onnxruntime` (CPU) "
          "or `pip install onnxruntime-gpu` (GPU).")
    raise

try:
    from huggingface_hub import hf_hub_download
except Exception as e:
    print("ERROR: huggingface_hub is not installed. Please `pip install huggingface_hub`.")
    raise


@dataclass
class PreprocConfig:
    size: Tuple[int, int]  # (height, width)
    rescale_factor: float
    image_mean: Tuple[float, float, float]
    image_std: Tuple[float, float, float]
    resample: int  # PIL resample enum


def load_preprocessor(repo_id: str, filename: str = "preprocessor_config.json") -> PreprocConfig:
    """Download and parse the preprocessor config to match the JS pipeline."""
    cfg_path = hf_hub_download(repo_id=repo_id, filename=filename)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    size = (int(cfg["size"]["height"]), int(cfg["size"]["width"]))
    rescale_factor = float(cfg.get("rescale_factor", 1.0 / 255.0))
    image_mean = tuple(float(x) for x in cfg.get("image_mean", [0.485, 0.456, 0.406]))
    image_std = tuple(float(x) for x in cfg.get("image_std", [0.229, 0.224, 0.225]))
    # Default to BILINEAR if unspecified
    pil_resample = int(cfg.get("resample", 2))
    return PreprocConfig(size=size, rescale_factor=rescale_factor, image_mean=image_mean,
                         image_std=image_std, resample=pil_resample)


def load_image(img_path_or_url: str) -> Image.Image:
    if img_path_or_url.startswith("http://") or img_path_or_url.startswith("https://"):
        resp = requests.get(img_path_or_url, timeout=30)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    else:
        return Image.open(img_path_or_url).convert("RGB")


def preprocess(img: Image.Image, pp: PreprocConfig) -> Tuple[np.ndarray, Tuple[int, int]]:
    orig_w, orig_h = img.width, img.height
    # Resize to model's expected input
    img_resized = img.resize((pp.size[1], pp.size[0]), resample=pp.resample)
    arr = np.array(img_resized).astype(np.float32)
    # to [0,1]
    arr *= pp.rescale_factor
    # normalize
    mean = np.array(pp.image_mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(pp.image_std, dtype=np.float32).reshape(1, 1, 3)
    arr = (arr - mean) / std
    # HWC -> CHW
    arr = np.transpose(arr, (2, 0, 1))
    # add batch dim
    arr = np.expand_dims(arr, 0).astype(np.float32)
    return arr, (orig_w, orig_h)


def make_session(model_repo: str, model_filename: str = "onnx/model.onnx", providers: Optional[list] = None) -> ort.InferenceSession:
    model_path = hf_hub_download(repo_id=model_repo, filename=model_filename)
    sess_opts = ort.SessionOptions()
    # Optionally, you can tune threads for CPU
    intra = int(os.environ.get("ORT_NUM_THREADS", "0"))
    if intra > 0:
        sess_opts.intra_op_num_threads = intra
    if providers is None:
        # Try CUDA if available, else CPU
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    try:
        session = ort.InferenceSession(model_path, sess_options=sess_opts, providers=providers)
    except Exception as e:
        print(f"Falling back to CPU due to provider error: {e}")
        session = ort.InferenceSession(model_path, sess_options=sess_opts, providers=["CPUExecutionProvider"])
    return session


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def postprocess(logits: np.ndarray, out_size: Tuple[int, int]) -> Image.Image:
    """
    logits: [1, 1, H, W] float32
    """
    if logits.ndim == 4:
        logits = logits[0, 0]
    elif logits.ndim == 3:
        logits = logits[0]
    # Apply sigmoid as per transformers.js example
    mask = sigmoid(logits)
    # to 0-255 uint8
    mask = (mask * 255.0).clip(0, 255).astype(np.uint8)
    mask_img = Image.fromarray(mask, mode="L")
    mask_img = mask_img.resize(out_size, resample=Image.BILINEAR)
    return mask_img


def save_cutout(rgb: Image.Image, alpha: Image.Image, out_path: str = "cutout.png") -> None:
    rgba = rgb.copy()
    rgba.putalpha(alpha)
    rgba.save(out_path)
    print(f"Saved RGBA cutout to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="BiRefNet ONNX inference (alpha matte).")
    parser.add_argument("--repo", default="onnx-community/BiRefNet-ONNX",
                        help="HF repo id. Examples: onnx-community/BiRefNet-ONNX, onnx-community/BiRefNet_lite-ONNX, onnx-community/BiRefNet-portrait-ONNX")
    parser.add_argument("--image", default="https://images.pexels.com/photos/5965592/pexels-photo-5965592.jpeg?auto=compress&cs=tinysrgb&w=1024",
                        help="Local path or URL to an image.")
    parser.add_argument("--save-mask", default="mask.png", help="Output path for the predicted mask PNG.")
    parser.add_argument("--save-cutout", default=None, help="Optional path for RGBA cutout PNG. If provided, saves a composited PNG with alpha.")
    parser.add_argument("--providers", default=None, help="Comma-separated ORT providers, e.g., 'CUDAExecutionProvider,CPUExecutionProvider'")
    args = parser.parse_args()

    print(f"Loading preprocessor from: {args.repo}")
    pp = load_preprocessor(args.repo)
    print(f"Preprocess config: size={pp.size}, rescale_factor={pp.rescale_factor}, mean={pp.image_mean}, std={pp.image_std}")

    print(f"Loading image: {args.image}")
    img = load_image(args.image)
    arr, (orig_w, orig_h) = preprocess(img, pp)
    print(f"Preprocessed to: {arr.shape} (N,C,H,W); original size: {(orig_w, orig_h)}")

    providers = None
    if args.providers:
        providers = [p.strip() for p in args.providers.split(",") if p.strip()]
        print(f"Using custom ORT providers: {providers}")
    print(f"Loading ONNX model from: {args.repo}")
    session = make_session(args.repo, providers=providers)

    in_name = session.get_inputs()[0].name
    out_name = session.get_outputs()[0].name
    print(f"Model IO: input='{in_name}', output='{out_name}'")

    outputs = session.run([out_name], {in_name: arr})
    logits = outputs[0]
    print(f"Raw output shape: {np.array(logits).shape}, dtype={np.array(logits).dtype}")

    mask_img = postprocess(logits, (orig_w, orig_h))
    mask_img.save(args.save_mask)
    print(f"Saved mask to: {args.save_mask}")

    if args.save_cutout is not None:
        save_cutout(img, mask_img, args.save_cutout)

    # Basic stats
    mask_np = np.array(mask_img)
    print(f"Mask stats -> min: {mask_np.min()}, max: {mask_np.max()}, mean: {mask_np.mean():.2f}")


if __name__ == "__main__":
    main()
