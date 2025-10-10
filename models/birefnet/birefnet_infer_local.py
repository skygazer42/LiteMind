
import argparse
import io
import json
import os
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
from PIL import Image
import requests

# ORT import
try:
    import onnxruntime as ort
except Exception as e:
    raise RuntimeError("onnxruntime not installed. `pip install onnxruntime` (CPU) or `pip install onnxruntime-gpu` (GPU).") from e

# Optional HF import for repo-based downloads
try:
    from huggingface_hub import hf_hub_download
except Exception:
    hf_hub_download = None  # allow purely-local runs


@dataclass
class PreprocConfig:
    size: Tuple[int, int]  # (height, width)
    rescale_factor: float
    image_mean: Tuple[float, float, float]
    image_std: Tuple[float, float, float]
    resample: int  # PIL resample enum


DEFAULT_PP = PreprocConfig(
    size=(512, 512),
    rescale_factor=1.0 / 255.0,
    image_mean=(0.485, 0.456, 0.406),
    image_std=(0.229, 0.224, 0.225),
    resample=Image.BILINEAR,
)


def load_preprocessor_from_repo(repo_id: str, filename: str = "preprocessor_config.json") -> PreprocConfig:
    if hf_hub_download is None:
        raise RuntimeError("huggingface_hub not installed but repo_id was provided. Install with `pip install huggingface_hub`, or pass --pp-json/--use-default-pp for local run.")
    cfg_path = hf_hub_download(repo_id=repo_id, filename=filename)
    return load_preprocessor_from_json(cfg_path)


def load_preprocessor_from_json(path: str) -> PreprocConfig:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    size = (int(cfg["size"]["height"]), int(cfg["size"]["width"])) if "size" in cfg else DEFAULT_PP.size
    rescale_factor = float(cfg.get("rescale_factor", DEFAULT_PP.rescale_factor))
    image_mean = tuple(float(x) for x in cfg.get("image_mean", list(DEFAULT_PP.image_mean)))
    image_std = tuple(float(x) for x in cfg.get("image_std", list(DEFAULT_PP.image_std)))
    pil_resample = int(cfg.get("resample", DEFAULT_PP.resample))
    return PreprocConfig(size=size, rescale_factor=rescale_factor, image_mean=image_mean,
                         image_std=image_std, resample=pil_resample)


def load_image(img_path_or_url: str) -> Image.Image:
    if img_path_or_url.startswith(("http://", "https://")):
        resp = requests.get(img_path_or_url, timeout=30)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    return Image.open(img_path_or_url).convert("RGB")


def preprocess(img: Image.Image, pp: PreprocConfig):
    orig_w, orig_h = img.width, img.height
    img_resized = img.resize((pp.size[1], pp.size[0]), resample=pp.resample)
    arr = np.array(img_resized).astype(np.float32)
    arr *= pp.rescale_factor
    mean = np.array(pp.image_mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(pp.image_std, dtype=np.float32).reshape(1, 1, 3)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))  # HWC->CHW
    arr = np.expand_dims(arr, 0).astype(np.float32)  # NCHW
    return arr, (orig_w, orig_h)


def make_session(model_path: str, providers: Optional[list] = None) -> ort.InferenceSession:
    sess_opts = ort.SessionOptions()
    intra = int(os.environ.get("ORT_NUM_THREADS", "0"))
    if intra > 0:
        sess_opts.intra_op_num_threads = intra
    if providers is None:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    try:
        return ort.InferenceSession(model_path, sess_options=sess_opts, providers=providers)
    except Exception as e:
        print(f"[WARN] Provider error '{e}'. Falling back to CPUExecutionProvider.")
        return ort.InferenceSession(model_path, sess_options=sess_opts, providers=["CPUExecutionProvider"])


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def postprocess(logits: np.ndarray, out_size: Tuple[int, int]) -> Image.Image:
    if logits.ndim == 4:
        logits = logits[0, 0]
    elif logits.ndim == 3:
        logits = logits[0]
    mask = sigmoid(logits)
    mask = (mask * 255.0).clip(0, 255).astype(np.uint8)
    mask_img = Image.fromarray(mask, mode="L").resize(out_size, resample=Image.BILINEAR)
    return mask_img


def save_cutout(rgb: Image.Image, alpha: Image.Image, out_path: str):
    rgba = rgb.copy()
    rgba.putalpha(alpha)
    rgba.save(out_path)
    print(f"Saved RGBA cutout: {out_path}")


def resolve_model_path(repo: Optional[str], local_model: Optional[str]) -> str:
    if local_model:
        if not os.path.exists(local_model):
            raise FileNotFoundError(f"Local model not found: {local_model}")
        return local_model
    if repo:
        if hf_hub_download is None:
            raise RuntimeError("huggingface_hub not installed and no local --model provided.")
        return hf_hub_download(repo_id=repo, filename="onnx/model.onnx")
    raise ValueError("Please provide either --model (local .onnx) or --repo (HF repo id).")


def resolve_preproc(repo: Optional[str], pp_json: Optional[str], use_default: bool) -> PreprocConfig:
    if pp_json:
        print(f"Loading preprocessor config from local JSON: {pp_json}")
        return load_preprocessor_from_json(pp_json)
    if repo:
        print(f"Loading preprocessor config from repo: {repo}")
        return load_preprocessor_from_repo(repo)
    if use_default:
        print("Using DEFAULT preprocessor config (size=1024x1024, ImageNet mean/std).")
        return DEFAULT_PP
    raise ValueError("No preprocessor provided. Use --pp-json, or --repo, or --use-default-pp.")


def main():
    ap = argparse.ArgumentParser(description="BiRefNet ONNX inference (supports local .onnx).")
    ap.add_argument("--repo", help="HF repo id (e.g., onnx-community/BiRefNet-ONNX).")
    ap.add_argument("--model", help="Local path to model.onnx (e.g., ./model_fp16.onnx).")
    ap.add_argument("--pp-json", help="Path to preprocessor_config.json if running locally.")
    ap.add_argument("--use-default-pp", action="store_true", help="Fallback to default preprocess if pp-json not provided.")
    ap.add_argument("--image", required=True, help="URL or local image path.")
    ap.add_argument("--save-mask", default="mask.png")
    ap.add_argument("--save-cutout", default=None)
    ap.add_argument("--providers", default=None, help="Comma-separated ORT EPs, e.g. 'CUDAExecutionProvider,CPUExecutionProvider'")
    args = ap.parse_args()

    pp = resolve_preproc(args.repo, args.pp_json, args.use_default_pp)
    img = load_image(args.image)
    arr, (ow, oh) = preprocess(img, pp)

    if args.providers:
        providers = [p.strip() for p in args.providers.split(",") if p.strip()]
    else:
        providers = None

    model_path = resolve_model_path(args.repo, args.model)
    print(f"Using model: {model_path}")
    sess = make_session(model_path, providers=providers)

    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    print(f"Model IO -> input: '{in_name}', output: '{out_name}'")

    outputs = sess.run([out_name], {in_name: arr})
    logits = outputs[0]

    mask_img = postprocess(logits, (ow, oh))
    mask_img.save(args.save_mask)
    print(f"Saved mask: {args.save_mask}")

    if args.save_cutout:
        save_cutout(img, mask_img, args.save_cutout)

    # Stats
    m = np.array(mask_img)
    print(f"Mask stats -> min {m.min()}  max {m.max()}  mean {m.mean():.2f}")


if __name__ == "__main__":
    main()
