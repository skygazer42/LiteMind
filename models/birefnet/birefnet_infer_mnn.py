import argparse, io, os, numpy as np
from PIL import Image
import requests

import MNN

MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)
H, W = 512, 512                  # 你的模型固定输入

def load_image(p):
    if p.startswith("http"):
        r = requests.get(p, timeout=30); r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
    else:
        img = Image.open(p).convert("RGB")
    return img

def preprocess(img):
    ow, oh = img.width, img.height
    img = img.resize((W, H), Image.BILINEAR)
    x = np.asarray(img).astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    x = x.transpose(2,0,1)[None, ...]   # NCHW
    return x, (ow, oh)

def sigmoid(x): return 1/(1+np.exp(-x))

def run_mnn(mnn_path, x, in_name="input_image", out_name="output_image", threads=4):
    import MNN
    x = np.ascontiguousarray(x.astype(np.float32))
    interp = MNN.Interpreter(mnn_path)

    # 兼容不同版本 API：有 Session_Config 就用，没有就用全局线程数
    if hasattr(MNN, "Session_Config"):
        conf = MNN.Session_Config()
        conf.numThread = int(threads)
        if hasattr(MNN, "BackendConfig"):
            bc = MNN.BackendConfig()
            bc.precision = getattr(MNN.BackendConfig, "Precision_Low", 0)  # 可改 High/Normal
            conf.backendConfig = bc
        sess = interp.createSession(conf)
    else:
        if hasattr(MNN, "setCPUThreads"):
            MNN.setCPUThreads(int(threads))
        sess = interp.createSession()

    # 取输入张量（若名字不匹配，退化为第一个）
    try:
        tin = interp.getSessionInput(sess, in_name)
    except Exception:
        tin = interp.getSessionInput(sess)

    in_shape = tuple(int(d) for d in x.shape)  # 一定要 tuple
    interp.resizeTensor(tin, in_shape)
    interp.resizeSession(sess)

    # 拷贝输入
    tmp = MNN.Tensor(in_shape, MNN.Halide_Type_Float, x, MNN.Tensor_DimensionType_Caffe)
    tin.copyFrom(tmp)

    # 执行
    interp.runSession(sess)

    # 取输出（同理，名字不匹配就取第一个）
    try:
        tout = interp.getSessionOutput(sess, out_name)
    except Exception:
        tout = interp.getSessionOutput(sess)

    out_shape = tuple(int(d) for d in tout.getShape())
    host = MNN.Tensor(out_shape, MNN.Halide_Type_Float,
                      np.empty(out_shape, dtype=np.float32),
                      MNN.Tensor_DimensionType_Caffe)
    tout.copyToHostTensor(host)
    y = np.array(host.getData(), dtype=np.float32).reshape(out_shape)
    return y


def postprocess(logits, out_size):
    if logits.ndim == 4:
        logits = logits[0,0]
    elif logits.ndim == 3:
        logits = logits[0]
    m = (sigmoid(logits) * 255).clip(0,255).astype(np.uint8)
    return Image.fromarray(m, "L").resize(out_size, Image.BILINEAR)

def save_cutout(rgb, alpha, path):
    rgba = rgb.copy(); rgba.putalpha(alpha); rgba.save(path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mnn", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--save-mask", default="mask_mnn.png")
    ap.add_argument("--save-cutout", default="cutout_mnn.png")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    img = load_image(args.image)
    x, (ow,oh) = preprocess(img)
    y = run_mnn(args.mnn, x, threads=args.threads)
    mask = postprocess(y, (ow,oh))
    mask.save(args.save_mask)
    print("Saved mask:", args.save_mask)
    if args.save_cutout:
        save_cutout(img, mask, args.save_cutout)
        print("Saved cutout:", args.save_cutout)
    a = np.array(mask)
    print(f"Mask stats -> min {a.min()} max {a.max()} mean {a.mean():.2f}")
