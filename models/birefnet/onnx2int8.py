
import argparse, glob, io, os, random
from typing import List, Dict, Optional
import numpy as np
from PIL import Image
import onnx
from onnx import numpy_helper
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat, CalibrationMethod
try:
    import requests
except Exception:
    requests = None

MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(1,1,3)
STD  = np.array([0.229, 0.224, 0.225], np.float32).reshape(1,1,3)

def upcast_fp16_to_fp32(in_path, out_path):
    m = onnx.load(in_path); ch=0
    inits=list(m.graph.initializer)
    for i,init in enumerate(inits):
        if init.data_type == onnx.TensorProto.FLOAT16:
            arr = numpy_helper.to_array(init).astype(np.float32)
            new = numpy_helper.from_array(arr, name=init.name)
            m.graph.initializer.remove(init); m.graph.initializer.insert(i,new); ch+=1
    onnx.checker.check_model(m); onnx.save(m,out_path)
    print(f"[fp16->fp32] {ch} tensors -> {out_path}")

def detect_input(model_path):
    m=onnx.load(model_path)
    for inp in m.graph.input:
        t=inp.type.tensor_type
        if t.shape.dim and len(t.shape.dim) in (3,4): return inp.name
    return m.graph.input[0].name

def read_image(path_or_url):
    if path_or_url and path_or_url.startswith(("http://","https://")) and requests is not None:
        r=requests.get(path_or_url,timeout=30); r.raise_for_status()
        from PIL import Image; import io as _io
        return Image.open(_io.BytesIO(r.content)).convert("RGB")
    return Image.open(path_or_url).convert("RGB")

def preprocess(im,h,w):
    im=im.resize((w,h), Image.Resampling.BILINEAR)
    arr=np.asarray(im,np.float32)/255.0
    arr=(arr-MEAN)/STD
    arr=np.transpose(arr,(2,0,1))[None,...].astype(np.float32)
    return arr

def synth_image(h,w):
    yy,xx=np.meshgrid(np.linspace(0,1,h),np.linspace(0,1,w),indexing="ij")
    base=(0.6*xx+0.4*yy)[...,None]
    noise=0.15*np.random.randn(h,w,3)
    arr=(base+noise+0.2)*255.0
    arr=np.clip(arr,0,255).astype(np.uint8)
    return Image.fromarray(arr,"RGB")

class ComboReader(CalibrationDataReader):
    def __init__(self,input_name,h,w,calib_dir,image,repeats,synthetic):
        self.input_name=input_name; self.samples=[]
        def add(im): self.samples.append({input_name: preprocess(im,h,w)})
        added=0
        if calib_dir:
            paths=[];
            for e in ("*.jpg","*.jpeg","*.png","*.bmp","*.webp"): paths+=glob.glob(os.path.join(calib_dir,e))
            for p in sorted(paths):
                try: add(Image.open(p).convert("RGB")); added+=1
                except: pass
        if added==0 and image:
            try:
                im=read_image(image)
                for _ in range(max(1,repeats)): add(im); added+=1
            except Exception as e:
                print("[warn] load --image failed:", e)
        if added==0 and synthetic>0:
            for _ in range(synthetic): add(synth_image(h,w)); added+=1
        self.i=0
        print(f"[calib] {added} samples prepared.")
    def get_next(self):
        if self.i>=len(self.samples): return None
        x=self.samples[self.i]; self.i+=1; return x
    def rewind(self): self.i=0

def main():
    ap=argparse.ArgumentParser("INT8 QDQ quant without real dataset")
    ap.add_argument("--float-model"); ap.add_argument("--fp16-model"); ap.add_argument("--upcast",action="store_true")
    ap.add_argument("--out",default="model_int8_qdq.onnx")
    ap.add_argument("--input-name",default=None)
    ap.add_argument("--height",type=int,default=1024); ap.add_argument("--width",type=int,default=1024)
    ap.add_argument("--calib-dir",default=None); ap.add_argument("--image",default=None)
    ap.add_argument("--repeats",type=int,default=32); ap.add_argument("--synthetic",type=int,default=64)
    ap.add_argument("--per-channel",action="store_true")
    ap.add_argument("--method",choices=["minmax","entropy"],default="minmax")
    a=ap.parse_args()
    if a.upcast:
        if not a.fp16_model: raise SystemExit("need --fp16-model with --upcast")
        tmp=os.path.splitext(a.fp16_model)[0]+"_fp32_tmp.onnx"; upcast_fp16_to_fp32(a.fp16_model,tmp); fm=tmp
    else:
        if not a.float_model: raise SystemExit("provide --float-model or use --fp16-model + --upcast")
        fm=a.float_model
    input_name=a.input_name or detect_input(fm); print("[info] input:", input_name)
    dr=ComboReader(input_name,a.height,a.width,a.calib_dir,a.image,a.repeats,a.synthetic)
    if not dr.samples: raise SystemExit("no calibration samples; provide --calib-dir or --image or --synthetic>0")
    qfmt=QuantFormat.QDQ; calib=CalibrationMethod.MinMax if a.method=="minmax" else CalibrationMethod.Entropy
    print("[quant] quantize_static...")
    quantize_static(model_input=fm, model_output=a.out, calibration_data_reader=dr, quant_format=qfmt,
                    per_channel=a.per_channel, activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    calibrate_method=calib)
    print("[done] saved:", a.out)

if __name__=="__main__":
    main()