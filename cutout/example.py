"""Lumen Cutout: the subject mask of one photo, with LiteRT.

usage: python example.py photo.jpg mask.png
pip install ai-edge-litert pillow numpy

This runs the 8-bit model on the CPU, which is fine for trying it out. On a desktop CPU the 8-bit
model is less faithful than on a phone NPU (see the model card); for phones, use a file from npu/.
"""
import sys
import numpy as np
from PIL import Image
from ai_edge_litert.interpreter import Interpreter

MODEL = "lumen_cutout_1024_int8.tflite"


def quantize(pixels, detail):
    scale, zero = detail["quantization"]  # 0.99369 and 0.99515 for the two inputs, zero point -128
    return np.clip(np.round(pixels / scale) + zero, -128, 127).astype(np.int8)


photo = Image.open(sys.argv[1]).convert("RGB")
interpreter = Interpreter(model_path=MODEL)
interpreter.allocate_tensors()
for detail in interpreter.get_input_details():
    side = detail["shape"][1]  # 1024: the full-size photo; 640: the backbone's copy
    pixels = np.asarray(photo.resize((side, side), Image.BILINEAR), np.float32)  # raw RGB 0-255, stretched
    interpreter.set_tensor(detail["index"], quantize(pixels, detail)[None])
interpreter.invoke()

out = interpreter.get_output_details()[0]
q = interpreter.get_tensor(out["index"])[0, :, :, 0].astype(np.int16)
mask = (q + 128) / 256.0  # 0..1, 0.5 at q = 0
Image.fromarray((mask * 255).round().astype(np.uint8)).resize(photo.size, Image.BILINEAR).save(sys.argv[2])
print("mask saved:", sys.argv[2], f"({(mask > 0.5).mean():.0%} of the photo is subject)")
