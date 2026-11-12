---
license: mit
base_model: ZhengPeng7/BiRefNet_lite
pipeline_tag: image-segmentation
library_name: litert
tags:
  - subject-segmentation
  - background-removal
  - salient-object-detection
  - birefnet
  - litert
  - tflite
  - int8
  - qualcomm
  - npu
  - on-device
  - android
  - distillation
datasets:
  - detection-datasets/coco
---

# Lumen Cutout: subject segmentation on mobile NPUs

**Lumen Cutout** lifts the subject of a photo off its background, on the phone's NPU: a 1024×1024 mask in
**23.7 ms on a Snapdragon 8 Elite Gen 5** and **182 ms on a mid-range Snapdragon 7 Gen 1**, all 8-bit, with every
layer on the NPU. It's the model behind "touch and hold to cut out" in [Lumen](https://github.com/subhashhhhhh/Lumen),
a local-first Android gallery.

It is based on [BiRefNet-lite](https://huggingface.co/ZhengPeng7/BiRefNet_lite) (Zheng et al.), which doesn't compile
for a phone NPU at all: its deformable convolutions, 5- and 6-dimensional tensors and cyclic rolls are not supported,
and its full-resolution activations don't fit the NPU's on-chip memory. Lumen Cutout rewrites what can be rewritten
exactly, changes the architecture where it can't, and gets the accuracy back with quantization-aware distillation
through the exact NPU graph.

📄 **Paper:** [*High-Resolution Subject Segmentation Entirely on Mobile NPUs: Exact Graph Rewrites, Split Resolution and 8-bit Distillation*](https://doi.org/10.5281/zenodo.24152780) (Zenodo, 2026) · 🛠️ **Code:** [lumen-npu-build](https://github.com/subhashhhhhh/lumen-npu-build)

## Results

**Against human labels**, on test sets that neither the teacher nor Lumen Cutout trained on (S-measure, max F, MAE;
higher, higher, lower is better):

| | ECSSD (1,000) | DUT-OMRON (5,168) |
|---|---|---|
| **Lumen Cutout, int8, on the 7 Gen 1 NPU** | **0.939** / 0.945 / 0.028 | **0.830** / 0.760 / 0.062 |
| The teacher (the original model, float32) | 0.938 / 0.947 / 0.025 | 0.826 / 0.763 / 0.057 |
| Apple Vision foreground mask | 0.940 / 0.946 / 0.023 | 0.842 / 0.757 / 0.037 |
| ML Kit Subject Segmentation | 0.928 / 0.938 / 0.033 | 0.789 / 0.719 / 0.091 |
| U²-Net | 0.928 / 0.940 / 0.033 | 0.848 / 0.795 / 0.054 |

The all-8-bit model on the NPU matches its float teacher, with slightly softer edges. Against 17 public models and two
platform APIs it is second on ECSSD; on DUT-OMRON six models score higher (the paper has the full table and the
training-data caveats). On 150 held-out photos its median IoU against the float teacher is 0.993.

**Speed** for one 1024×1024 mask (Qualcomm AI Hub, LiteRT on the NPU):

| Phone | Chip | Time | Peak memory |
|---|---|---|---|
| Samsung Galaxy S26 | Snapdragon 8 Elite Gen 5 | **23.7 ms** | 299 MB |
| Samsung Galaxy S25 | Snapdragon 8 Elite | 31.1 ms | 293 MB |
| Snapdragon 7 Gen 4 QRD | Snapdragon 7 Gen 4 | 73.8 ms | 286 MB |
| Samsung Galaxy S21 | Snapdragon 888 | 106.8 ms | 286 MB |
| Samsung Galaxy A73 5G | Snapdragon 778G | 188.7 ms | 474 MB |
| Motorola Edge 50 | Snapdragon 7 Gen 1 | 182 ms (`qnn-net-run`), 190 ms in the Lumen app | |

On the same phones the original model runs out of memory at 1024 pixels; at 512 it takes 677 ms (S26) and
3,653 ms (A73) on the CPU.

## Files

| File | |
|---|---|
| `lumen_cutout_1024_int8.tflite` | The model, all 8-bit (activations and weights), 61 MB. Runs anywhere LiteRT does; compile it for your chip, or take a file from `npu/`. |
| `npu/<chip>/lumen_cutout_1024_int8_<chip>.tflite` | Compiled ahead of time for 42 Snapdragon chips (`SM7450` is the 7 Gen 1, `SM8850` the 8 Elite Gen 5, …). Each is one NPU partition: no layer falls back to the CPU. |
| `pytorch/lumen_cutout.pt` | The weights of the final training round (float16), for PyTorch. [`convert/birefnet_lite.py`](https://github.com/subhashhhhhh/lumen-npu-build/blob/main/convert/birefnet_lite.py) builds the model from them; the command is in the [code's README](https://github.com/subhashhhhhh/lumen-npu-build#rebuild-the-final-model). |
| `example.py` | A mask for one photo with LiteRT, in 30 lines. |
| `cutout_512.tflite`, `pytorch/decoder_v2.pt` | Legacy: the earlier 512-pixel float model (v2, for GPUs) and its weights, kept for older versions of Lumen. |

## How to use it

Two inputs and one output, all int8:

| Tensor | Shape | Scale | Zero point |
|---|---|---|---|
| Input 0: the photo | 1×1024×1024×3 | 0.9936917 | −128 |
| Input 1: the same photo, smaller (the backbone's copy) | 1×640×640×3 | 0.9951496 | −128 |
| Output: the mask | 1×1024×1024×1 | 1/256 | −128 |

- Stretch the photo to a square (no padding), as raw RGB 0–255, and quantize: `q = clamp(round(v / scale) − 128, −128, 127)`.
  There are only 256 values, so a phone can use a 256-byte lookup table per input.
- The 640-pixel copy is a bilinear resize of the photo, with or without antialiasing (the model was trained with both).
- The mask is `(q + 128) / 256`, so 0.5 is `q = 0`. Stretch it back over the photo.

```bash
pip install ai-edge-litert pillow numpy
python example.py photo.jpg mask.png
```

**Run it on the NPU.** On a desktop CPU, LiteRT's reference 8-bit kernels round differently from the NPU, and the
8-bit model agrees less well with the float model there (median IoU 0.96 instead of 0.993, with some bad photos).
The model was trained for the NPU's arithmetic; the CPU is fine for trying it out.

On Android, load a file from `npu/` with LiteRT's `CompiledModel` and `Accelerator.NPU` (Qualcomm dispatch library).
It needs the chip it was compiled for. It doesn't compile for MediaTek NPUs, and old chips such as the Snapdragon
765G run it on the CPU.

## How it was made

- **Exact rewrites** (same maths, checked against the original): 4-dimensional window attention, shifted windows
  without a roll, channel-split convolutions, and attention one head at a time.
- **Architecture changes**: plain convolutions instead of deformable ones; the Swin backbone on a 640-pixel copy and
  the decoder and mask at 1024 (split resolution, two inputs); a slim full-resolution tail; one global window in
  stage 4; no half-resolution pass.
- **Training**: float distillation from the original model, then quantization-aware distillation through the exact NPU
  graph (every operator fake-quantized, as the NPU computes it). Distillation silently closed the gate of the
  deepest decoder branch (σ ≈ 10⁻¹¹); the final round restores it from the teacher and keeps it open with a
  feature loss.
- **Data** (teacher masks only, never human labels): 45% the author's own photos (not released), 20% DUTS-TR, 15%
  DIS5K training and validation images, 10% Supervisely Person, 10% AISegment matting images. Earlier rounds used
  COCO 2017.

## Limitations

- Quality is measured mostly against the teacher; the public test sets used here are about 400 pixels wide, so
  they can't show the benefit of a 1024-pixel mask.
- On very fine structures (DIS-VD: wires, fences, branches) it loses 0.017 in S-measure against the teacher.
- The teacher's public checkpoint was trained on DIS-TE1–4 and DUTS-TE, so those sets can't measure it (see the paper).

## License and credit

MIT (see `LICENSE`), like the original. Lumen Cutout is built on BiRefNet:

```bibtex
@article{zheng2024birefnet,
  title={Bilateral Reference for High-Resolution Dichotomous Image Segmentation},
  author={Zheng, Peng and Gao, Dehong and Fan, Deng-Ping and Liu, Li and Laaksonen, Jorma and Ouyang, Wanli and Sebe, Nicu},
  journal={CAAI Artificial Intelligence Research},
  volume={3},
  pages={9150038},
  year={2024}
}
```

If you use Lumen Cutout, please cite the paper:

```bibtex
@misc{gottumukkala2026lumencutout,
  title={High-Resolution Subject Segmentation Entirely on Mobile NPUs: Exact Graph Rewrites, Split Resolution and 8-bit Distillation},
  author={Gottumukkala, Subhash},
  year={2026},
  publisher={Zenodo},
  doi={10.5281/zenodo.24152780},
  url={https://doi.org/10.5281/zenodo.24152780}
}
```
