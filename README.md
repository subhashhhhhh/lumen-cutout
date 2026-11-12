# lumen-npu-build

Conversion, 8-bit quantization and ahead-of-time NPU compilation for the on-device models of
[Lumen](https://github.com/subhashhhhhh/Lumen), a local-first Android gallery. Most of it is
**Lumen Cutout**: subject segmentation that runs fully on phone NPUs, 8-bit, with every layer on the NPU. It is based on
[BiRefNet-lite](https://huggingface.co/ZhengPeng7/BiRefNet_lite) (Zheng et al.).

📄 **Paper:** [*High-Resolution Subject Segmentation Entirely on Mobile NPUs: Exact Graph Rewrites, Split Resolution and 8-bit Distillation*](https://doi.org/10.5281/zenodo.24152780) (Zenodo, 2026) · 🤗 **Model:** [subhashhhhhh/lumen-cutout](https://huggingface.co/subhashhhhhh/lumen-cutout)

| Phone | Chip | One 1024×1024 mask |
|---|---|---|
| Samsung Galaxy S26 | Snapdragon 8 Elite Gen 5 | 23.7 ms |
| Samsung Galaxy S25 | Snapdragon 8 Elite | 31.1 ms |
| Samsung Galaxy A73 5G | Snapdragon 778G | 188.7 ms |
| Motorola Edge 50 | Snapdragon 7 Gen 1 | 182 ms |

The original model doesn't compile for these NPUs and runs out of memory at 1024 pixels.

## What's here

| Path | |
|---|---|
| `convert/birefnet_npu_rewrites.py` | Exact rewrites of the original for the NPU: no tensor above 4 dimensions, no deformable convolution or `grid_sample`. |
| `convert/swin_v69.py` | Swin shifted windows without `torch.roll` (pad, aligned windows, a fixed mask), for the 7 Gen 1's NPU. |
| `convert/hybrid.py`, `convert/v4.py` | Split resolution (backbone on a 640-pixel copy, mask at 1024) and the final graph: two inputs, slim tail, global stage 4, no half pass. |
| `convert/qat.py` | Quantization-aware distillation through the exact NPU graph: every operator fake-quantized as the NPU computes it. |
| `convert/birefnet_lite.py` | PyTorch to LiteRT (`.tflite`, float32). |
| `convert/int8.py` | All-8-bit LiteRT model (`a8w8`), calibrated on COCO and checked against the float model. |
| `compile.py` | Ahead-of-time compilation for one Qualcomm or MediaTek chip with LiteRT's compiler. |
| `train/` | The distillation runs (Kaggle notebooks), one script per round; `train/v4r.py` made the final model. |
| `cutout/README.md` | The model card on Hugging Face. |

## Workflows

Everything runs on GitHub Actions; files move through releases.

1. **Convert** (`convert.yml`): runs a script from `convert/` and puts the `.tflite` in the `inputs` release.
2. **8-bit** (`int8.yml`): calibrates a float model from `inputs` on COCO photos and adds `<model>_a8w8.tflite`.
3. **Compile** (`compile.yml`): compiles models from `inputs` for every phone chip of a vendor (42 Snapdragon chips),
   one job per chip, into a `compiled-<chip>` release (and optionally the Hugging Face repo).

The final Lumen Cutout is `birefnet_v4r_snap_12000`: its PyTorch weights (`.pt`), float and 8-bit models are in the
[`inputs`](https://github.com/subhashhhhhh/lumen-npu-build/releases/tag/inputs) release, and the compiled files are
`birefnet_v4r_snap_12000_1024_a8w8_Qualcomm_<chip>_apply_plugin.tflite` in each `compiled-<chip>` release. The other
files in the releases are the experiments of the paper (earlier rounds, probes and ablations).

## Rebuild the final model

From its PyTorch weights to a compiled file for one chip, on your machine:

```bash
# float LiteRT: mask 1024, plain 7x7 convolutions at dilation 2, per-head attention, backbone 640, final graph
python convert/birefnet_lite.py birefnet_v4r_snap_12000_1024.tflite 1024 \
  https://github.com/subhashhhhhh/lumen-npu-build/releases/download/inputs/birefnet_v4r_snap_12000.pt \
  2 heads 640 tail two,slim,global4,nohalf
python convert/int8.py birefnet_v4r_snap_12000_1024.tflite birefnet_v4r_snap_12000_1024_a8w8.tflite a8w8 <COCO val2017 dir> 64
python compile.py birefnet_v4r_snap_12000_1024_a8w8.tflite SM7450 compiled/   # or any chip of compile.yml
```

To retrain it, `train/v4r.py` runs on Kaggle (two T4s): it clones this repository and starts from round 2's
checkpoint in the `inputs` release. The personal photos (45% of its data) are not released, so a rerun needs your own.

## Versions

| | |
|---|---|
| PyTorch to LiteRT | litert-torch 0.10.0, Python 3.11 |
| 8-bit | ai-edge-quantizer, a8w8, 64 COCO val2017 images |
| Compilation | ai-edge-litert 2.3.0, ai-edge-litert-sdk-qualcomm 2.3.0 (QAIRT 2.50), Python 3.12 |
| Teacher | `ZhengPeng7/BiRefNet_lite`, revision `aa62cd87eafb` |

## Limits

- The MediaTek compiler rejects the per-head attention (3-dimensional batched matrix products).
- The Snapdragon 865/870 (SM8250) is not known to the compiler.
- Compiled files run only on the chip they were compiled for, with LiteRT's Qualcomm dispatch library.

## Citation

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

## License

MIT, like the original ([ZhengPeng7/BiRefNet](https://github.com/ZhengPeng7/BiRefNet)). The personal photos used in
distillation are not released.
