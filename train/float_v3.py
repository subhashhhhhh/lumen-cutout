"""Lumen: BiRefNet-lite without deformable convolutions, distilled from the original.

Qualcomm's Hexagon NPU can't run BiRefNet's deformable convolutions (their sampling needs gathers
it can't tile). The student is the same model with each one replaced by a plain convolution of the
same size, started from its weights; it learns to reproduce the original's masks (no labels: the
original is the teacher). The Swin backbone stays frozen and shared in spirit: only the decoder
side (decoder and squeeze module) trains. Data: COCO 2017 images, mostly at 512, some at 768.
Saves the best student (by agreement with the teacher on held-out images) to student.pt.

v2: resumes from v1's student, loads images with 4 worker processes (v1 spent most of its time
decoding JPEGs on the main thread, about 3 s a step), runs up to 10 hours with a cosine-decayed rate.

v3: the 7x7 plain convolutions get a dilation (DILATE). v2 plateaued at ~3.85% flips: the original's
deformable 7x7s sampled 6-10 px off-grid (p99), a plain 7x7 reaches 3 px, dilation d reaches 3d. The
weights keep their shape, so it resumes from v2's student. Runs on both T4s: one process per GPU
(torchrun, relaunched by this script), gradients averaged by hand after each step; rank 0 evaluates,
keeps the EMA and saves. 6 hours.
"""
import glob, os, random, subprocess, sys, time
DILATE = int(os.environ.get("LUMEN_DILATE", "2"))
if "LOCAL_RANK" not in os.environ:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm", "einops", "kornia"], check=False)
    import torch
    n = torch.cuda.device_count()
    if n > 1:
        print("relaunching on", n, "GPUs", flush=True)
        r = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={n}", os.path.abspath(__file__)])
        sys.exit(r.returncode)
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torch.distributed as dist
from PIL import Image
from transformers import AutoModelForImageSegmentation

torch.backends.cudnn.benchmark = True
Rank = int(os.environ.get("RANK", "0")); World = int(os.environ.get("WORLD_SIZE", "1"))
if World > 1:
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
dev = f"cuda:{torch.cuda.current_device()}"
Lead = Rank == 0
_print = print
def print(*a, **k):
    if Lead: _print(*a, **k)
Budget = float(os.environ.get("LUMEN_MINUTES", "360")) * 60
start = time.time()
teacher = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet_lite", trust_remote_code=True).float().to(dev).eval()
student = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet_lite", trust_remote_code=True).float().to(dev).eval()
for p in teacher.parameters(): p.requires_grad_(False)


class PlainConv(nn.Module):
    """A deformable convolution's regular kernel, used plainly (no offsets, no modulator)."""
    def __init__(self, deform):
        super().__init__()
        rc = deform.regular_conv
        k = rc.kernel_size[0]; d = DILATE if k == 7 else 1
        self.conv = nn.Conv2d(rc.in_channels, rc.out_channels, rc.kernel_size, padding=d * (k // 2), dilation=d, bias=rc.bias is not None)
        self.conv.weight.data.copy_(rc.weight.data)
        if rc.bias is not None: self.conv.bias.data.copy_(rc.bias.data)

    def forward(self, x):
        return self.conv(x)


replaced = 0
for name, mod in list(student.named_modules()):
    for child_name, child in list(mod.named_children()):
        if type(child).__name__ == "DeformableConv2d":
            setattr(mod, child_name, PlainConv(child).to(dev)); replaced += 1
print("replaced", replaced, "deformable convolutions; 7x7 dilation", DILATE, "; GPUs", World, flush=True)
for name, p in student.named_parameters():
    p.requires_grad_(not name.startswith("bb."))
trainable = [p for p in student.parameters() if p.requires_grad]
prev = sorted(glob.glob("/kaggle/input/**/student.pt", recursive=True))
if prev:
    sd = torch.load(prev[0], map_location="cpu")
    missing, unexpected = student.load_state_dict({k: v.float() for k, v in sd.items()}, strict=False)
    assert not unexpected and all(k.startswith("bb.") for k in missing)
    print("resumed from", prev[0], flush=True)
print("gpu", torch.cuda.get_device_name(0), "x", World, flush=True)
print("trainable params", sum(p.numel() for p in trainable) / 1e6, "M", flush=True)

files = sorted(glob.glob("/kaggle/input/**/train2017/*.jpg", recursive=True))
val_files = sorted(glob.glob("/kaggle/input/**/val2017/*.jpg", recursive=True))[:200]
print(len(files), "train images,", len(val_files), "val", flush=True)
mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)


def load(path, side, augment):
    img = Image.open(path); img.draft("RGB", (int(side / 0.6), int(side / 0.6))); img = img.convert("RGB")  # fast JPEG decode, still sharp after a 0.6 crop
    if augment:
        w, h = img.size; s = random.uniform(0.6, 1.0)
        cw, ch = int(w * s), int(h * s); x0, y0 = random.randint(0, w - cw), random.randint(0, h - ch)
        img = img.crop((x0, y0, x0 + cw, y0 + ch))
        if random.random() < 0.5: img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return torch.from_numpy(np.asarray(img.resize((side, side), Image.BILINEAR))).permute(2, 0, 1).float() / 255


def batch(paths, side, augment):
    x = torch.stack([load(p, side, augment) for p in paths]).to(dev, non_blocking=True)
    return (x - mean) / std


def logits(model, x):
    return model(x)[-1]


opt = torch.optim.AdamW(trainable, lr=5e-5, weight_decay=1e-4)
# v1 plateaued at ~5.7% flips with noisy steps: evaluate and save an exponential average of the weights.
import copy
ema = copy.deepcopy(student).eval()
for p in ema.parameters(): p.requires_grad_(False)
ema_pairs = [(e, s) for e, s in zip(ema.parameters(), student.parameters()) if s.requires_grad]
ema_bufs = list(zip(ema.buffers(), student.buffers()))


class Images(torch.utils.data.IterableDataset):
    """Endless random batches (side, images): 30% at 768 (4 a batch), the rest at 512 (8)."""
    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = random.Random(1000 + 100 * Rank + (info.id if info else 0) + int(time.time()))
        random.seed(rng.random())
        while True:
            side, bs = (768, 4) if rng.random() < 0.3 else (512, 8)
            yield torch.stack([load(p, side, True) for p in rng.sample(files, bs)])


loader = iter(torch.utils.data.DataLoader(Images(), batch_size=None, num_workers=4, pin_memory=True, prefetch_factor=4, persistent_workers=True))
scaler = torch.cuda.amp.GradScaler()


@torch.no_grad()
def evaluate(side=512, model=None):
    model = model or student; flips = []; mae = []
    for i in range(0, len(val_files), 8):
        x = batch(val_files[i:i + 8], side, False)
        with torch.autocast("cuda", dtype=torch.float16):
            t = logits(teacher, x).float().sigmoid(); s = logits(model, x).float().sigmoid()
        flips.append(((t > 0.5) != (s > 0.5)).float().mean().item()); mae.append((t - s).abs().mean().item())
    return float(np.mean(flips)), float(np.mean(mae))


def sync():
    if World > 1: dist.barrier()


def average_grads():
    """Each GPU saw its own batch: average the trainable gradients (same set on every rank)."""
    if World == 1: return
    gs = [p.grad for p in trainable if p.grad is not None]
    flat = torch.cat([g.reshape(-1) for g in gs]); dist.all_reduce(flat); flat /= World
    i = 0
    for g in gs: n = g.numel(); g.copy_(flat[i:i + n].view_as(g)); i += n


def elapsed():
    """Rank 0's clock, so every rank uses the same rate and stops at the same step."""
    t = torch.tensor([time.time() - start], device=dev, dtype=torch.float64)
    if World > 1: dist.broadcast(t, 0)
    return t.item()


f0, m0 = evaluate() if Lead else (0.0, 0.0)
print(f"before training: pixels changing side {f0:.3%}, mean diff {m0:.4f}", flush=True)
sync()
best = f0
step = 0
while (now := elapsed()) < Budget:
    x = (next(loader).to(dev, non_blocking=True) - mean) / std
    for g in opt.param_groups:
        g["lr"] = 2e-6 + 0.5 * (5e-5 - 2e-6) * (1 + np.cos(np.pi * min(1.0, now / Budget)))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        t = logits(teacher, x).float().sigmoid()
    with torch.autocast("cuda", dtype=torch.float16):
        s = logits(student, x).float()
    # Edges count more: where the teacher is unsure, and next to where it changes.
    edge = (F.max_pool2d(t, 7, 1, 3) - (-F.max_pool2d(-t, 7, 1, 3))).clamp(0, 1)
    weight = 1 + 4 * edge
    loss = (F.binary_cross_entropy_with_logits(s, t, reduction="none") * weight).mean() + ((s.sigmoid() - t).abs() * weight).mean()
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward(); average_grads(); scaler.step(opt); scaler.update()
    step += 1
    with torch.no_grad():
        d = min(0.999, (1 + step) / (10 + step))
        for e, w in ema_pairs: e.mul_(d).add_(w.detach(), alpha=1 - d)
        for e, w in ema_bufs: e.copy_(w)
    if step % 100 == 0:
        print(f"step {step} loss {loss.item():.4f} ({(time.time() - start) / 60:.0f} min)", flush=True)
    if step % 1000 == 0 and Lead:
        f, m = evaluate(model=ema); fr, _ = evaluate()
        print(f"step {step}: pixels changing side {f:.3%} (EMA), {fr:.3%} (raw), mean diff {m:.4f}", flush=True)
        if f < best:
            best = f
            torch.save({k: v.half().cpu() for k, v in ema.state_dict().items() if not k.startswith("bb.")}, "/kaggle/working/student.pt")
            print("saved", flush=True)
        torch.save({k: v.half().cpu() for k, v in student.state_dict().items() if not k.startswith("bb.")}, "/kaggle/working/last.pt")
    if step % 1000 == 0: sync()
if not Lead:
    dist.barrier(); dist.destroy_process_group(); sys.exit(0)
f, m = evaluate(model=ema); f768, _ = evaluate(768, model=ema)
print(f"final: 512 flips {f:.3%} (best {best:.3%}), 768 flips {f768:.3%}, steps {step}", flush=True)
if f < best:
    torch.save({k: v.half().cpu() for k, v in ema.state_dict().items() if not k.startswith("bb.")}, "/kaggle/working/student.pt")
if World > 1:
    dist.barrier(); dist.destroy_process_group()
