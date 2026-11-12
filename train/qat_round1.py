"""Lumen Cutout, all-8-bit: quantization-aware distillation (QAT) of v3 against BiRefNet-lite.

v3's float student (7x7 plain convolutions at dilation 2) loses badly when every tensor is rounded to
8 bits after training (median IoU 0.920, worst 0.006 vs the float model). Here the student trains
*through* that rounding: the NPU graph is captured with torch.export and gets 8-bit fake
quantization on every float tensor and every conv/linear weight (lumen-npu-build convert/qat.py),
then it learns to match the original BiRefNet-lite's masks anyway. One change against v3, nothing
else: same COCO data, same loss (BCE + L1, edge weight 1 + 4 x the teacher's 7x7 local range), EMA.

- Swin trains too (the rounding is everywhere), at a tenth of the decoder's rate (2e-6 vs 2e-5).
- Observers calibrate on ~200 COCO photos first, then follow slowly, then freeze at FREEZE steps.
- Fixed 512 x 512, 2 photos per GPU (the exported graph has a fixed shape), 2 x T4 via torchrun.
- Every 500 steps: the EMA student, fake-quantized, vs the teacher on the first 200 val2017 photos:
  pixels changing side, and IoU on photos where the teacher's mask covers >= 1%. The best by mean
  IoU is saved as student.pt: plain float weights under the original names, all of them (Swin too),
  for birefnet_lite.py -> int8.py a8w8 -> compile.yml.
LUMEN_SMOKE=1 runs a few CPU steps on local photos at 256 to catch code errors.
"""
import glob, os, random, subprocess, sys, time, copy
SMOKE = os.environ.get("LUMEN_SMOKE") == "1"
OUT = os.environ.get("LUMEN_OUT", "/kaggle/working")
NPB = os.environ.get("LUMEN_NPB", "/tmp/npb")
if "LOCAL_RANK" not in os.environ and not SMOKE:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm", "einops", "kornia"], check=False)
    if not os.path.isdir(NPB):
        subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/subhashhhhhh/lumen-npu-build", NPB], check=True)
    import torch
    n = torch.cuda.device_count()
    if n > 1:
        print("relaunching on", n, "GPUs", flush=True)
        r = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={n}", os.path.abspath(__file__)])
        sys.exit(r.returncode)
sys.path.insert(0, os.path.join(NPB, "convert"))
import numpy as np, torch, torch.nn.functional as F
import torch.distributed as dist
from PIL import Image
from transformers import AutoModelForImageSegmentation
import qat

torch.backends.cudnn.benchmark = True
Rank = int(os.environ.get("RANK", "0")); World = int(os.environ.get("WORLD_SIZE", "1"))
if World > 1:
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
dev = "cpu" if SMOKE else f"cuda:{torch.cuda.current_device()}"
Lead = Rank == 0
_print = print
def print(*a, **k):
    if Lead: _print(*a, **k)

SIDE = 256 if SMOKE else 512
B = 1 if SMOKE else int(os.environ.get("LUMEN_BATCH", "2"))  # 4 a GPU runs out of a T4's 15 GB
DILATE = 2
LR, BB_LR = 2e-5, 2e-6
FREEZE = int(os.environ.get("LUMEN_FREEZE", "3000"))
EVAL = 2 if SMOKE else 500
Budget = float(os.environ.get("LUMEN_MINUTES", "330")) * 60
MODEL = os.environ.get("LUMEN_MODEL", "ZhengPeng7/BiRefNet_lite")
start = time.time()

teacher = qat.Wrapped(AutoModelForImageSegmentation.from_pretrained(MODEL, trust_remote_code=True).float().eval()).to(dev).eval()
for p in teacher.parameters(): p.requires_grad_(False)
prev = sorted(glob.glob(os.environ.get("LUMEN_RESUME", "/kaggle/input/**/student.pt"), recursive=True))
assert prev, "no v3 student.pt in the inputs"
sd = torch.load(prev[0], map_location="cpu")
print("resuming from", prev[0], len(sd), "tensors", flush=True)
gm, ref_keys, counts = qat.build(sd, DILATE, SIDE, B, MODEL, device=dev)
print("fake-quantized graph:", counts[0], "activation points,", counts[1], "weights; batch", B, "x", World, "GPUs; side", SIDE, flush=True)
bb = [p for n, p in gm.named_parameters() if n.startswith("net.bb.")]
dec = [p for n, p in gm.named_parameters() if not n.startswith("net.bb.")]
for p in bb + dec: p.requires_grad_(True)
print("trainable:", sum(p.numel() for p in dec) / 1e6, "M decoder side,", sum(p.numel() for p in bb) / 1e6, "M Swin", flush=True)
opt = torch.optim.AdamW([{"params": dec, "peak": LR}, {"params": bb, "peak": BB_LR}], lr=LR, weight_decay=1e-4)
trainable = dec + bb

files = sorted(glob.glob(os.environ.get("LUMEN_TRAIN", "/kaggle/input/**/train2017/*.jpg"), recursive=True))
val_files = sorted(glob.glob(os.environ.get("LUMEN_VAL", "/kaggle/input/**/val2017/*.jpg"), recursive=True))[:200]
if SMOKE: val_files = val_files[:2]
val_files = val_files[:len(val_files) // B * B]
print(len(files), "train images,", len(val_files), "val", flush=True)


def load(path, augment):
    """Raw RGB 0-255, HWC, as Lumen feeds the model."""
    img = Image.open(path); img.draft("RGB", (int(SIDE / 0.6), int(SIDE / 0.6))); img = img.convert("RGB")
    if augment:
        w, h = img.size; s = random.uniform(0.6, 1.0)
        cw, ch = int(w * s), int(h * s); x0, y0 = random.randint(0, w - cw), random.randint(0, h - ch)
        img = img.crop((x0, y0, x0 + cw, y0 + ch))
        if random.random() < 0.5: img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return torch.from_numpy(np.asarray(img.resize((SIDE, SIDE), Image.BILINEAR), np.float32).copy())


class Images(torch.utils.data.IterableDataset):
    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = random.Random(1000 + 100 * Rank + (info.id if info else 0) + int(time.time()))
        random.seed(rng.random())
        while True:
            yield torch.stack([load(p, True) for p in rng.sample(files, B)])


loader = iter(torch.utils.data.DataLoader(Images(), batch_size=None, num_workers=0 if SMOKE else 4, pin_memory=not SMOKE,
                                          **({} if SMOKE else {"prefetch_factor": 4, "persistent_workers": True})))
amp = not SMOKE
scaler = torch.amp.GradScaler("cuda", enabled=amp)


def autocast():
    return torch.autocast("cuda", dtype=torch.float16, enabled=amp)


def fq_buffers(m):
    return [b for name, b in m.named_buffers() if ".fq." in name]


def sync_observers():
    """Rank 0's quantization ranges everywhere (each GPU observed its own photos)."""
    if World > 1:
        for b in fq_buffers(gm): dist.broadcast(b, 0)


FloatMasks = []  # float v3's masks on the val photos: the brief's target is agreement with them


def _iou(a, b):
    return ((a & b).sum() / (a | b).sum().clamp(min=1)).item()


@torch.no_grad()
def evaluate(model, keep_float=False):
    flips, mae, ious, fious = [], [], [], []
    k = 0
    for i in range(0, len(val_files), B):
        x = torch.stack([load(p, False) for p in val_files[i:i + B]]).to(dev)
        with autocast():
            t = teacher(x).float(); s = model(x).float()
        for tt, ss in zip(t, s):
            a, b = tt > 0.5, ss > 0.5
            flips.append((a != b).float().mean().item()); mae.append((tt - ss).abs().mean().item())
            if a.float().mean() >= 0.01:  # IoU only where the reference finds a subject
                ious.append(_iou(a, b))
            if keep_float:
                FloatMasks.append(b.cpu())
            elif FloatMasks:
                f = FloatMasks[k].to(dev)
                if f.float().mean() >= 0.01: fious.append(_iou(f, b))
            k += 1
    ious = np.array(ious if ious else [float("nan")]); fious = np.array(fious if fious else [float("nan")])
    return dict(flips=float(np.mean(flips)), mae=float(np.mean(mae)), iou_med=float(np.median(ious)),
                iou_mean=float(ious.mean()), iou_min=float(ious.min()), n=len(ious),
                f_med=float(np.median(fious)), f_mean=float(fious.mean()), f_min=float(fious.min()), fn=len(fious))


def show(tag, r):
    return (f"{tag}: vs teacher: pixels changing side {r['flips']:.3%}, mean diff {r['mae']:.4f}, IoU median {r['iou_med']:.4f} "
            f"mean {r['iou_mean']:.4f} worst {r['iou_min']:.4f} ({r['n']}) | vs float v3: IoU median {r['f_med']:.4f} "
            f"mean {r['f_mean']:.4f} worst {r['f_min']:.4f} ({r['fn']})")


def save(model, name):
    w, missing = qat.float_weights(model, ref_keys)
    assert not missing, sorted(missing)[:5]
    torch.save({k: v.half().cpu() for k, v in w.items()}, f"{OUT}/{name}")


# Float (no rounding) and calibrated-but-untrained 8-bit: the two starting points.
qat.observe(gm, False); qat.quantize(gm, False)
if Lead: print(show("float v3 (no rounding)", evaluate(gm, keep_float=True)), flush=True)
qat.observe(gm, True); qat.averaging(gm, 0.05)
with torch.no_grad():
    for i in range(1 if SMOKE else 200 // (B * World)):
        x = next(loader).to(dev)
        with autocast(): gm(x)
sync_observers()
qat.quantize(gm, True); qat.averaging(gm, 0.01)
print(f"calibrated on {200 if not SMOKE else 1} photos ({time.time() - start:.0f} s in)", flush=True)
ema = copy.deepcopy(gm); qat.observe(ema, False)
for p in ema.parameters(): p.requires_grad_(False)
ema_pairs = [(e, s) for e, s in zip(ema.parameters(), gm.parameters())]
ema_bufs = list(zip(ema.buffers(), gm.buffers()))
best = -1.0
if Lead:
    qat.observe(gm, False); r0 = evaluate(gm); qat.observe(gm, True)
    print(show("8-bit before training", r0), flush=True)
    best = r0["f_mean"]
if World > 1: dist.barrier()


def average_grads():
    if World == 1: return
    gs = [p.grad for p in trainable if p.grad is not None]
    flat = torch.cat([g.reshape(-1) for g in gs]); dist.all_reduce(flat); flat /= World
    i = 0
    for g in gs: n = g.numel(); g.copy_(flat[i:i + n].view_as(g)); i += n


def elapsed():
    t = torch.tensor([time.time() - start], device=dev, dtype=torch.float64)
    if World > 1: dist.broadcast(t, 0)
    return t.item()


step = 0
while (now := elapsed()) < Budget:
    x = next(loader).to(dev, non_blocking=True)
    cos = 0.5 * (1 + np.cos(np.pi * min(1.0, now / Budget)))
    for g in opt.param_groups:
        g["lr"] = g["peak"] * (0.05 + 0.95 * cos) * min(1.0, (step + 1) / 300)
    with torch.no_grad(), autocast():
        t = teacher(x).float()
    with autocast():
        s = gm(x)
    s = torch.nan_to_num(s.float(), nan=0.5).clamp(1e-6, 1 - 1e-6)
    tn = t.permute(0, 3, 1, 2)
    edge = (F.max_pool2d(tn, 7, 1, 3) + F.max_pool2d(-tn, 7, 1, 3)).clamp(0, 1).permute(0, 2, 3, 1)
    weight = 1 + 4 * edge
    loss = (F.binary_cross_entropy(s, t, reduction="none") * weight).mean() + ((s - t).abs() * weight).mean()
    opt.zero_grad(set_to_none=True)
    # Average first, then unscale: every GPU then sees the same overflow and skips the same steps.
    scaler.scale(loss).backward(); average_grads(); scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    scaler.step(opt); scaler.update()
    step += 1
    if step % 100 == 0: sync_observers()
    if step == FREEZE:
        sync_observers(); qat.observe(gm, False)
        print(f"step {step}: quantization ranges frozen", flush=True)
    with torch.no_grad():
        d = min(0.999, (1 + step) / (10 + step))
        for e, w in ema_pairs: e.mul_(d).add_(w.detach(), alpha=1 - d)
        for e, w in ema_bufs: e.copy_(w)
    qat.observe(ema, False)
    if step == 20 and not SMOKE:
        print(f"memory after 20 steps: {torch.cuda.max_memory_allocated() / 2**30:.1f} GB; {(time.time() - start) / 60:.0f} min in", flush=True)
    if step % 100 == 0:
        print(f"step {step} loss scale {scaler.get_scale():.0f}", end="; ", flush=True) if step % 1000 == 0 else None
        print(f"step {step} loss {loss.item():.4f} ({(time.time() - start) / 60:.0f} min)", flush=True)
    if step % EVAL == 0:
        if Lead:
            r = evaluate(ema)
            print(show(f"step {step} (EMA, 8-bit)", r), flush=True)
            if r["f_mean"] > best:  # the brief's target: agreement with the float model
                best = r["f_mean"]; save(ema, "student.pt"); print("saved", flush=True)
            save(gm, "last.pt")
        if World > 1: dist.barrier()
    if SMOKE and step >= 4:
        print("smoke: ok", flush=True); break
if Lead:
    r = evaluate(ema)
    print(show(f"final (EMA, 8-bit), steps {step}", r) + f"; best mean IoU vs float {max(best, r['f_mean']):.4f}", flush=True)
    if r["f_mean"] > best: save(ema, "student.pt"); print("saved", flush=True)
    if not os.path.exists(f"{OUT}/student.pt"): save(ema, "student.pt")  # never better than the start: keep the last anyway
if World > 1:
    dist.barrier(); dist.destroy_process_group()
