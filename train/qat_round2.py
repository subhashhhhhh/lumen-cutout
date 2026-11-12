"""Lumen Cutout, all-8-bit, round 2: QAT on the per-head attention graph (141 ms on the SM7450 NPU),
on a non-COCO mix, starting from round 1's best (median IoU 0.996 vs float v3 on the phone NPU).

Changes against round 1, nothing else:
- Graph: per-head window attention (birefnet_npu_rewrites "heads"), the layout the phone runs.
- Data, sampled per source by weight: the author's own photos (not released; the 150 test photos
  removed by content hash) 45%, DUTS-TR 20%, DIS5K TR+VD 15%,
  Supervisely Person 10%, AISegment portraits 10%. No COCO.
- Val: 200 held-out photos from the same mix (100 own, 100 public), never trained on.
- Reference "float v3" masks come from the actual v3 weights (not the starting point).
- Snapshots every 2000 steps after 4000 (snap_<step>.pt) besides the best; the final pick is made on
  the phone NPU, not here (simulated 8-bit and the phone disagree).
Teacher, loss (BCE + L1, edge weight 1 + 4 x the teacher's 7x7 local range), EMA, observers: as round 1.
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
prev = sorted(glob.glob(os.environ.get("LUMEN_RESUME", "/kaggle/input/**/qat1_student.pt"), recursive=True))
v3p = sorted(glob.glob(os.environ.get("LUMEN_V3", "/kaggle/input/**/v3_student.pt"), recursive=True))
assert prev and v3p, ("need qat1_student.pt and v3_student.pt in the inputs", prev, v3p)
sd = torch.load(prev[0], map_location="cpu")
print("resuming from", prev[0], len(sd), "tensors; float reference", v3p[0], flush=True)
ATT = os.environ.get("LUMEN_ATTENTION", "heads")
gm, ref_keys, counts = qat.build(sd, DILATE, SIDE, B, MODEL, device=dev, attention=ATT)
print("attention layout:", ATT, flush=True)
print("fake-quantized graph:", counts[0], "activation points,", counts[1], "weights; batch", B, "x", World, "GPUs; side", SIDE, flush=True)
bb = [p for n, p in gm.named_parameters() if n.startswith("net.bb.")]
dec = [p for n, p in gm.named_parameters() if not n.startswith("net.bb.")]
for p in bb + dec: p.requires_grad_(True)
print("trainable:", sum(p.numel() for p in dec) / 1e6, "M decoder side,", sum(p.numel() for p in bb) / 1e6, "M Swin", flush=True)
opt = torch.optim.AdamW([{"params": dec, "peak": LR}, {"params": bb, "peak": BB_LR}], lr=LR, weight_decay=1e-4)
trainable = dec + bb

IMG = (".jpg", ".jpeg", ".png", ".webp")


def pick(pattern):
    return sorted(f for f in glob.glob(pattern, recursive=True) if f.lower().endswith(IMG) and "/._" not in f)


ROOT = os.environ.get("LUMEN_INPUT", "/kaggle/input")
SOURCES = {  # name: (files, weight)
    "own": (pick(f"{ROOT}/**/lumen-own-photos/**/*"), 0.45),
    "duts": (pick(f"{ROOT}/**/DUTS-TR-Image/*"), 0.20),
    "dis5k": (pick(f"{ROOT}/**/DIS-TR/im/*") + pick(f"{ROOT}/**/DIS-VD/im/*"), 0.15),
    "supervisely": ([f for f in pick(f"{ROOT}/**/Supervisely_Person/**/img/**/*")], 0.10),
    "aisegment": (pick(f"{ROOT}/**/clip_img/**/*"), 0.10),
}
if SMOKE:
    local = pick(os.environ.get("LUMEN_TRAIN", "/nonexistent/*"))
    SOURCES = {"own": (local, 1.0)}
SOURCES = {k: v for k, v in SOURCES.items() if v[0]}
# Held-out val: every 40th own photo (up to 100) and an even spread of 100 public ones; removed from training.
rv = random.Random(7)
own_all = SOURCES["own"][0]
val_own = own_all[::40][:100]
pub = [f for k, (fs, _) in SOURCES.items() if k != "own" for f in fs]
val_pub = rv.sample(pub, min(100, len(pub))) if pub else []
val_files = (val_own + val_pub) if not SMOKE else own_all[:2]
held = set(val_files)
SOURCES = {k: ([f for f in fs if f not in held], w) for k, (fs, w) in SOURCES.items()}
names = list(SOURCES); weights = [SOURCES[k][1] for k in names]
val_files = val_files[:len(val_files) // B * B]
print("train:", ", ".join(f"{k} {len(SOURCES[k][0])} ({SOURCES[k][1]:.0%})" for k in names), "| val", len(val_files), flush=True)


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
            yield torch.stack([load(rng.choice(SOURCES[rng.choices(names, weights)[0]][0]), True) for _ in range(B)])


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
if Lead:
    v3, _, _ = qat.build(torch.load(v3p[0], map_location="cpu"), DILATE, SIDE, B, MODEL, device=dev, attention=ATT)
    qat.observe(v3, False); qat.quantize(v3, False)
    print(show("float v3 (no rounding)", evaluate(v3, keep_float=True)), flush=True)
    del v3; torch.cuda.empty_cache() if not SMOKE else None
qat.observe(gm, False); qat.quantize(gm, False)
if Lead: print(show("round 1, float (no rounding)", evaluate(gm)), flush=True)
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
            if step >= 4000 and step % 2000 == 0: save(ema, f"snap_{step}.pt")
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
