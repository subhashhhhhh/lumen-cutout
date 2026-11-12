"""Lumen Cutout v4, all-8-bit QAT: 640 backbone, 1024 mask, on the graph the phone runs.

Graph (convert/v4.py "two,slim,global4,nohalf"): the photo at 1024 and at 640 (two inputs), no
half-resolution backbone pass, a slim full-size last stage, Swin stage 4 as one global window.
Starts from round 2's snap_8000 (best on the phone NPU). Its stage-4 branch is dead (the decoder's
gate collapsed to ~1e-11 during the first distillation); here it comes back from the teacher's
weights, with a feature loss on the gated stage-4 output so the gate can't quietly die again.
Teacher: full BiRefNet-lite float at 1024. Loss: BCE + L1 with edge weight (1 + 4 x the teacher's
local range), plus 0.5 x the normalised stage-4 feature error. EMA, observers, freeze: as round 2.
The 640 copy is made on the GPU, with and without antialiasing at random (Lumen's resize varies).
Data: as round 2 (the author's photos 45%, DUTS 20%, DIS5K 15%, Supervisely 10%, AISegment 10%; no COCO).
Val: the same 200 held-out photos, scored against the teacher at 1024 (IoU, edge-band error).
Snapshots every 2000 steps from 4000; the pick is made on the phone NPU.
LUMEN_SMOKE=1: a few CPU steps at 256/160 on local photos.
"""
import glob, os, random, subprocess, sys, time, copy, urllib.request
SMOKE = os.environ.get("LUMEN_SMOKE") == "1"
OUT = os.environ.get("LUMEN_OUT", "/kaggle/working")
NPB = os.environ.get("LUMEN_NPB", "/tmp/npb")
START_URL = "https://github.com/subhashhhhhh/lumen-cutout/releases/download/inputs/birefnet_qat2_snap_8000.pt"
if "LOCAL_RANK" not in os.environ and not SMOKE:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm", "einops", "kornia"], check=False)
    if not os.path.isdir(NPB):
        subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/subhashhhhhh/lumen-cutout", NPB], check=True)
    if not os.path.exists("/tmp/start.pt"):
        urllib.request.urlretrieve(START_URL, "/tmp/start.pt")
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

SIDE, BBS = (256, 160) if SMOKE else (1024, 640)
OPTS = os.environ.get("LUMEN_OPTS", "two,slim,global4,nohalf")
B = 1 if SMOKE else int(os.environ.get("LUMEN_BATCH", "1"))
ACC = 1 if SMOKE else int(os.environ.get("LUMEN_ACC", "2"))
DILATE = 2
LR = {"new": 2e-4, "revived": 5e-5, "decoder": 2e-5, "bb": 2e-6}
FEAT = 0.5
FREEZE = int(os.environ.get("LUMEN_FREEZE", "3000"))
EVAL = 2 if SMOKE else 1000
Budget = float(os.environ.get("LUMEN_MINUTES", "540")) * 60
MODEL = os.environ.get("LUMEN_MODEL", "ZhengPeng7/BiRefNet_lite")
start = time.time()

base = AutoModelForImageSegmentation.from_pretrained(MODEL, trust_remote_code=True).float().eval()
teacher = qat.Wrapped(copy.deepcopy(base)).to(dev).eval()
for p in teacher.parameters(): p.requires_grad_(False)
TF = {}
teacher.net.decoder.decoder_block4.register_forward_hook(lambda m, i, o: TF.__setitem__("p4", o))
teacher.net.decoder.gdt_convs_attn_4.register_forward_hook(lambda m, i, o: TF.__setitem__("gate", o))
start_pt = os.environ.get("LUMEN_START", "/tmp/start.pt")
sd = torch.load(start_pt, map_location="cpu")
print("starting from", start_pt, len(sd), "tensors; options", OPTS, flush=True)
gm, ref_keys, counts = qat.build_v4(sd, DILATE, SIDE, BBS, B, OPTS, MODEL, device=dev, attention="heads", revive=base, aux=True)
del base
print("fake-quantized graph:", counts[0], "activation points,", counts[1], "weights;", counts[2], "stage-4 tensors from the teacher;",
      "batch", B, "x", ACC, "x", World, "GPUs; sides", SIDE, BBS, flush=True)
NEW = ("net.decoder.tail_squeeze.", "net.decoder.ipt_blk1.", "net.decoder.conv_out1.")
REV = tuple("net." + p for p in ("squeeze_module.", "decoder.decoder_block4.", "decoder.gdt_convs_4.", "decoder.gdt_convs_attn_4.",
                                 "decoder.ipt_blk5.", "bb.layers.3.", "bb.norm3."))
groups = {k: [] for k in LR}
for n, p in gm.named_parameters():
    p.requires_grad_(True)
    groups["new" if n.startswith(NEW) else "revived" if n.startswith(REV) else "bb" if n.startswith("net.bb.") else "decoder"].append(p)
print("trainable (M):", {k: round(sum(p.numel() for p in v) / 1e6, 2) for k, v in groups.items()}, flush=True)
opt = torch.optim.AdamW([{"params": v, "peak": LR[k]} for k, v in groups.items() if v], lr=1e-5, weight_decay=1e-4)
trainable = [p for v in groups.values() for p in v]

IMG = (".jpg", ".jpeg", ".png", ".webp")


def pick(pattern):
    return sorted(f for f in glob.glob(pattern, recursive=True) if f.lower().endswith(IMG) and "/._" not in f)


ROOT = os.environ.get("LUMEN_INPUT", "/kaggle/input")
SOURCES = {
    "own": (pick(f"{ROOT}/**/lumen-own-photos/**/*"), 0.45),
    "duts": (pick(f"{ROOT}/**/DUTS-TR-Image/*"), 0.20),
    "dis5k": (pick(f"{ROOT}/**/DIS-TR/im/*") + pick(f"{ROOT}/**/DIS-VD/im/*"), 0.15),
    "supervisely": (pick(f"{ROOT}/**/Supervisely_Person/**/img/**/*"), 0.10),
    "aisegment": (pick(f"{ROOT}/**/clip_img/**/*"), 0.10),
}
if SMOKE:
    SOURCES = {"own": (pick(os.environ.get("LUMEN_TRAIN", "/nonexistent/*")), 1.0)}
SOURCES = {k: v for k, v in SOURCES.items() if v[0]}
rv = random.Random(7)  # the same held-out val as round 2
own_all = SOURCES["own"][0]
val_own = own_all[::40][:100]
pub = [f for k, (fs, _) in SOURCES.items() if k != "own" for f in fs]
val_pub = rv.sample(pub, min(100, len(pub))) if pub else []
val_files = (val_own + val_pub) if not SMOKE else own_all[:2]
held = set(val_files)
SOURCES = {k: ([f for f in fs if f not in held], w) for k, (fs, w) in SOURCES.items()}
names = list(SOURCES); weights = [SOURCES[k][1] for k in names]
print("train:", ", ".join(f"{k} {len(SOURCES[k][0])} ({SOURCES[k][1]:.0%})" for k in names), "| val", len(val_files), flush=True)


def load(path, augment):
    img = Image.open(path); img.draft("RGB", (int(SIDE / 0.6), int(SIDE / 0.6))); img = img.convert("RGB")
    if augment:
        w, h = img.size; s = random.uniform(0.6, 1.0)
        cw, ch = int(w * s), int(h * s); x0, y0 = random.randint(0, w - cw), random.randint(0, h - ch)
        img = img.crop((x0, y0, x0 + cw, y0 + ch))
        if random.random() < 0.5: img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return torch.from_numpy(np.asarray(img.resize((SIDE, SIDE), Image.BILINEAR), np.float32).copy())


def small(x, antialias):
    """The backbone's copy, NHWC 0-255, as the app makes it on the GPU."""
    return F.interpolate(x.permute(0, 3, 1, 2), size=(BBS, BBS), mode="bilinear", align_corners=False,
                         antialias=antialias).clamp(0, 255).permute(0, 2, 3, 1).contiguous()


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
    if World > 1:
        for b in fq_buffers(gm): dist.broadcast(b, 0)


def run_teacher(x):
    with torch.no_grad(), autocast():
        t = teacher(x).float()
    tp4 = (TF["p4"] * TF["gate"].sigmoid()).float()
    return t, tp4


def feat_loss(sp4, tp4):
    tp4 = F.adaptive_avg_pool2d(tp4, sp4.shape[2:])
    return ((sp4.float() - tp4) ** 2).mean() / (tp4 ** 2).mean().clamp(min=1e-6)


def band(t, r=8):
    tt = (t > 0.5).float().permute(0, 3, 1, 2)
    k = 2 * r + 1
    return (F.max_pool2d(tt, k, 1, r) - (-F.max_pool2d(-tt, k, 1, r))).bool().permute(0, 2, 3, 1)


ValT = []  # the teacher's val masks (uint8, CPU), made once


@torch.no_grad()
def evaluate(model):
    ious, errs, gates = [], [], []
    for i in range(0, len(val_files) - B + 1, B):
        x = torch.stack([load(p, False) for p in val_files[i:i + B]]).to(dev)
        if len(ValT) * B <= i:
            t, tp4 = run_teacher(x); ValT.append(((t * 255).round().byte().cpu(), tp4.abs().mean().item()))
        t = ValT[i // B][0].to(dev).float() / 255
        with autocast():
            s, sp4 = model(x, small(x, True))
        s = s.float()
        bd = band(t)
        for tt, ss, bb in zip(t, s, bd):
            a, b = tt > 0.5, ss > 0.5
            if a.float().mean() >= 0.01:
                ious.append(((a & b).sum() / (a | b).sum().clamp(min=1)).item())
            if bb.any(): errs.append((tt - ss).abs()[bb].mean().item())
        gates.append(sp4.float().abs().mean().item() / max(ValT[i // B][1], 1e-9))
    ious = np.array(ious or [float("nan")])
    return dict(iou_med=float(np.median(ious)), iou_mean=float(ious.mean()), iou_min=float(ious.min()),
                below=int((ious < 0.9).sum()), n=len(ious), edge=float(np.mean(errs)), s4=float(np.mean(gates)))


def show(tag, r):
    return (f"{tag}: vs teacher@{SIDE}: IoU median {r['iou_med']:.4f} mean {r['iou_mean']:.4f} worst {r['iou_min']:.4f} "
            f"below 0.9: {r['below']}/{r['n']} | edge-band error {r['edge']:.4f} | stage-4 strength vs teacher {r['s4']:.3f}")


def save(model, name):
    w, missing = qat.float_weights(model, ref_keys)
    assert not missing, sorted(missing)[:5]
    torch.save({k: v.half().cpu() for k, v in w.items()}, f"{OUT}/{name}")


# Float (no rounding), then calibrated 8-bit, before training.
qat.observe(gm, False); qat.quantize(gm, False)
if Lead: print(show("start, float (no rounding)", evaluate(gm)), flush=True)
qat.observe(gm, True); qat.averaging(gm, 0.05)
with torch.no_grad():
    for i in range(1 if SMOKE else 200 // (B * World)):
        x = next(loader).to(dev)
        with autocast(): gm(x, small(x, True))
sync_observers()
qat.quantize(gm, True); qat.averaging(gm, 0.01)
print(f"calibrated ({time.time() - start:.0f} s in)", flush=True)
ema = copy.deepcopy(gm); qat.observe(ema, False)
for p in ema.parameters(): p.requires_grad_(False)
ema_pairs = list(zip(ema.parameters(), gm.parameters()))
ema_bufs = list(zip(ema.buffers(), gm.buffers()))
best = -1.0
if Lead:
    qat.observe(gm, False); r0 = evaluate(gm); qat.observe(gm, True)
    print(show("8-bit before training", r0), flush=True)
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


step, dead = 0, 0
while (now := elapsed()) < Budget:
    cos = 0.5 * (1 + np.cos(np.pi * min(1.0, now / Budget)))
    for g in opt.param_groups:
        g["lr"] = g["peak"] * (0.05 + 0.95 * cos) * min(1.0, (step + 1) / 300)
    opt.zero_grad(set_to_none=True)
    for _ in range(ACC):
        x = next(loader).to(dev, non_blocking=True)
        t, tp4 = run_teacher(x)
        with autocast():
            s, sp4 = gm(x, small(x, random.random() < 0.5))
        s = torch.nan_to_num(s.float(), nan=0.5).clamp(1e-6, 1 - 1e-6)
        tn = t.permute(0, 3, 1, 2)
        edge = (F.max_pool2d(tn, 11, 1, 5) + F.max_pool2d(-tn, 11, 1, 5)).clamp(0, 1).permute(0, 2, 3, 1)
        weight = 1 + 4 * edge
        mask_loss = (F.binary_cross_entropy(s, t, reduction="none") * weight).mean() + ((s - t).abs() * weight).mean()
        fl = feat_loss(sp4, tp4)
        loss = (mask_loss + FEAT * fl) / ACC
        scaler.scale(loss).backward()
    average_grads(); scaler.unscale_(opt)
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
    if step % 100 == 0 or SMOKE:
        strength = sp4.float().abs().mean().item() / max(tp4.abs().mean().item(), 1e-9)
        dead = dead + 1 if strength < 1e-3 else 0
        print(f"step {step} loss {mask_loss.item():.4f} feat {fl.item():.3f} stage-4 strength {strength:.3f}"
              f"{' GATE DYING' if dead else ''} ({(time.time() - start) / 60:.0f} min)", flush=True)
    if step % EVAL == 0:
        if Lead:
            r = evaluate(ema)
            print(show(f"step {step} (EMA, 8-bit)", r), flush=True)
            if r["iou_mean"] > best:
                best = r["iou_mean"]; save(ema, "student.pt"); print("saved", flush=True)
            save(gm, "last.pt")
            if step >= 4000 and step % 2000 == 0: save(ema, f"snap_{step}.pt")
        if World > 1: dist.barrier()
    if SMOKE and step >= 4:
        print("smoke: ok", flush=True); break
if Lead:
    r = evaluate(ema)
    print(show(f"final (EMA, 8-bit), steps {step}", r), flush=True)
    if r["iou_mean"] > best: save(ema, "student.pt"); print("saved", flush=True)
    if not os.path.exists(f"{OUT}/student.pt"): save(ema, "student.pt")
if World > 1:
    dist.barrier(); dist.destroy_process_group()
