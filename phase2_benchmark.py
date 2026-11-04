# %% [markdown]
# # Phase 2: Why do lightweight TB models fail across datasets?
#
# Three experiments, each with an on/off switch (Section 1). **All of them save predictions and labels** (Phase 1 forgot the labels).
#
# | Switch | What it tests | Rough GPU time |
# |---|---|---|
# | `RUN_LODO` | **Leave-one-dataset-out**: train on two datasets, test on the third (plus Shenzhen→Montgomery). Is the problem TBX11K itself, or any change of site? | ~2.5 h |
# | `RUN_BASE` | **10 seeds** per model on the original TBX11K→external setup, each tested under 5 image conditions (original, border masked, corners masked, both, centre crop). Does masking the borders/corner markers change the predictions? | ~4.5 h |
# | `RUN_MITIGATION` | **Train with borders and corners masked** (3 models, 5 seeds). Does removing the suspected shortcut improve external AUC? | ~1 h |
#
# **How to run:** first with `QUICK = True` and all three switches on (about 10 minutes). Then `QUICK = False` and run the switches in separate
# *Save & Run All* commits (e.g. LODO + MITIGATION first, BASE second). Each commit writes its own `results_phase2/` folder.
# Settings: **GPU on, Internet on** (pretrained weights).

# %% [markdown]
# ## 1. Configuration

# %%
import os, re, io, sys, json, time, random, platform, warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms as T
import timm
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")

# ---------------- EDIT IF NEEDED ----------------
TBX_ROOT = "/kaggle/input/datasets/vbookshelf/tbx11k-simplified"
SZ_ROOT  = "/kaggle/input/datasets/kmader/pulmonary-chest-xray-abnormalities"
MC_ROOT  = "/kaggle/input/datasets/kmader/pulmonary-chest-xray-abnormalities"
# ------------------------------------------------

QUICK          = True     # True = smoke test (tiny, ~10 min). False = real experiments.
RUN_LODO       = True
RUN_BASE       = True
RUN_MITIGATION = True

PRETRAINED = True
IMG_CACHE, IMG_SIZE = 256, 224
BATCH, PATIENCE = 32, 4
EPOCHS = 2 if QUICK else 15
LR, WD = 3e-4, 1e-4
NEAR_DUP_BITS = 10
NUM_WORKERS = 2

LIGHT = ["MobileNetV3-L", "EfficientNet-B0", "EfficientNet-Lite0", "MobileViT-S", "ShuffleNetV2", "SqueezeNet1.1"]
BASE_MODELS = LIGHT
BASE_SEEDS = [0] if QUICK else list(range(10))
MITIG_MODELS = ["MobileNetV3-L", "ShuffleNetV2", "SqueezeNet1.1"]
MITIG_SEEDS = [0] if QUICK else list(range(5))
LODO_MODELS = ["MobileNetV3-L", "ShuffleNetV2", "SqueezeNet1.1", "EfficientNet-B0"]
LODO_SEEDS = [0] if QUICK else [0, 1, 2]
if QUICK:
    BASE_MODELS, MITIG_MODELS, LODO_MODELS = ["MobileNetV3-L", "ShuffleNetV2"], ["ShuffleNetV2"], ["ShuffleNetV2"]

# image conditions applied to the cached 256x256 grayscale arrays
ALL_CONDS = ["orig", "border", "corners", "border+corners", "centercrop80"]
MASK_TRAIN_COND = "border+corners"

MODELS = {
    "MobileNetV3-L":      ("timm", "mobilenetv3_large_100"),
    "EfficientNet-B0":    ("timm", "efficientnet_b0"),
    "EfficientNet-Lite0": ("timm", "efficientnet_lite0"),
    "MobileViT-S":        ("timm", "mobilevit_s"),
    "ShuffleNetV2":       ("tv",   "shufflenet_v2_x1_0"),
    "SqueezeNet1.1":      ("tv",   "squeezenet1_1"),
}
BASE = Path("/kaggle/working") if Path("/kaggle").exists() else Path(".")
OUT = BASE / "results_phase2"
(OUT / "preds").mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"
print("device:", DEVICE, "| QUICK =", QUICK)

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

# %% [markdown]
# ## 2. Data manifest, cache, leakage check, splits (same as Phase 1)

# %%
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp"}
def list_images(root):
    out = []
    for p in Path(root).rglob("*"):
        if p.suffix.lower() not in IMG_EXT: continue
        parts = [x.lower() for x in p.parts]
        if p.name.startswith(".") or "__macosx" in parts or any("mask" in x for x in parts): continue
        out.append(p)
    return sorted(out)

def build_nlm(root, pattern, name):
    rows = []
    for p in list_images(root):
        m = re.match(pattern, p.stem)
        if m: rows.append((str(p), int(m.group(1)), name))
    return pd.DataFrame(rows, columns=["path", "y", "source"])

def build_tbx(root):
    rows = []
    for p in list_images(root):
        if "test" in [x.lower() for x in p.parts]: continue
        s = p.stem.lower()
        if s.startswith("tb"): y = 1
        elif s.startswith("h") or s.startswith("s"): y = 0
        else: continue
        rows.append((str(p), y, "tbx11k"))
    return pd.DataFrame(rows, columns=["path", "y", "source"])

df_all = pd.concat([build_tbx(TBX_ROOT),
                    build_nlm(SZ_ROOT, r"CHNCXR_\d+_([01])", "shenzhen"),
                    build_nlm(MC_ROOT, r"MCUCXR_\d+_([01])", "montgomery")], ignore_index=True)
df_all["i"] = np.arange(len(df_all))
print(df_all.groupby("source").y.agg(["size", "sum"]))
assert df_all.source.nunique() == 3, "A dataset path is wrong."

def load_gray(path):
    return np.asarray(Image.open(path).convert("L").resize((IMG_CACHE, IMG_CACHE), Image.BILINEAR), dtype=np.uint8)
t0 = time.time()
with ThreadPoolExecutor(4) as ex:
    IMGS = np.stack(list(ex.map(load_gray, df_all.path)))
print(f"cached {IMGS.shape} in {time.time()-t0:.0f}s")

def ahash(arr, k=16):
    a = np.asarray(Image.fromarray(arr).resize((k, k), Image.BILINEAR), dtype=np.float32)
    return (a > a.mean()).ravel()
H = np.stack([ahash(a) for a in IMGS])
is_tbx = (df_all.source == "tbx11k").values
H_tbx, H_ext = H[is_tbx], H[~is_tbx]
min_d = np.full(len(H_tbx), 999)
for s in range(0, len(H_tbx), 500):
    min_d[s:s+500] = (H_tbx[s:s+500, None, :] != H_ext[None, :, :]).sum(-1).min(1)
dup = min_d <= NEAR_DUP_BITS
print(f"near-duplicates removed from TBX11K: {dup.sum()}")
df_tbx_clean = df_all[is_tbx].reset_index(drop=True)[~dup].reset_index(drop=True)

def split_tbx(df, seed=42):
    tr, tmp = train_test_split(df, test_size=0.30, stratify=df.y, random_state=seed)
    va, te = train_test_split(tmp, test_size=0.50, stratify=tmp.y, random_state=seed)
    return tr.reset_index(drop=True), va.reset_index(drop=True), te.reset_index(drop=True)
df_tr, df_va, df_te = split_tbx(df_tbx_clean)
df_sz_i = df_all[df_all.source == "shenzhen"].reset_index(drop=True)
df_mc_i = df_all[df_all.source == "montgomery"].reset_index(drop=True)
for n, d in [("train", df_tr), ("val", df_va), ("TBX-test", df_te), ("Shenzhen", df_sz_i), ("Montgomery", df_mc_i)]:
    print(f"{n:10s} n={len(d):5d} TB={int(d.y.sum()):4d}")

# %% [markdown]
# ## 3. Image conditions, dataset, models
# `apply_cond` modifies the cached image *before* the usual transforms:
# * `border`: outer 10% frame replaced by the image's mean grey level
# * `corners`: top-left and top-right boxes (16% high, 24% wide), where text markers and ID boxes usually sit
# * `centercrop80`: keep the central 80%, resized back (removes the periphery entirely)

# %%
def apply_cond(a, cond):
    if cond == "orig": return a
    Hh, Ww = a.shape
    if cond == "centercrop80":
        m = int(0.10 * Hh)
        return np.asarray(Image.fromarray(a[m:-m, m:-m]).resize((Ww, Hh), Image.BILINEAR))
    a = a.copy(); mu = int(a.mean())
    if "border" in cond:
        b = int(0.10 * Hh); a[:b, :] = mu; a[-b:, :] = mu; a[:, :b] = mu; a[:, -b:] = mu
    if "corners" in cond:
        h, w = int(0.16 * Hh), int(0.24 * Ww); a[:h, :w] = mu; a[:h, -w:] = mu
    return a

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
tf_train = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)),
                      T.RandomAffine(degrees=8, translate=(0.05, 0.05), scale=(0.95, 1.05)),
                      T.ColorJitter(brightness=0.15, contrast=0.15), T.ToTensor(), T.Normalize(MEAN, STD)])
tf_eval = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(), T.Normalize(MEAN, STD)])

class CXR(Dataset):
    def __init__(self, df, tf, cond="orig"):
        self.i, self.y, self.tf, self.cond = df.i.values, df.y.values.astype(np.float32), tf, cond
    def __len__(self): return len(self.i)
    def __getitem__(self, k):
        img = Image.fromarray(apply_cond(IMGS[self.i[k]], self.cond)).convert("RGB")
        return self.tf(img), self.y[k]

class TVNet(nn.Module):
    def __init__(self, features, dim, drop=0.2):
        super().__init__()
        self.features = features
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(drop), nn.Linear(dim, 1))
    def forward(self, x): return self.head(self.features(x))

def build_model(name):
    lib, arch = MODELS[name]
    if lib == "timm":
        return timm.create_model(arch, pretrained=PRETRAINED, num_classes=1)
    if arch == "shufflenet_v2_x1_0":
        w = torchvision.models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1 if PRETRAINED else None
        b = torchvision.models.shufflenet_v2_x1_0(weights=w)
        return TVNet(nn.Sequential(b.conv1, b.maxpool, b.stage2, b.stage3, b.stage4, b.conv5), 1024)
    w = torchvision.models.SqueezeNet1_1_Weights.IMAGENET1K_V1 if PRETRAINED else None
    return TVNet(torchvision.models.squeezenet1_1(weights=w).features, 512)

# %% [markdown]
# ## 4. Training, prediction, evaluation

# %%
@torch.no_grad()
def predict(model, df, cond="orig", bs=64):
    model.eval()
    dl = DataLoader(CXR(df, tf_eval, cond), batch_size=bs, shuffle=False, num_workers=NUM_WORKERS)
    out = []
    for x, _ in dl:
        with torch.autocast(device_type=DEVICE.type, enabled=USE_AMP):
            lo = model(x.to(DEVICE, non_blocking=True)).squeeze(-1)
        out.append(torch.sigmoid(lo.float()).cpu().numpy())
    return np.concatenate(out)

def train_one(name, seed, d_tr, d_va, train_cond):
    set_seed(seed)
    model = build_model(name).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - d_tr.y.mean()) / d_tr.y.mean(), device=DEVICE))
    g = torch.Generator(); g.manual_seed(seed)
    dl = DataLoader(CXR(d_tr, tf_train, train_cond), batch_size=BATCH, shuffle=True, drop_last=True,
                    num_workers=NUM_WORKERS, pin_memory=True, generator=g)
    best, best_state, bad, n_ep = -1, None, 0, 0
    for ep in range(EPOCHS):
        model.train()
        for x, y in dl:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=DEVICE.type, enabled=USE_AMP):
                loss = crit(model(x).squeeze(-1).float(), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sched.step(); n_ep = ep + 1
        va_auc = roc_auc_score(d_va.y, predict(model, d_va, train_cond))
        if va_auc > best:
            best, bad = va_auc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE: break
    model.load_state_dict(best_state)
    return model, best, n_ep

def youden_threshold(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])

def evaluate_run(model, d_va, test_sets, train_cond, eval_conds, meta):
    p_val = predict(model, d_va, train_cond)
    thr = youden_threshold(d_va.y.values, p_val)
    rows, store = [], {"val_y": d_va.y.values, "val_p": p_val, "thr": thr}
    for tname, tdf in test_sets.items():
        store[f"{tname}|y"] = tdf.y.values
        y = tdf.y.values
        for cond in eval_conds:
            p = predict(model, tdf, cond); store[f"{tname}|{cond}"] = p
            pred = p >= thr
            rows.append(dict(meta, test_set=tname, cond=cond, n=len(y), prevalence=y.mean(), thr=thr,
                             auc=roc_auc_score(y, p), sens=pred[y == 1].mean(), spec=(~pred[y == 0]).mean(),
                             flagged=pred.mean(), mean_prob=p.mean()))
    return rows, store

# %% [markdown]
# ## 5. Experiment setups
# * `tbx`: train on TBX11K, test on TBX11K-test, Shenzhen, Montgomery (the Phase 1 setting).
# * `lodo-*`: train on the union of the other datasets (85/15 train/val, stratified by dataset and label), test on the held-out dataset.

# %%
def sub(df, n):
    return df.sample(min(len(df), n), random_state=0).reset_index(drop=True) if QUICK else df.reset_index(drop=True)

POOLS = {"TBX11K": df_tbx_clean, "Shenzhen": df_sz_i, "Montgomery": df_mc_i}
LODO_FOLDS = {"lodo-hold-SZ (train TBX+MC)": ("Shenzhen", ["TBX11K", "Montgomery"]),
              "lodo-hold-MC (train TBX+SZ)": ("Montgomery", ["TBX11K", "Shenzhen"]),
              "lodo-hold-TBX (train SZ+MC)": ("TBX11K", ["Shenzhen", "Montgomery"]),
              "SZ->MC (train SZ only)":       ("Montgomery", ["Shenzhen"])}
_cache = {}
def get_setup(tag):
    if tag in _cache: return _cache[tag]
    if tag == "tbx":
        s = (sub(df_tr, 400), sub(df_va, 150), {"TBX11K-test": sub(df_te, 150), "Shenzhen": sub(df_sz_i, 200), "Montgomery": sub(df_mc_i, 138)})
    else:
        held, train_names = LODO_FOLDS[tag]
        pool = pd.concat([POOLS[n] for n in train_names], ignore_index=True)
        strat = pool.source + pool.y.astype(str)
        tr, va = train_test_split(pool, test_size=0.15, stratify=strat, random_state=42)
        s = (sub(tr, 400), sub(va, 150), {held: sub(POOLS[held], 300)})
    _cache[tag] = s
    return s

jobs = []
if RUN_LODO:
    for seed in LODO_SEEDS:
        for tag in LODO_FOLDS:
            for m in LODO_MODELS:
                jobs.append(dict(plan="lodo", tag=tag, model=m, seed=seed, train_cond="orig", eval_conds=["orig"]))
if RUN_MITIGATION:
    for seed in MITIG_SEEDS:
        for m in MITIG_MODELS:
            jobs.append(dict(plan="mitig", tag="tbx", model=m, seed=seed, train_cond=MASK_TRAIN_COND, eval_conds=[MASK_TRAIN_COND, "orig"]))
if RUN_BASE:
    for seed in BASE_SEEDS:
        for m in BASE_MODELS:
            jobs.append(dict(plan="base", tag="tbx", model=m, seed=seed, train_cond="orig", eval_conds=ALL_CONDS))
print(len(jobs), "training runs planned")

# %% [markdown]
# ## 6. Run everything (resumable)

# %%
def safe(s): return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
t_start = time.time(); n_done = 0
for k, job in enumerate(jobs, 1):
    plan, tag, name, seed = job["plan"], job["tag"], job["model"], job["seed"]
    f_csv = OUT / f"{plan}_{safe(tag)}_{safe(name)}_s{seed}.csv"
    if f_csv.exists():
        print(f"[{k}/{len(jobs)}] skip {plan}/{tag}/{name}/s{seed}"); continue
    d_tr, d_va, test_sets = get_setup(tag)
    t0 = time.time()
    model, best_val, n_ep = train_one(name, seed, d_tr, d_va, job["train_cond"])
    meta = dict(plan=plan, tag=tag, model=name, seed=seed, train_cond=job["train_cond"], val_auc=best_val, epochs=n_ep, n_train=len(d_tr))
    rows, store = evaluate_run(model, d_va, test_sets, job["train_cond"], job["eval_conds"], meta)
    np.savez_compressed(OUT / "preds" / f"{plan}_{safe(tag)}_{safe(name)}_s{seed}.npz", **store)
    pd.DataFrame(rows).to_csv(f_csv, index=False)   # written last = marks the run as finished
    n_done += 1
    el = time.time() - t_start
    print(f"[{k}/{len(jobs)}] {plan}/{tag}/{name}/s{seed}: {n_ep} ep, {time.time()-t0:.0f}s | "
          + " ".join(f"{r['test_set']}:{r['auc']:.3f}" for r in rows if r["cond"] == job["eval_conds"][0])
          + f" | elapsed {el/60:.0f} min, ETA {(el/n_done)*(len(jobs)-k)/60:.0f} min")
    del model; torch.cuda.empty_cache()
print("all runs finished")

# %% [markdown]
# ## 7. Summary tables (printed for you to paste back)

# %%
M = pd.concat([pd.read_csv(f) for f in sorted(OUT.glob("*.csv")) if f.name != "all_metrics.csv"], ignore_index=True)
M.to_csv(OUT / "all_metrics.csv", index=False)
def ms(s): return f"{s.mean():.3f} ± {s.std(ddof=1):.3f}" if len(s) > 1 else f"{s.mean():.3f}"
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)

if (M.plan == "base").any():
    b = M[M.plan == "base"]
    print("=== BASE: AUC (orig), mean ± SD over seeds ===")
    print(b[b.cond == "orig"].groupby(["model", "test_set"]).auc.agg(ms).unstack(), "\n")
    pv = b.pivot_table(index=["model", "seed", "test_set"], columns="cond", values="auc")
    for c in [c for c in ALL_CONDS if c != "orig" and c in pv.columns]: pv["d_" + c] = pv[c] - pv["orig"]
    print("=== BASE: mean change in AUC when the image is modified (negative = AUC drops) ===")
    print(pv.filter(like="d_").groupby(["test_set", "model"]).mean().round(3), "\n")
    fl = b.pivot_table(index=["model", "test_set"], columns="cond", values="flagged").round(3)
    print("=== BASE: fraction flagged TB at the fixed threshold ===\n", fl, "\n")
if (M.plan == "mitig").any():
    mi, ba = M[M.plan == "mitig"], M[M.plan == "base"]
    print("=== MITIGATION (trained with masked borders+corners, tested masked) vs BASE (same seeds, original images) ===")
    seeds = sorted(mi.seed.unique())
    a = mi[mi.cond == MASK_TRAIN_COND].groupby(["model", "test_set"]).auc.agg(ms)
    c = ba[(ba.cond == "orig") & ba.seed.isin(seeds) & ba.model.isin(mi.model.unique())].groupby(["model", "test_set"]).auc.agg(ms) if len(ba) else None
    print(pd.concat({"masked-train": a, "baseline": c}, axis=1) if c is not None else a, "\n")
if (M.plan == "lodo").any():
    lo = M[M.plan == "lodo"]
    print("=== LODO: AUC on the held-out dataset, mean ± SD over seeds ===")
    print(lo.groupby(["tag", "model"]).auc.agg(ms), "\n")
    print("=== LODO: fraction flagged TB / prevalence ===")
    print(lo.groupby(["tag", "model"])[["flagged", "prevalence"]].mean().round(3), "\n")
print("files in", OUT, ":", len(list(OUT.glob("*.csv"))), "csv,", len(list((OUT / "preds").glob("*.npz"))), "npz")

# %% [markdown]
# ## 8. What to send back
# Download the whole `results_phase2` folder from the notebook's **Output** tab (it holds only small CSV and NPZ files, no model weights),
# zip it if needed, and upload it to the chat. If the run is split across several commits, upload one zip per commit.
