"""
Experiment 3: Calibration gap — does adding D reduce ECE?

Uses LOCAL Flickr30k from Kaggle. Edit DATASET_PATH below.

Outputs in bni_results/:
  exp3_ece.csv
  exp3_calibration.png
  exp3_summary.txt
"""

import os, random
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import open_clip
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

# ── EDIT THIS PATH ───────────────────────────────────────────────────────────
DATASET_PATH = "/Users/aksha/.cache/kagglehub/datasets/hsankesara/flickr-image-dataset/versions/1/flickr30k_images"
# ─────────────────────────────────────────────────────────────────────────────

N_SAMPLES     = 1000
K_DISTRACTORS = 15
N_BINS        = 10
SEED          = 42
OUT_DIR       = "bni_results"

os.makedirs(OUT_DIR, exist_ok=True)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print("Device:", DEVICE)

# ── resolve paths ─────────────────────────────────────────────────────────────
captions_file = os.path.join(DATASET_PATH, "results.csv")
for candidate in [
    os.path.join(DATASET_PATH, "flickr30k_images", "flickr30k_images"),
    os.path.join(DATASET_PATH, "flickr30k_images"),
]:
    if os.path.isdir(candidate) and any(f.endswith(".jpg") for f in os.listdir(candidate)):
        images_folder = candidate
        break
else:
    raise FileNotFoundError("Cannot find image folder under DATASET_PATH")

# ── load captions ─────────────────────────────────────────────────────────────
def load_captions(path):
    rows = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        next(f)
        for line in f:
            parts = line.strip().split("|", 2)
            if len(parts) != 3: continue
            name, num, cap = [p.strip() for p in parts]
            try: rows.append({"image_name": name, "comment_number": int(num), "caption": cap})
            except ValueError: continue
    return pd.DataFrame(rows)

df_all = load_captions(captions_file)
pairs  = df_all[df_all["comment_number"] == 0][["image_name","caption"]].dropna().copy()
pairs["image_path"] = pairs["image_name"].apply(
    lambda x: os.path.join(images_folder, x))

def readable(p):
    if not os.path.exists(p): return False
    try: Image.open(p).verify(); return True
    except: return False

pairs = pairs[pairs["image_path"].apply(readable)].reset_index(drop=True)
pairs = pairs.sample(min(N_SAMPLES, len(pairs)), random_state=SEED).reset_index(drop=True)
N = len(pairs)
print(f"Using {N} samples")

# ── load CLIP ─────────────────────────────────────────────────────────────────
model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
tokenizer = open_clip.get_tokenizer("ViT-B-32")
model = model.to(DEVICE).eval()
logit_scale = model.logit_scale.exp().detach().cpu().item()

@torch.no_grad()
def enc_img(path):
    img = Image.open(path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(DEVICE)
    f = model.encode_image(x)
    return (f / f.norm(dim=-1, keepdim=True)).squeeze(0).cpu()

@torch.no_grad()
def enc_txt(texts):
    tok = tokenizer(texts).to(DEVICE)
    f = model.encode_text(tok)
    return (f / f.norm(dim=-1, keepdim=True)).cpu()

print("Encoding all captions...")
all_txt = enc_txt(pairs["caption"].tolist())

# ── main loop ─────────────────────────────────────────────────────────────────
print("Computing D and confidence...")
records = []
for i in tqdm(range(N)):
    img_e = enc_img(pairs.loc[i, "image_path"])

    d = float(torch.dot(img_e, all_txt[i]))

    dist_idx = random.sample([x for x in range(N) if x != i], K_DISTRACTORS - 1)
    cand     = torch.cat([all_txt[i].unsqueeze(0), all_txt[dist_idx]], dim=0)
    logits   = logit_scale * (cand @ img_e)
    probs    = torch.softmax(logits, dim=0).numpy()
    confidence = float(probs[0])
    correct    = int(probs.argmax() == 0)

    records.append({"d": d, "confidence": confidence, "correct": correct})

df_r = pd.DataFrame(records)

# ── normalise D → [0,1] ───────────────────────────────────────────────────────
d_min, d_max = df_r["d"].min(), df_r["d"].max()
df_r["d_norm"] = (df_r["d"] - d_min) / (d_max - d_min + 1e-9)
df_r["conf_d_blend"] = 0.5 * df_r["confidence"] + 0.5 * df_r["d_norm"]

# ── logistic regression ───────────────────────────────────────────────────────
y  = df_r["correct"].values
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

proba_conf = cross_val_predict(LogisticRegression(max_iter=1000),
    df_r[["confidence"]].values, y, cv=cv, method="predict_proba")[:,1]
proba_both = cross_val_predict(LogisticRegression(max_iter=1000),
    df_r[["confidence","d"]].values, y, cv=cv, method="predict_proba")[:,1]

# ── ECE ───────────────────────────────────────────────────────────────────────
def ece(probs, labels, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    err  = 0.0
    for b in range(n_bins):
        m = (probs >= bins[b]) & (probs < bins[b+1])
        if m.sum() == 0: continue
        err += (m.sum() / len(probs)) * abs(probs[m].mean() - labels[m].mean())
    return err

ece_raw   = ece(df_r["confidence"].values, y, N_BINS)
ece_blend = ece(df_r["conf_d_blend"].values, y, N_BINS)
ece_lr_c  = ece(proba_conf, y, N_BINS)
ece_lr    = ece(proba_both, y, N_BINS)

print(f"\nECE — Confidence (raw):       {ece_raw:.4f}")
print(f"ECE — Confidence (LR):        {ece_lr_c:.4f}")
print(f"ECE — D+Conf blend:           {ece_blend:.4f}")
print(f"ECE — D+Conf (LR):            {ece_lr:.4f}")
improvement = (ece_raw - ece_lr) / ece_raw * 100
print(f"Improvement (raw→LR D+Conf):  {improvement:.1f}%")

# ── plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
for label, probs, color in [
    (f"Conf alone (ECE={ece_raw:.3f})",   df_r["confidence"].values, "#1f77b4"),
    (f"D+Conf blend (ECE={ece_blend:.3f})",df_r["conf_d_blend"].values,"#ff7f0e"),
    (f"D+Conf LR (ECE={ece_lr:.3f})",      proba_both,                "#2ca02c"),
]:
    fp, mp = calibration_curve(y, probs, n_bins=N_BINS, strategy="uniform")
    ax.plot(mp, fp, "o-", label=label, lw=2, ms=6, color=color)
ax.plot([0,1],[0,1],"k--",lw=1,label="Perfect calibration")
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Fraction of positives")
ax.set_title("Reliability diagram")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax2 = axes[1]
vals   = [ece_raw, ece_lr_c, ece_blend, ece_lr]
labels = ["Conf\n(raw)","Conf\n(LR)","D+Conf\nblend","D+Conf\n(LR)"]
colors = ["#1f77b4","#aec7e8","#ff7f0e","#2ca02c"]
bars = ax2.bar(labels, vals, color=colors, edgecolor="white", width=0.5)
for bar, val in zip(bars, vals):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001,
             f"{val:.4f}", ha="center", va="bottom", fontsize=10)
ax2.set_ylabel("ECE (lower = better calibrated)")
ax2.set_title("Expected Calibration Error comparison")
ax2.grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/exp3_calibration.png", dpi=150)
print(f"Saved {OUT_DIR}/exp3_calibration.png")

# ── save CSV ──────────────────────────────────────────────────────────────────
pd.DataFrame({
    "method": labels, "ECE": vals
}).to_csv(f"{OUT_DIR}/exp3_ece.csv", index=False)

# ── summary ───────────────────────────────────────────────────────────────────
with open(f"{OUT_DIR}/exp3_summary.txt","w") as f:
    f.write("=== Experiment 3 Summary ===\n\n")
    f.write(f"N={N}  K={K_DISTRACTORS}  bins={N_BINS}\n\n")
    f.write(f"ECE Confidence (raw):    {ece_raw:.4f}\n")
    f.write(f"ECE Confidence (LR):     {ece_lr_c:.4f}\n")
    f.write(f"ECE D+Conf blend:        {ece_blend:.4f}\n")
    f.write(f"ECE D+Conf (LR):         {ece_lr:.4f}\n")
    f.write(f"Improvement:             {improvement:.1f}%\n\n")
    if ece_lr < ece_raw:
        f.write("Adding D IMPROVES calibration.\n")
        f.write("D carries information that model confidence alone does not.\n")
    else:
        f.write("D did not improve calibration here.\n")

print(f"Saved {OUT_DIR}/exp3_summary.txt")
print("\nExp 3 complete.")
