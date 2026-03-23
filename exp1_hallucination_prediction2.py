"""
Experiment 1: Does D predict retrieval error better than model confidence alone?

Uses LOCAL Flickr30k from Kaggle (no HuggingFace download needed).
Edit DATASET_PATH below to match your machine.

Outputs in bni_results/:
  exp1_raw.csv
  exp1_roc.png
  exp1_summary.txt
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
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.calibration import calibration_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import brier_score_loss

# ── EDIT THIS PATH ───────────────────────────────────────────────────────────
DATASET_PATH = "/Users/aksha/.cache/kagglehub/datasets/hsankesara/flickr-image-dataset/versions/1/flickr30k_images"
# ─────────────────────────────────────────────────────────────────────────────

N_SAMPLES    = 1000
K_DISTRACTORS = 15
SEED         = 42
OUT_DIR      = "bni_results"

os.makedirs(OUT_DIR, exist_ok=True)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print("Device:", DEVICE)

# ── resolve paths ─────────────────────────────────────────────────────────────
captions_file = os.path.join(DATASET_PATH, "results.csv")
if not os.path.exists(captions_file):
    raise FileNotFoundError(f"Missing: {captions_file}")

for candidate in [
    os.path.join(DATASET_PATH, "flickr30k_images", "flickr30k_images"),
    os.path.join(DATASET_PATH, "flickr30k_images"),
]:
    if os.path.isdir(candidate) and any(f.endswith(".jpg") for f in os.listdir(candidate)):
        images_folder = candidate
        break
else:
    raise FileNotFoundError("Cannot find image folder under DATASET_PATH")

print("Images:", images_folder)

# ── load captions ─────────────────────────────────────────────────────────────
def load_captions(path):
    rows = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        next(f)
        for line in f:
            parts = line.strip().split("|", 2)
            if len(parts) != 3:
                continue
            name, num, cap = [p.strip() for p in parts]
            try:
                rows.append({"image_name": name, "comment_number": int(num), "caption": cap})
            except ValueError:
                continue
    return pd.DataFrame(rows)

df = load_captions(captions_file)
print(f"Loaded {len(df)} caption rows")

# one caption per image (comment_number == 0)
pairs = df[df["comment_number"] == 0][["image_name","caption"]].dropna().copy()
pairs["image_path"] = pairs["image_name"].apply(lambda x: os.path.join(images_folder, x))

def readable(p):
    if not os.path.exists(p): return False
    try:
        Image.open(p).verify(); return True
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
print("CLIP loaded")

@torch.no_grad()
def encode_image(path):
    img = Image.open(path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(DEVICE)
    f = model.encode_image(x)
    return (f / f.norm(dim=-1, keepdim=True)).squeeze(0).cpu()

@torch.no_grad()
def encode_texts(texts):
    tok = tokenizer(texts).to(DEVICE)
    f = model.encode_text(tok)
    return (f / f.norm(dim=-1, keepdim=True)).cpu()

# ── precompute all text embeddings ────────────────────────────────────────────
print("Encoding all captions...")
all_txt = encode_texts(pairs["caption"].tolist())   # (N, 512)

# ── main loop ─────────────────────────────────────────────────────────────────
print("Running experiment 1...")
records = []
for i in tqdm(range(N)):
    img_e = encode_image(pairs.loc[i, "image_path"])   # (512,)

    # D = cosine sim to true caption
    d = float(torch.dot(img_e, all_txt[i]))

    # retrieval: true caption vs K-1 distractors
    dist_idx = random.sample([x for x in range(N) if x != i], K_DISTRACTORS - 1)
    cand = torch.cat([all_txt[i].unsqueeze(0), all_txt[dist_idx]], dim=0)  # (K, 512)
    logits = logit_scale * (cand @ img_e)
    probs  = torch.softmax(logits, dim=0).numpy()
    confidence = float(probs[0])
    correct    = int(probs.argmax() == 0)

    records.append({"d": d, "confidence": confidence, "correct": correct})

df_r = pd.DataFrame(records)
df_r.to_csv(f"{OUT_DIR}/exp1_raw.csv", index=False)
print(f"Retrieval accuracy: {df_r['correct'].mean():.3f}")
print(f"Corr(D, confidence): {df_r[['d','confidence']].corr().iloc[0,1]:.3f}")

# ── logistic regression comparison ───────────────────────────────────────────
y = df_r["correct"].values
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

results = {}
for name, cols in [
    ("Confidence alone",     ["confidence"]),
    ("D alone",              ["d"]),
    ("D + Confidence",       ["d","confidence"]),
]:
    proba = cross_val_predict(LogisticRegression(max_iter=1000),
                              df_r[cols].values, y, cv=cv, method="predict_proba")[:,1]
    auc   = roc_auc_score(y, proba)
    brier = brier_score_loss(y, proba)
    results[name] = {"AUC": auc, "Brier": brier, "proba": proba}
    print(f"  {name:25s}  AUC={auc:.4f}  Brier={brier:.4f}")

# ── ROC plot ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
colors = ["#1f77b4","#ff7f0e","#2ca02c"]
ax = axes[0]
for (name, res), col in zip(results.items(), colors):
    fpr, tpr, _ = roc_curve(y, res["proba"])
    ax.plot(fpr, tpr, color=col, lw=2, label=f"{name} (AUC={res['AUC']:.3f})")
ax.plot([0,1],[0,1],"k--",lw=1)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC: predicting retrieval correctness")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

ax2 = axes[1]
corr = df_r[["d","confidence"]].corr().iloc[0,1]
ax2.scatter(df_r["d"], df_r["confidence"], alpha=0.3, s=10, color="#1f77b4")
ax2.set_xlabel("D (cosine sim to true caption)")
ax2.set_ylabel("Model confidence (softmax)")
ax2.set_title(f"Confidence vs D  (r={corr:.3f})")
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/exp1_roc.png", dpi=150)
print(f"Saved {OUT_DIR}/exp1_roc.png")

# ── summary ───────────────────────────────────────────────────────────────────
with open(f"{OUT_DIR}/exp1_summary.txt","w") as f:
    f.write("=== Experiment 1 Summary ===\n\n")
    f.write(f"N={N}  K={K_DISTRACTORS}\n")
    f.write(f"Retrieval accuracy: {df_r['correct'].mean():.3f}\n")
    f.write(f"Corr(D, confidence): {corr:.3f}\n\n")
    for name, res in results.items():
        f.write(f"  {name:25s}  AUC={res['AUC']:.4f}  Brier={res['Brier']:.4f}\n")
    d_auc = results["D alone"]["AUC"]
    c_auc = results["Confidence alone"]["AUC"]
    b_auc = results["D + Confidence"]["AUC"]
    f.write("\nInterpretation:\n")
    if b_auc > c_auc:
        f.write(f"  D + Confidence (AUC={b_auc:.4f}) > Confidence alone (AUC={c_auc:.4f})\n")
        f.write("  D carries information that confidence alone does not. Core claim supported.\n")
    else:
        f.write("  Combined did not outperform confidence alone. Revisit.\n")

print(f"Saved {OUT_DIR}/exp1_summary.txt")
print("\nExp 1 complete.")
