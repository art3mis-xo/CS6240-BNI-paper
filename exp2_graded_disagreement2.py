"""
Experiment 2: Graded disagreement across 6 mismatch levels.

Uses LOCAL Flickr30k from Kaggle (no HuggingFace needed).
Edit DATASET_PATH below.

Levels:
  0 — Matched:          true caption
  1 — Alt caption:      different caption for same image (Flickr has 5 per image)
  2 — Shared nouns:     caption from different image sharing 2+ content words
  3 — No shared nouns:  caption from different image with 0 shared content words
  4 — Fully random:     random caption
  5 — Null:             empty string (missing modality)

Outputs in bni_results/:
  exp2_levels.csv
  exp2_graded.png
  exp2_boxplot.png
  exp2_summary.txt
"""

import os, random, re
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import open_clip
from scipy.stats import spearmanr

# ── EDIT THIS PATH ───────────────────────────────────────────────────────────
DATASET_PATH = "/Users/aksha/.cache/kagglehub/datasets/hsankesara/flickr-image-dataset/versions/1/flickr30k_images"
# ─────────────────────────────────────────────────────────────────────────────

N_SAMPLES    = 500
K_DISTRACTORS = 15
SEED         = 42
OUT_DIR      = "bni_results"

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

print("Images:", images_folder)

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
print(f"Loaded {len(df_all)} caption rows")

# group: image_name -> list of all captions (up to 5)
cap_map = df_all.groupby("image_name")["caption"].apply(list).to_dict()

# base pairs: one per image, comment_number == 0
pairs = df_all[df_all["comment_number"] == 0][["image_name","caption"]].dropna().copy()
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

# precompute true caption embeddings
print("Encoding true captions...")
all_txt = enc_txt(pairs["caption"].tolist())   # (N, 512)

# null embedding
null_emb = enc_txt([""])[0]   # (512,)

# ── noun helper ───────────────────────────────────────────────────────────────
STOP = {"a","an","the","is","are","was","were","on","in","at","to","of","and",
        "or","with","by","for","it","its","this","that","from","as","into","his",
        "her","their","there","some","two","three","four","while","near","next"}

def nouns(text):
    return set(w for w in re.findall(r"\b[a-z]+\b", text.lower())
               if w not in STOP and len(w) > 3)

# precompute noun sets
pair_nouns = [nouns(pairs.loc[i,"caption"]) for i in range(N)]

# ── retrieve alt caption for same image ──────────────────────────────────────
def get_alt_cap(image_name, true_cap):
    caps = cap_map.get(image_name, [true_cap])
    alts = [c for c in caps if c != true_cap]
    return random.choice(alts) if alts else true_cap

# ── retrieval confidence ──────────────────────────────────────────────────────
def retrieval_conf(img_e, txt_e):
    dist_idx = random.sample(range(N), K_DISTRACTORS - 1)
    cand = torch.cat([txt_e.unsqueeze(0), all_txt[dist_idx]], dim=0)
    logits = logit_scale * (cand @ img_e)
    probs = torch.softmax(logits, dim=0).numpy()
    return float(probs[0])

# ── main loop ─────────────────────────────────────────────────────────────────
print("Building graded disagreement levels...")
records = []

for i in tqdm(range(N)):
    img_e      = enc_img(pairs.loc[i, "image_path"])
    true_cap   = pairs.loc[i, "caption"]
    true_nouns = pair_nouns[i]
    img_name   = pairs.loc[i, "image_name"]

    # level 0 — matched
    e0 = all_txt[i]
    d0 = float(torch.dot(img_e, e0))

    # level 1 — alt caption same image
    alt = get_alt_cap(img_name, true_cap)
    e1 = enc_txt([alt])[0]
    d1 = float(torch.dot(img_e, e1))

    # level 2 — shared nouns (>=2) from different image
    shared_pool = [pairs.loc[j,"caption"] for j in range(N)
                   if j != i and len(pair_nouns[j] & true_nouns) >= 2]
    e2_txt = random.choice(shared_pool) if shared_pool else true_cap
    e2 = enc_txt([e2_txt])[0]
    d2 = float(torch.dot(img_e, e2))

    # level 3 — no shared nouns from different image
    diff_pool = [pairs.loc[j,"caption"] for j in range(N)
                 if j != i and len(pair_nouns[j] & true_nouns) == 0]
    e3_txt = random.choice(diff_pool) if diff_pool else pairs.loc[
        random.choice([x for x in range(N) if x != i]), "caption"]
    e3 = enc_txt([e3_txt])[0]
    d3 = float(torch.dot(img_e, e3))

    # level 4 — fully random
    j4 = random.choice([x for x in range(N) if x != i])
    e4 = all_txt[j4]
    d4 = float(torch.dot(img_e, e4))

    # level 5 — null
    d5 = float(torch.dot(img_e, null_emb))

    for level, (d_val, emb) in enumerate([
        (d0,e0),(d1,e1),(d2,e2),(d3,e3),(d4,e4),(d5,null_emb)
    ]):
        conf = retrieval_conf(img_e, emb)
        records.append({"sample_i": i, "level": level, "d": d_val, "confidence": conf})

df_r = pd.DataFrame(records)
df_r.to_csv(f"{OUT_DIR}/exp2_levels.csv", index=False)

# ── stats ─────────────────────────────────────────────────────────────────────
sp_d,    p_d    = spearmanr(df_r["level"], df_r["d"])
sp_conf, p_conf = spearmanr(df_r["level"], df_r["confidence"])
print(f"Spearman(level, D):          rho={sp_d:.3f}  p={p_d:.4f}")
print(f"Spearman(level, confidence): rho={sp_conf:.3f}  p={p_conf:.4f}")

level_means = df_r.groupby("level")[["d","confidence"]].mean().reset_index()
level_labels = ["L0\nMatched","L1\nAlt cap","L2\nShared\nnouns",
                "L3\nNo shared\nnouns","L4\nRandom","L5\nNull"]

# ── plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.plot(level_means["level"], level_means["d"], "o-", color="#1f77b4", lw=2, ms=7)
ax.set_xticks(range(6)); ax.set_xticklabels(level_labels, fontsize=8)
ax.set_ylabel("Mean D (cosine similarity)")
ax.set_title(f"D across mismatch levels (rho={sp_d:.3f}, p={p_d:.3g})")
ax.grid(alpha=0.3)

ax2 = axes[1]
ax2.plot(level_means["level"], level_means["confidence"], "o-", color="#ff7f0e", lw=2, ms=7)
ax2.set_xticks(range(6)); ax2.set_xticklabels(level_labels, fontsize=8)
ax2.set_ylabel("Mean confidence")
ax2.set_title(f"Confidence across mismatch levels (rho={sp_conf:.3f}, p={p_conf:.3g})")
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/exp2_graded.png", dpi=150, bbox_inches="tight")
print(f"Saved {OUT_DIR}/exp2_graded.png")

fig2, ax3 = plt.subplots(figsize=(10, 5))
data_by_level = [df_r[df_r["level"]==l]["d"].values for l in range(6)]
bp = ax3.boxplot(data_by_level, patch_artist=True,
                 medianprops=dict(color="black", lw=2))
for patch in bp["boxes"]: patch.set_facecolor("#a8c8e8")
ax3.set_xticklabels(level_labels, fontsize=8)
ax3.set_ylabel("D (cosine similarity)")
ax3.set_title("Distribution of D at each mismatch level")
ax3.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/exp2_boxplot.png", dpi=150)
print(f"Saved {OUT_DIR}/exp2_boxplot.png")

# ── summary ───────────────────────────────────────────────────────────────────
with open(f"{OUT_DIR}/exp2_summary.txt","w") as f:
    f.write("=== Experiment 2 Summary ===\n\n")
    f.write(f"N={N}\nLevels: 0=matched,1=alt cap,2=shared nouns,"
            f"3=no shared nouns,4=random,5=null\n\n")
    for _, row in level_means.iterrows():
        f.write(f"  Level {int(row.level)}: D={row.d:.4f}  conf={row.confidence:.4f}\n")
    f.write(f"\nSpearman(level,D):          rho={sp_d:.3f} p={p_d:.4f}\n")
    f.write(f"Spearman(level,confidence): rho={sp_conf:.3f} p={p_conf:.4f}\n")
    if abs(sp_d) > abs(sp_conf):
        f.write("\nD tracks mismatch level MORE strongly than confidence. Core claim supported.\n")
    else:
        f.write("\nConfidence tracks mismatch level more strongly than D. Revisit framing.\n")

print(f"Saved {OUT_DIR}/exp2_summary.txt")
print("\nExp 2 complete.")
