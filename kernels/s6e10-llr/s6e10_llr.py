# S6E10 | generator-surrogate LLR features (distilgpt2 fine-tuned GReaT-style on the ORIGINAL rows only; label-free)
# Kaggle 2xT4 run of src/train_llr.py from github.com/TranBaDat2607/Predicting-Airline-Satisfaction
import sys, subprocess
print(subprocess.run("nvidia-smi; nproc; free -g", shell=True, capture_output=True, text=True).stdout)
sys.argv = [sys.argv[0], "--orig", "auto", "--comp", "auto", "--out", "/kaggle/working",
            "--model", "distilgpt2", "--epochs", "5", "--bs", "64", "--lr", "5e-5", "--score-bs", "256", "--hours", "5"]
"""Generator-surrogate LLR features (label-free): distilgpt2 fine-tuned GReaT-style on the ORIGINAL rows only.

The S6 Playground rows look like samples from a language model fine-tuned on the original data (every value
is copied from the original's vocabulary, numbers behave like GPT-2 BPE tokens, the marginals are sharpened as
by temperature sampling). A similar LM trained on the original alone approximates the generator's own label
probability without seeing any competition label (idea: S6E9 2nd place, Team Alicia).

Training: every row of the original (129,880; `Inflight service` dropped to match the competition columns) is
written as "col is value, col is value, ..." in a fresh random column order each epoch; plain LM loss.
Scoring, for every competition row (train + test):
  llr_last  = log p(" satisfied" | features, "satisfaction is") - log p(" neutral or dissatisfied" | same)
  llr_first = log p(features | "satisfaction is satisfied, ") - log p(features | "satisfaction is neutral ...")
one fixed random column order each (seeded). No competition label is used anywhere, so both columns are
ordinary label-free features (OOF = train rows, test = test rows).

    python src/train_llr.py --orig data/ext/airline-passenger-satisfaction --comp data --out artifacts/llr
    python src/train_llr.py ... --tiny --max-orig 3000 --max-comp 2000 --epochs 1     # CPU smoke test
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--orig", required=True)
ap.add_argument("--comp", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--model", default="distilgpt2")
ap.add_argument("--epochs", type=float, default=5)
ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--score-bs", type=int, default=512)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--max-orig", type=int, default=0)
ap.add_argument("--max-comp", type=int, default=0)
ap.add_argument("--tiny", action="store_true", help="random tiny GPT-2 (smoke test, no download)")
ap.add_argument("--hours", type=float, default=11.0, help="stop training early to leave time for scoring")
a = ap.parse_args()
T0 = time.time()
if a.orig == "auto" or a.comp == "auto":  # Kaggle: find the mounted inputs (layouts differ between images)
    import glob
    hits = sorted(glob.glob("/kaggle/input/**/train.csv", recursive=True))
    print("inputs:", hits)
    if a.comp == "auto":
        a.comp = os.path.dirname([h for h in hits if "playground-series-s6e10" in h][0])
    if a.orig == "auto":
        a.orig = os.path.dirname([h for h in hits if "airline-passenger-satisfaction" in h][0])
os.makedirs(a.out, exist_ok=True)
torch.manual_seed(a.seed)
rng = np.random.default_rng(a.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"
ngpu = torch.cuda.device_count()

from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast

TARGET = "satisfaction"
LABELS = ["satisfied", "neutral or dissatisfied"]


def log(*x):
    print(f"[{(time.time() - T0) / 60:6.1f} min]", *x, flush=True)


def load_orig():
    fs = [os.path.join(a.orig, f) for f in ("train.csv", "test.csv") if os.path.exists(os.path.join(a.orig, f))]
    d = pd.concat([pd.read_csv(f, dtype=str, keep_default_na=False) for f in fs], ignore_index=True)
    d = d.drop(columns=[c for c in d.columns if c in ("", "Unnamed: 0", "id", "Inflight service")])
    return d


comp_tr = pd.read_csv(os.path.join(a.comp, "train.csv"), dtype=str, keep_default_na=False)
comp_te = pd.read_csv(os.path.join(a.comp, "test.csv"), dtype=str, keep_default_na=False)
FEATS = [c for c in comp_tr.columns if c not in ("id", TARGET)]
orig = load_orig()
assert set(FEATS) <= set(orig.columns), set(FEATS) - set(orig.columns)
orig = orig[FEATS + [TARGET]]
for d in (orig, comp_tr, comp_te):
    for c in FEATS:
        d[c] = d[c].replace("", "nan")
if a.max_orig:
    orig = orig.sample(min(a.max_orig, len(orig)), random_state=0).reset_index(drop=True)
log("original rows", len(orig), "features", len(FEATS), "| label counts", orig[TARGET].value_counts().to_dict())

tok = GPT2TokenizerFast.from_pretrained("gpt2" if a.tiny else a.model)
tok.pad_token = tok.eos_token
EOS = tok.eos_token_id
if a.tiny:
    model = GPT2LMHeadModel(GPT2Config(n_layer=2, n_head=2, n_embd=64, vocab_size=len(tok)))
else:
    model = GPT2LMHeadModel.from_pretrained(a.model)
model.to(dev)
net = torch.nn.DataParallel(model) if ngpu > 1 else model
log("model", "tiny" if a.tiny else a.model, "params", sum(p.numel() for p in model.parameters()), "gpus", ngpu)

COLS = FEATS + [TARGET]
vals = orig[COLS].to_numpy()


def row_text(row, order):
    return ", ".join(f"{COLS[j]} is {row[j]}" for j in order)


# ---------------------------------------------------------------- training
n = len(vals)
steps_per_epoch = (n + a.bs - 1) // a.bs
total = int(a.epochs * steps_per_epoch)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1 - s / max(total, 1)))
scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")
model.train()
step, t_train = 0, time.time()
while step < total:
    perm = rng.permutation(n)
    for b in range(0, n, a.bs):
        if step >= total or time.time() - T0 > a.hours * 3600:
            break
        idx = perm[b:b + a.bs]
        texts = [row_text(vals[i], rng.permutation(len(COLS))) + tok.eos_token for i in idx]
        enc = tok(texts, return_tensors="pt", padding=True)
        ids, att = enc.input_ids.to(dev), enc.attention_mask.to(dev)
        labels = ids.masked_fill(att == 0, -100)
        labels[:, 0] = -100
        # keep the first eos (end of row) as a target
        first_pad = att.sum(1)
        labels[torch.arange(len(ids)), (first_pad - 1).clamp(max=ids.shape[1] - 1)] = EOS
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev == "cuda"):
            out = net(input_ids=ids, attention_mask=att, labels=labels)
            loss = out.loss.mean()
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        step += 1
        if step % 200 == 0 or step == 1:
            el = time.time() - t_train
            log(f"step {step}/{total} loss {loss.item():.4f}  {el / step:.3f} s/step  ETA {el / step * (total - step) / 60:.1f} min")
    if time.time() - T0 > a.hours * 3600:
        log("training time limit reached")
        break
log("training done, steps", step)
model.eval()


# ---------------------------------------------------------------- scoring
tnet = torch.nn.DataParallel(model.transformer) if ngpu > 1 else model.transformer


@torch.no_grad()
def seq_logprob(texts, n_prefix_chars):
    """Sum of token log-probs of each text after its first n_prefix_chars characters. The LM head is applied
    only at the scored positions, in chunks (a full-vocabulary softmax over every position would not fit)."""
    enc = tok(texts, return_tensors="pt", padding=True, return_offsets_mapping=True)
    ids, att = enc.input_ids.to(dev), enc.attention_mask.to(dev)
    start = enc.offset_mapping[:, 1:, 0].to(dev)
    npre = torch.tensor(n_prefix_chars, device=dev)[:, None]
    m = (att[:, 1:] == 1) & (start >= npre)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev == "cuda"):
        hid = tnet(input_ids=ids, attention_mask=att).last_hidden_state[:, :-1]
    rows = torch.nonzero(m, as_tuple=True)
    h, tgt = hid[rows], ids[:, 1:][rows]
    lp = torch.empty(len(tgt), device=dev)
    for s in range(0, len(tgt), 8192):
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev == "cuda"):
            lg = model.lm_head(h[s:s + 8192]).float()
        lp[s:s + 8192] = lg.gather(1, tgt[s:s + 8192, None])[:, 0] - torch.logsumexp(lg, 1)
    out = torch.zeros(len(ids), device=dev).index_add_(0, rows[0], lp)
    return out.cpu().numpy()


def score(df, tag):
    v = df[FEATS].to_numpy()
    if a.max_comp:
        v = v[:a.max_comp]
    srng = np.random.default_rng(1234)
    o_last = srng.permutation(len(FEATS))
    o_first = srng.permutation(len(FEATS))
    feat_last = [", ".join(f"{FEATS[j]} is {r[j]}" for j in o_last) for r in v]
    feat_first = [", ".join(f"{FEATS[j]} is {r[j]}" for j in o_first) for r in v]
    out = np.zeros((len(v), 2), dtype=np.float32)
    order = np.argsort([len(s) for s in feat_last])
    t = time.time()
    for b in range(0, len(v), a.score_bs):
        idx = order[b:b + a.score_bs]
        res = []
        for lab in LABELS:  # target last: score only the label tokens
            pre = [feat_last[i] + f", {TARGET} is" for i in idx]
            res.append(seq_logprob([p + " " + lab for p in pre], [len(p) for p in pre]))
        out[idx, 0] = res[0] - res[1]
        res = []
        for lab in LABELS:  # target first: score only the feature tokens
            pre = f"{TARGET} is {lab}, "
            res.append(seq_logprob([pre + feat_first[i] for i in idx], [len(pre)] * len(idx)))
        out[idx, 1] = res[0] - res[1]
        if (b // a.score_bs) % 100 == 0:
            el = time.time() - t
            log(f"{tag}: {b + len(idx)}/{len(v)} rows  ETA {el / (b + len(idx)) * (len(v) - b - len(idx)) / 60:.1f} min")
    return out


tr = score(comp_tr, "train")
np.save(os.path.join(a.out, "llr_train.npy"), tr)
te = score(comp_te, "test")
np.save(os.path.join(a.out, "llr_test.npy"), te)
y = (comp_tr[TARGET].iloc[:len(tr)].str.lower() == "true").astype(int).to_numpy()
from sklearn.metrics import roc_auc_score
log("label-free AUC on competition train: llr_last %.6f  llr_first %.6f" % (roc_auc_score(y, tr[:, 0]), roc_auc_score(y, tr[:, 1])))
log("done")
