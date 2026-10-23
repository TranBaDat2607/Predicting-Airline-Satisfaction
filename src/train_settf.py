"""Set transformer member: every column is one token, the encoder sees the row as an unordered set.

Token = column embedding + value embedding (values seen >= MIN_COUNT times in train+test get their own vector, so
Flight Distance acts as a route id) + periodic numeric embedding (numeric columns) + side features of the value
(log count in train+test, smoothed log-odds of "satisfied" for that exact value in the original dataset, and for the
13 ratings sachith7's label-free expected value and the residual). One summary token carries row statistics.
No positional encoding: a bidirectional encoder over the tokens is permutation invariant.

Pretraining: replaced-token detection on train+test features (no labels): 20 % of tokens are swapped with the same
column of a random row and the model predicts which ones. Fine-tuning per fold on the frozen folds, epoch picked on a
5 % inner holdout of the fold's training rows; the scored fold and test are only predicted.

    python src/train_settf.py --name settf --folds 0 2 4      # pretrains once per device, then fine-tunes the folds
    python src/train_settf.py --name settf --assemble          # all folds done: OOF/test + ledger record
    python -m kgkit gpu build --name settf --hours 3 --cmd "python src/train_settf.py --name settf --folds {fold}" ...
    (--smoke: 20k rows, 1 epoch; --device cpu for a local test). Outputs follow the kgkit runner layout:
    artifacts/<name>/fold{k}_oof.npy (logits of the scored fold), fold{k}_idx.npy, fold{k}_test.npy.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

TARGET = "satisfaction"
MIN_COUNT = 5
NUMERIC = ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]
CATS = ["Gender", "Customer Type", "Type of Travel", "Class"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ----------------------------------------------------------------------------------------------- data
def load_tables(args):
    tr = pd.read_csv(args.train)
    te = pd.read_csv(args.test)
    y = tr[TARGET].astype(str).str.lower().map({"true": 1, "false": 0}).to_numpy(np.float32)
    cols = [c for c in tr.columns if c not in ("id", TARGET)]
    orig = None
    if args.orig_dir and Path(args.orig_dir).exists():
        orig = pd.concat([pd.read_csv(p) for p in sorted(Path(args.orig_dir).glob("*.csv"))], ignore_index=True)
        orig["_y"] = (orig[TARGET] == "satisfied").astype(float)
    aux = pd.read_parquet(args.aux).to_numpy(np.float32) if args.aux and Path(args.aux).exists() else None
    return tr, te, y, cols, orig, aux


def build_features(tr, te, cols, orig, aux):
    """Integer value ids (global, offset per column), standardised numeric values, per-value side tables."""
    both = pd.concat([tr[cols], te[cols]], ignore_index=True)
    n = len(both)
    ratings = [c for c in cols if c not in NUMERIC and c not in CATS]
    vid = np.zeros((n, len(cols)), np.int64)
    xnum = np.zeros((n, len(cols)), np.float32)
    side_lc, side_lo, offsets, sizes = [], [], [], []
    off = 0
    prior = orig["_y"].mean() if orig is not None else 0.5
    for j, c in enumerate(cols):
        s = both[c]
        key = s.astype(str).where(s.notna(), "__nan__")
        vc = key.value_counts()
        keep = vc[vc >= MIN_COUNT].index
        lut = {k: i + 2 for i, k in enumerate(keep)}           # 0 = rare, 1 = nan
        lut["__nan__"] = 1
        ids = key.map(lut).fillna(0).astype(np.int64).to_numpy()
        vid[:, j] = ids + off
        size = len(keep) + 2
        cnt = np.zeros(size, np.float32)
        cnt[2:] = np.log1p(vc[keep].to_numpy())
        cnt[1] = np.log1p((key == "__nan__").sum())
        cnt[0] = np.log1p(MIN_COUNT / 2)
        lo = np.zeros(size, np.float32)
        if orig is not None and c in orig.columns:
            ok = orig[c].astype(str).where(orig[c].notna(), "__nan__")
            g = orig.groupby(ok)["_y"].agg(["sum", "count"])
            m = 20.0
            p = (g["sum"] + m * prior) / (g["count"] + m)
            lom = np.log(p / (1 - p)) - math.log(prior / (1 - prior))
            for k, v in lom.items():
                if k in lut:
                    lo[lut[k]] = v
        side_lc.append(cnt)
        side_lo.append(lo)
        offsets.append(off)
        sizes.append(size)
        off += size
        if c in CATS:
            continue
        v = pd.to_numeric(s, errors="coerce").to_numpy(np.float64)
        if "Delay" in c:
            v = np.log1p(np.nan_to_num(v, nan=0.0))
        r = pd.Series(v).rank(pct=True).to_numpy()          # quantile transform -> [0, 1]
        xnum[:, j] = np.nan_to_num(r, nan=0.0).astype(np.float32)
    lc = np.concatenate(side_lc)
    lc = (lc - lc.mean()) / (lc.std() + 1e-6)
    lo = np.concatenate(side_lo)
    num_mask = np.array([c not in CATS for c in cols])
    # expected value of each rating from the other columns (label free) and the row's residual
    expv = np.zeros((n, len(cols)), np.float32)
    if aux is not None:
        assert len(aux) == n
        for k, c in enumerate(ratings):
            expv[:, cols.index(c)] = (aux[:, k] - 2.5) / 1.5
    raw = np.zeros((n, len(cols)), np.float32)
    for c in ratings:
        raw[:, cols.index(c)] = (both[c].to_numpy(np.float32) - 2.5) / 1.5
    rating_mask = np.array([c in ratings for c in cols])
    rt = both[ratings].to_numpy(np.float32)
    dd, ad = both["Departure Delay in Minutes"].to_numpy(np.float32), both["Arrival Delay in Minutes"].fillna(0).to_numpy(np.float32)
    summ = np.column_stack([rt.mean(1), rt.std(1), (rt == 0).sum(1), (rt == 5).sum(1), (rt <= 2).sum(1),
                            np.log1p(dd) - np.log1p(ad), np.log1p(both["Flight Distance"].to_numpy(np.float32)),
                            both["Arrival Delay in Minutes"].isna().to_numpy(np.float32)]).astype(np.float32)
    summ = (summ - summ.mean(0)) / (summ.std(0) + 1e-6)
    return dict(vid=vid, xnum=xnum, raw=raw, expv=expv, summ=summ, lc=lc.astype(np.float32), lo=lo.astype(np.float32),
                num_mask=num_mask, rating_mask=rating_mask, n_vals=off, n_cols=len(cols))


# ----------------------------------------------------------------------------------------------- model
class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.w


class SetTF(nn.Module):
    N_SIDE = 4

    def __init__(self, n_vals, n_cols, num_mask, n_summ, d=256, layers=6, heads=8, nf=24, sigma=8.0, drop=0.1):
        super().__init__()
        self.col = nn.Parameter(torch.randn(n_cols, d) * 0.02)
        self.val = nn.Embedding(n_vals, d)
        nn.init.normal_(self.val.weight, std=0.02)
        self.freq = nn.Parameter(torch.randn(n_cols, nf) * sigma)
        self.pw = nn.Parameter(torch.randn(n_cols, 2 * nf, d) * (1 / math.sqrt(2 * nf)))
        self.pb = nn.Parameter(torch.zeros(n_cols, d))
        self.sw = nn.Parameter(torch.randn(n_cols, self.N_SIDE, d) * 0.02)
        self.register_buffer("num_mask", torch.as_tensor(num_mask, dtype=torch.float32))
        self.summ = nn.Sequential(nn.Linear(n_summ, d), nn.GELU(), nn.Linear(d, d))
        self.norm = RMSNorm(d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, drop, activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out_norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)
        self.rtd = nn.Linear(d, 1)

    def forward(self, vid, xnum, side, summ):
        ang = 2 * math.pi * xnum.unsqueeze(-1) * self.freq                      # B C nf
        per = torch.cat([torch.sin(ang), torch.cos(ang)], -1)                    # B C 2nf
        num = F.relu(torch.einsum("bcf,cfd->bcd", per, self.pw) + self.pb)
        tok = self.col + self.val(vid) + num * self.num_mask[None, :, None] + torch.einsum("bcs,csd->bcd", side, self.sw)
        seq = torch.cat([self.norm(tok), self.summ(summ).unsqueeze(1)], 1)
        h = self.out_norm(self.enc(seq))
        return self.head(h.mean(1)).squeeze(-1), self.rtd(h[:, :-1]).squeeze(-1)


# ----------------------------------------------------------------------------------------------- batches
class Batcher:
    def __init__(self, f, device):
        t = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt)
        self.vid, self.xnum, self.raw = t(f["vid"], torch.long), t(f["xnum"]), t(f["raw"])
        self.expv, self.summ = t(f["expv"]), t(f["summ"])
        self.lc, self.lo = t(f["lc"]).to(device), t(f["lo"]).to(device)
        self.rmask = t(f["rating_mask"]).to(device)
        self.device = device

    def get(self, idx, corrupt=0.0, gen=None):
        idx_t = torch.as_tensor(idx)
        vid, xnum, raw = self.vid[idx_t], self.xnum[idx_t].clone(), self.raw[idx_t].clone()
        expv, summ = self.expv[idx_t], self.summ[idx_t]
        repl = None
        if corrupt > 0:
            B, C = vid.shape
            repl = torch.rand(B, C, generator=gen) < corrupt
            donor = torch.randint(0, len(self.vid), (B, C), generator=gen)
            cols = torch.arange(C).expand(B, C)
            dv = self.vid[donor, cols]
            repl &= dv != vid                                                    # a swap to the same value is no swap
            vid = torch.where(repl, dv, vid)
            xnum = torch.where(repl, self.xnum[donor, cols], xnum)
            raw = torch.where(repl, self.raw[donor, cols], raw)
        d = self.device
        vid, xnum, raw, expv, summ = vid.to(d), xnum.to(d), raw.to(d), expv.to(d), summ.to(d)
        side = torch.stack([self.lc[vid], self.lo[vid], expv * self.rmask, (raw - expv) * self.rmask], -1)
        return vid, xnum, side, summ, (repl.to(d).float() if repl is not None else None)


def make_model(f, args):
    return SetTF(f["n_vals"], f["n_cols"], f["num_mask"], f["summ"].shape[1], d=args.d, layers=args.layers,
                 heads=args.heads, sigma=args.sigma, drop=args.drop)


def cosine(opt, total, warm):
    return torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, total)))))


def predict(model, bt, idx, bs=8192):
    model.eval()
    out = []
    with torch.no_grad(), torch.autocast(bt.device.split(":")[0], enabled=bt.device.startswith("cuda")):
        for s in range(0, len(idx), bs):
            vid, xnum, side, summ, _ = bt.get(idx[s:s + bs])
            out.append(model(vid, xnum, side, summ)[0].float().cpu().numpy())
    model.train()
    return np.concatenate(out) if out else np.zeros(0)


# ----------------------------------------------------------------------------------------------- stages
def pretrain(f, args, out):
    dev = args.device
    torch.manual_seed(0)
    bt = Batcher(f, dev)
    model = make_model(f, args).to(dev)
    n = len(f["vid"])
    steps = args.pre_epochs * (n // args.bs)
    opt = torch.optim.AdamW(model.parameters(), lr=args.pre_lr, weight_decay=0.01)
    sch = cosine(opt, steps, max(1, steps // 20))
    scaler = torch.amp.GradScaler(enabled=dev.startswith("cuda"))
    gen = torch.Generator().manual_seed(1)
    step, t0 = 0, time.time()
    for ep in range(args.pre_epochs):
        perm = np.random.default_rng(ep).permutation(n)
        for s in range(0, n - args.bs + 1, args.bs):
            vid, xnum, side, summ, repl = bt.get(perm[s:s + args.bs], corrupt=args.corrupt, gen=gen)
            with torch.autocast(dev.split(":")[0], enabled=dev.startswith("cuda")):
                _, rl = model(vid, xnum, side, summ)
                loss = F.binary_cross_entropy_with_logits(rl.float(), repl)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sch.step()
            step += 1
            if step % 200 == 0:
                log(f"pretrain ep {ep} step {step}/{steps} rtd loss {loss.item():.4f} ({time.time() - t0:.0f}s)")
    torch.save(model.state_dict(), pre_path(out))
    log(f"pretrain done in {time.time() - t0:.0f}s")


def finetune(f, y, folds, args, out, fold):
    dev = args.device
    torch.manual_seed(100 + fold)
    bt = Batcher(f, dev)
    model = make_model(f, args).to(dev)
    pre = pre_path(out)
    if not args.no_pretrain:
        model.load_state_dict(torch.load(pre, map_location=dev))
        log(f"fold {fold}: loaded {pre}")
    ntr = len(y)
    trn = np.where(folds != fold)[0]
    val = np.where(folds == fold)[0]
    rng = np.random.default_rng(fold)
    inner = rng.random(len(trn)) < 0.05
    fit, hold = trn[~inner], trn[inner]
    yt = torch.as_tensor(y)
    steps_ep = len(fit) // args.bs
    total = int(args.epochs * steps_ep)
    head_params = list(model.head.parameters())
    other = [p for n_, p in model.named_parameters() if not n_.startswith("head.")]
    opt = torch.optim.AdamW([{"params": other, "lr": args.lr}, {"params": head_params, "lr": args.lr * 5}], weight_decay=0.01)
    sch = cosine(opt, total, max(1, total // 20))
    scaler = torch.amp.GradScaler(enabled=dev.startswith("cuda"))
    evals = max(1, int(args.evals_per_epoch))
    every = max(1, steps_ep // evals)
    best, best_state, step, t0 = -1.0, None, 0, time.time()
    hist = []
    while step < total:
        perm = rng.permutation(fit)
        for s in range(0, len(perm) - args.bs + 1, args.bs):
            idx = perm[s:s + args.bs]
            vid, xnum, side, summ, _ = bt.get(idx)
            yb = yt[idx].to(dev)
            if args.label_smooth > 0:
                yb = yb * (1 - args.label_smooth) + 0.5 * args.label_smooth
            with torch.autocast(dev.split(":")[0], enabled=dev.startswith("cuda")):
                logit_, _ = model(vid, xnum, side, summ)
                loss = F.binary_cross_entropy_with_logits(logit_.float(), yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sch.step()
            step += 1
            if step % every == 0 or step >= total:
                auc = roc_auc_score(y[hold], predict(model, bt, hold))
                hist.append((step / steps_ep, auc))
                log(f"fold {fold} ep {step / steps_ep:.2f} loss {loss.item():.4f} inner AUC {auc:.6f} ({time.time() - t0:.0f}s)")
                if auc > best:
                    best = auc
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if step >= total:
                break
    model.load_state_dict(best_state)
    pv = predict(model, bt, val)
    pt = predict(model, bt, np.arange(ntr, len(f["vid"])))
    auc = roc_auc_score(y[val], pv)
    np.save(out / f"fold{fold}_test.npy", pt)
    np.save(out / f"fold{fold}_idx.npy", val)
    json.dump(dict(fold=fold, auc=auc, inner_best=best, hist=hist, secs=time.time() - t0),
              open(out / f"fold{fold}_meta.json", "w"))
    np.save(out / f"fold{fold}_oof.npy", pv)                                    # last: marks the fold done
    log(f"fold {fold}: VAL AUC {auc:.6f} (inner best {best:.6f}) in {time.time() - t0:.0f}s")


def pre_path(out):
    return out / f"pretrained_{os.environ.get('CUDA_VISIBLE_DEVICES', 'cpu').replace(',', '-')}.pt"


def assemble(y, folds, out, args):
    oof = np.full(len(y), np.nan)
    tests, scores = [], []
    for k in range(folds.max() + 1):
        p = out / f"fold{k}_oof.npy"
        if p.exists():
            oof[np.load(out / f"fold{k}_idx.npy")] = np.load(p)
            tests.append(np.load(out / f"fold{k}_test.npy"))
            scores.append(roc_auc_score(y[folds == k], oof[folds == k]))
    ok = np.isfinite(oof)
    log(f"folds done {len(tests)}; OOF AUC {roc_auc_score(y[ok], oof[ok]):.6f} on {ok.mean():.0%} rows")
    if not ok.all() or args.smoke or args.subsample:
        return
    cv = roc_auc_score(y, oof)
    sig = lambda z: 1 / (1 + np.exp(-z))
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from kgkit.experiment import Ledger
    params = {k: v for k, v in vars(args).items() if k not in ("folds", "train", "test", "fold_file", "orig_dir", "aux")}
    rec = Ledger().log(args.name, cv, fold_scores=scores, params=params, model="settf",
                       notes=args.notes or "set transformer: value-embedding tokens + side features, RTD pretraining",
                       oof=sig(oof), test_pred=sig(np.mean(tests, 0)), task="binary")
    log(f"logged {rec['id']} CV {cv:.6f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="settf")
    ap.add_argument("--notes", default="")
    ap.add_argument("--assemble", action="store_true", help="only assemble finished folds and log the experiment")
    ap.add_argument("--stage", choices=["pretrain", "finetune", "assemble", "all"], default="all")
    ap.add_argument("--folds", type=int, nargs="*")
    ap.add_argument("--train", default="data/train.csv")
    ap.add_argument("--test", default="data/test.csv")
    ap.add_argument("--fold-file", default="data/folds.csv")
    ap.add_argument("--orig-dir", default="data/ext/airline-passenger-satisfaction")
    ap.add_argument("--aux", default="data/ext/sachith7/aux_ev.parquet")
    ap.add_argument("--out", help="default artifacts/<name>")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--sigma", type=float, default=8.0)
    ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--evals-per-epoch", type=float, default=4)
    ap.add_argument("--label-smooth", type=float, default=0.0)
    ap.add_argument("--pre-epochs", type=int, default=2)
    ap.add_argument("--pre-lr", type=float, default=3e-4)
    ap.add_argument("--corrupt", type=float, default=0.2)
    ap.add_argument("--no-pretrain", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--subsample", type=int, default=0, help="random train rows (with --smoke: default 20000)")
    args = ap.parse_args()
    if args.assemble:
        args.stage = "assemble"
    out = Path(args.out or f"artifacts/{args.name}")
    if not Path(args.orig_dir).exists():                      # Kaggle: the dataset is mounted under /kaggle/input
        hits = sorted(Path("/kaggle/input").rglob("airline-passenger-satisfaction")) if Path("/kaggle/input").exists() else []
        args.orig_dir = str(hits[0]) if hits else ""
    if not Path(args.aux).exists():
        hits = sorted(Path("/kaggle/input").rglob("aux_ev.parquet")) if Path("/kaggle/input").exists() else []
        hits = hits or sorted(Path("data").rglob("aux_ev.parquet"))
        args.aux = str(hits[0]) if hits else ""
    out.mkdir(parents=True, exist_ok=True)
    tr, te, y, cols, orig, aux = load_tables(args)
    folds = pd.read_csv(args.fold_file)["fold"].to_numpy()
    if args.smoke or args.subsample:
        keep = np.random.default_rng(0).choice(len(tr), args.subsample or 20000, replace=False)
        keep.sort()
        tkeep = np.arange(5000)
        if aux is not None:
            aux = np.concatenate([aux[keep], aux[len(tr) + tkeep]])
        tr, te, y, folds = tr.iloc[keep].reset_index(drop=True), te.iloc[tkeep].reset_index(drop=True), y[keep], folds[keep]
        if args.smoke:
            args.epochs, args.pre_epochs = min(args.epochs, 1.0), min(args.pre_epochs, 1)
    f = build_features(tr, te, cols, orig, aux)
    log(f"{len(tr)} train + {len(te)} test rows, {f['n_cols']} tokens + summary, {f['n_vals']} value ids, device {args.device}")
    if args.stage in ("pretrain", "all") and not args.no_pretrain and not pre_path(out).exists():
        pretrain(f, args, out)
    if args.stage in ("finetune", "all"):
        for k in (args.folds if args.folds is not None else range(folds.max() + 1)):
            finetune(f, y, folds, args, out, k)
    if args.stage in ("assemble", "all") and not os.environ.get("KG_DEFER_ASSEMBLE"):
        assemble(y, folds, out, args)


if __name__ == "__main__":
    main()
