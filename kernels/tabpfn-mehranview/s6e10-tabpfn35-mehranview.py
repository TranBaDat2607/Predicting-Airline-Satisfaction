# S6E10 | TabPFN-3.5 member on Mehran Kazeminia's view, made honest: raw columns + KMeans group of the 4 categoricals
# + 47 target-encoded value combos (sklearn TargetEncoder fit INSIDE each fold), frozen folds SKF(5, shuffle, rs=42).
# His full-train version (no OOF) is the 10 % TabPFN part of the public 0.96213 file and measurably helps the public LB
# (+3e-5 over the 0.96210 base); this run gives OOF so the view can be stacked and its gain measured.
# Worker + lean KV-cache patch from hemingweb/s6e10-exp01-tabpfn35-full-context; context subsampled to CTX rows.
import os, time, subprocess
os.environ["RUN_T0"] = str(time.time())
print(subprocess.run("nvidia-smi; nproc; free -g", shell=True, capture_output=True, text=True).stdout)
subprocess.run("pip install -q tabpfn==9.0.0", shell=True, check=True)
import torch, tabpfn
print("torch", torch.__version__, "tabpfn", tabpfn.__version__, "GPUs", torch.cuda.device_count())

# TabPFN-3.5 checkpoint (Prior Labs, TABPFN-3.5 License v1.0: non-commercial; Kaggle competitions are allowed use)
from huggingface_hub import hf_hub_download
os.makedirs("/tmp/tabpfn_ckpt", exist_ok=True)
os.environ["CKPT"] = hf_hub_download("Prior-Labs/tabpfn_3_5", "tabpfn-v3.5-20260909.safetensors", local_dir="/tmp/tabpfn_ckpt")
os.environ.update(WORK="/tmp/exp01", OUT="/kaggle/working", LIMIT_H=os.environ.get("LIMIT_H", "11.3"), N_SEEDS=os.environ.get("N_SEEDS", "2"), CTX=os.environ.get("CTX", "200000"), GPUS=",".join(str(i) for i in range(max(1, torch.cuda.device_count()))))
os.makedirs("/tmp/exp01", exist_ok=True)
print(os.environ["CKPT"], os.environ["GPUS"])


# worker (one (fold, seed) unit on one GPU) + memory-lean KV-cache patch
open('/tmp/exp01/worker.py', 'w').write('# TabPFN-3.5 full-context worker: one (fold, seed) unit on one GPU.\n# Context = ALL labelled rows outside the fold (no subsampling). Predicts the\n# validation fold (OOF) and the full test set with the same KV cache.\nimport os, sys, time, json, argparse, gc\nimport numpy as np\n\nap = argparse.ArgumentParser()\nap.add_argument("--fold", type=int, required=True)\nap.add_argument("--seed", type=int, required=True)\nap.add_argument("--work", required=True)        # dir with data.npz\nap.add_argument("--out", required=True)         # dir for unit outputs\nap.add_argument("--ckpt", required=True)\nap.add_argument("--batch", type=int, default=32768)\nap.add_argument("--deadline", type=float, required=True)  # unix time: abort test pred after this\nargs = ap.parse_args()\n\nimport torch\nfrom sklearn.metrics import roc_auc_score\nfrom tabpfn import TabPFNClassifier\nimport lean_patch; lean_patch.apply()   # chunked KV-cache build -> full context fits a T4\n\nTAG = f"[f{args.fold} s{args.seed} gpu{os.environ.get(\'CUDA_VISIBLE_DEVICES\',\'?\')}]"\nDEV = os.environ.get("TABPFN_DEVICE", "cuda")\nCUDA = DEV.startswith("cuda")\ndef empty():\n    if CUDA: torch.cuda.empty_cache()\ndef peak():\n    return torch.cuda.max_memory_allocated() / 2**30 if CUDA else 0.0\ndef log(*a):\n    print(TAG, time.strftime("%H:%M:%S"), *a, flush=True)\n\nd = np.load(os.path.join(args.work, f"data_f{args.fold}.npz"))\nXtr, ytr, Xva, yva, XT, cat_idx = d["Xtr"], d["ytr"], d["Xva"], d["yva"], d["XT"], list(d["cat_idx"])\nCTX = int(os.environ.get("CTX", 0))\nif CTX and CTX < len(Xtr):\n    from sklearn.model_selection import train_test_split\n    keep, _ = train_test_split(np.arange(len(Xtr)), train_size=CTX, stratify=ytr, random_state=1000 + args.seed)\n    keep = np.sort(keep); Xtr, ytr = Xtr[keep], ytr[keep]\nlog(f"context={len(Xtr):,} val={len(Xva):,} test={len(XT):,} feats={X.shape[1]}")\n\n# Fallback chain if the full-context fit does not fit in T4 memory.\nCONFIGS = [\n    dict(memory_saving_mode="auto", keep_cache_on_device=True),\n    dict(memory_saving_mode=True, keep_cache_on_device=True),\n    dict(memory_saving_mode=True, keep_cache_on_device=False),\n]\n\ndef is_oom(e):\n    s = str(e).lower()\n    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in s or "cuda error: out of memory" in s\n\ndef make(cfg):\n    return TabPFNClassifier(\n        model_path=args.ckpt,\n        n_estimators=1,                       # one estimator per unit; seeds = extra ensemble members\n        random_state=args.seed,\n        device=DEV,\n        ignore_pretraining_limits=True,\n        fit_mode="fit_with_cache",            # KV cache built once, reused for every batch\n        kv_cache_precision="int8",            # needed on a 16 GB T4 at ~560k context\n        inference_precision="autocast",\n        categorical_features_indices=cat_idx,\n        **cfg,\n    )\n\nclf, used = None, None\nt0 = time.time()\nfor i, cfg in enumerate(CONFIGS):\n    try:\n        empty()\n        if CUDA: torch.cuda.reset_peak_memory_stats()\n        clf = make(cfg)\n        clf.fit(Xtr, ytr)\n        used = cfg\n        break\n    except Exception as e:\n        if is_oom(e) and i + 1 < len(CONFIGS):\n            log(f"OOM in fit with {cfg} -> next fallback"); clf = None; gc.collect(); empty()\n            continue\n        raise\nt_fit = time.time() - t0\nlog(f"fit done in {t_fit/60:.1f} min, cfg={used}, peak={peak():.2f} GB")\n\ndef predict(Xq, name, deadline=None):\n    out = np.empty(len(Xq), dtype=np.float32)\n    bs, i, t = args.batch, 0, time.time()\n    while i < len(Xq):\n        if deadline and time.time() > deadline:\n            log(f"{name}: deadline reached at {i:,}/{len(Xq):,} -> abort"); return None\n        j = min(i + bs, len(Xq))\n        try:\n            out[i:j] = clf.predict_proba(Xq[i:j])[:, 1]\n        except Exception as e:\n            if is_oom(e) and bs > 1024:\n                bs //= 2; empty(); log(f"{name}: OOM -> batch {bs}"); continue\n            raise\n        i = j\n        el = time.time() - t\n        log(f"{name}: {i:,}/{len(Xq):,} rows  {el/60:.1f} min  ETA {el/i*(len(Xq)-i)/60:.1f} min")\n    return out\n\ntag = f"f{args.fold}_s{args.seed}"\nt1 = time.time()\npv = predict(Xva, "val")\nt_val = time.time() - t1\nauc = roc_auc_score(yva, pv)\nnp.save(os.path.join(args.out, f"{tag}_val.npy"), pv)\nlog(f"VAL AUC = {auc:.6f}  (val pred {t_val/60:.1f} min)")\n\nt2 = time.time()\npt = predict(XT, "test", deadline=args.deadline)\nt_test = time.time() - t2\nif pt is not None:\n    np.save(os.path.join(args.out, f"{tag}_test.npy"), pt)\n\nmeta = dict(fold=args.fold, seed=args.seed, auc=float(auc), n_ctx=int(len(Xtr)),\n            t_fit=t_fit, t_val=t_val, t_test=t_test, total=time.time() - t0,\n            test_done=pt is not None, cfg=str(used),\n            peak_gb=peak())\njson.dump(meta, open(os.path.join(args.out, f"{tag}_meta.json"), "w"), indent=1)\nlog("DONE", json.dumps(meta))\n')
open('/tmp/exp01/lean_patch.py', 'w').write('# Memory-lean KV-cache build for TabPFN-3.5 (tabpfn==9.0.0).\n# The stock fit_with_cache path runs every ICL layer over all ~560k context rows\n# at once (full q/k/v + fp32 norm copies + unchunked MLP) -> OOM on a 16 GB T4.\n# This patch computes K/V for the context in row chunks, then the attention\n# output and the MLP in row chunks, updating the residual stream in place.\n# Same math as the original (queries are independent given K/V; SSMax scaling\n# depends only on the K length), so predictions are unchanged.\nimport os\nimport torch\nfrom tabpfn.architectures import tabpfn_v3_5 as A\n\nimport time\nCH = int(os.environ.get("LEAN_CHUNK", 16384))\n_calls = [0, None]\n_orig_forward = A.ICLTransformerBlock.forward\n\n\ndef _lean_forward(self, x_BRE, single_eval_pos, save_peak_memory_factor=None, *,\n                  cached_kv=None, return_kv=False):\n    if not return_kv or cached_kv is not None:\n        return _orig_forward(self, x_BRE, single_eval_pos, save_peak_memory_factor,\n                             cached_kv=cached_kv, return_kv=return_kv)\n    att = self.icl_attention\n    x = x_BRE.contiguous()\n    B, R, _ = x.shape\n    N = R if single_eval_pos is None else single_eval_pos\n    nh = att.num_kv_heads_test\n    with torch.no_grad():\n        K = V = None\n        for s in range(0, N, CH):\n            e = min(s + CH, N)\n            h = self.layernorm(x[:, s:e])\n            k = att.k_projection(h).view(B, e - s, att.num_kv_heads, att.head_dim)\n            v = att.v_projection(h).view(B, e - s, att.num_kv_heads, att.head_dim)\n            k = att.k_norm(k)\n            if K is None:\n                K = torch.empty((B, N, att.num_kv_heads, att.head_dim), dtype=v.dtype, device=x.device)\n                V = torch.empty_like(K)\n            K[:, s:e] = k.to(v.dtype)\n            V[:, s:e] = v\n            del h, k, v\n        starts = list(range(0, N, CH)) + list(range(N, R, CH))\n        for s in starts:\n            e = min(s + CH, N if s < N else R)\n            xc = x[:, s:e]\n            q = att.q_norm(att.q_projection(self.layernorm(xc)).view(B, e - s, att.num_heads, att.head_dim))\n            if s < N or nh is None or N == R:\n                out = A._batched_scaled_dot_product_attention(q, K, V, softmax_scaling_layer=att.softmax_scaling_layer)\n            else:\n                out = A._batched_scaled_dot_product_attention(q, K[:, :, :nh], V[:, :, :nh],\n                                                              softmax_scaling_layer=att.softmax_scaling_layer)\n            xc.add_(att.out_projection(out.reshape(B, e - s, att.head_dim * att.num_heads)))\n            del q, out\n        if nh is not None:\n            k_cache, v_cache = K[:, :, :nh].contiguous(), V[:, :, :nh].contiguous()\n        else:\n            k_cache, v_cache = K, V\n        del K, V\n        kv_entry = A.KVCacheEntry(key=k_cache.detach(), value=v_cache.detach())\n        for s in range(0, R, CH):\n            xc = x[:, s:min(s + CH, R)]\n            xc.add_(self.mlp(self.layernorm_mlp(xc)))\n    _calls[0] += 1\n    if os.environ.get("LEAN_LOG") == "1":\n        now = time.time()\n        if _calls[1] is not None and _calls[0] % 6 == 0:\n            print(f"    [lean] ICL layer call {_calls[0]}  last {now - _calls[1]:.1f}s  rows={R:,}  "\n                  f"mem={torch.cuda.memory_allocated()/2**30 if torch.cuda.is_available() else 0:.2f} GB", flush=True)\n        _calls[1] = now\n    return x, kv_entry\n\n\ndef apply():\n    A.ICLTransformerBlock.forward = _lean_forward\n')
print('worker + patch written')


# ==== S6E10 | EXP01 | TabPFN-3.5, full labelled context, 5 fixed folds, 2xT4 ====
# Lessons applied from the S6E9 2nd-place write-up (Team Alicia / URAD):
#  * validation first: one fixed fold file, saved and reused by every future experiment
#  * TabPFN-3.5 with the FULL labelled context (no subsampling): ~560k rows per fold
#  * fit_with_cache + int8 KV cache + autocast  (only way a 16 GB T4 holds this context)
#  * OOF + test preds saved per (fold, seed) so later level-2/3 blends stay honest
#  * no public-LB fitting: the CV here is the decision metric
import os, sys, time, json, glob, threading, subprocess, collections
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

T0 = float(os.environ.get("RUN_T0", time.time()))
LIMIT_H = float(os.environ.get("LIMIT_H", 11.5))          # Kaggle hard limit is 12 h
HARD = T0 + LIMIT_H * 3600
LOCAL = os.environ.get("LOCAL_TEST") == "1"
WORK = os.environ.get("WORK", "/tmp/exp01")
OUT = os.environ.get("OUT", "/kaggle/working")
UNITS = os.path.join(OUT, "units")
CKPT = os.environ["CKPT"]
N_SEEDS = int(os.environ.get("N_SEEDS", 10))
N_FOLDS = 5
os.makedirs(WORK, exist_ok=True); os.makedirs(UNITS, exist_ok=True)

def say(*a): print(time.strftime("%H:%M:%S"), f"[+{(time.time()-T0)/3600:.2f}h]", *a, flush=True)

# ---------------- data ----------------
if LOCAL:
    tr_path, te_path = os.environ["TRAIN_CSV"], os.environ["TEST_CSV"]
else:
    tr_path = [p for p in glob.glob("/kaggle/input/**/train.csv", recursive=True) if "s6e10" in p][0]
    te_path = tr_path.replace("train.csv", "test.csv")
train, test = pd.read_csv(tr_path), pd.read_csv(te_path)
if LOCAL:
    train = train.sample(int(os.environ.get("LOCAL_N", 3000)), random_state=0).reset_index(drop=True)
    test = test.head(int(os.environ.get("LOCAL_N", 3000)) // 2)
TARGET = "satisfaction"
y = train[TARGET].astype(str).str.lower().map({"true": 1, "false": 0}).values.astype(np.int64)
feats = [c for c in train.columns if c not in ("id", TARGET)]
SHORT = {"Age": "Age", "Distance": "Flight Distance", "Wifi": "Inflight wifi service",
         "Convenient": "Departure/Arrival time convenient", "Booking": "Ease of Online booking", "Gate": "Gate location",
         "Food": "Food and drink", "Boarding": "Online boarding", "Seat": "Seat comfort",
         "Entertain": "Inflight entertainment", "Onboard": "On-board service", "Leg": "Leg room service",
         "Baggage": "Baggage handling", "Checkin": "Checkin service", "Clean": "Cleanliness",
         "Departure": "Departure Delay in Minutes", "Arrival": "Arrival Delay in Minutes", "Gender": "Gender",
         "Customer": "Customer Type", "Travel": "Type of Travel", "Class": "Class"}
COMBOS = [("Class","Travel","Customer"), ("Age","Class","Travel","Customer"), ("Gender","Class","Travel","Customer"),
    ("Distance","Class","Travel","Customer"), ("Age","Gender","Class","Travel","Customer"),
    ("Distance","Gender","Class","Travel","Customer")] + \
    [(a, r, "Class", "Travel", "Customer") for r in ("Wifi", "Booking", "Baggage", "Boarding", "Convenient", "Leg", "Gate",
                                                    "Onboard", "Seat", "Checkin", "Clean", "Entertain", "Food")
     for a in ("Age", "Distance")] + \
    [("Wifi","Age"), ("Wifi","Seat"), ("Wifi","Booking"), ("Wifi","Boarding"), ("Seat","Boarding"), ("Wifi","Entertain"),
     ("Distance","Age"), ("Distance","Wifi"), ("Distance","Seat"), ("Distance","Booking"), ("Distance","Boarding"),
     ("Distance","Entertain"), ("Departure","Arrival"), ("Departure","Arrival","Convenient"),
     ("Wifi","Boarding","Entertain","Seat")]
assert len(COMBOS) == 47, len(COMBOS)
df = pd.concat([train[feats], test[feats]], axis=0, ignore_index=True)
df["Arrival Delay in Minutes"] = df["Arrival Delay in Minutes"].fillna(df["Arrival Delay in Minutes"].mode()[0])
cats = ["Gender", "Customer Type", "Type of Travel", "Class"]
for c in cats:
    df[c] = pd.factorize(df[c])[0]
from sklearn.cluster import KMeans
df["cat_groups"] = KMeans(n_clusters=12, random_state=420, n_init=20).fit(df[cats]).labels_
combo_codes = np.column_stack([pd.factorize(df[[SHORT[k] for k in cb]].astype(str).agg("_".join, axis=1))[0]
                               for cb in COMBOS])
base = df[feats + ["cat_groups"]].to_numpy(np.float32)
cat_idx = np.array([feats.index(c) for c in cats] + [len(feats)])
ntr = len(train)
say(f"base {base.shape}, combos {combo_codes.shape}, pos rate {y.mean():.4f}")

skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=42)
folds = np.zeros(ntr, dtype=np.int64)
for k, (_, v) in enumerate(skf.split(np.zeros(ntr), y)):
    folds[v] = k
pd.DataFrame({"id": train["id"], "fold": folds}).to_csv(os.path.join(OUT, "folds_skf5_rs42.csv"), index=False)
from sklearn.preprocessing import TargetEncoder
for k in range(N_FOLDS):
    tr_, va_ = folds != k, folds == k
    enc = TargetEncoder(target_type="binary", random_state=420)
    te_tr = enc.fit_transform(combo_codes[:ntr][tr_], y[tr_])            # cross-fitted inside the fold's training rows
    te_va = enc.transform(combo_codes[:ntr][va_])
    te_te = enc.transform(combo_codes[ntr:])
    np.savez(os.path.join(WORK, f"data_f{k}.npz"),
             Xtr=np.hstack([base[:ntr][tr_], te_tr]).astype(np.float32), ytr=y[tr_],
             Xva=np.hstack([base[:ntr][va_], te_va]).astype(np.float32), yva=y[va_],
             XT=np.hstack([base[ntr:], te_te]).astype(np.float32), cat_idx=cat_idx)
    say(f"fold {k}: features {base.shape[1] + te_tr.shape[1]} (TE fit on {tr_.sum():,} rows)")

# ---------------- scheduler: (seed, fold) units, one GPU each ----------------
gpus = os.environ.get("GPUS", "0,1").split(",")
queue = collections.deque((s, f) for s in range(N_SEEDS) for f in range(N_FOLDS))
lock = threading.Lock()
durations, fails = [], [0]
WORKER = os.path.join(WORK, "worker.py")   # written by the cell above (a script kernel runs from /kaggle/src)

def gpu_loop(g):
    while True:
        with lock:
            if not queue or fails[0] >= 3:
                return
            est = float(np.median(durations)) if durations else None
            if est is not None and time.time() + 1.10 * est > HARD:
                say(f"gpu{g}: not enough time for another unit (est {est/60:.0f} min) -> stop")
                return
            s, f = queue.popleft()
        say(f"gpu{g}: START fold {f} seed {s}")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=g, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", LEAN_LOG="1")
        cmd = [sys.executable, "-u", WORKER, "--fold", str(f), "--seed", str(s), "--work", WORK,
               "--out", UNITS, "--ckpt", CKPT, "--deadline", str(HARD - 300)]
        t = time.time()
        p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        tail = collections.deque(maxlen=40)
        for line in p.stdout:
            tail.append(line.rstrip())
            if "warn" not in line.lower():
                print(line.rstrip(), flush=True)
            if time.time() > HARD:
                p.kill(); say(f"gpu{g}: HARD deadline -> killed fold {f} seed {s}"); return
        p.wait()
        dt = time.time() - t
        with lock:
            if p.returncode == 0:
                durations.append(dt); fails[0] = 0
                say(f"gpu{g}: END fold {f} seed {s} in {dt/60:.1f} min")
            else:
                fails[0] += 1
                say(f"gpu{g}: FAILED fold {f} seed {s} rc={p.returncode}\n" + "\n".join(tail))

threads = [threading.Thread(target=gpu_loop, args=(g,)) for g in gpus]
for th in threads: th.start()
for th in threads: th.join()

# ---------------- assemble ----------------
metas = [json.load(open(m)) for m in sorted(glob.glob(os.path.join(UNITS, "*_meta.json")))]
say(f"units finished: {len(metas)}")
oof = np.full(len(y), np.nan); nseed = np.zeros(len(y), dtype=int)
per_fold = {}
for f in range(N_FOLDS):
    vs = sorted(glob.glob(os.path.join(UNITS, f"f{f}_s*_val.npy")))
    if not vs: continue
    m = folds == f
    oof[m] = np.mean([np.load(v) for v in vs], axis=0); nseed[m] = len(vs)
    per_fold[f] = dict(auc=float(roc_auc_score(y[m], oof[m])), n_seeds=len(vs))
cov = ~np.isnan(oof)
summary = dict(exp="exp01_tabpfn35_fullctx", per_fold=per_fold, units=metas, coverage=float(cov.mean()))
if cov.any():
    summary["oof_auc"] = float(roc_auc_score(y[cov], oof[cov]))
    s0 = [os.path.join(UNITS, f"f{f}_s0_val.npy") for f in range(N_FOLDS)]
    if all(os.path.exists(p) for p in s0):
        o0 = np.zeros(len(y))
        for f, p in enumerate(s0): o0[folds == f] = np.load(p)
        summary["oof_auc_seed0"] = float(roc_auc_score(y, o0))
    aucs = [v["auc"] for v in per_fold.values()]
    summary["fold_auc_mean"], summary["fold_auc_sd"] = float(np.mean(aucs)), float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0
pd.DataFrame({"id": train["id"], "fold": folds, "target": y, "pred": oof, "n_seeds": nseed}).to_csv(
    os.path.join(OUT, "oof_tabpfn35_mehranview.csv"), index=False)
ts = sorted(glob.glob(os.path.join(UNITS, "*_test.npy")))
if ts:
    pt = np.mean([np.load(t) for t in ts], axis=0)
    pd.DataFrame({"id": test["id"], TARGET: pt}).to_csv(os.path.join(OUT, "test_tabpfn35_mehranview.csv"), index=False)
    pd.DataFrame({"id": test["id"], TARGET: pt}).to_csv(os.path.join(OUT, "submission.csv"), index=False)
    summary["n_test_units"] = len(ts)
json.dump(summary, open(os.path.join(OUT, "summary_exp01.json"), "w"), indent=1)
say("SUMMARY", json.dumps({k: v for k, v in summary.items() if k != "units"}, indent=1))
