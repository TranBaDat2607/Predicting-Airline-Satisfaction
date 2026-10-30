"""Scored files mined from what public notebooks document: every entry is a public file or a recipe from public inputs
whose public score its author measured and published (goodpjw2008's Meta Stack / Twin Stacks pages, mayankraaz's Mirror
Pair Climb, tkms0816's Push Model). Unlike a notebook's search-API score these pairs are exact: no version ambiguity.

Each recipe is rebuilt and checked against the author's pinned file where one exists (rank corr printed).

    python src/lb_mined.py      # -> data/tilt/mined/<name>.npy (rank vectors), data/tilt/mined.csv (name, score, note)
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

G = Path("data/ext/goodpjw-pub")
OUT = Path("data/tilt/mined")
OUT.mkdir(parents=True, exist_ok=True)
ids = pd.read_csv("data/test.csv", usecols=["id"])["id"].to_numpy()
N = len(ids)
rows = []


def pr(v):
    return (rankdata(np.asarray(v, np.float64)) - 0.5) / len(v)


def csv_(path):
    d = pd.read_csv(path)
    return d.set_index("id").iloc[:, 0].reindex(ids).to_numpy(np.float64)


def add(name, score, v, note=""):
    r = pr(v).astype(np.float32)
    np.save(OUT / f"{name}.npy", r)
    rows.append({"name": name, "score": score, "note": note})


def check(a, b, label):
    print(f"check {label}: rank corr {np.corrcoef(pr(a), pr(b))[0, 1]:.8f}", flush=True)


# ---------------------------------------------------------------- goodpjw2008, plain scored files in his dataset
PLAIN = {"probe_comp_xgb_base_raw.csv": 0.95808, "probe_comp_gen_v2_o12.csv": 0.95343, "probe_comp_realmlp_v4_s1.csv": 0.96082,
         "probe_comp_gen3_pilot.csv": 0.95662, "probe_comp_revisit_p2_viewb.csv": 0.96129, "probe_comp_ens_v4_k10.csv": 0.96107,
         "probe_comp_ne4f_single.csv": 0.96110, "hanifnoerrofiq_agentic_1019.csv": 0.96198, "mayankraaz_surface_v2_1019.csv": 0.96196,
         "surface_step2_1019.csv": 0.96196, "meta_part6_climb1_096217_1030.csv": 0.96217, "twin_submission_1015.csv": 0.96187,
         "mehrankazeminia_tabpfn_1023.csv": 0.96100}
for f, s in PLAIN.items():
    add("gp_" + f[:-4], s, csv_(G / f), "goodpjw2008 dataset, score from his notebook")

# series of his 16 submitted stacks (Parts 1-8)
ser = np.load(G / "series_stacks_p1_p8.npz", allow_pickle=True)
for n, s in zip(list(ser["names"]), ser["public"]):
    add(f"gp_series_{n}", float(s), ser[f"test_{n}"], "goodpjw2008 series_stacks: public as stored")

# dated public files he pinned, with their public scores
ch = np.load(G / "public_chain_1013_1022.npz", allow_pickle=True)
for i in range(len(ch["public"])):
    add(f"gp_chain_{i:02d}_{ch['author'][i]}", float(ch["public"][i]), ch["ranks"][i], f"public chain {ch['date'][i]}")

# Part 3 (10-20): 8 strong moves on the 182-member LR stack (0.96180)
t = np.load(G / "stack182_lr_lgb_test.npz")
At, Gt = pr(t["lr"]), pr(t["lgb"])
add("gp_stack182_lr", 0.96180, At, "182 LR stack")
for g, s in ((0.5, 0.96176), (1.0, 0.96161), (1.5, 0.96131)):
    add(f"gp_p3_away_lgb_{g:g}", s, At + g * (At - Gt), "Part 3: away from LGB stacker")
add("gp_p3_lgb_alone", 0.96154, Gt, "Part 3: LGB stacker alone")
MEM = {m: pr(np.load(G / f"test_{m}.npy").astype(np.float64).ravel()) for m in ("tabpfn_ne4", "tabm_v4aux", "xgb_k10", "cat_ctr_k")}
U = {m: v - At for m, v in MEM.items()}
U["lgb"] = Gt - At
for m, w, s in (("tabpfn_ne4", 0.60, 0.96152), ("tabm_v4aux", 0.35, 0.96157), ("xgb_k10", 0.55, 0.96153), ("cat_ctr_k", 0.40, 0.96152)):
    add(f"gp_p3_toward_{m}", s, At + w * U[m], "Part 3: toward member")
# Part 14 (10-21): the home test and the moves on the 0.96208 file
add("gp_p14_home", 0.96182, At - 0.05 * U["tabpfn_ne4"] - 0.15 * U["cat_ctr_k"], "Part 14 home test")
M208 = pr(csv_(G / "mayankraaz_surface_v4_096208_1021.csv"))
add("gp_p14_m208_cat5", 0.96208, M208 - 0.05 * U["cat_ctr_k"], "Part 14 transfer")
add("gp_p14_m208_tab5cat10", 0.96208, M208 - 0.05 * U["tabpfn_ne4"] - 0.10 * U["cat_ctr_k"], "Part 14 transfer")
add("gp_p14_m208_tab5cat15", 0.96207, M208 - 0.05 * U["tabpfn_ne4"] - 0.15 * U["cat_ctr_k"], "Part 14 transfer")
add("gp_p14_m208_lgb08", 0.96192, M208 + 0.8 * U["lgb"], "Part 14 probe")
add("gp_p14_m208_xgb04", 0.96195, M208 + 0.4 * U["xgb_k10"], "Part 14 probe")
add("gp_p14_m208_tab045", 0.96193, M208 + 0.45 * U["tabpfn_ne4"], "Part 14 probe")
add("gp_p14_m208_lgbm06", 0.96199, M208 - 0.6 * U["lgb"], "Part 14 check")
# Part 2 (10-19)
H198 = pr(csv_(G / "hanifnoerrofiq_agentic_1019.csv"))
step2 = pr(csv_(G / "surface_step2_1019.csv"))
add("gp_p2_h198_lgb02", 0.96196, H198 + 0.2 * U["lgb"], "Part 2")
add("gp_p2_h198_lgb03", 0.96195, H198 + 0.3 * U["lgb"], "Part 2")
add("gp_p2_final", 0.96197, 0.5 * step2 + 0.5 * H198, "Part 2 file")
# Section 3 (10-18)
anchor = pr(csv_(G / "anchor_twin_dp30_1017.csv"))
T188 = pr(csv_("data/ext/ps-s6e10-7/0.96188.txt"))
E188p = Path("data/tilt/k/ewtherka__s6e10-0-96188-lb-l2-family-wise-actually-helps/submission.csv")
if E188p.exists():
    E188 = pr(csv_(E188p))
    add("gp_s3_measure", 0.96191, anchor + 1.5 * (T188 - E188), "Section 3 transfer (E188 is the notebook's current output)")
# Part 4 (10-23): the TabPFN dose map on the 0.96213 file
M210 = pr(csv_(G / "mayankraaz_summit_composer_096210_1023.csv"))
PT = csv_(G / "mehrankazeminia_tabpfn_1023.csv")
NE4 = np.load(G / "test_tabpfn_ne4.npy").astype(np.float64).ravel()
Y213 = pr(csv_(G / "public_096213_1023.csv"))
check(0.90 * M210 + 0.10 * PT, Y213, "0.96213 recipe vs pinned")
for tt, s in ((-1, 0.96211), (-0.5, 0.96212), (0.5, 0.96211), (1, 0.96208)):
    add(f"gp_p4_tab_{tt:g}", s, 0.90 * M210 + 0.10 * (PT + tt * (PT - NE4)), "Part 4 dose map")
# Part 6 (10-28): probes A-F (E, G pinned), mirror side of the climb
P6 = pr(csv_(G / "meta_part4_096213_1023.csv"))
COMP = {"xgb_base": ("probe_comp_xgb_base_raw.csv", 0.95808, 0.0909), "gen12": ("probe_comp_gen_v2_o12.csv", 0.95343, 0.1021),
        "realmlp": ("probe_comp_realmlp_v4_s1.csv", 0.96082, 0.0858), "p2viewb": ("probe_comp_revisit_p2_viewb.csv", 0.96129, 0.085),
        "ens_v4": ("probe_comp_ens_v4_k10.csv", 0.96107, 0.085), "ne4f": ("probe_comp_ne4f_single.csv", 0.96110, 0.0880),
        "gen3": ("probe_comp_gen3_pilot.csv", 0.95662, 0.0927)}
D6 = {k: pr(csv_(G / f)) - P6 for k, (f, _, _) in COMP.items()}
g6 = {k: 0.96213 - p - 4 * c * D6[k].std() ** 2 for k, (_, p, c) in COMP.items()}


def push(keys, K, c=0.09):
    sig = np.array([[np.mean((D6[a] - D6[a].mean()) * (D6[b] - D6[b].mean())) for b in keys] for a in keys])
    return np.linalg.solve(sig, np.array([g6[k] for k in keys])) / (2 * K * c)


for tag, keys, K, s in [("A", ["xgb_base"], 8, 0.96214), ("B", ["gen12"], 8, 0.96213), ("C", ["xgb_base", "gen12"], 8, 0.96214),
                        ("D", ["xgb_base", "gen12", "realmlp", "p2viewb", "ens_v4", "ne4f"], 12, 0.96214),
                        ("F", ["xgb_base"], 12, 0.96214), ("G", ["xgb_base", "gen12", "realmlp", "gen3"], 12, 0.96216)]:
    eps = push(keys, K)
    b = P6 - sum(e * D6[k] for k, e in zip(keys, eps))
    if tag == "G":
        check(b, csv_(G / "meta_part6_probeG_096216_1028.csv"), "probe G rebuilt vs pinned")
    else:
        add(f"gp_p6_probe{tag}", s, b, "Part 6 probe")
Gr = pr(csv_(G / "meta_part6_probeG_096216_1028.csv"))
st = np.clip(np.load(G / "test_stack245.npy").astype(np.float64).ravel(), 1e-6, 1 - 1e-6)
add("gp_stack245", 0.96186, st, "frozen 245 stack")
scale = np.sort(np.log(st / (1 - st)))
z = scale[np.rint(Gr * N + 0.5).astype(int) - 1]
rng = np.random.default_rng(1001)
mask = rng.random(N) < 0.25
noise = rng.standard_normal(N) * 0.04 * mask
check(z - noise, csv_(G / "meta_part6_climb1_096217_1030.csv"), "climb rebuilt vs pinned")
add("gp_p6_climb_plus", 0.96213, z + noise, "Part 6 mirror + side")
# Part 16 (10-27): 7 rank blends
T16 = csv_(G / "twin_submission_1015.csv")
P16 = csv_(G / "meta_part4_096213_1023.csv")
runs = [("twin", T16, w) for w in (0.10, 0.25, 0.50, 0.75)] + [("p8", ser["test_ens_v18"], 0.5), ("p1", ser["test_ens_v3"], 0.5), ("p1", ser["test_ens_v3"], 0.25)]
for (nm, X, w), s in zip(runs, [0.96211, 0.96209, 0.96204, 0.96196, 0.96197, 0.96182, 0.96203]):
    add(f"gp_p16_{nm}_{w:g}", s, (1 - w) * pr(P16) + w * pr(X), "Part 16 blend")

# ---------------------------------------------------------------- tkms0816 Push Model
T = Path("data/ext/tkms-push")
for f, s in {"m_kumolarge.csv": 0.95956, "m_tabpfn35.csv": 0.96064, "m_xgb_te2f50.csv": 0.96067, "ref_r1_push_096217.csv": 0.96217,
             "ref_t3_096223.csv": 0.96223}.items():
    add("tk_" + f[:-4], s, csv_(T / f), "tkms0816 Push Model")
R1 = pr(csv_(T / "ref_r1_push_096217.csv"))
Mp = Path("data/tilt/k/mayankraaz__s6e10-mirror-pair-climb-0-96218/submission.csv")
M = pr(csv_(Mp))
Q = pr(csv_(G / "meta_part6_climb1_096217_1030.csv"))
A = np.column_stack([R1 - P6, M - P6])
Qp = (Q - P6) - A @ np.linalg.lstsq(A, Q - P6, rcond=None)[0]
check(R1 + 1.3 * (M - P6) + Qp, csv_(T / "ref_t3_096223.csv"), "T3 rebuilt vs pinned")
for nm, c1, c2, s in (("a", 1, 0, 0.96222), ("b", 1.3, 0, 0.96222), ("c", 1, 1, 0.96222), ("d", 1.6, 1.5, 0.96222), ("e", 1.3, 1.6, 0.96222)):
    add(f"tk_combo_{nm}", s, R1 + c1 * (M - P6) + c2 * Qp, "tkms0816 table")

# ---------------------------------------------------------------- mayankraaz Mirror Pair Climb: every scored step, forward
A7, AT = Path("data/ext/ps-s6e10-7"), Path("data/ext/s6e10-lb-surface-atlas")
raw7 = lambda f: csv_(A7 / f)  # noqa: E731
RC = pr(raw7("0.96206.csv"))
T188r = raw7("0.96188.txt")
BASE3 = [raw7(f) for f in ("0.96201.csv", "0.96203.csv", "0.96204.csv")]
V41, V43, V54 = (pr(csv_(AT / f"rabari_v{v}_submission.csv")) for v in (41, 43, 54))
OWN = np.load(AT / "own_cv_stack_181_test.npy").astype(np.float64)
TABPFN = PT
PUBLIC213 = Y213
PUB35 = np.load("data/ext/s6e10-tfm-oof-and-test-predictions/test_tabpfn35.npy").astype(np.float64).ravel()


def hblend(m, main, corr, mix=(0.3, 0.7)):
    n, sides = len(m), []
    for rev in (False, True):
        order = np.argsort(-m if rev else m, axis=1, kind="stable")
        s = np.zeros(n)
        for j in range(m.shape[1]):
            k = order[:, j]
            s += m[np.arange(n), k] * (main[k] + corr[j])
        sides.append(s.astype(np.float32))
    return (np.float32(mix[0]) * sides[0] + np.float32(mix[1]) * sides[1]).astype(np.float32)


def line(tv):
    a125, a450 = V41 - RC, V43 - RC
    return pr(RC + a125 + (tv - 1.25) / 3.25 * (a450 - a125))


def compose(fourth, teacher, tw, main, corr, gate=0.0):
    m = np.column_stack(BASE3 + [fourth])
    w = tw + gate * (pr(m.std(axis=1)) - 0.5)
    return pr(hblend(m * (1 - w[:, None]) + w[:, None] * teacher[:, None], np.asarray(main, float), np.asarray(corr, float)))


def recipe(tv, tw, own, main, corr, gate):
    return compose(line(tv), (1 - own) * T188r + own * OWN, tw, main, corr, gate)


base = recipe(12.0, 0.45, -1.0, [-0.25, 0.10, 0.15, 1.00], [0.14, -0.02, -0.04, -0.08], 0.0)
moved = recipe(13.0, 0.58, -1.5, [-0.35, 0.10, 0.15, 1.10], [0.175, -0.025, -0.05, -0.10], -0.05)
P2 = pr(V54 + (moved - base))
q = np.sort(0.10 * TABPFN + 0.90 * P2)
k = np.rint(PUBLIC213 * N + 0.5).astype(int) - 1
S_raw = (q[k] - 0.10 * TABPFN) / 0.90
S = pr(S_raw)
based = pr(S + 4.0 * (S - P2))
spec = TABPFN + 0.25 * (TABPFN - (PUB35 + NE4) / 2)
L = pr(0.095 * spec + 0.905 * based)
add("my_layer", 0.96213, L, "Mirror Pair step 4")
sc = np.sort(np.log(np.clip(OWN, 1e-6, 1 - 1e-6) / (1 - np.clip(OWN, 1e-6, 1 - 1e-6))))


def nudge(rank_file, noise_):
    zz = sc[np.rint(rank_file * N + 0.5).astype(int) - 1]
    return pr(zz + noise_)


def block_noise(kk, size, n_blocks, block_seed, noise_seed):
    blocks = np.random.default_rng(block_seed).integers(0, n_blocks, N)
    return np.random.default_rng(noise_seed).standard_normal(N) * size * (blocks == kk)


# (block_seed, block, kept sign, noise_seed, kept score, other side's score or None)
STEPS = [(2026, 0, +1, 100, 0.96213, None), (2026, 1, +1, 101, 0.96214, 0.96212), (2026, 2, -1, 102, 0.96215, 0.96212),
         (2026, 3, -1, 103, 0.96215, 0.96214), (2027, 0, +1, 2027000, 0.96216, None), (2027, 1, -1, 2027001, 0.96217, 0.96215),
         (2027, 2, +1, 2027002, 0.96217, 0.96216), (2027, 3, -1, 2027003, 0.96218, 0.96217)]
cur = L
for i, (bs, kk, sgn, ns, sk, so) in enumerate(STEPS):
    n_ = block_noise(kk, 0.02, 4, bs, ns)
    kept = nudge(cur, sgn * n_)
    add(f"my_step{i + 5}_kept", sk, kept, "Mirror Pair kept side")
    if so is not None:
        add(f"my_step{i + 5}_other", so, nudge(cur, -sgn * n_), "Mirror Pair other side")
    cur = kept
check(cur, csv_(Mp), "Mirror Pair final rebuilt vs its notebook output")

pd.DataFrame(rows).to_csv("data/tilt/mined.csv", index=False)
print(f"{len(rows)} mined files")
print(pd.DataFrame(rows).groupby(pd.DataFrame(rows).name.str[:3]).score.describe().to_string())
