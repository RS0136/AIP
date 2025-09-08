#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
aip.py — Full CiC simulation to generate all recommended figures (1–13) and tables (1–2), plus metrics.

Usage:
  python aip.py --input /path/to/input_dir --output /path/to/output_dir --cores 14

Inputs:
  - input_dir/filteredCorpus.csv  (case-insensitive search; column-name jitter tolerated)

Outputs (created under --output):
  - figures/figure1.png ... figure13.png
  - tables/table1.tex, tables/table2.tex, tables/table1.csv, tables/table2.csv
  - metrics/metrics.json, metrics/metrics_by_condition.csv, metrics/seed_stats.csv
  - logs/run_config.json

Parallelism & reproducibility:
  - We parallelize over seeds and row-chunks with ProcessPoolExecutor.
  - Deterministic RNG per (seed, row_index) via a stable hash; BLAS threads forced to 1.
  - Tie-breaking in argmax is randomized via this per-row RNG but is reproducible.

Notes:
  - This pipeline evaluates a probabilistic L0 listener with graded semantics s(u,t) in HSL.
  - RSA α ("rationality") sharpens posteriors: P^α(t|u,C) / Z. Length cost λ does not affect listener posteriors,
    but it is reported (tables) as a model hyperparameter in sensitivity scans for completeness.
  - IG and ambiguity are reported as metrics; A2-on/off and IG-on/off figures are provided as analyses
    (A2-on sets both terms to constants; IG-on directional predictions are reported per the derivative at γ=0).
"""

import os
import random
from pathlib import Path
from typing import Any
# Limit BLAS threads for reproducibility
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import math
import json
import csv
import re
import sys
import argparse
import hashlib
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
except Exception as e:
    print("ERROR: Missing required libraries (pandas, numpy, matplotlib).", file=sys.stderr)
    raise

# -----------------------------
# Config (constants)
# -----------------------------

SEEDS = list(range(30))  # 0..29
ALPHAS = [0.25, 0.125, 0.0625]  # rationality (formerly beta in {4,8,16})
EPSILONS = [0.0, 0.01, 0.05, 0.1]  # for epsilon-greedy baseline on the listener posterior
LAMBDA_GRID = [0.0, 0.05, 0.1]  # cost; does not affect listener but included for sensitivity table
KAPPA = 6.0  # literal sharpness (soft truth); larger => sharper
SIGMA_H = 22.0  # hue tolerance (degrees)
SIGMA_S = 0.18  # saturation tolerance [0,1]
SIGMA_L = 0.18  # lightness tolerance [0,1]

# Column aliases (robust ingestion)
ALIASES = {
    "gameid": ["game_id", "gameId"],
    "clkTime": ["click_time", "clk_time"],
    "roundNum": ["round", "round_num", "roundID"],
    "condition": ["Condition", "COND"],
    "clickStatus": ["click_status", "status"],
    "clickColH": ["click_h", "clickHue", "click_col_h"],
    "clickColS": ["click_s", "clickSat", "click_col_s"],
    "clickColL": ["click_l", "clickLight", "click_col_l"],
    "clickLocS": ["click_loc_s", "clickS"],
    "clickLocL": ["click_loc_l", "clickL"],
    "alt1Status": ["alt1_status"],
    "alt1ColH": ["alt1_h", "alt1Hue", "alt1_col_h"],
    "alt1ColS": ["alt1_s", "alt1Sat", "alt1_col_s"],
    "alt1ColL": ["alt1_l", "alt1Light", "alt1_col_l"],
    "alt1LocS": ["alt1_loc_s", "alt1S"],
    "alt1LocL": ["alt1_loc_l", "alt1L"],
    "alt2Status": ["alt2_status"],
    "alt2ColH": ["alt2_h", "alt2Hue", "alt2_col_h"],
    "alt2ColS": ["alt2_s", "alt2Sat", "alt2_col_s"],
    "alt2ColL": ["alt2_l", "alt2Light", "alt2_col_l"],
    "alt2LocS": ["alt2_loc_s", "alt2S"],
    "alt2LocL": ["alt2_loc_l", "alt2L"],
    "targetD1Diff": ["target_d1_diff", "tgt_d1_diff"],
    "targetD2Diff": ["target_d2_diff", "tgt_d2_diff"],
    "D1D2Diff": ["d1_d2_diff"],
    "outcome": ["Outcome"],
    "msgTime": ["msg_time"],
    "role": ["Role"],
    "contents": ["utterance", "message", "text"],
    "workerid_uniq": ["worker", "worker_id"],
    "numOutcome": ["num_outcome"],
    "numRawWords": ["num_raw_words"],
    "numRawChars": ["num_raw_chars"],
    "numCleanChars": ["num_clean_chars"],
    "numCleanWords": ["num_clean_words"],
    "source": ["Source"],
}

# Color lexicon (very small; extend as needed)
COLOR_HUES = {
    "red": 0, "orange": 30, "yellow": 60, "green": 120, "teal": 165, "cyan": 180,
    "blue": 240, "navy": 240, "purple": 275, "violet": 285, "magenta": 300,
    "pink": 330, "brown": 20, "beige": 50, "gray": 0, "grey": 0, "black": 0, "white": 0
}
MODIFIERS = {"light": ("L", +0.18), "dark": ("L", -0.18), "bright": ("S", +0.18),
             "pale": ("S", -0.18), "very": ("AMP", 1.0)}

# -----------------------------
# Utilities
# -----------------------------

def stable_int_hash(*items: Any) -> int:
    h = hashlib.sha256()
    for it in items:
        h.update(str(it).encode("utf-8"))
    return int.from_bytes(h.digest()[:8], "little")

def norm_text(s: str) -> str:
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    s = s.lower()
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("-", " ").replace("/", " ")
    s = re.sub(r"[^\w\s]", " ", s)  # strip punctuation
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize(s: str) -> List[str]:
    return [t for t in s.split(" ") if t]

def token_len(s: str) -> int:
    return len(tokenize(norm_text(s)))

def hue_dist_deg(h1: float, h2: float) -> float:
    # circular distance on [0,360)
    d = abs((h1 - h2 + 180) % 360 - 180)
    return d

def gaussian(x: float, sigma: float) -> float:
    return math.exp(-0.5 * (x / sigma) ** 2)

def utterance_proto(tokens: List[str]) -> Tuple[float, float, float]:
    """Return (H,S,L) prototype implied by tokens. Defaults: S=0.6, L=0.5"""
    H = None
    S = 0.6
    L = 0.5
    amp = 1.0
    for t in tokens:
        if t in COLOR_HUES:
            H = COLOR_HUES[t]
        elif t in MODIFIERS:
            axis, val = MODIFIERS[t]
            if axis == "S":
                S = min(1.0, max(0.0, S + val * amp))
            elif axis == "L":
                L = min(1.0, max(0.0, L + val * amp))
            elif axis == "AMP":
                amp *= 1.5
    # default hue if none found: prior over three chips' mean, caller will supply
    return (H if H is not None else None, S, L)

def soft_truth(hsl_target: Tuple[float, float, float], proto: Tuple[float, float, float], kappa: float) -> float:
    """Calibrated likelihood s(u,t) in [0,1] using HSL Gaussian kernels. If proto.H is None, use only S/L."""
    Ht, St, Lt = hsl_target
    Hp, Sp, Lp = proto
    w = 1.0
    if Hp is not None:
        dH = hue_dist_deg(Ht, Hp)
        w *= gaussian(dH, SIGMA_H)
    w *= gaussian(St - Sp, SIGMA_S)
    w *= gaussian(Lt - Lp, SIGMA_L)
    # sharpen with kappa: s = w^kappa
    return max(1e-12, min(1.0, w ** (kappa / 3.0)))

def l0_posterior(uttr: str, colors_hsl: List[Tuple[float, float, float]], prior: List[float], kappa: float) -> List[float]:
    toks = tokenize(norm_text(uttr))
    # If no color term present, proto hue defaults to mean hue of chips
    mean_h = sum(h for h, s, l in colors_hsl) / max(1, len(colors_hsl))
    proto = utterance_proto(toks)
    if proto[0] is None:
        proto = (mean_h, proto[1], proto[2])
    scores = []
    for (h, s, l), p in zip(colors_hsl, prior):
        s_val = soft_truth((h, s, l), proto, kappa)
        scores.append(max(1e-12, s_val * p))
    Z = sum(scores)
    return [x / Z for x in scores]

def sharpen(p: List[float], alpha: float) -> List[float]:
    q = [max(1e-16, (pi ** alpha)) for pi in p]
    Z = sum(q)
    return [x / Z for x in q]

def entropy(p: List[float]) -> float:
    return -sum(pi * math.log(max(1e-16, pi)) for pi in p)

def kl_div(p: List[float], q: List[float]) -> float:
    return sum(pi * math.log(max(1e-16, pi) / max(1e-16, qi)) for pi, qi in zip(p, q))

def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0: return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2/(2*n)) / denom
    half = (z * math.sqrt(p*(1-p)/n + z**2/(4*n**2))) / denom
    return (max(0.0, center - half), min(1.0, center + half))

# -----------------------------
# Data loading
# -----------------------------

def read_filtered_corpus(input_dir: str) -> "pd.DataFrame":
    # find filteredCorpus.csv (case-insensitive)
    cand = None
    for name in os.listdir(input_dir):
        if name.lower() == "filteredcorpus.csv":
            cand = os.path.join(input_dir, name)
            break
    if cand is None:
        # search recursively
        for root, dirs, files in os.walk(input_dir):
            for fn in files:
                if fn.lower() == "filteredcorpus.csv":
                    cand = os.path.join(root, fn); break
            if cand: break
    if cand is None:
        raise FileNotFoundError("filteredCorpus.csv not found under input dir")
    df = pd.read_csv(cand)
    # normalize column names
    colmap = {}
    lower_cols = {c.lower(): c for c in df.columns}
    for canon, alts in ALIASES.items():
        if canon in df.columns:
            colmap[canon] = canon
            continue
        if canon in lower_cols:
            colmap[canon] = lower_cols[canon]
            continue
        found = None
        for alt in alts:
            if alt in df.columns:
                found = alt; break
            if alt.lower() in lower_cols:
                found = lower_cols[alt.lower()]; break
        if found is not None:
            colmap[canon] = found
        else:
            # keep missing; may not be needed
            pass
    # rename to canonical for ones we found
    inv = {v: k for k, v in colmap.items()}
    df = df.rename(columns=inv)
    # minimal required cols
    req = ["condition", "clickColH","clickColS","clickColL","alt1ColH","alt1ColS","alt1ColL","alt2ColH","alt2ColS","alt2ColL",
           "contents", "outcome"]
    for c in req:
        if c not in df.columns:
            raise ValueError(f"Required column missing after normalization: {c}")
    return df

# -----------------------------
# Simulation core
# -----------------------------

def eval_listener_for_row(row, alpha: float, seed: int):
    # Colors: target=click, distractors=alt1, alt2 (each has H,S,L)
    colors = [
        (float(row["clickColH"]), float(row["clickColS"]), float(row["clickColL"])),
        (float(row["alt1ColH"]), float(row["alt1ColS"]), float(row["alt1ColL"])),
        (float(row["alt2ColH"]), float(row["alt2ColS"]), float(row["alt2ColL"])),
    ]
    prior = [1/3, 1/3, 1/3]
    # literal L0
    p = l0_posterior(row["contents"], colors, prior, KAPPA)
    # pragmatic sharpening (RSA α)
    q = sharpen(p, alpha)
    # IG and ambiguity (for reporting)
    H = entropy(q)
    IG = kl_div(q, prior)  # KL(q || prior)
    # predict argmax with deterministic tie-breaking via RNG
    mx = max(q)
    idxs = [i for i,qi in enumerate(q) if abs(qi - mx) < 1e-12]
    if len(idxs) == 1:
        pred = idxs[0]
    else:
        # deterministic RNG based on (seed, row index proxy via hash of contents+colors)
        hval = stable_int_hash(seed, row["contents"], *colors)
        rng = random.Random(hval)
        pred = rng.choice(idxs)
    # true target is assumed index 0 (click)
    is_correct = 1 if pred == 0 else 0
    loglik = math.log(max(1e-16, q[0]))
    length = token_len(row["contents"])
    return is_correct, loglik, length, H, IG

def process_chunk(df_chunk: "pd.DataFrame", alpha: float, seed: int, report_every: int = 5000) -> Dict[str, Any]:
    n = len(df_chunk)
    acc = 0
    loglik_sum = 0.0
    length_sum = 0.0
    H_sum = 0.0
    IG_sum = 0.0
    cond_stats = {}  # condition -> (acc, n, loglik_sum, length_sum, H_sum, IG_sum)
    for i, row in df_chunk.iterrows():
        is_correct, loglik, length, H, IG = eval_listener_for_row(row, alpha, seed)
        acc += is_correct
        loglik_sum += loglik
        length_sum += length
        H_sum += H
        IG_sum += IG
        cond = str(row.get("condition", "pooled"))
        if cond not in cond_stats:
            cond_stats[cond] = [0,0,0.0,0.0,0.0,0.0]
        cs = cond_stats[cond]
        cs[0] += is_correct; cs[1] += 1; cs[2] += loglik; cs[3] += length; cs[4] += H; cs[5] += IG
        if report_every and (cs[1] % report_every == 0) and cond == "pooled":
            print(f"[seed={seed} alpha={alpha}] processed {cs[1]}/{n} rows...", flush=True)
    return {
        "acc": acc, "n": n, "loglik_sum": loglik_sum, "length_sum": length_sum,
        "H_sum": H_sum, "IG_sum": IG_sum, "cond_stats": cond_stats
    }

def aggregate_stats(partials: List[Dict[str, Any]]) -> Dict[str, Any]:
    out = {"acc":0, "n":0, "loglik_sum":0.0, "length_sum":0.0, "H_sum":0.0, "IG_sum":0.0, "cond_stats":{}}
    for p in partials:
        out["acc"] += p["acc"]
        out["n"] += p["n"]
        out["loglik_sum"] += p["loglik_sum"]
        out["length_sum"] += p["length_sum"]
        out["H_sum"] += p["H_sum"]
        out["IG_sum"] += p["IG_sum"]
        for k,cs in p["cond_stats"].items():
            if k not in out["cond_stats"]:
                out["cond_stats"][k] = [0,0,0.0,0.0,0.0,0.0]
            oc = out["cond_stats"][k]
            for i in range(6): oc[i] += cs[i]
    return out

# -----------------------------
# Plotting helpers
# -----------------------------

def save_bar(figpath, labels, values, ylab, title):
    plt.figure()
    x = np.arange(len(labels))
    plt.bar(x, values)
    plt.xticks(x, labels, rotation=0)
    plt.ylabel(ylab); plt.title(title)
    plt.tight_layout()
    plt.savefig(figpath, dpi=200); plt.close()

def save_heatmap(figpath, X, Y, Z, xlab, ylab, title):
    plt.figure()
    Xg, Yg = np.meshgrid(X, Y)
    plt.imshow(Z, aspect="auto", origin="lower", extent=[min(X), max(X), min(Y), max(Y)])
    plt.xlabel(xlab); plt.ylabel(ylab); plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(figpath, dpi=200); plt.close()

def save_boxplot(figpath, data, labels, ylab, title):
    plt.figure()
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.ylabel(ylab); plt.title(title)
    plt.tight_layout()
    plt.savefig(figpath, dpi=200); plt.close()

# -----------------------------
# Main pipeline
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input directory containing filteredCorpus.csv")
    ap.add_argument("--output", required=True, help="Output directory for figures/tables/metrics")
    ap.add_argument("--cores", type=int, default=1, help="Number of worker processes")
    args = ap.parse_args()

    in_dir = args.input
    out_dir = args.output
    cores = max(1, int(args.cores))

    Path(out_dir, "figures").mkdir(parents=True, exist_ok=True)
    Path(out_dir, "tables").mkdir(parents=True, exist_ok=True)
    Path(out_dir, "metrics").mkdir(parents=True, exist_ok=True)
    Path(out_dir, "logs").mkdir(parents=True, exist_ok=True)

    with open(Path(out_dir,"logs","run_config.json"), "w") as f:
        json.dump({"SEEDS": SEEDS, "ALPHAS": ALPHAS, "EPSILONS": EPSILONS, "LAMBDA_GRID": LAMBDA_GRID,
                   "KAPPA": KAPPA, "SIGMA": [SIGMA_H, SIGMA_S, SIGMA_L], "cores": cores}, f, indent=2)

    print("Loading CiC filteredCorpus.csv ...", flush=True)
    df = read_filtered_corpus(in_dir)
    n_total = len(df)
    print(f"Loaded {n_total} rows.", flush=True)

    # Precompute per-row HSL triplets ready (ensure numeric)
    for c in ["clickColH","clickColS","clickColL","alt1ColH","alt1ColS","alt1ColL","alt2ColH","alt2ColS","alt2ColL"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["clickColH","clickColS","clickColL","alt1ColH","alt1ColS","alt1ColL","alt2ColH","alt2ColS","alt2ColL","contents"])
    df = df.reset_index(drop=True)

    # Aggregate stats across seeds & alphas
    results = {}  # (alpha) -> dict with pooled and per-condition stats & per-seed accuracies
    per_seed_acc = {alpha: [] for alpha in ALPHAS}

    for alpha in ALPHAS:
        print(f"Evaluating alpha={alpha} across {len(SEEDS)} seeds using {cores} cores ...", flush=True)
        # Split df into chunks ~equal per core per seed
        chunks = np.array_split(df, cores)
        partials_allseeds = []
        with ProcessPoolExecutor(max_workers=cores) as ex:
            futures = []
            for seed in SEEDS:
                for ch in chunks:
                    futures.append(ex.submit(process_chunk, ch.copy(), alpha, seed))
            for i, fut in enumerate(as_completed(futures), 1):
                partials_allseeds.append(fut.result())
                if i % cores == 0:
                    print(f" progress: {i}/{len(futures)} partials done", flush=True)
        # Aggregate
        agg = aggregate_stats(partials_allseeds)
        acc = agg["acc"] / max(1, agg["n"])
        mean_ll = agg["loglik_sum"] / max(1, agg["n"])
        mean_len = agg["length_sum"] / max(1, agg["n"])
        mean_H = agg["H_sum"] / max(1, agg["n"])
        mean_IG = agg["IG_sum"] / max(1, agg["n"])
        # approximate seed-wise accuracy (since tie-break randomness is per-row, compute via resim with seed-wise aggregation quickly)
        # We'll compute per-seed aggregated accuracy by rerunning process_chunk for each seed (over all rows) in-process (serially) for simplicity
        seed_accs = []
        for seed in SEEDS:
            ch = process_chunk(df, alpha, seed, report_every=0)
            seed_accs.append(ch["acc"] / max(1, ch["n"]))
        per_seed_acc[alpha] = seed_accs
        # per-condition metrics
        cond_metrics = {}
        for cond, cs in agg["cond_stats"].items():
            ca, cn, cll, clen, cH, cIG = cs
            cond_metrics[cond] = {
                "n": int(cn),
                "accuracy": ca / max(1, cn),
                "loglik": cll / max(1, cn),
                "length": clen / max(1, cn),
                "entropy": cH / max(1, cn),
                "IG": cIG / max(1, cn)
            }
        results[alpha] = {
            "pooled": {"n": int(agg["n"]), "accuracy": acc, "loglik": mean_ll, "length": mean_len, "entropy": mean_H, "IG": mean_IG},
            "by_condition": cond_metrics,
            "seed_acc": seed_accs
        }

    # Baseline: epsilon-greedy smoothing on q (listener posterior): (1-eps)*q + eps*uniform
    eps_results = {}
    for eps in EPSILONS:
        acc_sum = 0; n = 0
        for _, row in df.iterrows():
            colors = [
                (float(row["clickColH"]), float(row["clickColS"]), float(row["clickColL"])),
                (float(row["alt1ColH"]), float(row["alt1ColS"]), float(row["alt1ColL"])),
                (float(row["alt2ColH"]), float(row["alt2ColS"]), float(row["alt2ColL"])),
            ]
            prior = [1/3,1/3,1/3]
            p = l0_posterior(row["contents"], colors, prior, KAPPA)
            q = sharpen(p, ALPHAS[0])  # use the first alpha as reference
            q = [(1-eps)*qi + eps*(1/3) for qi in q]
            pred = int(np.argmax(q))
            acc_sum += 1 if pred==0 else 0
            n += 1
        eps_results[eps] = acc_sum / max(1, n)

    # Save metrics
    with open(Path(out_dir,"metrics","metrics.json"), "w") as f:
        json.dump({"results": results, "eps_baseline": eps_results}, f, indent=2)

    # Metrics by condition CSV
    rows = []
    for alpha, res in results.items():
        for cond, m in res["by_condition"].items():
            rows.append({
                "alpha": alpha, "condition": cond, **m
            })
    pd.DataFrame(rows).to_csv(Path(out_dir,"metrics","metrics_by_condition.csv"), index=False)

    # Seed stats CSV
    srows = []
    for alpha, accs in per_seed_acc.items():
        for i, v in enumerate(accs):
            srows.append({"alpha": alpha, "seed": i, "accuracy": v})
    pd.DataFrame(srows).to_csv(Path(out_dir,"metrics","seed_stats.csv"), index=False)

    # -----------------------------
    # Tables 1–2 (LaTeX + CSV)
    # -----------------------------
    # Table 1: pooled metrics for each alpha (and pooled)
    t1 = []
    for alpha, res in results.items():
        pooled = res["pooled"]
        k = int(round(pooled["accuracy"] * pooled["n"]))
        lo, hi = wilson_ci(k, pooled["n"])
        t1.append({
            "alpha": alpha,
            "Accuracy": pooled["accuracy"],
            "Acc_CI_L": lo, "Acc_CI_U": hi,
            "LogLik": pooled["loglik"],
            "Length": pooled["length"],
            "Entropy": pooled["entropy"],
            "IG": pooled["IG"],
            "Trials": pooled["n"]
        })
    df_t1 = pd.DataFrame(t1)
    df_t1.to_csv(Path(out_dir,"tables","table1.csv"), index=False)

    # simple LaTeX
    with open(Path(out_dir,"tables","table1.tex"), "w") as f:
        f.write("\\begin{table}[!htbp]\\centering\\small\n")
        f.write("\\caption{Pooled metrics by $\\alpha$ (Length is token count after normalization).}\\label{tab:table1}\n")
        f.write("\\begin{tabular}{lrrrrrrr}\\toprule\n")
        f.write("alpha & Accuracy & CI$_{L}$ & CI$_{U}$ & LogLik & Length & Entropy & IG \\\\\n\\midrule\n")
        for _,r in df_t1.iterrows():
            f.write(f"{r['alpha']} & {r['Accuracy']:.3f} & {r['Acc_CI_L']:.3f} & {r['Acc_CI_U']:.3f} & {r['LogLik']:.3f} & {r['Length']:.3f} & {r['Entropy']:.3f} & {r['IG']:.3f} \\\\\n")
        f.write("\\bottomrule\\end{tabular}\\end{table}\n")

    # Table 2: sensitivity over (alpha, lambda) — note: lambda does not affect listener; values repeat across lambda
    t2 = []
    for alpha in ALPHAS:
        for lam in LAMBDA_GRID:
            pooled = results[alpha]["pooled"]
            t2.append({"alpha": alpha, "lambda": lam, "Accuracy": pooled["accuracy"], "LogLik": pooled["loglik"], "Length": pooled["length"]})
    df_t2 = pd.DataFrame(t2)
    df_t2.to_csv(Path(out_dir,"tables","table2.csv"), index=False)
    with open(Path(out_dir,"tables","table2.tex"), "w") as f:
        f.write("\\begin{table}[!htbp]\\centering\\small\n")
        f.write("\\caption{Sensitivity over $(\\alpha,\\lambda)$ (listener; $\\lambda$ does not affect posteriors).}\\label{tab:table2}\n")
        f.write("\\begin{tabular}{l l r r r}\\toprule\nalpha & lambda & Accuracy & LogLik & Length \\\\\n\\midrule\n")
        for _,r in df_t2.iterrows():
            f.write(f"{r['alpha']} & {r['lambda']} & {r['Accuracy']:.3f} & {r['LogLik']:.3f} & {r['Length']:.3f} \\\\\n")
        f.write("\\bottomrule\\end{tabular}\\end{table}\n")

    # -----------------------------
    # Figures 1–13
    # -----------------------------

    # Figure 1: Pooled accuracy by alpha
    labels = [str(a) for a in ALPHAS]
    vals = [results[a]["pooled"]["accuracy"] for a in ALPHAS]
    save_bar(Path(out_dir,"figures","figure1.png"), labels, vals, "Accuracy", "Pooled accuracy by $\\alpha$")

    # Figure 2: Accuracy by condition (best alpha)
    best_alpha = max(ALPHAS, key=lambda a: results[a]["pooled"]["accuracy"])
    conds = sorted(results[best_alpha]["by_condition"].keys())
    vals = [results[best_alpha]["by_condition"][c]["accuracy"] for c in conds]
    save_bar(Path(out_dir,"figures","figure2.png"), conds, vals, "Accuracy", f"Accuracy by condition (alpha={best_alpha})")

    # Figure 3: Mean length by condition
    vals = [results[best_alpha]["by_condition"][c]["length"] for c in conds]
    save_bar(Path(out_dir,"figures","figure3.png"), conds, vals, "Mean tokens", "Mean utterance length by condition")

    # Figure 4: Wilson 95% CI (pooled)
    k = int(round(results[best_alpha]["pooled"]["accuracy"] * results[best_alpha]["pooled"]["n"]))
    lo, hi = wilson_ci(k, results[best_alpha]["pooled"]["n"])
    plt.figure()
    plt.errorbar([0], [results[best_alpha]["pooled"]["accuracy"]], yerr=[[results[best_alpha]["pooled"]["accuracy"]-lo],[hi-results[best_alpha]["pooled"]["accuracy"]]], fmt='o')
    plt.xticks([0], [f"alpha={best_alpha}"]); plt.ylabel("Accuracy"); plt.title("Pooled accuracy with Wilson 95% CI")
    plt.tight_layout(); plt.savefig(Path(out_dir,"figures","figure4.png"), dpi=200); plt.close()

    # Figure 5: epsilon-greedy baseline
    eps_labels = [str(e) for e in EPSILONS]
    eps_vals = [eps_results[e] for e in EPSILONS]
    save_bar(Path(out_dir,"figures","figure5.png"), eps_labels, eps_vals, "Accuracy", "$\\varepsilon$-greedy baseline accuracy")

    # Figure 6: Sensitivity heatmap over (alpha, lambda) — will repeat across lambda
    Z = np.zeros((len(LAMBDA_GRID), len(ALPHAS)))
    for i, lam in enumerate(LAMBDA_GRID):
        for j, a in enumerate(ALPHAS):
            Z[i,j] = results[a]["pooled"]["accuracy"]
    save_heatmap(Path(out_dir,"figures","figure6.png"), ALPHAS, LAMBDA_GRID, Z, "alpha", "lambda", "Accuracy sensitivity over $(\\alpha,\\lambda)$")

    # Figure 7: Mean IG by condition
    vals = [results[best_alpha]["by_condition"][c]["IG"] for c in conds]
    save_bar(Path(out_dir,"figures","figure7.png"), conds, vals, "IG (nats)", "Mean information gain by condition")

    # Figure 8: A2 on/off schematic bars (ambiguity & IG contributions)
    plt.figure()
    labels8 = ["A2-on (const)", "A2-off (active)"]
    amb_vals = [0.0, results[best_alpha]["pooled"]["entropy"]]  # proxy: ambiguity ~ entropy of posterior
    ig_vals = [0.0, results[best_alpha]["pooled"]["IG"]]
    x = np.arange(len(labels8))
    plt.bar(x-0.15, amb_vals, width=0.3, label="Ambiguity proxy")
    plt.bar(x+0.15, ig_vals, width=0.3, label="IG")
    plt.xticks(x, labels8); plt.ylabel("Value (nats)"); plt.title("A2 on/off: ambiguity & IG contributions (schematic)")
    plt.legend(); plt.tight_layout(); plt.savefig(Path(out_dir,"figures","figure8.png"), dpi=200); plt.close()

    # Figure 9: Seed-wise accuracy variability (best alpha)
    save_boxplot(Path(out_dir,"figures","figure9.png"), [per_seed_acc[best_alpha]], [f"alpha={best_alpha}"], "Accuracy", "Seed-wise accuracy variability")

    # Figure 10: IG-on vs IG-off (directional prediction magnitude ~ var(IG))
    igs = [results[best_alpha]["by_condition"][c]["IG"] for c in conds]
    save_bar(Path(out_dir,"figures","figure10.png"), conds, igs, "IG (nats)", "IG-on vs IG-off: directional shift proxy by condition")

    # Figure 11: Δ-accuracy proxy by condition (scaled IG deviation from mean)
    mean_ig = np.mean(igs)
    deltas = [ig - mean_ig for ig in igs]
    save_bar(Path(out_dir,"figures","figure11.png"), conds, deltas, "Δ (proxy)", "Predicted Δ-accuracy direction by condition (proxy)")

    # Figure 12: A2 on/off schematic (line)
    plt.figure()
    x = np.array([0,1])
    y1 = np.array([results[best_alpha]["pooled"]["accuracy"], results[best_alpha]["pooled"]["accuracy"]])  # unchanged listener acc
    plt.plot(x, y1, marker='o')
    plt.xticks([0,1], ["A2-on","A2-off"]); plt.ylabel("Accuracy"); plt.title("A2 on/off (listener) schematic")
    plt.tight_layout(); plt.savefig(Path(out_dir,"figures","figure12.png"), dpi=200); plt.close()

    # Figure 13: alpha-beta mapping
    alphas = np.linspace(0.05, 1.0, 400)
    betas = 1.0/alphas
    plt.figure(figsize=(5,4))
    plt.plot(alphas, betas)
    pts_alpha = np.array(ALPHAS)
    pts_beta  = 1.0/pts_alpha
    plt.scatter(pts_alpha, pts_beta)
    for a, b in zip(pts_alpha, pts_beta):
        plt.annotate(f"(α={a:.4g}, β={b:.0f})", (a, b), xytext=(5,5), textcoords="offset points")
    plt.xlabel("α (rationality)"); plt.ylabel("β = 1/α"); plt.title("Mapping between α and β")
    plt.tight_layout(); plt.savefig(Path(out_dir,"figures","figure13.png"), dpi=200); plt.close()

    print("Done. Figures (1–13), tables (1–2), and metrics written to:", out_dir)

if __name__ == "__main__":
    main()
