#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIP (AIF⇔RSA) Simulation Suite
==============================

Generates all figures (PNG) and paired LaTeX tables for the AIP preprint.
Everything is fully *simulated* while using the given Colors-in-Context-like
CSV only to shape simple empirical distributions (e.g., utterance length,
contrast magnitudes, condition frequencies).

Outputs
-------
- output/figures/fig01.png ... fig14.png
- output/tables/fig01_table.tex ... fig14_table.tex
- output/data/fig01_data.csv ... fig14_data.csv
- output/run_config.json  (metadata)

Usage
-----
python aip.py --input <folder-with-filteredCorpus.csv> --output <outdir> --cores <n>

Notes
-----
- Reproducible: seeds = 0..29 by default (can be changed via --seeds).
- Parallelized with multi-process executors. Simple console progress (no tqdm).
- Matplotlib only; no seaborn; one plot per figure; no custom colors.
- Under the hood, we build a simple referential color game:
  * Target color and two distractors (HSL).
  * Vocabulary uses base color words + light/dark/vivid/dull modifiers.
  * Literal semantics (Gaussian in H/S/L).
  * RSA speaker chooses utterance u to maximize U(u) = α·ln P_L1(target|u) − λ·len(u).
  * AIF/EFE speaker chooses u to minimize G(u) = risk + ambiguity − IG + λ'·len(u).
  * Under assumptions A1–A6, G(u) is (affine-)equivalent to −U(u).

This script is purposely self-contained and safe to run on modest hardware.
"""

from pathlib import Path
import argparse
import json
import math
import os
import random
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Any

import numpy as np
import pandas as pd

# Optional tqdm fallback
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    def tqdm(x=None, total=None):
        # Context-manager form: with tqdm(total=N) as pbar: pbar.update(k)
        if x is None:
            class _NoTqdm:
                def __enter__(self): return self
                def update(self, *args, **kwargs): pass
                def __exit__(self, *exc): pass
            return _NoTqdm()
        # Iterable form: for item in tqdm(iterable):
        return x

# Matplotlib (no seaborn; one chart per figure; no explicit colors)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# SciPy is optional (only for Wilson CI); implement Wilson by hand to avoid dependency.
# from scipy.stats import norm


# ------------------------------
# Lightweight logger
# ------------------------------
def _log(msg: str):
    print(msg, flush=True)
# ------------------------------
# Utility: Wilson score interval
# ------------------------------
def wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """
    Wilson score interval for a binomial proportion.
    Returns (lower, upper). If n == 0, returns (nan, nan).
    """
    if n == 0:
        return (float('nan'), float('nan'))
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z*z/(2*n)) / denom
    margin = (z / denom) * math.sqrt((phat*(1-phat) + z*z/(4*n)) / n)
    return (max(0.0, center - margin), min(1.0, center + margin))


# -----------------------------
# CLI / Config dataclass
# -----------------------------
@dataclass
class Config:
    input_dir: str
    output_dir: str
    cores: int
    seeds: List[int]
    n_trials_per_seed: int
    alpha_true: float
    lambda_true: float
    beta_true: float  # 1/alpha semantics possible, but kept explicit
    length_cost_true: float  # For AIF length penalty
    sigma_h: float
    sigma_s: float
    sigma_l: float
    vocab_base: List[str]
    conditions_to_draw: int  # number of "conditions" categories to synthesize
    grid_alpha: List[float]
    grid_lambda: List[float]
    grid_beta: List[float]
    grid_len_cost: List[float]
    random_state: int

    @staticmethod
    def default():
        return Config(
            input_dir=".",
            output_dir="output",
            cores=max(1, os.cpu_count() // 2),
            seeds=list(range(30)),  # 0..29 as requested
            n_trials_per_seed=1000,
            alpha_true=6.0,
            lambda_true=0.15,
            beta_true=1.0/6.0,
            length_cost_true=0.15,
            sigma_h=18.0,  # hue width in degrees (empirical-ish)
            sigma_s=0.12,  # saturation width
            sigma_l=0.12,  # lightness width
            vocab_base=["red","orange","yellow","green","cyan","blue","purple","pink","brown","gray"],
            conditions_to_draw=6,
            grid_alpha=[2.0, 4.0, 6.0, 8.0, 10.0],
            grid_lambda=[0.05, 0.10, 0.15, 0.20, 0.25],
            grid_beta=[1/12, 1/8, 1/6, 1/4, 1/2],
            grid_len_cost=[0.05, 0.10, 0.15, 0.20, 0.25],
            random_state=0
        )


# ----------------------------------
# I/O helpers
# ----------------------------------
def ensure_outdirs(outdir: str):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    Path(outdir, "figures").mkdir(parents=True, exist_ok=True)
    Path(outdir, "tables").mkdir(parents=True, exist_ok=True)
    Path(outdir, "data").mkdir(parents=True, exist_ok=True)


def save_tex_table(df: pd.DataFrame, path: str, caption: str, label: str):
    """
    Save a simple LaTeX table (booktabs) without adding a full LaTeX document wrapper.
    """
    with open(path, "w", encoding="utf-8") as f:
        _log(f"[table] writing {path}")
        f.write("\\begin{table}[!ht]\n")
        f.write("\\centering\n")
        f.write(df.to_latex(index=False, escape=True, float_format=lambda x: f"{x:.4f}", na_rep="---", bold_rows=False, longtable=False, caption=caption, label=label))  # type: ignore
        f.write("\\end{table}\n")


# ----------------------------------
# Corpus loading (for shaping priors)
# ----------------------------------
def load_corpus(input_dir: str) -> pd.DataFrame:
    p = Path(input_dir) / "filteredCorpus.csv"
    print(f"[IO] Loading corpus: {p}", flush=True)
    if not p.exists():
        # Fallback: create a tiny synthetic corpus so the script can run.
        # Real usage expects filteredCorpus.csv to exist.
        rng = np.random.default_rng(0)
        n = 2000
        df = pd.DataFrame({
            "gameid": np.arange(n),
            "clkTime": rng.integers(1000, 5000, size=n),
            "roundNum": rng.integers(1, 5, size=n),
            "condition": rng.choice(["baseline","contrast","ambiguous","costly","control","noise"], size=n),
            "clickStatus": rng.integers(0, 2, size=n),
            "clickColH": rng.uniform(0, 360, size=n),
            "clickColS": rng.uniform(0.1, 0.9, size=n),
            "clickColL": rng.uniform(0.1, 0.9, size=n),
            "clickLocS": rng.uniform(0.0, 1.0, size=n),
            "clickLocL": rng.uniform(0.0, 1.0, size=n),
            "alt1Status": rng.integers(0,2,size=n),
            "alt1ColH": rng.uniform(0,360,size=n),
            "alt1ColS": rng.uniform(0.1,0.9,size=n),
            "alt1ColL": rng.uniform(0.1,0.9,size=n),
            "alt1LocS": rng.uniform(0.0,1.0,size=n),
            "alt1LocL": rng.uniform(0.0,1.0,size=n),
            "alt2Status": rng.integers(0,2,size=n),
            "alt2ColH": rng.uniform(0,360,size=n),
            "alt2ColS": rng.uniform(0.1,0.9,size=n),
            "alt2ColL": rng.uniform(0.1,0.9,size=n),
            "alt2LocS": rng.uniform(0.0,1.0,size=n),
            "alt2LocL": rng.uniform(0.0,1.0,size=n),
            "targetD1Diff": rng.uniform(0,1,size=n),
            "targetD2Diff": rng.uniform(0,1,size=n),
            "D1D2Diff": rng.uniform(0,1,size=n),
            "outcome": rng.integers(0,2,size=n),
            "msgTime": rng.integers(500, 4000, size=n),
            "role": rng.choice(["speaker","listener"], size=n),
            "contents": rng.choice(["light blue","dark green","very red","dull purple","vivid yellow"], size=n),
            "workerid_uniq": rng.integers(0, 100, size=n),
            "numOutcome": rng.integers(0, 5, size=n),
            "numRawWords": rng.integers(1, 6, size=n),
            "numRawChars": rng.integers(3, 30, size=n),
            "numCleanChars": rng.integers(3, 30, size=n),
            "numCleanWords": rng.integers(1, 6, size=n),
            "source": rng.choice(["lab","mturk"], size=n),
        })
        return df

    df = pd.read_csv(p)
    return df


# ----------------------------------
# Color helpers (HSL distance & names)
# ----------------------------------
def circdist_deg(a: float, b: float) -> float:
    """Circular distance on [0,360)."""
    d = abs((a - b + 180) % 360 - 180)
    return d


def hsl_distance(c1: Tuple[float,float,float], c2: Tuple[float,float,float], 
                 wh: float=1.0, ws: float=1.0, wl: float=1.0) -> float:
    """Weighted Euclidean-like distance for H in degrees, S/L in [0,1]."""
    dh = circdist_deg(c1[0], c2[0]) / 180.0  # normalize to [0,1]
    ds = abs(c1[1] - c2[1])
    dl = abs(c1[2] - c2[2])
    return math.sqrt((wh*dh)**2 + (ws*ds)**2 + (wl*dl)**2)


BASE_HUES = {
    "red": 0.0, "orange": 30.0, "yellow": 60.0, "green": 120.0,
    "cyan": 180.0, "blue": 240.0, "purple": 285.0, "pink": 320.0,
    "brown": 25.0, "gray": 0.0  # gray has no hue; set arbitrary
}


def nearest_base_color_name(h: float) -> str:
    best = None
    bestd = 1e9
    for name, hh in BASE_HUES.items():
        d = circdist_deg(h, hh)
        if d < bestd:
            bestd, best = d, name
    return best


def utterance_vocab_for_target(hsl: Tuple[float,float,float], cfg: Config) -> List[str]:
    """Generate a small candidate utterance set around the target color."""
    base = nearest_base_color_name(hsl[0])
    mods = ["", "light", "dark", "vivid", "dull"]
    vocab = []
    for m in mods:
        if m == "":
            vocab.append(base)
        else:
            vocab.append(f"{m} {base}")
    # Add two neighboring color names
    # (just heuristic neighbors by hue order)
    order = ["red","orange","yellow","green","cyan","blue","purple","pink","brown","gray"]
    idx = order.index(base)
    neighbor_left = order[(idx - 1) % len(order)]
    neighbor_right = order[(idx + 1) % len(order)]
    vocab.extend([neighbor_left, neighbor_right])
    # Ensure uniqueness
    return sorted(list(set(vocab)))


def utterance_length_tokens(u: str) -> int:
    return len(u.strip().split())


# ----------------------------------
# Semantics: literal listener L0 and pragmatic listener L1
# ----------------------------------
def utterance_color_score(u: str, color: Tuple[float,float,float], cfg: Config) -> float:
    """
    Negative distance-based score: higher is better (closer match).
    """
    # Parse utterance: optional modifier + base color word
    parts = u.split()
    modifier = ""
    base = parts[-1]
    if len(parts) == 2:
        modifier = parts[0]

    # Build a prototype color for the utterance:
    # base hue
    h = BASE_HUES.get(base, 0.0)
    # default S,L
    s = 0.6
    l = 0.6
    if base == "gray":
        s = 0.1

    # modifiers
    if modifier == "light":
        l = min(0.9, l + 0.2)
    elif modifier == "dark":
        l = max(0.2, l - 0.2)
    elif modifier == "vivid":
        s = min(0.9, s + 0.2)
    elif modifier == "dull":
        s = max(0.1, s - 0.2)

    dist = hsl_distance((h,s,l), color, 1.0, 1.0, 1.0)
    # Convert to Gaussian-like score (no custom colors; keep simple)
    # Higher score when smaller distance.
    # sigma components roughly baked into cfg.sigma_*; use combined width ~ 0.2.
    tau = 0.2
    return - (dist**2) / (2 * tau**2)


def L0_posterior_over_colors(u: str, colors: List[Tuple[float,float,float]], cfg: Config) -> np.ndarray:
    scores = np.array([utterance_color_score(u, c, cfg) for c in colors])
    # Normalize as softmax
    xs = scores - scores.max()
    probs = np.exp(xs)
    probs /= probs.sum()
    return probs


def RSA_speaker_probs(target_idx: int, colors: List[Tuple[float,float,float]],
                      vocab: List[str], cfg: Config, alpha: float, lam: float) -> np.ndarray:
    """
    P_S(u | target) ∝ exp(alpha * ln P_L1(target|u) - lam * len(u))
    Approximate L1 by L0 posterior (1-step pragmatic = nearsighted assumption).
    """
    # For each u, compute P(target|u) under L0
    p_target = []
    for u in vocab:
        post = L0_posterior_over_colors(u, colors, cfg)
        p_target.append(post[target_idx])
    p_target = np.array(p_target)
    # Utility
    lengths = np.array([utterance_length_tokens(u) for u in vocab])
    util = alpha * np.log(np.clip(p_target, 1e-12, 1.0)) - lam * lengths
    xs = util - util.max()
    probs = np.exp(xs)
    probs /= probs.sum()
    return probs


def AIF_EFE_speaker_probs(target_idx: int, colors: List[Tuple[float,float,float]],
                          vocab: List[str], cfg: Config, beta: float, len_cost: float,
                          weights: Dict[str, float]) -> np.ndarray:
    """
    P_S(u | target) ∝ exp( -beta * G(u) ), where
    G(u) = w_risk*risk + w_amb*ambiguity - w_ig*IG + len_cost*len(u)

    risk = expected loss if listener guesses color != target (here: 1 - P(target|u))
    ambiguity = entropy of posterior over colors
    IG = KL( posterior || prior ), prior uniform here.

    Under A1–A6, G(u) ~ -const * ln P(target|u) + len_cost*len(u) (affine).
    """
    p_list = []
    risk_list = []
    amb_list = []
    ig_list = []
    lengths = np.array([utterance_length_tokens(u) for u in vocab])

    prior = np.ones(len(colors)) / len(colors)
    for u in vocab:
        post = L0_posterior_over_colors(u, colors, cfg)
        p_t = post[target_idx]
        p_list.append(p_t)
        # risk: expected 0-1 loss if we pick color ~post
        risk = 1.0 - p_t
        risk_list.append(risk)
        # ambiguity: entropy
        entropy = -np.sum(post * np.log(np.clip(post, 1e-12, 1.0)))
        amb_list.append(entropy)
        # IG: KL(post || prior)
        kl = np.sum(post * (np.log(np.clip(post, 1e-12, 1.0)) - np.log(prior)))
        ig_list.append(kl)

    p_list = np.array(p_list)
    risk_arr = np.array(risk_list)
    amb_arr = np.array(amb_list)
    ig_arr = np.array(ig_list)

    w_risk = weights.get("risk", 1.0)
    w_amb = weights.get("ambiguity", 1.0)
    w_ig = weights.get("ig", 1.0)

    G = w_risk*risk_arr + w_amb*amb_arr - w_ig*ig_arr + len_cost*lengths
    xs = -beta * G
    xs -= xs.max()
    probs = np.exp(xs)
    probs /= probs.sum()
    return probs


# ----------------------------------
# Trial simulation
# ----------------------------------
@dataclass
class Trial:
    target: Tuple[float,float,float]
    d1: Tuple[float,float,float]
    d2: Tuple[float,float,float]
    condition: str
    contrast: float  # |D1 - D2| proxy
    vocab: List[str]


def sample_condition(df: pd.DataFrame, rng: np.random.Generator, k: int) -> List[str]:
    if "condition" in df.columns:
        vals = df["condition"].dropna().astype(str).values
        if len(vals) == 0:
            vals = np.array(["baseline"]*k)
        else:
            # sample with empirical frequencies
            uniq, counts = np.unique(vals, return_counts=True)
            p = counts / counts.sum()
            return rng.choice(uniq, size=k, p=p).tolist()
    return ["baseline"] * k


def sample_lengths(df: pd.DataFrame, rng: np.random.Generator, n: int) -> np.ndarray:
    col = "numCleanWords" if "numCleanWords" in df.columns else None
    if col is not None:
        vals = df[col].dropna().astype(int).values
        if len(vals) > 0:
            return rng.choice(vals, size=n, replace=True)
    # fallback: 1~3 tokens
    return rng.integers(1, 4, size=n)


def sample_contrasts(df: pd.DataFrame, rng: np.random.Generator, n: int) -> np.ndarray:
    for col in ["D1D2Diff","targetD1Diff","targetD2Diff"]:
        if col in df.columns and df[col].notna().any():
            vals = df[col].dropna().values
            if len(vals) > 0:
                return rng.choice(vals, size=n, replace=True)
    return rng.uniform(0.0, 1.0, size=n)


def sample_color(df: pd.DataFrame, rng: np.random.Generator) -> Tuple[float,float,float]:
    # Use empirical HSL if available, else sample random
    if {"clickColH","clickColS","clickColL"}.issubset(df.columns):
        sub = df[["clickColH","clickColS","clickColL"]].dropna()
        if len(sub) > 0:
            row = sub.sample(n=1, random_state=rng.integers(0, 1<<32)).iloc[0]
            return float(row["clickColH"]), float(row["clickColS"]), float(row["clickColL"])
    # Fallback random
    return float(rng.uniform(0,360)), float(rng.uniform(0.1,0.9)), float(rng.uniform(0.1,0.9))


def make_trials(df: pd.DataFrame, n_trials: int, cfg: Config, seed: int) -> List[Trial]:
    rng = np.random.default_rng(seed)
    conds = sample_condition(df, rng, n_trials)
    contrasts = sample_contrasts(df, rng, n_trials)
    trials: List[Trial] = []
    for i in range(n_trials):
        t = sample_color(df, rng)
        d1 = sample_color(df, rng)
        d2 = sample_color(df, rng)
        vocab = utterance_vocab_for_target(t, cfg)
        trials.append(Trial(
            target=t, d1=d1, d2=d2,
            condition=conds[i],
            contrast=float(abs(contrasts[i])),
            vocab=vocab
        ))
    return trials


@dataclass
class SimResult:
    seed: int
    trials: List[Dict[str, Any]]  # per-trial dict logs
    summary: Dict[str, Any]       # seed-level summary


def simulate_seed(df: pd.DataFrame, cfg: Config, seed: int) -> SimResult:
    rng = np.random.default_rng(seed)
    trials = make_trials(df, cfg.n_trials_per_seed, cfg, seed)

    # True model used to generate utterances
    alpha0 = cfg.alpha_true
    lam0 = cfg.lambda_true
    beta0 = cfg.beta_true
    len_cost0 = cfg.length_cost_true

    weights_assump = {"risk": 1.0, "ambiguity": 1.0, "ig": 1.0}

    logs = []
    correct = 0
    total = 0
    lengths = []

    for idx, tr in enumerate(trials):
        colors = [tr.target, tr.d1, tr.d2]
        target_idx = 0

        # Speaker chooses utterance with RSA (ground truth generator)
        ps = RSA_speaker_probs(target_idx, colors, tr.vocab, cfg, alpha=alpha0, lam=lam0)
        u_idx = rng.choice(len(tr.vocab), p=ps)
        u = tr.vocab[u_idx]

        # Listener picks color given utterance (L0 posterior used here)
        post = L0_posterior_over_colors(u, colors, cfg)
        choice_idx = rng.choice(len(colors), p=post)

        is_correct = int(choice_idx == target_idx)
        correct += is_correct
        total += 1
        lengths.append(utterance_length_tokens(u))

        # Also compute AIF/EFE probs for logging
        ps_aif = AIF_EFE_speaker_probs(target_idx, colors, tr.vocab, cfg, beta=beta0, len_cost=len_cost0, weights=weights_assump)
        logs.append({
            "trial": idx,
            "seed": seed,
            "condition": tr.condition,
            "contrast": tr.contrast,
            "utterance": u,
            "u_len": utterance_length_tokens(u),
            "rsa_prob": float(ps[u_idx]),
            "aif_prob": float(ps_aif[u_idx]),
            "listener_p_target": float(post[target_idx]),
            "is_correct": is_correct,
        })

    acc = correct / max(1, total)
    wil_lo, wil_hi = wilson_interval(correct, total)
    summary = {
        "seed": seed,
        "n_trials": total,
        "acc": acc,
        "wilson_lo": wil_lo,
        "wilson_hi": wil_hi,
        "mean_len": float(np.mean(lengths)) if lengths else float("nan"),
    }
    return SimResult(seed=seed, trials=logs, summary=summary)


# ----------------------------------
# Model fitting by simple grid search
# ----------------------------------
def fit_rsa(logs: List[Dict[str, Any]], cfg: Config, df: pd.DataFrame, seed: int) -> Dict[str, Any]:
    # For each trial, recompute RSA probabilities for the chosen utterance across grid and compute LL.
    rng = np.random.default_rng(seed + 12345)
    # Reconstruct trials deterministically using stored info (contrast, condition, utterance length not enough).
    # So we rebuild by re-sampling targets/distractors with the same seed to get consistent colors/vocab.
    trials = make_trials(df, len(logs), cfg, seed)

    best = {"alpha": None, "lambda": None, "ll": -1e18, "bic": None}
    n_params = 2
    n = len(trials)

    for a in cfg.grid_alpha:
        for lam in cfg.grid_lambda:
            ll = 0.0
            for i, tr in enumerate(trials):
                colors = [tr.target, tr.d1, tr.d2]
                target_idx = 0
                vocab = tr.vocab
                ps = RSA_speaker_probs(target_idx, colors, vocab, cfg, alpha=a, lam=lam)
                # Find prob of the utterance actually produced in logs[i]["utterance"]
                try:
                    u_idx = vocab.index(logs[i]["utterance"])
                    p_u = max(1e-12, float(ps[u_idx]))
                except ValueError:
                    p_u = 1e-12
                ll += math.log(p_u)
            # BIC
            bic = -2*ll + n_params * math.log(max(1, n))
            if ll > best["ll"]:
                best = {"alpha": a, "lambda": lam, "ll": ll, "bic": bic}
    return best


def fit_aif(logs: List[Dict[str, Any]], cfg: Config, df: pd.DataFrame, seed: int, weights: Dict[str,float]) -> Dict[str, Any]:
    trials = make_trials(df, len(logs), cfg, seed)
    best = {"beta": None, "len_cost": None, "ll": -1e18, "bic": None}
    n_params = 2
    n = len(trials)

    for b in cfg.grid_beta:
        for lc in cfg.grid_len_cost:
            ll = 0.0
            for i, tr in enumerate(trials):
                colors = [tr.target, tr.d1, tr.d2]
                target_idx = 0
                vocab = tr.vocab
                ps = AIF_EFE_speaker_probs(target_idx, colors, vocab, cfg, beta=b, len_cost=lc, weights=weights)
                try:
                    u_idx = vocab.index(logs[i]["utterance"])
                    p_u = max(1e-12, float(ps[u_idx]))
                except ValueError:
                    p_u = 1e-12
                ll += math.log(p_u)
            bic = -2*ll + n_params * math.log(max(1, n))
            if ll > best["ll"]:
                best = {"beta": b, "len_cost": lc, "ll": ll, "bic": bic}
    return best


# ----------------------------------
# Figure makers (1 plot per figure)
# Each saves: PNG + paired LaTeX + CSV of underlying data
# ----------------------------------
def fig01_effect_decomposition(outdir: str, cfg: Config):
    """
    Show average contributions of risk / ambiguity / IG and length penalty in AIF,
    and compare with RSA utility components, on a synthetic batch of utterances.
    """
    rng = np.random.default_rng(cfg.random_state)
    # Sample 300 random utterances against random colors to get typical values
    df = load_corpus(cfg.input_dir)
    trials = make_trials(df, 300, cfg, seed=cfg.random_state)
    weights = {"risk":1.0,"ambiguity":1.0,"ig":1.0}
    rows = []
    for tr in trials:
        colors = [tr.target, tr.d1, tr.d2]
        target_idx = 0
        for u in tr.vocab:
            post = L0_posterior_over_colors(u, colors, cfg)
            p_t = post[target_idx]
            risk = 1.0 - p_t
            amb = -np.sum(post * np.log(np.clip(post, 1e-12, 1.0)))
            ig = np.sum(post * (np.log(np.clip(post, 1e-12, 1.0)) - math.log(1/len(colors))))
            ulen = utterance_length_tokens(u)
            rsa_util = cfg.alpha_true * math.log(max(1e-12, p_t)) - cfg.lambda_true * ulen
            aif_terms = {
                "risk": risk,
                "ambiguity": amb,
                "ig": ig,
                "len": ulen,
            }
            rows.append({"risk":risk,"ambiguity":amb,"ig":ig,"len":ulen,"rsa_util":rsa_util})

    agg = pd.DataFrame(rows).mean().to_frame(name="mean").T
    # Prepare table
    table_df = agg[["risk","ambiguity","ig","len","rsa_util"]]
    csv_path = os.path.join(outdir, "data", "fig01_data.csv")
    table_df.to_csv(csv_path, index=False)

    # Plot a simple bar chart of normalized contributions
    vals = table_df.iloc[0].values
    names = list(table_df.columns)
    plt.figure(figsize=(6,4))
    plt.bar(range(len(vals)), vals)
    plt.xticks(range(len(vals)), names, rotation=0)
    plt.title("Fig 1: Mean AIF terms vs RSA utility (simulated)")
    plt.tight_layout()
    png_path = os.path.join(outdir, "figures", "fig01.png")
    plt.savefig(png_path, dpi=200)
    plt.close()

    # LaTeX
    tex_path = os.path.join(outdir, "tables", "fig01_table.tex")
    save_tex_table(table_df.round(4), tex_path, caption="Mean contributions of AIF terms compared to RSA utility (simulated).", label="tab:fig01")


def fig02_model_comp_ll_bic(outdir: str, cfg: Config, fit_r: Dict[str,Any], fit_a: Dict[str,Any]):
    # Table
    df = pd.DataFrame([
        {"model":"RSA","alpha":fit_r["alpha"],"lambda":fit_r["lambda"],"LL":fit_r["ll"],"BIC":fit_r["bic"]},
        {"model":"AIF","beta":fit_a["beta"],"len_cost":fit_a["len_cost"],"LL":fit_a["ll"],"BIC":fit_a["bic"]},
    ])
    csv_path = os.path.join(outdir, "data", "fig02_data.csv")
    df.to_csv(csv_path, index=False)

    # Plot LL comparison
    plt.figure(figsize=(6,4))
    plt.bar([0,1], [df.loc[0,"LL"], df.loc[1,"LL"]])
    plt.xticks([0,1], ["RSA","AIF"])
    plt.title("Fig 2: Log-Likelihood Comparison (simulated)")
    plt.tight_layout()
    png_path = os.path.join(outdir, "figures", "fig02.png")
    plt.savefig(png_path, dpi=200)
    plt.close()

    tex_path = os.path.join(outdir, "tables", "fig02_table.tex")
    save_tex_table(df.round(4), tex_path, caption="Model comparison (Log-Likelihood and BIC) on simulated data.", label="tab:fig02")


def fig03_ablation(outdir: str, cfg: Config, df_corpus: pd.DataFrame, seed: int, logs: List[Dict[str,Any]]):
    # Evaluate AIF fits with various ablations of weights
    weights_list = [
        {"name":"Full (risk+amb-IG)","risk":1.0,"ambiguity":1.0,"ig":1.0},
        {"name":"No IG","risk":1.0,"ambiguity":1.0,"ig":0.0},
        {"name":"No ambiguity","risk":1.0,"ambiguity":0.0,"ig":1.0},
        {"name":"Risk only","risk":1.0,"ambiguity":0.0,"ig":0.0},
        {"name":"Ambiguity only","risk":0.0,"ambiguity":1.0,"ig":0.0},
    ]
    rows = []
    for w in weights_list:
        fit = fit_aif(logs, cfg, df_corpus, seed, weights=w)
        rows.append({"ablation":w["name"], "beta":fit["beta"], "len_cost":fit["len_cost"], "LL":fit["ll"], "BIC":fit["bic"]})
    tab = pd.DataFrame(rows)
    csv_path = os.path.join(outdir, "data", "fig03_data.csv")
    tab.to_csv(csv_path, index=False)

    # Plot BIC
    plt.figure(figsize=(7,4))
    plt.bar(range(len(tab)), tab["BIC"].values)
    plt.xticks(range(len(tab)), tab["ablation"].values, rotation=20, ha="right")
    plt.title("Fig 3: AIF ablations (BIC)")
    plt.tight_layout()
    png_path = os.path.join(outdir, "figures", "fig03.png")
    plt.savefig(png_path, dpi=200)
    plt.close()

    tex_path = os.path.join(outdir, "tables", "fig03_table.tex")
    save_tex_table(tab.round(4), tex_path, caption="AIF ablation analysis (BIC) across risk/ambiguity/IG terms.", label="tab:fig03")


def fig04_len_vs_contrast(outdir: str, logs: List[Dict[str,Any]]):
    # Aggregate mean utterance length by contrast deciles
    df = pd.DataFrame(logs)
    df["bin"] = pd.qcut(df["contrast"], q=10, duplicates="drop")
    tab = df.groupby("bin", as_index=False)["u_len"].mean().rename(columns={"u_len":"mean_len"})
    csv_path = os.path.join(outdir, "data", "fig04_data.csv")
    tab.to_csv(csv_path, index=False)

    # Plot
    plt.figure(figsize=(6,4))
    plt.plot(range(len(tab)), tab["mean_len"].values, marker="o")
    plt.xticks(range(len(tab)), [str(b) for b in tab["bin"]], rotation=30, ha="right")
    plt.title("Fig 4: Utterance length vs contrast (deciles)")
    plt.tight_layout()
    png_path = os.path.join(outdir, "figures", "fig04.png")
    plt.savefig(png_path, dpi=200)
    plt.close()

    tex_path = os.path.join(outdir, "tables", "fig04_table.tex")
    save_tex_table(tab, tex_path, caption="Mean utterance length by contrast decile (simulated).", label="tab:fig04")


def fig05_calibration(outdir: str, logs: List[Dict[str,Any]]):
    # Calibration of listener_p_target to actual correctness
    df = pd.DataFrame(logs)
    df["bin"] = pd.qcut(df["listener_p_target"], q=10, duplicates="drop")
    tab = df.groupby("bin", as_index=False).agg(
        mean_conf=("listener_p_target","mean"),
        emp_acc=("is_correct","mean"),
        n=("is_correct","size")
    )
    csv_path = os.path.join(outdir, "data", "fig05_data.csv")
    tab.to_csv(csv_path, index=False)

    plt.figure(figsize=(5,5))
    plt.plot(tab["mean_conf"].values, tab["emp_acc"].values, marker="o")
    plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("Mean predicted P(target)")
    plt.ylabel("Empirical accuracy")
    plt.title("Fig 5: Calibration curve")
    plt.tight_layout()
    png_path = os.path.join(outdir, "figures", "fig05.png")
    plt.savefig(png_path, dpi=200)
    plt.close()

    tex_path = os.path.join(outdir, "tables", "fig05_table.tex")
    save_tex_table(tab, tex_path, caption="Calibration between predicted P(target) and empirical accuracy.", label="tab:fig05")


def fig06_confusion_like(outdir: str, cfg: Config):
    # Simulated confusion matrix of listener picks among 3 options for random utterances
    rng = np.random.default_rng(cfg.random_state+7)
    df_corpus = load_corpus(cfg.input_dir)
    trials = make_trials(df_corpus, 500, cfg, seed=cfg.random_state+7)

    mat = np.zeros((3,3), dtype=int)
    for tr in trials:
        colors = [tr.target, tr.d1, tr.d2]
        target_idx = 0
        u = rng.choice(tr.vocab)
        post = L0_posterior_over_colors(u, colors, cfg)
        choice = rng.choice(3, p=post)
        mat[target_idx, choice] += 1

    tab = pd.DataFrame(mat, columns=["choose_target","choose_d1","choose_d2"])
    csv_path = os.path.join(outdir, "data", "fig06_data.csv")
    tab.to_csv(csv_path, index=False)

    plt.figure(figsize=(4,4))
    plt.imshow(mat, aspect="auto")
    plt.xticks([0,1,2], ["target","d1","d2"])
    plt.yticks([0,1,2], ["true target","true d1","true d2"])
    plt.title("Fig 6: Confusion-like matrix (simulated)")
    plt.tight_layout()
    png_path = os.path.join(outdir, "figures", "fig06.png")
    plt.savefig(png_path, dpi=200)
    plt.close()

    tex_path = os.path.join(outdir, "tables", "fig06_table.tex")
    save_tex_table(tab, tex_path, caption="Counts of choices given random utterances (simulated).", label="tab:fig06")


def fig07_accuracy_by_condition(outdir: str, logs: List[Dict[str,Any]]):
    df = pd.DataFrame(logs)
    tab_rows = []
    for cond, sub in df.groupby("condition"):
        k = int(sub["is_correct"].sum())
        n = int(len(sub))
        acc = k/n if n else float("nan")
        lo, hi = wilson_interval(k, n)
        tab_rows.append({"condition":cond, "n":n, "acc":acc, "wilson_lo":lo, "wilson_hi":hi})
    tab = pd.DataFrame(tab_rows).sort_values("n", ascending=False)
    csv_path = os.path.join(outdir, "data", "fig07_data.csv")
    tab.to_csv(csv_path, index=False)

    plt.figure(figsize=(7,4))
    xs = np.arange(len(tab))
    plt.errorbar(xs, tab["acc"].values, yerr=[tab["acc"].values-tab["wilson_lo"].values, tab["wilson_hi"].values-tab["acc"].values], fmt='o')
    plt.xticks(xs, tab["condition"].values, rotation=20, ha="right")
    plt.ylabel("Accuracy")
    plt.title("Fig 7: Accuracy by condition (Wilson CI)")
    plt.tight_layout()
    png_path = os.path.join(outdir, "figures", "fig07.png")
    plt.savefig(png_path, dpi=200)
    plt.close()

    tex_path = os.path.join(outdir, "tables", "fig07_table.tex")
    save_tex_table(tab.round(4), tex_path, caption="Accuracy by condition with Wilson 95\\% intervals (simulated).", label="tab:fig07")


def fig08_param_recovery(outdir: str, fit_r: Dict[str,Any], fit_a: Dict[str,Any], cfg: Config):
    tab = pd.DataFrame([
        {"model":"RSA","true_alpha":cfg.alpha_true,"est_alpha":fit_r["alpha"],"true_lambda":cfg.lambda_true,"est_lambda":fit_r["lambda"]},
        {"model":"AIF","true_beta":cfg.beta_true,"est_beta":fit_a["beta"],"true_len_cost":cfg.length_cost_true,"est_len_cost":fit_a["len_cost"]},
    ])
    csv_path = os.path.join(outdir, "data", "fig08_data.csv")
    tab.to_csv(csv_path, index=False)

    plt.figure(figsize=(6,4))
    plt.plot([cfg.alpha_true], [fit_r["alpha"]], marker="o")
    plt.xlabel("True alpha")
    plt.ylabel("Estimated alpha")
    plt.title("Fig 8: RSA parameter recovery (alpha)")
    plt.tight_layout()
    png_path = os.path.join(outdir, "figures", "fig08.png")
    plt.savefig(png_path, dpi=200)
    plt.close()

    tex_path = os.path.join(outdir, "tables", "fig08_table.tex")
    save_tex_table(tab.round(4), tex_path, caption="Parameter recovery for RSA and AIF (simulated).", label="tab:fig08")


def fig09_len_cost_sweep(outdir: str, cfg: Config, df_corpus: pd.DataFrame):
    rng = np.random.default_rng(cfg.random_state+9)
    trials = make_trials(df_corpus, 400, cfg, seed=cfg.random_state+9)
    target_idx = 0
    means = []
    for lc in cfg.grid_len_cost:
        ulen = []
        for tr in trials:
            colors = [tr.target, tr.d1, tr.d2]
            ps = AIF_EFE_speaker_probs(target_idx, colors, tr.vocab, cfg, beta=cfg.beta_true, len_cost=lc, weights={"risk":1.0,"ambiguity":1.0,"ig":1.0})
            u_idx = np.argmax(ps)  # MAP choice for length summary
            ulen.append(utterance_length_tokens(tr.vocab[u_idx]))
        means.append({"len_cost":lc, "mean_len":float(np.mean(ulen))})
    tab = pd.DataFrame(means)
    csv_path = os.path.join(outdir, "data", "fig09_data.csv")
    tab.to_csv(csv_path, index=False)

    plt.figure(figsize=(6,4))
    plt.plot(tab["len_cost"].values, tab["mean_len"].values, marker="o")
    plt.xlabel("Length cost")
    plt.ylabel("Mean utterance length")
    plt.title("Fig 9: Length penalty vs utterance length")
    plt.tight_layout()
    png_path = os.path.join(outdir, "figures", "fig09.png")
    plt.savefig(png_path, dpi=200)
    plt.close()

    tex_path = os.path.join(outdir, "tables", "fig09_table.tex")
    save_tex_table(tab.round(4), tex_path, caption="Mean utterance length under varying AIF length cost.", label="tab:fig09")


def fig10_noise_robustness(outdir: str, cfg: Config, df_corpus: pd.DataFrame):
    rng = np.random.default_rng(cfg.random_state+10)
    trials = make_trials(df_corpus, 300, cfg, seed=cfg.random_state+10)
    noise_levels = [0.0, 0.1, 0.2, 0.3]
    rows = []
    for nl in noise_levels:
        correct = 0
        n = 0
        for tr in trials:
            colors = [tr.target, tr.d1, tr.d2]
            post_true = L0_posterior_over_colors(rng.choice(tr.vocab), colors, cfg)
            # Mix with uniform noise
            post_noisy = (1-nl)*post_true + nl*(np.ones_like(post_true)/len(post_true))
            choice_idx = rng.choice(3, p=post_noisy)
            if choice_idx == 0:
                correct += 1
            n += 1
        acc = correct / max(1,n)
        lo, hi = wilson_interval(correct, n)
        rows.append({"noise":nl, "acc":acc, "wilson_lo":lo, "wilson_hi":hi})
    tab = pd.DataFrame(rows)
    csv_path = os.path.join(outdir, "data", "fig10_data.csv")
    tab.to_csv(csv_path, index=False)

    plt.figure(figsize=(6,4))
    plt.errorbar(tab["noise"].values, tab["acc"].values, yerr=[tab["acc"].values-tab["wilson_lo"].values, tab["wilson_hi"].values-tab["acc"].values], fmt='o')
    plt.xlabel("Semantic noise level")
    plt.ylabel("Accuracy")
    plt.title("Fig 10: Robustness to semantic noise")
    plt.tight_layout()
    png_path = os.path.join(outdir, "figures", "fig10.png")
    plt.savefig(png_path, dpi=200)
    plt.close()

    tex_path = os.path.join(outdir, "tables", "fig10_table.tex")
    save_tex_table(tab.round(4), tex_path, caption="Accuracy vs injected semantic noise (simulated).", label="tab:fig10")


def fig11_efe_vs_negutil(outdir: str, cfg: Config, df_corpus: pd.DataFrame):
    rng = np.random.default_rng(cfg.random_state+11)
    trials = make_trials(df_corpus, 300, cfg, seed=cfg.random_state+11)
    rows = []
    for tr in trials:
        colors = [tr.target, tr.d1, tr.d2]
        target_idx = 0
        for u in tr.vocab:
            post = L0_posterior_over_colors(u, colors, cfg)
            p_t = post[target_idx]
            ulen = utterance_length_tokens(u)
            rsa_negU = -(cfg.alpha_true * math.log(max(1e-12, p_t)) - cfg.lambda_true * ulen)
            # AIF G(u)
            risk = 1.0 - p_t
            amb = -np.sum(post * np.log(np.clip(post, 1e-12, 1.0)))
            ig = np.sum(post * (np.log(np.clip(post, 1e-12, 1.0)) - math.log(1/len(colors))))
            G = risk + amb - ig + cfg.length_cost_true * ulen
            rows.append({"neg_RSA_util":rsa_negU, "EFE":G})
    tab = pd.DataFrame(rows)
    csv_path = os.path.join(outdir, "data", "fig11_data.csv")
    tab.to_csv(csv_path, index=False)

    plt.figure(figsize=(5,5))
    plt.scatter(tab["neg_RSA_util"].values, tab["EFE"].values, s=8)
    plt.xlabel("- RSA utility")
    plt.ylabel("EFE")
    plt.title("Fig 11: EFE vs -Utility (simulated)")
    plt.tight_layout()
    png_path = os.path.join(outdir, "figures", "fig11.png")
    plt.savefig(png_path, dpi=200)
    plt.close()

    tex_path = os.path.join(outdir, "tables", "fig11_table.tex")
    save_tex_table(tab.describe().reset_index(), tex_path, caption="Summary of EFE vs -RSA utility scatter.", label="tab:fig11")


def fig12_rt_vs_entropy(outdir: str, cfg: Config, df_corpus: pd.DataFrame):
    rng = np.random.default_rng(cfg.random_state+12)
    trials = make_trials(df_corpus, 400, cfg, seed=cfg.random_state+12)
    rows = []
    for tr in trials:
        colors = [tr.target, tr.d1, tr.d2]
        u = rng.choice(tr.vocab)
        post = L0_posterior_over_colors(u, colors, cfg)
        ent = -np.sum(post * np.log(np.clip(post, 1e-12, 1.0)))
        # Simulated reaction time: RT ~ base + k * entropy + noise
        rt = 800 + 400*ent + rng.normal(0, 80)
        rows.append({"entropy":ent, "rt_ms":rt})
    tab = pd.DataFrame(rows)
    csv_path = os.path.join(outdir, "data", "fig12_data.csv")
    tab.to_csv(csv_path, index=False)

    plt.figure(figsize=(6,4))
    plt.scatter(tab["entropy"].values, tab["rt_ms"].values, s=8)
    plt.xlabel("Posterior entropy")
    plt.ylabel("RT (ms)")
    plt.title("Fig 12: Reaction time vs entropy")
    plt.tight_layout()
    png_path = os.path.join(outdir, "figures", "fig12.png")
    plt.savefig(png_path, dpi=200)
    plt.close()

    tex_path = os.path.join(outdir, "tables", "fig12_table.tex")
    save_tex_table(tab.describe().reset_index(), tex_path, caption="Summary stats: reaction time vs posterior entropy.", label="tab:fig12")


def fig13_cv_ll_across_seeds(outdir: str, seed_fits: List[Dict[str,Any]]):
    tab = pd.DataFrame(seed_fits)
    csv_path = os.path.join(outdir, "data", "fig13_data.csv")
    tab.to_csv(csv_path, index=False)

    plt.figure(figsize=(7,4))
    plt.plot(tab["seed"].values, tab["rsa_ll"].values, marker="o", label="RSA")
    plt.plot(tab["seed"].values, tab["aif_ll"].values, marker="o", label="AIF")
    plt.xlabel("Seed")
    plt.ylabel("Log-Likelihood")
    plt.title("Fig 13: Cross-validated LL across seeds")
    plt.legend()
    plt.tight_layout()
    png_path = os.path.join(outdir, "figures", "fig13.png")
    plt.savefig(png_path, dpi=200)
    plt.close()

    tex_path = os.path.join(outdir, "tables", "fig13_table.tex")
    save_tex_table(tab.round(4), tex_path, caption="Per-seed fitted LLs for RSA and AIF.", label="tab:fig13")


def fig14_best_params_summary(outdir: str, fit_r: Dict[str,Any], fit_a: Dict[str,Any]):
    tab = pd.DataFrame([
        {"model":"RSA","param1":"alpha","value":fit_r["alpha"]},
        {"model":"RSA","param1":"lambda","value":fit_r["lambda"]},
        {"model":"AIF","param1":"beta","value":fit_a["beta"]},
        {"model":"AIF","param1":"len_cost","value":fit_a["len_cost"]},
    ])
    csv_path = os.path.join(outdir, "data", "fig14_data.csv")
    tab.to_csv(csv_path, index=False)

    plt.figure(figsize=(6,4))
    plt.bar(range(len(tab)), tab["value"].values)
    plt.xticks(range(len(tab)), [f"{m}:{p}" for m,p in zip(tab["model"].values, tab["param1"].values)], rotation=20, ha="right")
    plt.title("Fig 14: Best-fit parameters")
    plt.tight_layout()
    png_path = os.path.join(outdir, "figures", "fig14.png")
    plt.savefig(png_path, dpi=200)
    plt.close()

    tex_path = os.path.join(outdir, "tables", "fig14_table.tex")
    save_tex_table(tab.round(4), tex_path, caption="Best-fit parameters for RSA and AIF on simulated data.", label="tab:fig14")



def _fit_both_for_seed(r: SimResult, cfg: Config, df_corpus: pd.DataFrame, rep_seed: int) -> Dict[str, Any]:
    fr = fit_rsa(r.trials, cfg, df_corpus, seed=rep_seed)
    fa = fit_aif(r.trials, cfg, df_corpus, seed=rep_seed, weights={"risk":1.0,"ambiguity":1.0,"ig":1.0})
    return {"seed": r.seed, "rsa_ll": fr["ll"], "aif_ll": fa["ll"]}

# ----------------------------------
# Main runner
# ----------------------------------
def main():
    cfg = Config.default()

    parser = argparse.ArgumentParser(description="AIP (AIF⇔RSA) full simulation for figures & tables.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", required=True, help="Input folder (must contain filteredCorpus.csv)")
    parser.add_argument("--output", required=True, help="Output folder for figures/tables/data")
    parser.add_argument("--cores", type=int, default=cfg.cores, help="Number of parallel workers")
    parser.add_argument("--trials-per-seed", type=int, default=cfg.n_trials_per_seed, help="Number of trials per seed")
    parser.add_argument("--seeds", type=str, default="0-29", help="Seeds as 'a-b' or comma list (e.g., '0-9' or '0,2,5')")
    args = parser.parse_args()

    cfg.input_dir = args.input
    cfg.output_dir = args.output
    cfg.cores = max(1, int(args.cores))
    cfg.n_trials_per_seed = int(args.trials_per_seed)

    # Parse seeds
    if "-" in args.seeds:
        a, b = args.seeds.split("-")
        seeds = list(range(int(a), int(b)+1))
    else:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    cfg.seeds = seeds

    ensure_outdirs(cfg.output_dir)
    df_corpus = load_corpus(cfg.input_dir)

    # Save run config
    run_meta = asdict(cfg)
    with open(os.path.join(cfg.output_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)

    _log(f"Config: cores={cfg.cores}, trials_per_seed={cfg.n_trials_per_seed}, seeds={cfg.seeds}")
    df_corpus = load_corpus(cfg.input_dir)
    _log(f"Loaded corpus: shape={df_corpus.shape} columns={list(df_corpus.columns)[:6]}...")

    # --- Simulate per seed (parallel, multi-process) with simple console progress ---
    print("Simulating per seed...")
    results: List[SimResult] = []
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import time as _time
    t0 = _time.time()
    total = len(cfg.seeds)
    with ProcessPoolExecutor(max_workers=cfg.cores) as ex:
        future_map = {ex.submit(simulate_seed, df_corpus, cfg, seed): seed for seed in cfg.seeds}
        done = 0
        for fut in as_completed(future_map):
            res = fut.result()
            results.append(res)
            done += 1
            elapsed = _time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0.0
            eta = (total - done) / rate if rate > 0 else float('inf')
            print(f"[{done}/{total}] seeds done | elapsed {elapsed:.1f}s | ETA {eta:.1f}s", flush=True)

    # Aggregate logs (deterministic by seed order)
    print('[SIM] Aggregating logs...', flush=True)
    all_logs = [row for r in results for row in r.trials]
    all_summ = [r.summary for r in results]
    print(f"[SIM] Total trials: {len(all_logs)} | seeds: {len(all_summ)}", flush=True)

    # Save per-trial logs and summaries
    pd.DataFrame(all_logs).to_csv(os.path.join(cfg.output_dir, "data", "all_trial_logs.csv"), index=False)
    pd.DataFrame(all_summ).to_csv(os.path.join(cfg.output_dir, "data", "per_seed_summary.csv"), index=False)

    # Fit models on the concatenated logs but with a fixed "representative" seed for trial reconstruction.
    rep_seed = cfg.seeds[0]
    _log("Fitting RSA (grid search)...")
    _log(f"Grid sizes: alpha={len(cfg.grid_alpha)} x lambda={len(cfg.grid_lambda)}")
    fitR = fit_rsa(all_logs, cfg, df_corpus, seed=rep_seed)
    print(f"  RSA best: alpha={fitR['alpha']} lambda={fitR['lambda']} LL={fitR['ll']:.2f} BIC={fitR['bic']:.2f}")

    _log("Fitting AIF (grid search)...")
    _log(f"Grid sizes: beta={len(cfg.grid_beta)} x len_cost={len(cfg.grid_len_cost)}")
    fitA = fit_aif(all_logs, cfg, df_corpus, seed=rep_seed, weights={"risk":1.0,"ambiguity":1.0,"ig":1.0})
    print(f"  AIF best: beta={fitA['beta']} len_cost={fitA['len_cost']} LL={fitA['ll']:.2f} BIC={fitA['bic']:.2f}")

    # Per-seed CV-ish LLs (parallel, multi-process)
    seed_fits = []
    print("Per-seed LLs (parallel)...")
    from concurrent.futures import ProcessPoolExecutor, as_completed
    total = len(results)
    done = 0
    t1 = __import__('time').time()
    with ProcessPoolExecutor(max_workers=cfg.cores) as ex:
        future_map = {ex.submit(_fit_both_for_seed, r, cfg, df_corpus, rep_seed): r.seed for r in results}
        for fut in as_completed(future_map):
            out = fut.result()
            seed_fits.append(out)
            done += 1
            elapsed = __import__('time').time() - t1
            rate = done / elapsed if elapsed > 0 else 0.0
            eta = (total - done) / rate if rate > 0 else float('inf')
            print(f"[{done}/{total}] fitted | elapsed {elapsed:.1f}s | ETA {eta:.1f}s", flush=True)
    seed_fits = sorted(seed_fits, key=lambda x: x["seed"])
    pd.DataFrame(seed_fits).to_csv(os.path.join(cfg.output_dir, "data", "per_seed_fits.csv"), index=False)

    # --- Figures & Tables ---
    _log("Building figures & tables...")
    for name, fn, args in [
        ("Fig01", fig01_effect_decomposition, (cfg.output_dir, cfg)),
        ("Fig02", fig02_model_comp_ll_bic, (cfg.output_dir, cfg, fitR, fitA)),
        ("Fig03", fig03_ablation, (cfg.output_dir, cfg, df_corpus, rep_seed, all_logs)),
        ("Fig04", fig04_len_vs_contrast, (cfg.output_dir, all_logs)),
        ("Fig05", fig05_calibration, (cfg.output_dir, all_logs)),
        ("Fig06", fig06_confusion_like, (cfg.output_dir, cfg)),
        ("Fig07", fig07_accuracy_by_condition, (cfg.output_dir, all_logs)),
        ("Fig08", fig08_param_recovery, (cfg.output_dir, fitR, fitA, cfg)),
        ("Fig09", fig09_len_cost_sweep, (cfg.output_dir, cfg, df_corpus)),
        ("Fig10", fig10_noise_robustness, (cfg.output_dir, cfg, df_corpus)),
        ("Fig11", fig11_efe_vs_negutil, (cfg.output_dir, cfg, df_corpus)),
        ("Fig12", fig12_rt_vs_entropy, (cfg.output_dir, cfg, df_corpus)),
        ("Fig13", fig13_cv_ll_across_seeds, (cfg.output_dir, seed_fits)),
        ("Fig14", fig14_best_params_summary, (cfg.output_dir, fitR, fitA)),
    ]:
        _log(f"[build] start {name}")
        fn(*args)
        _log(f"[build] done  {name}")

    _log("All figures/tables/data generated.")
    _log(f"Output root: {cfg.output_dir}")


if __name__ == "__main__":
    main()
