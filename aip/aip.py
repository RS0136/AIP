#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIP Simulation Runner (aip.py)
--------------------------------
- CLI to run AIP preprint simulations on the Colors in Context (CiC) dataset.
- Follows the output naming convention and prints step-by-step progress.
- Uses all available data (no subsampling) and supports parallel execution.
- Generates paired figures (PDF, PNG) and tables (CSV, TeX) per simulation.
"""
import argparse
import json
import math
import os
import sys
import time
import textwrap
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except Exception:
    JOBLIB_AVAILABLE = False

def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_table_csv_tex(df: pd.DataFrame, basepath: str):
    csv_path = f"{basepath}.csv"
    tex_path = f"{basepath}.tex"
    df.to_csv(csv_path, index=False)
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(df.to_latex(index=False, escape=True))
    log(f"Saved table: {csv_path} and {tex_path}")

def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + (z**2)/n
    center = (p + (z**2)/(2*n)) / denom
    margin = (z * math.sqrt((p*(1-p) + (z**2)/(4*n)) / n)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))

def bic(ll: float, k_params: int, n_obs: int) -> float:
    return k_params * math.log(max(1, n_obs)) - 2.0 * ll

def text_tokenize(s: str) -> List[str]:
    if not isinstance(s, str):
        return []
    s = s.lower()
    for ch in [",", ".", "!", "?", ":", ";", "(", ")", "[", "]", "{", "}", "\"", "'"]:
        s = s.replace(ch, " ")
    toks = [t for t in s.split() if t]
    return toks

@dataclass
class NBConfig:
    add_alpha: float = 1.0
    normalize_len: bool = True
    vocab_top_ratio: float = 1.0
    drop_rare_min_df: int = 1
    noise_std: float = 0.0
    temp_beta: float = 1.0
    drop_top_frac: float = 0.0

class NaiveBayesText:
    def __init__(self, config: NBConfig):
        self.cfg = config
        self.vocab_: List[str] = []
        self.class_log_prior_: Optional[np.ndarray] = None
        self.feature_log_prob_: Optional[np.ndarray] = None

    def _build_vocab(self, docs: List[List[str]]) -> List[str]:
        df_counts: Dict[str, int] = {}
        for toks in docs:
            for t in set(toks):
                df_counts[t] = df_counts.get(t, 0) + 1
        items = [(t, c) for t, c in df_counts.items() if c >= self.cfg.drop_rare_min_df]
        items.sort(key=lambda x: x[1], reverse=True)

        if self.cfg.drop_top_frac > 0.0 and len(items) > 0:
            kdrop = int(len(items) * self.cfg.drop_top_frac)
            kdrop = min(max(0, kdrop), len(items))
            items = items[kdrop:]

        if self.cfg.vocab_top_ratio < 1.0:
            k = max(1, int(len(items) * self.cfg.vocab_top_ratio))
            items = items[:k]
        return [t for t, _ in items]

    def _vectorize(self, docs: List[List[str]]) -> np.ndarray:
        term_index = {t: i for i, t in enumerate(self.vocab_)}
        X = np.zeros((len(docs), len(self.vocab_)), dtype=np.float64)
        for i, toks in enumerate(docs):
            for t in toks:
                j = term_index.get(t)
                if j is not None:
                    X[i, j] += 1.0
            if self.cfg.normalize_len:
                s = X[i].sum()
                if s > 0:
                    X[i] /= s
        return X

    def fit(self, texts: List[str], y: np.ndarray):
        docs = [text_tokenize(t) for t in texts]
        self.vocab_ = self._build_vocab(docs)
        X = self._vectorize(docs)

        y = np.asarray(y).astype(int)
        n0 = int((y == 0).sum())
        n1 = int((y == 1).sum())
        n = max(1, n0 + n1)
        self.class_log_prior_ = np.log(np.array([max(1, n0), max(1, n1)]) / n)

        add_a = self.cfg.add_alpha
        sum0 = X[y == 0].sum(axis=0) + add_a
        sum1 = X[y == 1].sum(axis=0) + add_a
        p0 = sum0 / sum0.sum()
        p1 = sum1 / sum1.sum()
        logp0 = np.log(p0 + 1e-12)
        logp1 = np.log(p1 + 1e-12)
        W = np.vstack([logp0, logp1])

        if self.cfg.noise_std > 0.0:
            noise = np.random.normal(0.0, self.cfg.noise_std, size=W.shape)
            W = W + noise
        self.feature_log_prob_ = W
        return self

    def base_scores(self, texts: List[str]) -> np.ndarray:
        docs = [text_tokenize(t) for t in texts]
        X = self._vectorize(docs)
        scores = self.class_log_prior_[None, :] + X @ self.feature_log_prob_.T
        return scores

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        scores = self.base_scores(texts)
        beta = max(1e-6, float(self.cfg.temp_beta))
        scores = scores / beta
        m = scores.max(axis=1, keepdims=True)
        exp = np.exp(scores - m)
        P = exp / exp.sum(axis=1, keepdims=True)
        return P

    def predict_proba_with_beta(self, texts: List[str], beta_array: np.ndarray) -> np.ndarray:
        scores = self.base_scores(texts)
        beta = np.clip(beta_array.reshape(-1, 1), 1e-6, None)
        scores = scores / beta
        m = scores.max(axis=1, keepdims=True)
        exp = np.exp(scores - m)
        P = exp / exp.sum(axis=1, keepdims=True)
        return P

def guess_binary_target(df: pd.DataFrame) -> np.ndarray:
    if "outcome" in df.columns:
        col = df["outcome"]
        if col.dropna().dtype.kind in "biufc":
            vals = col.fillna(0).astype(float)
            uniq = sorted(pd.unique(vals))
            if len(uniq) <= 3:
                y = (vals > 0).astype(int).values
                return y
        else:
            s = col.astype(str).str.lower()
            pos = s.isin(["1", "true", "yes", "target", "correct", "success", "hit"])
            if pos.any():
                y = (pos).astype(int).values
                return y

    if "clickStatus" in df.columns:
        col = df["clickStatus"]
        if col.dropna().dtype.kind in "biufc":
            vals = col.fillna(0).astype(float)
            uniq = sorted(pd.unique(vals))
            if len(uniq) <= 3:
                y = (vals > 0).astype(int).values
                return y
        else:
            s = col.astype(str).str.lower()
            pos = s.isin(["1", "true", "yes", "clicked", "success"])
            y = (pos).astype(int).values
            return y

    raise ValueError("Could not infer a binary target from 'outcome' or 'clickStatus'.")

def pick_text_column(df: pd.DataFrame) -> pd.Series:
    for cand in ["contents", "message", "utterance", "text"]:
        if cand in df.columns:
            return df[cand].fillna("")
    cols = [c for c in df.columns if c not in ("outcome", "clickStatus")]
    return df[cols].astype(str).agg(" ".join, axis=1)

def fivefold_groups_by_game(df: pd.DataFrame) -> List[np.ndarray]:
    n = len(df)
    if n == 0:
        return [np.zeros(0, dtype=bool)]
    if "gameid" in df.columns:
        gids = df["gameid"].astype(str).values
        unique = pd.unique(gids)
        unique_sorted = np.sort(unique)
        buckets = np.array_split(unique_sorted, min(5, max(1, len(unique_sorted))))
        folds = []
        for b in buckets:
            mask = np.isin(gids, b)
            if mask.sum() > 0:
                folds.append(mask)
        return folds if folds else [np.ones(n, dtype=bool)]
    else:
        idx = np.arange(n)
        chunks = np.array_split(idx, min(5, max(1, n)))
        masks = [np.isin(idx, ch) for ch in chunks if len(ch) > 0]
        return masks if masks else [np.ones(n, dtype=bool)]

@dataclass
class Metrics:
    acc: float
    acc_lo: float
    acc_hi: float
    ll: float
    bic: float
    n: int
    k_params: int
    def to_dict(self):
        d = asdict(self)
        d["delta_bic"] = np.nan
        return d

def evaluate_model(y_true: np.ndarray, proba: np.ndarray, k_params: int) -> Metrics:
    n = len(y_true)
    if n == 0:
        return Metrics(acc=np.nan, acc_lo=np.nan, acc_hi=np.nan, ll=0.0, bic=np.nan, n=0, k_params=k_params)
    y_pred = (proba[:, 1] >= 0.5).astype(int)
    k_correct = int((y_pred == y_true).sum())
    acc = k_correct / n
    lo, hi = wilson_ci(k_correct, n)
    eps = 1e-12
    p1 = proba[np.arange(n), y_true]
    ll = float(np.sum(np.log(np.clip(p1, eps, 1.0))))
    b = bic(ll, k_params=k_params, n_obs=n)
    return Metrics(acc=acc, acc_lo=lo, acc_hi=hi, ll=ll, bic=b, n=n, k_params=k_params)

def cv_eval_configs(text: pd.Series, y: np.ndarray, folds: List[np.ndarray],
                    named_cfgs: List[Tuple[str, NBConfig]], cores: int = 1) -> pd.DataFrame:
    def fit_eval(cfg_name: str, cfg: NBConfig) -> Dict[str, float]:
        log(f"  Config: {cfg_name}")
        metrics_list: List[Metrics] = []
        for i, test_mask in enumerate(folds):
            log(f"    CV fold {i+1}/{len(folds)}")
            train_mask = ~test_mask
            model = NaiveBayesText(cfg)
            model.fit(text[train_mask].tolist(), y[train_mask])
            proba = model.predict_proba(text[test_mask].tolist())
            m = evaluate_model(y_true=y[test_mask], proba=proba, k_params=len(model.vocab_) + 2)
            metrics_list.append(m)

        metrics_list = [m for m in metrics_list if m.n > 0]
        n_total = sum(m.n for m in metrics_list) if metrics_list else 0
        if n_total == 0:
            return {"config": cfg_name, "acc": np.nan, "acc_lo": np.nan, "acc_hi": np.nan,
                    "ll": 0.0, "bic": np.nan, "n": 0, "k_params": 0}

        ll_total = sum(m.ll for m in metrics_list)
        bic_total = sum(m.bic for m in metrics_list if not np.isnan(m.bic))
        acc_total = sum(m.acc * (m.n / n_total) for m in metrics_list if m.n > 0 and not np.isnan(m.acc))
        acc_lo = min(m.acc_lo for m in metrics_list if not np.isnan(m.acc_lo))
        acc_hi = max(m.acc_hi for m in metrics_list if not np.isnan(m.acc_hi))

        return {"config": cfg_name, "acc": acc_total, "acc_lo": acc_lo, "acc_hi": acc_hi,
                "ll": ll_total, "bic": bic_total, "n": n_total, "k_params": int(np.mean([m.k_params for m in metrics_list]))}

    items = named_cfgs
    if JOBLIB_AVAILABLE and cores > 1:
        results = Parallel(n_jobs=min(cores, len(items)), prefer="threads")(
            delayed(fit_eval)(name, cfg) for name, cfg in items
        )
    else:
        results = [fit_eval(name, cfg) for name, cfg in items]

    valid_bics = [r["bic"] for r in results if not np.isnan(r["bic"])]
    best_bic = min(valid_bics) if valid_bics else np.nan
    for r in results:
        r["delta_bic"] = (r["bic"] - best_bic) if not np.isnan(r["bic"]) and not np.isnan(best_bic) else np.nan
    return pd.DataFrame(results)

def run_sim_a0(out_dir: str):
    log("Sim-A0: start")
    mapping = [
        {"EFE_term": "risk (expected cost)", "RSA_term": "utility (−cost)", "Notes": "Sign alignment: cost vs utility"},
        {"EFE_term": "ambiguity (expected entropy)", "RSA_term": "uncertainty penalty", "Notes": "Preference for informative utterances"},
        {"EFE_term": "−information gain (−IG)", "RSA_term": "exploration bonus (with sign)", "Notes": "A1 nearsighted sets IG≈0"},
        {"EFE_term": "β (temperature)", "RSA_term": "α=1/β (rationality)", "Notes": "Softmax/Gibbs normalization"},
        {"EFE_term": "L0 (literal listener)", "RSA_term": "probabilistic, continuous semantics", "Notes": "Calibrated/shared semantics (A5)"},
        {"EFE_term": "Regularity (σ-finite, normalization)", "RSA_term": "finite candidate set", "Notes": "Assumption A6"},
    ]
    df_tab = pd.DataFrame(mapping)
    base = os.path.join(out_dir, "tables", "table01_terms_mapping")
    save_table_csv_tex(df_tab, base)

    fig_dir = os.path.join(out_dir, "figures")
    ensure_dir(fig_dir)
    plt.figure(figsize=(8, 5))
    txt = textwrap.dedent("""
        EFE ↔ RSA Bridge (AIP)
        ----------------------
        risk  ↔  −cost (utility)
        ambiguity  ↔  uncertainty penalty
        −IG  ↔  exploration (sign matters)
        β  ↔  α=1/β
        L0: probabilistic, continuous semantics
        Regularity: finite/σ-finite candidate set
    """).strip("\n")
    plt.axis("off")
    plt.text(0.02, 0.98, txt, va="top", ha="left", fontsize=12, family="monospace")
    pdf_path = os.path.join(fig_dir, "fig01_aip_bridge.pdf")
    png_path = os.path.join(fig_dir, "fig01_aip_bridge.png")
    plt.tight_layout()
    plt.savefig(pdf_path)
    plt.savefig(png_path, dpi=200)
    plt.close()
    log(f"Saved figures: {pdf_path}, {png_path}")
    log("Sim-A0: done")

def run_sim_ladder(df: pd.DataFrame, out_dir: str, cores: int, seed: int = 0):
    log("Sim-LADDER: start")
    text = pick_text_column(df)
    y = guess_binary_target(df)
    folds = fivefold_groups_by_game(df)

    ladder_cfgs = [
        ("A1_only", NBConfig(normalize_len=False, vocab_top_ratio=0.3, drop_rare_min_df=5, noise_std=0.0, temp_beta=1.0)),
        ("A1+A2", NBConfig(normalize_len=True,  vocab_top_ratio=0.3, drop_rare_min_df=5, noise_std=0.0, temp_beta=1.0)),
        ("A1..A4", NBConfig(normalize_len=True, vocab_top_ratio=0.5, drop_rare_min_df=3, noise_std=0.0, temp_beta=1.0)),
        ("A1..A5", NBConfig(normalize_len=True, vocab_top_ratio=0.7, drop_rare_min_df=2, noise_std=0.05, temp_beta=1.0)),
        ("A1..A6", NBConfig(normalize_len=True, vocab_top_ratio=1.0, drop_rare_min_df=2, noise_std=0.0, temp_beta=1.0)),
    ]

    df_tab = cv_eval_configs(text, y, folds, ladder_cfgs, cores=cores)
    base = os.path.join(out_dir, "tables", "table02_ladder")
    save_table_csv_tex(df_tab, base)

    fig_dir = os.path.join(out_dir, "figures")
    ensure_dir(fig_dir)
    plt.figure(figsize=(8, 4.5))
    x = np.arange(len(df_tab))
    plt.plot(x, df_tab["acc"].values, marker="o")
    plt.xticks(x, df_tab["config"].values, rotation=30, ha="right")
    plt.ylabel("Accuracy")
    pdf_path = os.path.join(fig_dir, "fig02_ladder.pdf")
    png_path = os.path.join(fig_dir, "fig02_ladder.png")
    plt.tight_layout()
    plt.savefig(pdf_path)
    plt.savefig(png_path, dpi=200)
    plt.close()
    log(f"Saved figures: {pdf_path}, {png_path}")
    log("Sim-LADDER: done")

def run_sim_cic_main(df: pd.DataFrame, out_dir: str, cores: int, seed: int = 0):
    log("Sim-CIC-MAIN: start")
    text = pick_text_column(df)
    y = guess_binary_target(df)
    folds = fivefold_groups_by_game(df)

    variants = [
        ("RSA_like", NBConfig(add_alpha=1.0, normalize_len=True, vocab_top_ratio=1.0, drop_rare_min_df=2, noise_std=0.0, temp_beta=1.0)),
        ("Ablation_no_len_norm", NBConfig(add_alpha=1.0, normalize_len=False, vocab_top_ratio=1.0, drop_rare_min_df=2, noise_std=0.0, temp_beta=1.0)),
        ("Ablation_vocab_cap_0.5", NBConfig(add_alpha=1.0, normalize_len=True, vocab_top_ratio=0.5, drop_rare_min_df=2, noise_std=0.0, temp_beta=1.0)),
        ("Ablation_noise_0.1", NBConfig(add_alpha=1.0, normalize_len=True, vocab_top_ratio=1.0, drop_rare_min_df=2, noise_std=0.1, temp_beta=1.0)),
    ]

    df_tab = cv_eval_configs(text, y, folds, variants, cores=cores)
    base = os.path.join(out_dir, "tables", "table12_cic_main")
    save_table_csv_tex(df_tab, base)

    fig_dir = os.path.join(out_dir, "figures")
    ensure_dir(fig_dir)
    plt.figure(figsize=(8, 4.5))
    x = np.arange(len(df_tab))
    acc = df_tab["acc"].values
    lo = df_tab["acc_lo"].values
    hi = df_tab["acc_hi"].values
    yerr = np.vstack([acc - lo, hi - acc])
    plt.bar(x, acc)
    plt.errorbar(x, acc, yerr=yerr, fmt="none", capsize=4)
    plt.xticks(x, df_tab["config"].values, rotation=30, ha="right")
    plt.ylabel("Accuracy")
    pdf_path = os.path.join(fig_dir, "fig12_cic_main.pdf")
    png_path = os.path.join(fig_dir, "fig12_cic_main.png")
    plt.tight_layout()
    plt.savefig(pdf_path)
    plt.savefig(png_path, dpi=200)
    plt.close()
    log(f"Saved figures: {pdf_path}, {png_path}")
    log("Sim-CIC-MAIN: done")

def run_sim_form_robust(df: pd.DataFrame, out_dir: str, cores: int, seed: int = 0):
    log("Sim-CIC-FORM-ROBUST: start")
    text = pick_text_column(df)
    y = guess_binary_target(df)
    toks_list = [text_tokenize(t) for t in text.tolist()]
    lengths = np.array([len(toks) for toks in toks_list])
    has_bigram = np.array([1 if len(toks) >= 2 else 0 for toks in toks_list])

    qs = [0.0, 0.25, 0.5, 0.75, 1.0]
    edges = np.quantile(lengths, qs)
    edges = np.unique(edges)
    bins = np.digitize(lengths, bins=edges[1:-1], right=True)

    groups = []
    for b in range(len(edges)):
        groups.append((f"len_bin_{b+1}", (bins == b)))

    groups.append(("has_bigram_0", has_bigram == 0))
    groups.append(("has_bigram_1", has_bigram == 1))

    cfg = NBConfig(add_alpha=1.0, normalize_len=True, vocab_top_ratio=1.0, drop_rare_min_df=2, noise_std=0.0, temp_beta=1.0)
    records = []
    for name, mask in groups:
        n = int(mask.sum())
        if n < 10:
            continue
        log(f"  Group {name}: n={n}")
        model = NaiveBayesText(cfg)
        model.fit([t for i, t in enumerate(text.tolist()) if mask[i]], y[mask])
        proba = model.predict_proba([t for i, t in enumerate(text.tolist()) if mask[i]])
        m = evaluate_model(y_true=y[mask], proba=proba, k_params=len(model.vocab_) + 2)
        rec = m.to_dict()
        rec["group"] = name
        records.append(rec)

    df_tab = pd.DataFrame(records)
    base = os.path.join(out_dir, "tables", "table13_form_robust")
    save_table_csv_tex(df_tab, base)

    fig_dir = os.path.join(out_dir, "figures")
    ensure_dir(fig_dir)
    plt.figure(figsize=(8, 4.5))
    if not df_tab.empty:
        x = np.arange(len(df_tab))
        plt.plot(x, df_tab["acc"].values, marker="o")
        plt.xticks(x, df_tab["group"].values, rotation=45, ha="right")
    plt.ylabel("Accuracy")
    pdf_path = os.path.join(fig_dir, "fig13_form_robust.pdf")
    png_path = os.path.join(fig_dir, "fig13_form_robust.png")
    plt.tight_layout()
    plt.savefig(pdf_path)
    plt.savefig(png_path, dpi=200)
    plt.close()
    log(f"Saved figures: {pdf_path}, {png_path}")
    log("Sim-CIC-FORM-ROBUST: done")

def run_sim_a1_nearsighted(df: pd.DataFrame, out_dir: str, cores: int, seed: int = 0):
    log("Sim-A1-NEARSIGHTED: start")
    text = pick_text_column(df)
    y = guess_binary_target(df)
    folds = fivefold_groups_by_game(df)

    grid = [1.0, 0.75, 0.5, 0.25, 0.1]
    named_cfgs = [(f"vocab_top_ratio_{r:.2f}", NBConfig(normalize_len=True, vocab_top_ratio=r, drop_rare_min_df=2)) for r in grid]
    df_tab = cv_eval_configs(text, y, folds, named_cfgs, cores=cores)

    base = os.path.join(out_dir, "tables", "table03_nearsighted")
    save_table_csv_tex(df_tab, base)

    fig_dir = os.path.join(out_dir, "figures")
    ensure_dir(fig_dir)
    plt.figure(figsize=(8, 4.5))
    x = np.arange(len(df_tab))
    plt.plot(x, df_tab["acc"].values, marker="o")
    plt.xticks(x, df_tab["config"].values, rotation=30, ha="right")
    plt.ylabel("Accuracy")
    pdf_path = os.path.join(fig_dir, "fig03_nearsighted.pdf")
    png_path = os.path.join(fig_dir, "fig03_nearsighted.png")
    plt.tight_layout()
    plt.savefig(pdf_path)
    plt.savefig(png_path, dpi=200)
    plt.close()
    log("Sim-A1-NEARSIGHTED: done")

def run_sim_a2_equal_entropy(df: pd.DataFrame, out_dir: str, cores: int, seed: int = 0):
    log("Sim-A2-EQUAL-ENTROPY: start")
    text = pick_text_column(df)
    y = guess_binary_target(df)

    # Tokenize once to measure lengths robustly
    toks_list = [text_tokenize(t) for t in text.tolist()]
    lengths = np.array([len(toks) for toks in toks_list])

    # Robust quantile binning
    q = 4
    _, edges = pd.qcut(lengths, q=q, retbins=True, duplicates="drop")
    k = max(1, len(edges) - 1)
    labels = [f"Q{i+1}" for i in range(k)]
    qbins = pd.cut(lengths, bins=edges, include_lowest=True, labels=labels)

    df_local = pd.DataFrame({"text": text, "y": y, "bin": qbins.astype(str)})

    records = []
    for bname in sorted(df_local["bin"].dropna().unique()):
        sel = (df_local["bin"] == bname).values
        n = int(sel.sum())
        if n < 30:
            log(f"  Skip {bname} (n={n}<30)")
            continue
        log(f"  Bin {bname}: n={n}")
        text_b = df_local.loc[sel, "text"]
        y_b = df_local.loc[sel, "y"].values
        if "gameid" in df.columns:
            folds_b = fivefold_groups_by_game(df.loc[sel, :])
        else:
            folds_b = fivefold_groups_by_game(pd.DataFrame({"idx": np.arange(n)}))

        named_cfgs = [
            (f"{bname}_len_norm_on", NBConfig(normalize_len=True, vocab_top_ratio=1.0, drop_rare_min_df=2)),
            (f"{bname}_len_norm_off", NBConfig(normalize_len=False, vocab_top_ratio=1.0, drop_rare_min_df=2)),
        ]
        df_tab_b = cv_eval_configs(text_b, y_b, folds_b, named_cfgs, cores=cores)
        records.append(df_tab_b)

    df_tab = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    base = os.path.join(out_dir, "tables", "table04_equal_entropy")
    save_table_csv_tex(df_tab, base)

    fig_dir = os.path.join(out_dir, "figures")
    ensure_dir(fig_dir)
    plt.figure(figsize=(9, 4.5))
    if not df_tab.empty:
        x = np.arange(len(df_tab))
        plt.bar(x, df_tab["acc"].values)
        plt.xticks(x, df_tab["config"].values, rotation=45, ha="right")
    plt.ylabel("Accuracy")
    pdf_path = os.path.join(fig_dir, "fig04_equal_entropy.pdf")
    png_path = os.path.join(fig_dir, "fig04_equal_entropy.png")
    plt.tight_layout()
    plt.savefig(pdf_path)
    plt.savefig(png_path, dpi=200)
    plt.close()
    log("Sim-A2-EQUAL-ENTROPY: done")

def _softmax_scores(scores: np.ndarray) -> np.ndarray:
    if scores.shape[0] == 0:
        return np.zeros((0, scores.shape[1]))
    m = scores.max(axis=1, keepdims=True)
    exp = np.exp(scores - m)
    return exp / exp.sum(axis=1, keepdims=True)

def run_sim_a3_state_indep_cost(df: pd.DataFrame, out_dir: str, cores: int, seed: int = 0):
    log("Sim-A3-STATE-INDEP-COST: start")
    text = pick_text_column(df)
    y = guess_binary_target(df)
    folds = fivefold_groups_by_game(df)

    state_feat = df["targetD1Diff"].fillna(0).astype(float).values if "targetD1Diff" in df.columns else np.zeros(len(df))

    amps = [0.0, 0.25, 0.5, 1.0, 2.0]
    records = []

    for amp in amps:
        log(f"  Amplitude amp={amp}")
        metrics_list = []
        for i, test_mask in enumerate(folds):
            log(f"    CV fold {i+1}/{len(folds)}")
            train_mask = ~test_mask
            cfg = NBConfig(normalize_len=True, vocab_top_ratio=1.0, drop_rare_min_df=2)
            model = NaiveBayesText(cfg)
            model.fit(text[train_mask].tolist(), y[train_mask])

            scores = model.base_scores(text[test_mask].tolist())
            bias = amp * (state_feat[test_mask] - np.mean(state_feat[train_mask]))
            scores[:, 1] += bias
            proba = _softmax_scores(scores)
            m = evaluate_model(y_true=y[test_mask], proba=proba, k_params=len(model.vocab_) + 3)
            metrics_list.append(m)

        metrics_list = [m for m in metrics_list if m.n > 0]
        n_total = sum(m.n for m in metrics_list) if metrics_list else 0
        if n_total == 0:
            continue
        ll_total = sum(m.ll for m in metrics_list)
        bic_total = sum(m.bic for m in metrics_list if not np.isnan(m.bic))
        acc_total = sum(m.acc * (m.n / n_total) for m in metrics_list if not np.isnan(m.acc))
        acc_lo = min(m.acc_lo for m in metrics_list if not np.isnan(m.acc_lo))
        acc_hi = max(m.acc_hi for m in metrics_list if not np.isnan(m.acc_hi))
        records.append({"amp": amp, "acc": acc_total, "acc_lo": acc_lo, "acc_hi": acc_hi, "ll": ll_total, "bic": bic_total, "n": n_total, "k_params": int(np.mean([m.k_params for m in metrics_list]))})

    df_tab = pd.DataFrame(records)
    base = os.path.join(out_dir, "tables", "table05_state_indep_cost")
    save_table_csv_tex(df_tab, base)

    fig_dir = os.path.join(out_dir, "figures")
    ensure_dir(fig_dir)
    plt.figure(figsize=(8, 4.5))
    x = np.arange(len(df_tab))
    plt.plot(x, df_tab["acc"].values, marker="o")
    plt.xticks(x, [f"amp={a}" for a in df_tab["amp"].values], rotation=0)
    plt.ylabel("Accuracy")
    pdf_path = os.path.join(fig_dir, "fig05_state_indep_cost.pdf")
    png_path = os.path.join(fig_dir, "fig05_state_indep_cost.png")
    plt.tight_layout()
    plt.savefig(pdf_path)
    plt.savefig(png_path, dpi=200)
    plt.close()
    log("Sim-A3-STATE-INDEP-COST: done")

def run_sim_a4_local_temp(df: pd.DataFrame, out_dir: str, cores: int, seed: int = 0):
    log("Sim-A4-LOCAL-TEMP: start")
    text = pick_text_column(df)
    y = guess_binary_target(df)
    folds = fivefold_groups_by_game(df)

    if "msgTime" in df.columns:
        tvals = pd.to_datetime(df["msgTime"], errors="coerce")
        order = (tvals - tvals.min()).dt.total_seconds().fillna(0).values
    else:
        order = np.arange(len(df))
    order = (order - order.min()) / (order.max() - order.min() + 1e-9)

    drifts = [
        ("beta_const_1.0", lambda o: np.ones_like(o) * 1.0),
        ("beta_linear_1.0_to_1.5", lambda o: 1.0 + 0.5 * o),
        ("beta_linear_1.5_to_1.0", lambda o: 1.5 - 0.5 * o),
        ("beta_randomwalk_sigma0.2", None),
    ]

    records = []
    for name, fn in drifts:
        log(f"  Drift mode: {name}")
        metrics_list = []
        for i, test_mask in enumerate(folds):
            log(f"    CV fold {i+1}/{len(folds)}")
            train_mask = ~test_mask
            cfg = NBConfig(normalize_len=True, vocab_top_ratio=1.0, drop_rare_min_df=2, temp_beta=1.0)
            model = NaiveBayesText(cfg)
            model.fit(text[train_mask].tolist(), y[train_mask])

            if fn is None:
                steps = np.random.normal(0.0, 0.2, size=np.sum(test_mask))
                beta_arr = 1.0 + np.cumsum(steps)
                beta_arr = np.clip(beta_arr, 0.5, 2.0)
            else:
                beta_arr = fn(order[test_mask])

            proba = model.predict_proba_with_beta(text[test_mask].tolist(), beta_arr)
            m = evaluate_model(y_true=y[test_mask], proba=proba, k_params=len(model.vocab_) + 2)
            metrics_list.append(m)

        metrics_list = [m for m in metrics_list if m.n > 0]
        n_total = sum(m.n for m in metrics_list) if metrics_list else 0
        if n_total == 0:
            continue
        ll_total = sum(m.ll for m in metrics_list)
        bic_total = sum(m.bic for m in metrics_list if not np.isnan(m.bic))
        acc_total = sum(m.acc * (m.n / n_total) for m in metrics_list if not np.isnan(m.acc))
        acc_lo = min(m.acc_lo for m in metrics_list if not np.isnan(m.acc_lo))
        acc_hi = max(m.acc_hi for m in metrics_list if not np.isnan(m.acc_hi))
        records.append({"drift": name, "acc": acc_total, "acc_lo": acc_lo, "acc_hi": acc_hi, "ll": ll_total, "bic": bic_total, "n": n_total, "k_params": int(np.mean([m.k_params for m in metrics_list]))})

    df_tab = pd.DataFrame(records)
    base = os.path.join(out_dir, "tables", "table06_local_temp")
    save_table_csv_tex(df_tab, base)
    try:
        df_tab.to_csv(base + ".csv", index=False)
    except Exception:
        pass

    # Pretty labels for A4
    display_map = {
        "beta_const_1.0": "const β=1.0",
        "beta_linear_1.0_to_1.5": "linear 1.0→1.5",
        "beta_linear_1.5_to_1.0": "linear 1.5→1.0",
        "beta_randomwalk_sigma0.2": "random walk σ=0.2",
    }
    df_tab["label"] = [display_map.get(d, d) for d in df_tab["drift"].values]

    fig_dir = os.path.join(out_dir, "figures")
    ensure_dir(fig_dir)
    plt.figure(figsize=(8, 4.5))
    x = np.arange(len(df_tab))
    plt.bar(x, df_tab["acc"].values)
    plt.xticks(x, df_tab["label"].values, rotation=30, ha="right")
    plt.ylabel("Accuracy")
    pdf_path = os.path.join(fig_dir, "fig06_local_temp.pdf")
    png_path = os.path.join(fig_dir, "fig06_local_temp.png")
    plt.tight_layout()
    plt.savefig(pdf_path)
    plt.savefig(png_path, dpi=200)
    plt.close()
    log("Sim-A4-LOCAL-TEMP: done")

def run_sim_a5_semantics_share(df: pd.DataFrame, out_dir: str, cores: int, seed: int = 0):
    log("Sim-A5-SEMANTICS-SHARE: start")
    text = pick_text_column(df)
    y = guess_binary_target(df)
    folds = fivefold_groups_by_game(df)

    grid = [0.0, 0.05, 0.1, 0.2, 0.3]
    named_cfgs = [(f"noise_{s:.2f}", NBConfig(normalize_len=True, vocab_top_ratio=1.0, drop_rare_min_df=2, noise_std=s)) for s in grid]
    df_tab = cv_eval_configs(text, y, folds, named_cfgs, cores=cores)

    base = os.path.join(out_dir, "tables", "table07_semantics_share")
    save_table_csv_tex(df_tab, base)

    fig_dir = os.path.join(out_dir, "figures")
    ensure_dir(fig_dir)
    plt.figure(figsize=(8, 4.5))
    x = np.arange(len(df_tab))
    plt.plot(x, df_tab["acc"].values, marker="o")
    plt.xticks(x, df_tab["config"].values, rotation=30, ha="right")
    plt.ylabel("Accuracy")
    pdf_path = os.path.join(fig_dir, "fig07_semantics_share.pdf")
    png_path = os.path.join(fig_dir, "fig07_semantics_share.png")
    plt.tight_layout()
    plt.savefig(pdf_path)
    plt.savefig(png_path, dpi=200)
    plt.close()
    log("Sim-A5-SEMANTICS-SHARE: done")

def run_sim_a6_regularity(df: pd.DataFrame, out_dir: str, cores: int, seed: int = 0):
    log("Sim-A6-REGULARITY: start")
    text = pick_text_column(df)
    y = guess_binary_target(df)
    folds = fivefold_groups_by_game(df)

    setups = [
        ("df>=2_no_drop_top", NBConfig(drop_rare_min_df=2, drop_top_frac=0.0)),
        ("df>=3_no_drop_top", NBConfig(drop_rare_min_df=3, drop_top_frac=0.0)),
        ("df>=5_no_drop_top", NBConfig(drop_rare_min_df=5, drop_top_frac=0.0)),
        ("df>=2_drop_top_5pct", NBConfig(drop_rare_min_df=2, drop_top_frac=0.05)),
        ("df>=2_drop_top_10pct", NBConfig(drop_rare_min_df=2, drop_top_frac=0.10)),
    ]
    named_cfgs = [(name, NBConfig(normalize_len=True, vocab_top_ratio=1.0,
                                  drop_rare_min_df=cfg.drop_rare_min_df, drop_top_frac=cfg.drop_top_frac))
                  for name, cfg in setups]

    df_tab = cv_eval_configs(text, y, folds, named_cfgs, cores=cores)

    base = os.path.join(out_dir, "tables", "table08_regularity")
    save_table_csv_tex(df_tab, base)

    fig_dir = os.path.join(out_dir, "figures")
    ensure_dir(fig_dir)
    plt.figure(figsize=(8, 4.5))
    x = np.arange(len(df_tab))
    plt.bar(x, df_tab["acc"].values)
    plt.xticks(x, df_tab["config"].values, rotation=30, ha="right")
    plt.ylabel("Accuracy")
    pdf_path = os.path.join(fig_dir, "fig08_regularity.pdf")
    png_path = os.path.join(fig_dir, "fig08_regularity.png")
    plt.tight_layout()
    plt.savefig(pdf_path)
    plt.savefig(png_path, dpi=200)
    plt.close()
    log("Sim-A6-REGULARITY: done")

def run_sim_factor_screen(df: pd.DataFrame, out_dir: str, cores: int, seed: int = 0):
    log("Sim-FACTOR-SCREEN: start")
    text = pick_text_column(df)
    y = guess_binary_target(df)
    folds = fivefold_groups_by_game(df)

    levels = {
        "A1_vocab_cap": [0.5, 1.0],
        "A2_len_norm": [False, True],
        "A3_state_cost_amp": [0.0, 1.0],
        "A4_beta_drift": ["const", "linear"],
        "A5_noise": [0.0, 0.1],
        "A6_rare_df": [5, 2],
    }

    designs = [
        {"A1_vocab_cap":0.5, "A2_len_norm":False, "A3_state_cost_amp":0.0, "A4_beta_drift":"const",  "A5_noise":0.0, "A6_rare_df":5},
        {"A1_vocab_cap":1.0, "A2_len_norm":False, "A3_state_cost_amp":1.0, "A4_beta_drift":"const",  "A5_noise":0.1, "A6_rare_df":2},
        {"A1_vocab_cap":0.5, "A2_len_norm":True,  "A3_state_cost_amp":1.0, "A4_beta_drift":"linear", "A5_noise":0.0, "A6_rare_df":2},
        {"A1_vocab_cap":1.0, "A2_len_norm":True,  "A3_state_cost_amp":0.0, "A4_beta_drift":"linear", "A5_noise":0.1, "A6_rare_df":5},
        {"A1_vocab_cap":0.5, "A2_len_norm":True,  "A3_state_cost_amp":0.0, "A4_beta_drift":"const",  "A5_noise":0.1, "A6_rare_df":2},
        {"A1_vocab_cap":1.0, "A2_len_norm":True,  "A3_state_cost_amp":1.0, "A4_beta_drift":"const",  "A5_noise":0.0, "A6_rare_df":5},
        {"A1_vocab_cap":0.5, "A2_len_norm":False, "A3_state_cost_amp":1.0, "A4_beta_drift":"linear","A5_noise":0.1, "A6_rare_df":5},
        {"A1_vocab_cap":1.0, "A2_len_norm":False, "A3_state_cost_amp":0.0, "A4_beta_drift":"linear","A5_noise":0.0, "A6_rare_df":2},
    ]

    if "msgTime" in df.columns:
        tvals = pd.to_datetime(df["msgTime"], errors="coerce")
        order = (tvals - tvals.min()).dt.total_seconds().fillna(0).values
    else:
        order = np.arange(len(df))
    order = (order - order.min()) / (order.max() - order.min() + 1e-9)
    state_feat = df["targetD1Diff"].fillna(0).astype(float).values if "targetD1Diff" in df.columns else np.zeros(len(df))

    def eval_design(d: Dict) -> Dict[str, float]:
        log(f"  Design: {d}")
        metrics_list = []
        for i, test_mask in enumerate(folds):
            train_mask = ~test_mask
            cfg = NBConfig(normalize_len=bool(d["A2_len_norm"]),
                           vocab_top_ratio=float(d["A1_vocab_cap"]),
                           drop_rare_min_df=int(d["A6_rare_df"]),
                           noise_std=float(d["A5_noise"]),
                           temp_beta=1.0)
            model = NaiveBayesText(cfg)
            model.fit(text[train_mask].tolist(), y[train_mask])

            scores = model.base_scores(text[test_mask].tolist())

            amp = float(d["A3_state_cost_amp"])
            bias = amp * (state_feat[test_mask] - np.mean(state_feat[train_mask]))
            scores[:, 1] += bias

            if d["A4_beta_drift"] == "linear":
                beta_arr = 1.0 + 0.5 * order[test_mask]
            else:
                beta_arr = np.ones(np.sum(test_mask)) * 1.0

            proba = _softmax_scores(scores / beta_arr.reshape(-1, 1))
            m = evaluate_model(y_true=y[test_mask], proba=proba, k_params=len(model.vocab_) + 4)
            metrics_list.append(m)

        metrics_list = [m for m in metrics_list if m.n > 0]
        n_total = sum(m.n for m in metrics_list) if metrics_list else 0
        if n_total == 0:
            return {"acc": np.nan, "ll": 0.0, "bic": np.nan, **d}
        acc_total = sum(m.acc * (m.n / n_total) for m in metrics_list if not np.isnan(m.acc))
        ll_total = sum(m.ll for m in metrics_list)
        bic_total = sum(m.bic for m in metrics_list if not np.isnan(m.bic))
        return {"acc": acc_total, "ll": ll_total, "bic": bic_total, **d}

    if JOBLIB_AVAILABLE and cores > 1:
        results = Parallel(n_jobs=min(len(designs), cores), prefer="threads")(delayed(eval_design)(d) for d in designs)
    else:
        results = [eval_design(d) for d in designs]

    df_tab = pd.DataFrame(results)
    base = os.path.join(out_dir, "tables", "table09_factor_screen")
    save_table_csv_tex(df_tab, base)

    effects = []
    for key, levels_ in levels.items():
        for level in levels_:
            mask = df_tab[key] == level
            effects.append({"factor_level": f"{key}={level}", "mean_acc": df_tab.loc[mask, "acc"].mean()})
    df_eff = pd.DataFrame(effects)

    fig_dir = os.path.join(out_dir, "figures"); ensure_dir(fig_dir)
    plt.figure(figsize=(9, 4.5))
    x = np.arange(len(df_eff)); plt.bar(x, df_eff["mean_acc"].values)
    plt.xticks(x, df_eff["factor_level"].values, rotation=45, ha="right")
    plt.ylabel("Mean Accuracy")
    pdf_path = os.path.join(fig_dir, "fig09_factor_screen.pdf")
    png_path = os.path.join(fig_dir, "fig09_factor_screen.png")
    plt.tight_layout()
    plt.savefig(pdf_path)
    plt.savefig(png_path, dpi=200)
    plt.close()
    log("Sim-FACTOR-SCREEN: done")


def run_sim_param_recovery(df: pd.DataFrame, out_dir: str, cores: int, seed: int = 0):
    log("Sim-PARAM-RECOVERY: start")
    text = pick_text_column(df)
    y_obs = guess_binary_target(df)
    folds = fivefold_groups_by_game(df)

    # True and candidate alphas (α = 1/β)
    alpha_true_grid = [0.5, 1.0, 2.0, 3.0]
    alpha_fit_grid = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]

    records = []
    for alpha_true in alpha_true_grid:
        beta_true = 1.0 / alpha_true
        log(f"  Alpha_true={alpha_true} (beta_true={beta_true:.3f})")
        fold_ests = []

        for i, test_mask in enumerate(folds):
            log(f"    CV fold {i+1}/{len(folds)}")
            train_mask = ~test_mask

            # Train TEACHER on observed labels (structure-identical to the fitted family)
            teacher = NaiveBayesText(NBConfig(normalize_len=True, vocab_top_ratio=1.0, drop_rare_min_df=2, temp_beta=1.0))
            teacher.fit(text[train_mask].tolist(), y_obs[train_mask])

            # TEACHER base scores (pre-temperature) on train/test
            scores_train = teacher.base_scores(text[train_mask].tolist())
            scores_test  = teacher.base_scores(text[test_mask].tolist())

            # Generate stochastic labels from teacher at β_true
            P_train = _softmax_scores(scores_train / beta_true)
            P_test  = _softmax_scores(scores_test  / beta_true)
            y_train_teach = (np.random.rand(P_train.shape[0]) < P_train[:, 1]).astype(int)
            y_test_teach  = (np.random.rand(P_test.shape[0])  < P_test[:, 1]).astype(int)

            # Parameter recovery: do NOT retrain a student.
            # Evaluate candidate α by applying β=1/α to the *same* teacher scores.
            best_alpha = None; best_ll = -np.inf
            for alpha_fit in alpha_fit_grid:
                beta_fit = 1.0 / alpha_fit
                proba_fit = _softmax_scores(scores_test / beta_fit)
                m = evaluate_model(y_true=y_test_teach, proba=proba_fit, k_params=len(teacher.vocab_) + 2)
                if m.ll > best_ll:
                    best_ll = m.ll
                    best_alpha = alpha_fit

            if best_alpha is not None:
                fold_ests.append(best_alpha)

        if len(fold_ests) == 0:
            continue
        rec = {
            "alpha_true": alpha_true,
            "alpha_hat_mean": float(np.mean(fold_ests)),
            "alpha_hat_std": float(np.std(fold_ests)),
            "abs_error_mean": float(np.mean([abs(a - alpha_true) for a in fold_ests])),
        }
        records.append(rec)

    df_tab = pd.DataFrame(records)
    base = os.path.join(out_dir, "tables", "table10_param_recovery")
    save_table_csv_tex(df_tab, base)

    # Plot α* vs α̂ (no title)
    fig_dir = os.path.join(out_dir, "figures"); ensure_dir(fig_dir)
    plt.figure(figsize=(7, 4.5))
    x = np.arange(len(df_tab))
    plt.plot(x, df_tab["alpha_true"].values, marker="o")
    plt.plot(x, df_tab["alpha_hat_mean"].values, marker="x")
    plt.xticks(x, [f"α*={a}" for a in df_tab["alpha_true"].values])
    plt.ylabel("Alpha")
    pdf_path = os.path.join(fig_dir, "fig10_param_recovery.pdf")
    png_path = os.path.join(fig_dir, "fig10_param_recovery.png")
    plt.tight_layout(); plt.savefig(pdf_path); plt.savefig(png_path, dpi=200); plt.close()
    log("Sim-PARAM-RECOVERY: done")

def run_sim_model_recovery(df: pd.DataFrame, out_dir: str, cores: int, seed: int = 0):
    log("Sim-MODEL-RECOVERY: start")
    text = pick_text_column(df)
    y_obs = guess_binary_target(df)
    folds = fivefold_groups_by_game(df)

    gen_cfgs = [
        ("GEN_A1heavy", NBConfig(normalize_len=True, vocab_top_ratio=0.3, drop_rare_min_df=5, noise_std=0.0, temp_beta=1.0)),
        ("GEN_full", NBConfig(normalize_len=True, vocab_top_ratio=1.0, drop_rare_min_df=2, noise_std=0.05, temp_beta=1.0)),
    ]
    fit_cfgs = [
        ("FIT_RSA_like", NBConfig(normalize_len=True, vocab_top_ratio=1.0, drop_rare_min_df=2)),
        ("FIT_vocab_cap_0.5", NBConfig(normalize_len=True, vocab_top_ratio=0.5, drop_rare_min_df=2)),
        ("FIT_no_len_norm", NBConfig(normalize_len=False, vocab_top_ratio=1.0, drop_rare_min_df=2)),
    ]

    conf = {(gname, fname): 0 for gname, _ in gen_cfgs for fname, _ in fit_cfgs}
    totals = {gname: 0 for gname, _ in gen_cfgs}

    for gname, gcfg in gen_cfgs:
        log(f"  Generator: {gname}")
        for i, test_mask in enumerate(folds):
            train_mask = ~test_mask
            gen = NaiveBayesText(gcfg)
            gen.fit(text[train_mask].tolist(), y_obs[train_mask])
            y_train_gen = np.argmax(gen.predict_proba(text[train_mask].tolist()), axis=1)
            y_test_gen = np.argmax(gen.predict_proba(text[test_mask].tolist()), axis=1)

            best_bic = np.inf; best_fit_name = None
            for fname, fcfg in fit_cfgs:
                model = NaiveBayesText(fcfg)
                model.fit(text[train_mask].tolist(), y_train_gen)
                proba = model.predict_proba(text[test_mask].tolist())
                m = evaluate_model(y_true=y_test_gen, proba=proba, k_params=len(model.vocab_) + 2)
                if not np.isnan(m.bic) and m.bic < best_bic:
                    best_bic = m.bic
                    best_fit_name = fname

            if best_fit_name is not None:
                conf[(gname, best_fit_name)] += 1
            totals[gname] += 1

    rows = []
    for gname, _ in gen_cfgs:
        row = {"generator": gname}
        for fname, _ in fit_cfgs:
            row[fname] = conf[(gname, fname)]
        rows.append(row)
    df_tab = pd.DataFrame(rows)
    base = os.path.join(out_dir, "tables", "table11_model_recovery")
    save_table_csv_tex(df_tab, base)

    fig_dir = os.path.join(out_dir, "figures"); ensure_dir(fig_dir)
    fig, ax = plt.subplots(figsize=(6, 3.8))
    mat = df_tab[[f for f, _ in fit_cfgs]].values
    im = ax.imshow(mat, aspect="auto")
    ax.set_xticks(np.arange(len(fit_cfgs)))
    ax.set_xticklabels([f for f, _ in fit_cfgs], rotation=30, ha="right")
    ax.set_yticks(np.arange(len(gen_cfgs)))
    ax.set_yticklabels([g for g, _ in gen_cfgs])
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("count (best-BIC picks)")
    pdf_path = os.path.join(fig_dir, "fig11_model_recovery.pdf")
    png_path = os.path.join(fig_dir, "fig11_model_recovery.png")
    plt.tight_layout()
    plt.savefig(pdf_path)
    plt.savefig(png_path, dpi=200)
    plt.close()
    log("Sim-MODEL-RECOVERY: done")

def run_sim_aip_two_stage(df: pd.DataFrame, out_dir: str, cores: int, seed: int = 0):
    """Proper cross-validated two-stage policy (confirm -> commit)."""
    log("Sim-AIP-TWO-STAGE: start")
    text = pick_text_column(df)
    y = guess_binary_target(df)
    folds = fivefold_groups_by_game(df)

    def to_token_prefix(val, name):
        try:
            return f"{name}_{str(val).lower()}"
        except Exception:
            return f"{name}_na"

    confirm_cols = [c for c in ["condition", "role", "source"] if c in df.columns]
    add_tokens = []
    for i in range(len(df)):
        toks = []
        for c in confirm_cols:
            toks.append(to_token_prefix(df.iloc[i][c], c))
        for c in ["clickColH", "clickColS", "clickColL"]:
            if c in df.columns:
                try:
                    v = float(df.iloc[i][c])
                    binv = int(np.clip(v // 10, 0, 10))
                    toks.append(f"{c}_b{binv}")
                except Exception:
                    pass
        add_tokens.append(" ".join(toks))
    add_tokens = pd.Series(add_tokens, index=text.index)

    thresholds = [0.05, 0.1, 0.2, 0.3]
    base_cfg = NBConfig(normalize_len=True, vocab_top_ratio=1.0, drop_rare_min_df=2)

    records = []
    for tau in thresholds:
        log(f"  Threshold τ={tau}")
        metrics_list = []
        n_uncertain_total = 0
        for i, test_mask in enumerate(folds):
            log(f"    CV fold {i+1}/{len(folds)}")
            train_mask = ~test_mask

            base_model = NaiveBayesText(base_cfg)
            base_model.fit(text[train_mask].tolist(), y[train_mask])

            base_proba_train = base_model.predict_proba(text[train_mask].tolist())
            unc_train = (np.abs(base_proba_train[:, 1] - 0.5) <= tau)
            n_uncertain_train = int(unc_train.sum())

            aug_train_text = text[train_mask].copy()
            if n_uncertain_train > 0:
                sel_idx = aug_train_text.index[unc_train]
                aug_train_text.loc[sel_idx] = (aug_train_text.loc[sel_idx] + " " + add_tokens.loc[sel_idx]).values

            confirm_model = NaiveBayesText(base_cfg)
            confirm_model.fit(aug_train_text.tolist(), y[train_mask])

            base_proba_test = base_model.predict_proba(text[test_mask].tolist())
            unc_test = (np.abs(base_proba_test[:, 1] - 0.5) <= tau)
            n_uncertain_test = int(unc_test.sum())
            n_uncertain_total += n_uncertain_test

            proba_comb = base_proba_test.copy()
            if n_uncertain_test > 0:
                sel_uncertain_abs = text.index[test_mask][unc_test]
                aug_test_texts = text.loc[sel_uncertain_abs] + " " + add_tokens.loc[sel_uncertain_abs]
                proba_unc = confirm_model.predict_proba(aug_test_texts.tolist())
                proba_comb[unc_test] = proba_unc

            k_total = len(base_model.vocab_) + len(confirm_model.vocab_) + 4
            m = evaluate_model(y_true=y[test_mask], proba=proba_comb, k_params=k_total)
            metrics_list.append(m)

        metrics_list = [m for m in metrics_list if m.n > 0]
        n_total = sum(m.n for m in metrics_list) if metrics_list else 0
        if n_total == 0:
            continue
        ll_total = sum(m.ll for m in metrics_list)
        bic_total = sum(m.bic for m in metrics_list if not np.isnan(m.bic))
        acc_total = sum(m.acc * (m.n / n_total) for m in metrics_list if not np.isnan(m.acc))
        acc_lo = min(m.acc_lo for m in metrics_list if not np.isnan(m.acc_lo))
        acc_hi = max(m.acc_hi for m in metrics_list if not np.isnan(m.acc_hi))

        records.append({"tau": tau, "acc": acc_total, "acc_lo": acc_lo, "acc_hi": acc_hi,
                        "ll": ll_total, "bic": bic_total, "n_uncertain_total": n_uncertain_total})

    df_tab = pd.DataFrame(records)
    base = os.path.join(out_dir, "tables", "table14_two_stage")
    save_table_csv_tex(df_tab, base)

    fig_dir = os.path.join(out_dir, "figures"); ensure_dir(fig_dir)
    plt.figure(figsize=(8, 4.5))
    x = np.arange(len(df_tab)); plt.plot(x, df_tab["acc"].values, marker="o")
    plt.xticks(x, [f"τ={t}" for t in df_tab["tau"].values])
    plt.ylabel("Accuracy")
    pdf_path = os.path.join(fig_dir, "fig14_two_stage.pdf")
    png_path = os.path.join(fig_dir, "fig14_two_stage.png")
    plt.tight_layout()
    plt.savefig(pdf_path)
    plt.savefig(png_path, dpi=200)
    plt.close()
    log("Sim-AIP-TWO-STAGE: done")


def run_sim_ladder_dll(df: pd.DataFrame, out_dir: str, cores: int, seed: int = 0):
    """
    Sim-LADDER-DLL: Compute per-example ΔLL between adjacent LADDER rungs.
    Requires table02_ladder.csv; will run Sim-LADDER first if missing.
    Outputs:
      - tables/table02b_ladder_dll_per_example.csv
      - figures/fig02b_ladder_dll_per_example.(pdf|png)
    """
    log("Sim-LADDER-DLL: start")
    ladder_csv = os.path.join(out_dir, "tables", "table02_ladder.csv")
    if not os.path.exists(ladder_csv):
        log("  Missing table02_ladder.csv; running Sim-LADDER first.")
        run_sim_ladder(df, out_dir, cores, seed)
    if not os.path.exists(ladder_csv):
        raise FileNotFoundError("Sim-LADDER-DLL requires table02_ladder.csv")
    lad = pd.read_csv(ladder_csv)
    desired = ["A1_only", "A1+A2", "A1..A4", "A1..A5", "A1..A6"]
    order = [c for c in desired if c in lad["config"].tolist()]
    if len(order) < 2:
        order = lad["config"].tolist()
    idx = lad.set_index("config")
    rows = []
    for i in range(len(order)-1):
        a, b = order[i], order[i+1]
        ll_a = float(idx.loc[a, "ll"]); ll_b = float(idx.loc[b, "ll"])
        n = int(idx.loc[a, "n"])
        dll = (ll_b - ll_a) / max(1, n)
        rows.append({"from": a, "to": b, "delta_ll_per_example": dll})
    df_out = pd.DataFrame(rows)
    base = os.path.join(out_dir, "tables", "table02b_ladder_dll_per_example")
    try:
        save_table_csv_tex(df_out, base)
    except Exception:
        pass
    try:
        df_out.to_csv(base + ".csv", index=False)
    except Exception as e:
        log(f"CSV write failed: {e}")
    ensure_dir(os.path.join(out_dir, "figures"))
    plt.figure(figsize=(8, 4.0))
    x = np.arange(len(df_out))
    plt.bar(x, df_out["delta_ll_per_example"].values)
    plt.xticks(x, [f"{r['from']}→{r['to']}" for _, r in df_out.iterrows()], rotation=20, ha="right")
    plt.ylabel("ΔLL per example")
    plt.title("LADDER: per-example ΔLL between adjacent rungs")
    pdf_path = os.path.join(out_dir, "figures", "fig02b_ladder_dll_per_example.pdf")
    png_path = os.path.join(out_dir, "figures", "fig02b_ladder_dll_per_example.png")
    plt.tight_layout(); plt.savefig(pdf_path); plt.savefig(png_path, dpi=200); plt.close()
    log(f"Saved figures: {pdf_path}, {png_path}")
    log("Sim-LADDER-DLL: done")


def run_sim_amb_entropy(df: pd.DataFrame, out_dir: str, cores: int, seed: int = 0):
    """
    Sim-AMB-ENTROPY: Ambiguity vs. Entropy sanity check from CV listener posteriors.
    - Builds 5 folds by game id (if present), otherwise by chunks.
    - Trains a simple NaiveBayesText model out-of-fold and collects P(y=1|u).
    - Saves both per-utterance detail and summary.
    Outputs:
      - tables/table15_amb_entropy_detail.csv  (utterance-level)
      - tables/table15_amb_entropy_summary.csv (global + per-context stats)
      - figures/fig15a_amb_entropy_scatter.(pdf|png)
      - figures/fig15b_amb_entropy_spearman_hist.(pdf|png)  (if context available)
    """
    log("Sim-AMB-ENTROPY: start")
    text = pick_text_column(df)
    y = guess_binary_target(df)
    folds = fivefold_groups_by_game(df)
    proba_oof = np.zeros((len(df), 2), dtype=float)
    seen = np.zeros(len(df), dtype=bool)
    base_cfg = NBConfig(normalize_len=True, vocab_top_ratio=1.0, drop_rare_min_df=2)
    for i, test_mask in enumerate(folds):
        train_mask = ~test_mask
        model = NaiveBayesText(base_cfg)
        model.fit(text[train_mask].tolist(), y[train_mask])
        proba = model.predict_proba(text[test_mask].tolist())
        proba_oof[test_mask] = proba
        seen[test_mask] = True
        log(f"  CV fold {i+1}/{len(folds)}: filled {int(test_mask.sum())} rows")
    if not seen.all():
        miss = np.where(~seen)[0]
        if len(miss) > 0:
            log(f"  Filling {len(miss)} missing with in-sample proba")
            model = NaiveBayesText(base_cfg).fit(text.tolist(), y)
            proba_oof[miss] = model.predict_proba(text.iloc[miss].tolist())
    # Per-utterance aggregation
    df_row = pd.DataFrame({"utterance": text.values, "p1": proba_oof[:, 1]})
    agg = df_row.groupby("utterance").agg(p1_mean=("p1","mean"), n_obs=("p1","size")).reset_index()
    p = np.clip(agg["p1_mean"].values, 1e-12, 1-1e-12)
    entropy = -(p*np.log(p) + (1-p)*np.log(1-p))
    ambiguity = 1.0 - 2.0*np.abs(p - 0.5)
    agg["entropy"] = entropy
    agg["ambiguity"] = ambiguity
    # Save detail
    base_detail = os.path.join(out_dir, "tables", "table15_amb_entropy_detail")
    try:
        save_table_csv_tex(agg, base_detail)
    except Exception:
        pass
    try:
        agg.to_csv(base_detail + ".csv", index=False)
    except Exception:
        pass
    # Summary
    pear = float(np.corrcoef(entropy, ambiguity)[0,1]) if len(agg)>1 else np.nan
    def _spearman(x, y):
        # rank transform + Pearson
        x = np.asarray(x); y = np.asarray(y)
        if len(x) < 2 or len(x) != len(y): return np.nan
        rx = pd.Series(x).rank(method="average").values
        ry = pd.Series(y).rank(method="average").values
        rx = (rx - rx.mean()) / (rx.std() + 1e-12)
        ry = (ry - ry.mean()) / (ry.std() + 1e-12)
        return float(np.mean(rx * ry))
    spear_all = _spearman(entropy, ambiguity)
    # Per-context Spearman
    ctx_col = None
    for c in ["gameid","context_id","conversation","conv_id"]:
        if c in df.columns:
            ctx_col = c; break
    rhos = []
    if ctx_col is not None:
        mp = {u:(e,a) for u,e,a in zip(agg["utterance"], agg["entropy"], agg["ambiguity"])}
        for ctx, us in df.groupby(ctx_col)["utterance"]:
            uniq = list(dict.fromkeys(us))
            xs=[]; ys=[]
            for u in uniq:
                v = mp.get(u, None)
                if v is not None: xs.append(v[0]); ys.append(v[1])
            if len(xs) >= 3:
                rhos.append(_spearman(np.array(xs), np.array(ys)))
    summary = pd.DataFrame([{
        "pearson_all": pear,
        "spearman_all": spear_all,
        "n_points": int(len(agg)),
        "mean_spearman_per_context": float(np.nanmean(rhos)) if len(rhos)>0 else np.nan,
        "n_contexts": int(len(rhos)),
        "has_context_col": bool(ctx_col is not None),
    }])
    base_summary = os.path.join(out_dir, "tables", "table15_amb_entropy_summary")
    try:
        save_table_csv_tex(summary, base_summary)
    except Exception:
        pass
    try:
        summary.to_csv(base_summary + ".csv", index=False)
    except Exception:
        pass
    # Figures
    fig_dir = os.path.join(out_dir, "figures"); ensure_dir(fig_dir)
    plt.figure(figsize=(6.5, 5.0))
    plt.scatter(agg["entropy"].values, agg["ambiguity"].values, s=8, alpha=0.4)
    plt.xlabel("Entropy (binary)"); plt.ylabel("Ambiguity (1 - 2|p-0.5|)")
    plt.title(f"Ambiguity vs Entropy  (Pearson={pear:.3f}, Spearman={spear_all:.3f})")
    pdf_path = os.path.join(fig_dir, "fig15a_amb_entropy_scatter.pdf")
    png_path = os.path.join(fig_dir, "fig15a_amb_entropy_scatter.png")
    plt.tight_layout(); plt.savefig(pdf_path); plt.savefig(png_path, dpi=200); plt.close()
    if len(rhos) > 0:
        plt.figure(figsize=(7.0, 4.2))
        plt.hist([r for r in rhos if np.isfinite(r)], bins=20, alpha=0.85)
        plt.xlabel("Spearman ρ (per-context rank agreement)"); plt.ylabel("Count")
        plt.title("Ambiguity vs Entropy: per-context rank agreement")
        pdf_path2 = os.path.join(fig_dir, "fig15b_amb_entropy_spearman_hist.pdf")
        png_path2 = os.path.join(fig_dir, "fig15b_amb_entropy_spearman_hist.png")
        plt.tight_layout(); plt.savefig(pdf_path2); plt.savefig(png_path2, dpi=200); plt.close()
    log("Sim-AMB-ENTROPY: done")

SIM_REGISTRY = {
    "Sim-AMB-ENTROPY": {"needs_data": True, "runner": lambda df, out, cores, seed: run_sim_amb_entropy(df, out, cores, seed)},
    "Sim-LADDER-DLL": {"needs_data": True, "runner": lambda df, out, cores, seed: run_sim_ladder_dll(df, out, cores, seed)},
    "Sim-A0": {"needs_data": False, "runner": lambda df, out, cores, seed: run_sim_a0(out)},
    "Sim-LADDER": {"needs_data": True, "runner": lambda df, out, cores, seed: run_sim_ladder(df, out, cores, seed)},
    "Sim-CIC-MAIN": {"needs_data": True, "runner": lambda df, out, cores, seed: run_sim_cic_main(df, out, cores, seed)},
    "Sim-CIC-FORM-ROBUST": {"needs_data": True, "runner": lambda df, out, cores, seed: run_sim_form_robust(df, out, cores, seed)},
    "Sim-A1-NEARSIGHTED": {"needs_data": True, "runner": lambda df, out, cores, seed: run_sim_a1_nearsighted(df, out, cores, seed)},
    "Sim-A2-EQUAL-ENTROPY": {"needs_data": True, "runner": lambda df, out, cores, seed: run_sim_a2_equal_entropy(df, out, cores, seed)},
    "Sim-A3-STATE-INDEP-COST": {"needs_data": True, "runner": lambda df, out, cores, seed: run_sim_a3_state_indep_cost(df, out, cores, seed)},
    "Sim-A4-LOCAL-TEMP": {"needs_data": True, "runner": lambda df, out, cores, seed: run_sim_a4_local_temp(df, out, cores, seed)},
    "Sim-A5-SEMANTICS-SHARE": {"needs_data": True, "runner": lambda df, out, cores, seed: run_sim_a5_semantics_share(df, out, cores, seed)},
    "Sim-A6-REGULARITY": {"needs_data": True, "runner": lambda df, out, cores, seed: run_sim_a6_regularity(df, out, cores, seed)},
    "Sim-FACTOR-SCREEN": {"needs_data": True, "runner": lambda df, out, cores, seed: run_sim_factor_screen(df, out, cores, seed)},
    "Sim-PARAM-RECOVERY": {"needs_data": True, "runner": lambda df, out, cores, seed: run_sim_param_recovery(df, out, cores, seed)},
    "Sim-MODEL-RECOVERY": {"needs_data": True, "runner": lambda df, out, cores, seed: run_sim_model_recovery(df, out, cores, seed)},
    "Sim-AIP-TWO-STAGE": {"needs_data": True, "runner": lambda df, out, cores, seed: run_sim_aip_two_stage(df, out, cores, seed)},
}

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AIP Simulation Runner (CiC)")
    p.add_argument("--input", required=False, help="Path to CiC filteredCorpus.csv")
    p.add_argument("--output", required=True, help="Output directory (will create subfolders)")
    p.add_argument("--cores", type=int, default=1, help="Number of parallel workers")
    p.add_argument("--sim", required=True, help="Simulation name (see registry) or 'ALL'")
    p.add_argument("--seeds", nargs=2, type=int, default=[0, 0], help="Seed range inclusive, e.g., 0 29")
    return p.parse_args(argv)

def load_cic_csv(path: str) -> pd.DataFrame:
    log(f"Loading dataset: {path}")
    df = pd.read_csv(path)
    log(f"Loaded rows: {len(df):,}")
    for c in ["clkTime", "msgTime"]:
        if c in df.columns:
            try:
                df[c] = pd.to_datetime(df[c], errors="coerce")
            except Exception:
                pass
    return df

def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)
    out_dir = args.output
    ensure_dir(out_dir); ensure_dir(os.path.join(out_dir, "figures"))
    ensure_dir(os.path.join(out_dir, "tables")); ensure_dir(os.path.join(out_dir, "logs"))

    run_cfg = {"argv": sys.argv, "output": out_dir, "sim": args.sim, "cores": args.cores, "seeds": args.seeds, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    log(f"Run config: {json.dumps(run_cfg)}")

    if args.sim.upper() == "ALL":
        sim_names = list(SIM_REGISTRY.keys())
    else:
        sim_names = [args.sim]

    df = None
    if any(SIM_REGISTRY[name]["needs_data"] for name in sim_names):
        if not args.input:
            log("ERROR: --input is required for the selected simulation(s)."); sys.exit(2)
        df = load_cic_csv(args.input)

    seed_start, seed_end = args.seeds
    seeds = list(range(int(seed_start), int(seed_end) + 1))

    for sim_name in sim_names:
        if sim_name not in SIM_REGISTRY:
            log(f"ERROR: Unknown simulation '{sim_name}'. Available: {list(SIM_REGISTRY.keys())}"); sys.exit(2)
        runner = SIM_REGISTRY[sim_name]["runner"]
        for sd in seeds:
            log(f"=== Running {sim_name} (seed={sd}) ===")
            np.random.seed(sd)
            try:
                runner(df, out_dir, args.cores, sd)
                log_dir = os.path.join(out_dir, "logs"); ensure_dir(log_dir)
                with open(os.path.join(log_dir, f"{sim_name}_seed{sd:02d}.jsonl"), "a", encoding="utf-8") as f:
                    f.write(json.dumps({"sim": sim_name, "seed": sd, "time": time.strftime("%Y-%m-%d %H:%M:%S")}, ensure_ascii=False) + "\n")
            except Exception as e:
                log_dir = os.path.join(out_dir, "logs"); ensure_dir(log_dir)
                err_path = os.path.join(log_dir, f"{sim_name}_seed{sd:02d}_error.txt")
                with open(err_path, "w", encoding="utf-8") as f:
                    f.write(str(e))
                log(f"ERROR in {sim_name} (seed={sd}): {e}")
            log(f"=== Finished {sim_name} (seed={sd}) ===")

    log("All requested simulations completed.")

if __name__ == "__main__":
    main()
