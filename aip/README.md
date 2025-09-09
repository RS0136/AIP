# AIP Simulation Suite — README

This document explains how to run **`aip.py`** and how to interpret the outputs it produces for the AIP study.

> **Numbering disclaimer**  
> Numeric suffixes in file names (e.g., `fig02_*`, `table06_*`) are **engineering labels** for automation.  
> **They may not match the final figure/table numbering in the preprint**, which is assigned by **appearance order**.  
> Always cite figures/tables using the **preprint’s numbering**.

> **Figure styling policy**  
> Figures are saved **without titles**. Use **axis labels**, **legends/colorbars**, and the **paper caption** to convey context.

---

## 1. Quick start

```bash
# Run everything
python aip.py --input path/to/filteredCorpus.csv --output ./output --cores 4 --sim ALL

# Run a single simulation
python aip.py --input path/to/filteredCorpus.csv --output ./output --cores 4 --sim Sim-LADDER
```

**Arguments**
- `--input`  : Path to the dataset (e.g., CiC `filteredCorpus.csv`).
- `--output` : Output directory root (subfolders `tables/`, `figures/`, `logs/` are created).
- `--cores`  : Parallel workers (use `1` if unsure).
- `--sim`    : Simulation key (see §3); `ALL` runs everything registered.
- `--seed`   : Random seed (optional).

**Output layout**
```
<output>/
  tables/   # CSV (and sometimes TeX) with metrics
  figures/  # PDF and PNG plots (no titles)
  logs/     # JSONL logs for select simulations
```

---

## 2. Simulation keys

### Core (main text)

- **`Sim-LADDER`** — Stepwise ablation from A1 to A6.  
  _Typical outputs_: `tables/table02_ladder.csv`, `figures/fig02_ladder.pdf`

- **`Sim-LADDER-DLL`** — Per‑example **ΔLL** between adjacent LADDER rungs.  
  _Typical outputs_: `tables/table02b_ladder_dll_per_example.csv`, `figures/fig02b_ladder_dll_per_example.pdf`

- **`Sim-A1-NEARSIGHTED`** — Nearsightedness under temperature/vocabulary‑cap sweeps.  
  _Typical outputs_: `tables/table03_nearsighted.csv`, `figures/fig03_nearsighted.pdf`

- **`Sim-A2-EQUAL-ENTROPY`** — Equal‑entropy over utterances; on/off comparisons.  
  _Typical outputs_: `tables/table04_equal_entropy.csv`, `figures/fig04_equal_entropy.pdf`

- **`Sim-AMB-ENTROPY`** — Ambiguity–Entropy sanity check using out‑of‑fold listener posteriors.  
  _Typical outputs_:  
  `tables/table15_amb_entropy_detail.csv` (utterance‑level),  
  `tables/table15_amb_entropy_summary.csv` (global/per‑context correlation stats),  
  `figures/fig15a_amb_entropy_scatter.pdf`, `figures/fig15b_amb_entropy_spearman_hist.pdf` (the latter only if a context id is available)

- **`Sim-A3-STATE-INDEP-COST`** — State‑independent cost vs. state‑dependent leakage amplitude.  
  _Typical outputs_: `tables/table05_state_indep_cost.csv`, `figures/fig05_state_indep_cost.pdf`

- **`Sim-A4-LOCAL-TEMP`** — Locally shared temperature (contrast‑local β) vs. constant/drifting baselines.  
  _Typical outputs_: `tables/table06_local_temp.csv`, `figures/fig06_local_temp.pdf`  
  _Note_: the table includes a human‑readable **`label`** column (e.g., `const β=1.0`, `linear 1.5→1.0`, `random walk σ=0.2`).

### Appendix / robustness (typical)

- **`Sim-A5-SEMANTICS-SHARE`** — L0–speaker semantic sharing (+ calibration).  
- **`Sim-A6-REGULARITY`** — Regularity/normalizability constraints.  
- **`Sim-FACTOR-SCREEN`** — Factorial screening across A1–A6.  
- **`Sim-PARAM-RECOVERY`** — Parameter recovery (true vs. estimated).  
- **`Sim-MODEL-RECOVERY`** — Model recovery (confusion counts of generators vs. fitters).  
  _Note_: **fig11 includes a right‑side colorbar legend** (“count (best‑BIC picks)”).  
- **`Sim-CIC-MAIN`** — Full CiC analysis with cross‑validation.  
- **`Sim-CIC-FORM-ROBUST`** — Robustness to utterance length and form cues.  
- **`Sim-AIP-TWO-STAGE`** — Confirm‑then‑commit two‑stage policy.

> `ALL` runs every key present in the internal `SIM_REGISTRY`.

---

## 3. What changed in this build (requested fixes)

1. **LADDER‑DLL table output is guaranteed**  
   `Sim-LADDER-DLL` always writes `tables/table02b_ladder_dll_per_example.csv` (along with the figure).

2. **A4 table includes a `label` column**  
   `table06_local_temp.csv` contains a human‑readable `label` column and the figure uses these labels on the x‑axis.

3. **AMB–ENTROPY writes both detail and summary tables**  
   `Sim-AMB-ENTROPY` produces:  
   - `table15_amb_entropy_detail.csv` — utterance‑level (`utterance, n_obs, p1_mean, entropy, ambiguity`)  
   - `table15_amb_entropy_summary.csv` — global Pearson/Spearman and per‑context statistics

Additionally, **all figures are saved without titles** (axis labels/legends only).

---

## 4. Interpreting common metrics

- **acc** — accuracy on held‑out or CV splits.  
- **ll** — log‑likelihood; prefer this for calibration/fit claims when accuracy is flat.  
- **bic** — Bayesian Information Criterion (lower is better).  
- Confidence intervals (when present): `acc_lo`, `acc_hi`. Sample size: `n`.

Typical patterns to expect:  
- **LADDER** — strong improvement up to **A1 + A2**, then diminishing or slightly negative ΔLL/example.  
- **A2** — “on” consistently beats “off” in Δacc and ΔBIC across splits.  
- **A3** — increasing state‑dependent leakage harms acc and BIC.  
- **A4** — local/shared β stabilizes fit vs. random‑walk drifts.

---

## 5. Troubleshooting

- **Missing `table02_ladder.csv` when running `Sim-LADDER-DLL`** → run `Sim-LADDER` first (or re‑run `ALL`).  
- **No `fig15b_*` from `Sim-AMB-ENTROPY`** → requires a context id (e.g., `gameid`, `context_id`) in the input.  
- **Syntax errors after local edits** → keep indentation consistent inside function bodies; the released file passes a syntax check.

---

## 6. Citation

In the manuscript, always cite figures/tables by the **preprint’s numbering** (appearance order).  
File suffix numbers are for automation and may diverge after section re‑ordering.
