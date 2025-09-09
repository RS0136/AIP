# AIP Simulation Suite — README

This repository provides a single command-line entry point, **`aip.py`**, that runs the simulations used in the AIP (Ambiguity-Informativeness Policy) study and produces all tables/figures referenced in the preprint.

> **Important numbering note**  
> The numeric suffixes in output filenames (e.g., `fig02_...`, `table06_...`) are **engineering labels** for automation.  
> **They may not match the final figure/table numbering in the preprint**, which is assigned by **appearance order**.  
> When writing the paper, always cite figures/tables using the **preprint’s numbering**, not the file suffixes.

---

## 1. Quick start

```bash
# Run all simulations (recommended after verifying Python deps)
python aip.py --input path/to/filteredCorpus.csv --output ./output --cores 4 --sim ALL

# Run an individual simulation only
python aip.py --input path/to/filteredCorpus.csv --output ./output --cores 4 --sim Sim-LADDER
```

**Required arguments**
- `--input`  : Path to the dataset (e.g., CiC’s `filteredCorpus.csv`).
- `--output` : Output directory. Subfolders `tables/`, `figures/`, and `logs/` are created here.
- `--cores`  : Parallel workers. Use `1` if unsure.
- `--sim`    : Which simulation to run (see §3 for keys). Use `ALL` to run everything in the registry.

**Optional**
- `--seed`   : Random seed for simulations that use stochastic components.

**Output layout**
```
<output>/
  tables/   # CSV (and sometimes TeX) of metrics
  figures/  # PDF and PNG plots
  logs/     # JSONL with run metadata (only for some sims)
```

---

## 2. What the suite produces

The suite covers both **main-text simulations** (demonstrating the AIP assumptions and incremental ablations) and **appendix simulations** (robustness, diagnostics). Each simulation writes **one or more CSV tables** and **one or more figures**. Filenames are stable across reruns, making it easy to track outputs programmatically.

Key metrics include **accuracy (`acc`)**, **log-likelihood (`ll`)**, and **Bayesian Information Criterion (`bic`)**. Some tables also include **confidence intervals** (`acc_lo`, `acc_hi`), sample sizes (`n`), and experiment-specific columns (e.g., `drift`, `amp`).

---

## 3. Simulation keys (CLI values for `--sim`)

### Core (main text)
- **`Sim-LADDER`** — Stepwise ablation from A1 to A6 (global picture).  
  _Typical outputs_: `tables/table02_ladder.csv`, `figures/fig02_ladder.pdf`

- **`Sim-LADDER-DLL`** — _Per-example_ **ΔLL** between adjacent LADDER rungs.  
  _Typical outputs_: `tables/table02b_ladder_dll_per_example.csv`, `figures/fig02b_ladder_dll_per_example.pdf`

- **`Sim-A1-NEARSIGHTED`** — Nearsightedness (softmax/Gibbs) with temperature / vocabulary-cap sweeps.  
  _Typical outputs_: `tables/table03_nearsighted.csv`, `figures/fig03_nearsighted.pdf`

- **`Sim-A2-EQUAL-ENTROPY`** — Equal-entropy over utterances to stabilize ambiguity scaling; on/off comparisons.  
  _Typical outputs_: `tables/table04_equal_entropy.csv`, `figures/fig04_equal_entropy.pdf`

- **`Sim-AMB-ENTROPY`** — Ambiguity–Entropy sanity check using out-of-fold listener posteriors.  
  _Typical outputs_:  
  `tables/table15_amb_entropy_detail.csv` (utterance-level),  
  `tables/table15_amb_entropy_summary.csv` (global/per-context correlations),  
  `figures/fig15a_amb_entropy_scatter.pdf`, `figures/fig15b_amb_entropy_spearman_hist.pdf` (the latter only if a context id is present)

- **`Sim-A3-STATE-INDEP-COST`** — State-independent cost vs. state-dependent “leakage” amplitude.  
  _Typical outputs_: `tables/table05_state_indep_cost.csv`, `figures/fig05_state_indep_cost.pdf`

- **`Sim-A4-LOCAL-TEMP`** — Locally shared temperature (contrast-local β) vs. constant/drifting baselines.  
  _Typical outputs_: `tables/table06_local_temp.csv`, `figures/fig06_local_temp.pdf`  
  _Note_: the table contains a human-readable **`label`** column (e.g., `const β=1.0`, `linear 1.5→1.0`, `random walk σ=0.2`).

### Appendix / robustness (typical)
- **`Sim-A5-SEMANTICS-SHARE`** — Share semantics between L0 and the speaker (+optional calibration).  
  _Outputs_: `table07_semantics_share.csv`, `fig07_semantics_share.pdf`

- **`Sim-A6-REGULARITY`** — Regularity/normalizability constraints.  
  _Outputs_: `table08_regularity.csv`, `fig08_regularity.pdf`

- **`Sim-FACTOR-SCREEN`** — Factorial screening across A1–A6 and related options.  
  _Outputs_: `table09_factor_screen.csv`, `fig09_factor_screen.pdf`

- **`Sim-PARAM-RECOVERY`** — Parameter recovery diagnostics (true vs. estimated).  
  _Outputs_: `table10_param_recovery.csv`, `fig10_param_recovery.pdf`

- **`Sim-MODEL-RECOVERY`** — Model recovery; confusion matrix over generators vs. fitters.  
  _Outputs_: `table11_model_recovery.csv`, `fig11_model_recovery.pdf`  
  _Note_: **fig11 includes a colorbar legend on the right** (“count (best‑BIC picks)”).

- **`Sim-CIC-MAIN`** — Full CiC analysis with cross-validation.  
  _Outputs_: `table12_cic_main.csv`, `fig12_cic_main.pdf`

- **`Sim-CIC-FORM-ROBUST`** — Robustness to utterance length and simple compositional cues.  
  _Outputs_: `table13_form_robust.csv`, `fig13_form_robust.pdf`

- **`Sim-AIP-TWO-STAGE`** — Confirm‑then‑commit AIP two‑stage policy.  
  _Outputs_: `table14_two_stage.csv`, `fig14_two_stage.pdf`

> `ALL` runs every key present in the internal `SIM_REGISTRY`.

---

## 4. Interpreting the key results

- **LADDER & ΔLL per example**  
  LADDER provides the big picture (A1→A6). The companion ΔLL/example bars quantify how much **each adjacent rung** improves (or slightly degrades) fit per observation. It is normal to see strong gains up to **A1 + A2**, with diminishing or even slightly negative returns when adding more complex modules.

- **A1 (nearsightedness)**  
  Accuracy may look nearly flat across caps/temperatures, while **LL/BIC** shifts meaningfully. In reporting, prefer LL/BIC to argue calibration/fit.

- **A2 (equal entropy)**  
  Expect consistent **on > off** improvements (Δacc/ΔBIC) across splits. The **AMB–ENTROPY** sanity check supports why length/entropy normalization helps.

- **A3 (state‑independent cost)**  
  Injecting state‑dependent leakage should **harm fit** (accuracy drops; BIC increases).

- **A4 (local β)**  
  Local/shared β often yields similar accuracy but **better LL/BIC stability** than unconstrained drifts; random walk drift typically performs worst.

---

## 5. Reproducibility & seeds

- Many simulations are deterministic given `--seed`, but some perform internal multi-seed loops or CV splits.  
- The **AMB–ENTROPY** simulation uses **out-of-fold** probabilities to avoid optimistic bias.  
- Table files include all numbers used in the paper; you can cite them directly in text.

---

## 6. Performance tips

- If you only need a subset for drafting, run individual keys (e.g., `Sim-LADDER`, `Sim-A2-EQUAL-ENTROPY`).  
- Reduce `--cores` if you experience over-subscription on shared servers.  
- Figures are saved as PDF and PNG; keep PDFs for the paper and PNGs for quick previews.

---

## 7. Troubleshooting

- **Missing `table02_ladder.csv` when running `Sim-LADDER-DLL`**  
  Run `Sim-LADDER` first (or use a version that auto-runs it).

- **No `fig15b_...` histogram from `Sim-AMB-ENTROPY`**  
  The per‑context rank‑agreement plot requires a context identifier in the input (e.g., `gameid`, `context_id`). Without it, only the scatter is produced.

- **Inconsistent labels in A4**  
  The code writes a human-readable `label` column in `table06_local_temp.csv` and uses those labels in the figure to keep the paper text and plots aligned.

- **IndentationError after local edits**  
  Ensure any custom additions keep consistent indentation inside functions (especially around A4 label mapping and fig11 plotting). The shipped version passes a syntax check.

---

## 8. Versioning & change highlights

- Added **`Sim-LADDER-DLL`**: always saves `table02b_ladder_dll_per_example.csv` (and companion figure).  
- Added **`Sim-AMB-ENTROPY`** detail (`*_detail.csv`) and summary (`*_summary.csv`) tables.  
- A4 tables now include a **`label`** column; figures display these labels on the x‑axis.  
- **fig11** (Model Recovery) now includes a **right‑side colorbar legend**.

---

## 9. Citation

When citing outputs in the manuscript, **use the preprint’s figure/table numbering** (by appearance order). The file suffix numbers are for scripting only and may diverge after section re‑ordering.

If you cite this codebase, please reference the AIP preprint.

