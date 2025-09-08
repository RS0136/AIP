# AIP Simulation Suite (AIF ⇔ RSA)

This repository contains a fully reproducible simulation pipeline that generates all figures and paired LaTeX tables for the AIP preprint. The code simulates a referential color game and demonstrates an affine equivalence between Expected Free Energy (AIF/EFE) and pragmatic utility in Rational Speech Acts (RSA) under assumptions A1–A6.

- **One command** produces **14 figures** (`PNG`) and **14 tables** (`LaTeX`) plus CSV data.
- **Deterministic parallelism**: independent per-seed RNG + result collation sorted by seed.
- **No seaborn**, **no color styling**, **one plot per figure** to keep artifacts simple and journal-friendly.

---

## Contents

- `aip.py` — main simulation script (CLI)
- `output/` — generated artifacts (created at runtime)
  - `figures/fig01.png` … `fig14.png`
  - `tables/fig01_table.tex` … `fig14_table.tex`
  - `data/*.csv`, `run_config.json`
- (optional) `input/filteredCorpus.csv` — used only to **shape** empirical distributions (utterance lengths, contrast magnitudes, condition frequencies). If missing, a small synthetic corpus is used so the script always runs.

---

## Requirements

- **Python** ≥ 3.9
- **Packages**
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `joblib`
- (Optional) `latexmk` if you plan to compile a LaTeX preprint that `\input`s the generated tables

Install:
```bash
pip install numpy pandas matplotlib joblib
