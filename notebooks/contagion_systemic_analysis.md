# Contagion / Systemic Risk Analysis

Date: 2026-03-03

## 1) `attack.py` vs `systemic_risk_simulation.ipynb`

### Scope and objective

- `interbank-0.1.0/methods/credit_rating_under_attack/attack_methods/attack.py`
  - Purpose: generate **attacked/perturbed datasets** (`nodes{attack_rate}`, `edges{attack_rate}`) for robustness tests in credit-rating training.
  - It modifies graph structure and node features.
- `notebooks/systemic_risk_simulation.ipynb`
  - Purpose: run **contagion Monte Carlo scenarios** and produce systemic-risk labels (`systemically_important`) plus scenario logs/summaries.
  - It does not create adversarial dataset variants like `nodes0.25/edges0.25`.

### How the "attack/shock" is done

- In `attack.py`:
  - `edge_attack(...)` injects extra edges between selected rating groups.
  - `feature_attack(...)` rescales selected feature columns for sampled banks of specific ranks.
  - Output files are written to `datasets/edges{attack_rate}/...` and `datasets/nodes{attack_rate}/...`.
- In `systemic_risk_simulation.ipynb`:
  - Scenarios sample seed banks (`single` or `multi`) and shock magnitudes from `shock_range`.
  - Defaults propagate through two channels:
    - liquidity channel (loss vs liquidity buffer)
    - exposure channel (loss vs capital buffer)
  - Output is impact metrics + node labels, not modified raw datasets.

### Pipeline integration

- `attack.py` is consumed by the credit-rating pipeline:
  - `methods/utils.py` loads attacked files when `attack_rate != 0`.
  - `methods/train.py` trains GNNs using those attacked inputs.
- `systemic_risk_simulation.ipynb` is a standalone labeling/simulation workflow under `notebooks/`.

### Bottom line

- `attack.py` = **data poisoning/perturbation step** for model robustness.
- `systemic_risk_simulation.ipynb` = **financial contagion simulation engine** for systemic-risk scoring and labels.

---

## 2) `02_contagion_simulations.ipynb` vs `systemic_risk_simulation.ipynb`

### High-level purpose

- `notebooks/02_contagion_simulations.ipynb`
  - Large-scale simulation data generation.
  - Stores normalized tables:
    - Table 1: runs
    - Table 2: bank end-state sparse
    - Table 3: round summary (optional)
- `notebooks/systemic_risk_simulation.ipynb`
  - Label generation pipeline for systemic importance (`0/1`) with scenario logs and quarter summaries.

### Core engine

- Notebook 02 calls:
  - `src.models.simulate_failure(...)` from `src/models/contagion.py`.
- Systemic-risk notebook implements its own:
  - `run_scenario(...)` + helper functions inside the notebook itself.

### Default/cascade logic differences

- Notebook 02 (`simulate_failure`):
  - One propagation mode per run (`Exposure` or `liquidity`).
  - Bank fails when remaining equity `<= 0`.
  - Optional `spread_without_default=True`: contagion can start even if shocked bank did not default.
- Systemic-risk notebook (`run_scenario`):
  - Two-stage defaults per round:
    - Stage A: liquidity loss > liquidity buffer.
    - Stage B: exposure loss > capital buffer.
  - Uses `capital_ratio`, `liquidity_ratio`, `lgd`, iterative bookkeeping by default sets.

### Shock design

- Notebook 02:
  - `N_SIMULATIONS = 500_000` per quarter.
  - One initial bank sampled per run.
  - Shock fraction sampled from `[0.05, 0.60]`.
- Systemic-risk notebook:
  - `n_scenarios = 10_000` per quarter.
  - Mix of `single` and `multi` seed scenarios.
  - Shock range `[0.20, 0.60]`.
  - Includes backfill scenarios for nodes never selected as seeds.

### Temporal coverage

- Notebook 02:
  - Iterates 2016Q1 to 2023Q4 (32 quarters).
- Systemic-risk notebook:
  - Configured for 2016Q1 to 2023Q1 (29 quarters).

### Outputs

- Notebook 02:
  - Writes Parquet tables under:
    - `src/data/sim_runs/`
    - `src/data/sim_bank_state/`
    - `src/data/sim_round_summary/` (if enabled)
- Systemic-risk notebook:
  - Writes to `outputs/`:
    - `systemic_labels/*_labels.csv`
    - `scenario_logs/*_scenarios.parquet` (or CSV fallback)
    - `summaries/*_summary.csv`

### Bottom line

- Notebook 02 = **simulation dataset generator** (rich run-level/bank-level audit trail).
- Systemic-risk notebook = **labeling workflow** (scenario impact aggregation -> trigger scores -> top-quantile systemic labels).
