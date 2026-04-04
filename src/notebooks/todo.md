# Simulation Notebooks — TODO

## `systemic_risk_simulation.ipynb` (priority notebook)

- [ ] Add missing quarters to `CONFIG["quarters"]`: `2023Q2`, `2023Q3`, `2023Q4`
- [ ] Set `RUN_FULL = True` to execute all 29+ quarters
- [ ] Verify outputs land in `outputs/systemic_labels/`, `outputs/scenario_logs/`, `outputs/summaries/`

## Stressed-network variants (replacing `attack.py`)

- [ ] Design a preprocessing step that perturbs **Equity** and/or edge **Weights** directly before simulation
  - Option A: Equity shock — reduce equity of a subset of banks by a factor (balance sheet stress)
  - Option B: Exposure amplification — scale edge `Weights` up by a factor (increased interconnectedness)
  - Option C: Network fragmentation — remove a fraction of edges (liquidity hoarding / market freeze)
- [ ] Do NOT use `attack.py` output — it drops edge `Weights`, breaking the contagion propagation

## `02_contagion_simulations.ipynb` (already executed)

- [ ] If labels are needed from this notebook's data, add a labeling step on top of the existing parquet outputs in `src/data/sim_runs/`
- [ ] Consider extending to multi-bank seeds (currently single-bank only)

## `03_systemic_importance_simulation.ipynb`

- [ ] Low priority — simpler model, superseded by `systemic_risk_simulation.ipynb`
- [ ] If attacked-data support is needed, fix `attack.py` first (injected edges are missing `Weights` column)
