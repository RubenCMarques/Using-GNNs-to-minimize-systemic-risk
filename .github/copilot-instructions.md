# Copilot Instructions

## Repository structure (what happens where)
- `src/`: core thesis code (data loading, contagion simulation, classical features, embedding pipelines, ML training helpers).
- `src/data/`: data loading helpers plus simulation table builders used to normalize contagion runs.
- `src/models/`: contagion model, embedding pipelines, classical algorithms, and ML training utilities.
- `src/notebooks/`: exploratory and analysis notebooks (not part of test runs).
- `tests/`: pytest tests for loader/contagion/centrality modules.
- `HFTCRNet/`: separate model implementation with its own `src/`, data, and training scripts.
- `outputs/`: generated artifacts (parquet/csv experiment outputs, plots, etc.).

## Commands
- Run all tests: `pytest`
- Run a single test: `pytest tests/test_loader.py::test_load_data`
- HFTCRNet subproject (separate codebase): `cd HFTCRNet/src && python generate_contagionlist.py`, then configure `HFTCRNet/src/run.sh` and run `bash run.sh`

## High-level architecture
- **Quarterly dataset loading** lives in `src/data/data_loader.py`, which reads `src/datasets/edges/edge_{year}Q{quarter}.csv` and `src/datasets/nodes/{year}Q{quarter}.csv` into pandas frames used across tests, simulations, and embedding pipelines.
- **Contagion simulations** are in `src/models/contagion.py`. `simulate_failure` models cascade dynamics; `run_default_contagion_analysis` runs one shock per bank, then `src/data/simulation_store.py` normalizes results into `runs`, `bank_state`, and optional `round_summary` tables and writes per-quarter outputs (parquet + target CSVs).
- **Classical centrality features** are computed in `src/models/classical_algorithms.py` (degree, betweenness, closeness, eigenvector, weighted degree, DebtRank, PageRank) and are later merged into classical ML datasets.
- **Embedding pipelines** are split between `src/models/embeddings.py` and `src/models/embedding_pipeline.py`. GraphSAGE/Node2Vec embeddings are trained per quarter and warm-started across time to keep a stable embedding space; pooled datasets merge embeddings with per-quarter targets from `src/datasets/targets/target_{year}Q{quarter}.csv`.
- **Fixed embeddings** in `src/models/fix_embeddings.py` replace link-prediction-only training with feature reconstruction (GraphSAGE) or enriched structural embeddings (Node2Vec) while keeping the same warm-started, quarter-by-quarter pipeline.
- **ML training utilities** in `src/models/ml_train_and_store.py` load classical/embedding datasets, apply a time-based split by quarter, and train regression pipelines with shared evaluation metrics.

## Key conventions
- **Quarter naming** is consistent across the repo: `YYYYQ#` is used for filenames and the `period` column (e.g., `2021Q3`).
- **Core edge/node schemas**: edges require `Sourceid`, `Targetid`, `Weights`; nodes must include `index` and `Equity`, and may include `Total_assets` or `Assets` for asset-loss tracking.
- **Feature selection for embeddings**: target-like columns (`systemic_risk_label`, `srisk_ratio`, `srisk_value`, etc.) are excluded from node features in embedding pipelines.
- **Warm-starting is expected** for embeddings: pass the previous quarter’s `state_dict` when moving to the next quarter to keep temporal coherence.
