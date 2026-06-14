# Code Structure Guide

> Temporary guide to the whole project layout, the methodology, and how the pieces fit
> together. (Separate from `README.md` / `src/notebooks/README.md`.)

## 1. What the project does

Predict the **systemic risk** of banks in interbank networks and compare three families of
predictors:

1. **Classical centrality** — hand-crafted network measures (degree, betweenness, PageRank, DebtRank, …).
2. **Feature-based embeddings** — GraphSAGE (uses node financial features **+** structure).
3. **Network-structure embeddings** — Node2Vec (structure only, no node features).

Plus an **ensemble/stacking** that combines centrality + embeddings, and a sensitivity analysis
over a **default-probability threshold `p`**.

The target is `systemic_risk_label` = the cascade size produced by a contagion simulation
(and `log_systemic_risk_label = log1p(...)`, the column actually used for modeling).

## 2. Two analytical dimensions

- **Dataset dimension** — three networks (below).
- **Threshold dimension** — `p` ∈ {0, 5, 10, …, 40}% (`p0 … p40`). `p` makes banks fail more
  easily, so cascade sizes (the target) grow with `p`.
  - Datasets 1 & 2 (equity model): `p` shrinks each bank's equity buffer → `Equity = (1 - p)·Equity`.
  - Dataset 3 (ER cascade model): `p` shrinks the failure buffer → `Ki = (1 - p)·0.04`.

## 3. Datasets

| id | network | type | nodes | notes |
|----|---------|------|-------|-------|
| **dataset_1** | AI4Risk interbank | temporal (2016Q1–2023Q4, 32 quarters) | 4 548/quarter | real-ish; **temporal** train/val/test split |
| **dataset_2** | synthetic European | static | 1 444 | **random 70/15/15** stratified split |
| **dataset_3** | Erdős–Rényi (avg degree 0.8) | static | 10 000 | structure-only; targets from a default-cascade; **random 70/15/15** |

Raw data: `src/datasets/dataset_{1,2,3}/`. The retired intercounty-migration network is parked in
`src/datasets/_archive_migration/`.

## 4. Repository layout (high level)

```
src/
├── datasets/dataset_{1,2,3}/          raw network data + targets (+ contagion sims for d1)
│   ├── edges/ nodes/ targets/         (d1: per-quarter; d2: network.xlsx+nodes.csv; d3: edges.csv)
│   └── targets/  target.csv (=p0) + target_p5..p40.csv
├── data/
│   ├── classical_features/dataset_{1,2,3}/      centrality feature parquets
│   ├── embeddings/dataset_{1,2}/{feature_based,network_based}/   GraphSAGE / Node2Vec parquets
│   │   └── dataset_3/network_based/              Node2Vec only
│   ├── predictions/                              model outputs, ensemble outputs, comparison figs
│   ├── data_loader.py  simulation_store.py
├── models/
│   ├── classical_algorithms.py   centrality measures
│   ├── contagion.py              contagion simulation + target table
│   ├── embeddings.py             GraphSAGE (link-pred) + Node2Vec configs/training
│   ├── fix_embeddings.py         GraphSAGE reconstruction-objective variant
│   ├── embedding_pipeline.py     quarter-by-quarter pooled embedding datasets (d1)
│   ├── ml_train_and_store.py     ModelTrainer, splits, metrics, load_gnn_dataset, load_model
│   └── dataset_{1,2}/{04_a,05_a,05_b}/   saved best models (joblib)
└── notebooks/                     the pipeline (see §6)
```

## 5. Core shared code (`src/models`, `src/data`)

- **`classical_algorithms.py`** — `degree_centrality`, `betweenness_centrality`, `closeness_centrality`,
  `eigenvector_centrality`, `weighted_degree`, `pagerank_centrality`, `debtrank`. All take
  `(edges, nodes)`, build a directed graph, and return per-bank values (isolated banks → 0).
- **`contagion.py`** — `simulate_failure`, `run_default_contagion_analysis*`,
  `build_systemic_importance_summary`, `build_target_table`. `systemic_risk_label = cascade_size`.
- **`embeddings.py`** — `GNNConfig`, `Node2VecConfig`, `prepare_graph_data`, `extract_embeddings`
  (dispatches GraphSAGE link-prediction vs Node2Vec random walks).
- **`fix_embeddings.py`** — `FixedGNNConfig`, `extract_fixed_embeddings` (GraphSAGE with a
  feature-reconstruction loss; also dispatches Node2Vec).
- **`embedding_pipeline.py`** — `build_pooled_dataset` / `build_fixed_pooled_dataset` for the
  **temporal** Dataset 1 (warm-started quarter by quarter).
- **`ml_train_and_store.py`** — the modeling backbone:
  - `ModelTrainer(split="temporal" | "random")` — `random` = stratified **70/15/15** (`random_split`),
    used for the static datasets; `temporal` = `time_split`, used for Dataset 1.
  - `load_classical_dataset`, `load_gnn_dataset` (routes a filename to
    `embeddings/dataset_{1,2,3}/{feature_based,network_based}/`), `make_pipeline`, `load_model`,
    `top1_bank_cohort`, `top1_metrics`, `CLASSICAL_FEATURE_CANDIDATES`.
- **`data/data_loader.py`** — `load_data(year, quarter)` for Dataset 1 (defaults to `dataset_1`).

## 6. Pipeline by stage (`src/notebooks`)

Organized **stage → dataset**. Notebooks auto-discover the project root, so paths resolve from any depth.

### 00_notes — exploration / utilities
- `bankruptcy_cascade_python_replica.ipynb` — exact Python port of the R contagion model
  (`financial-contagion-in-R/`), validated node-for-node. Fast dict/array version.
- `bankruptcy_cascade_networkx.ipynb` — same model via networkx (readable, slower).
- `bankruptcy_cascade_p_threshold_sweep.ipynb` / `cascade_ki_sweep.ipynb` — `p`/`Ki` sweeps used to
  build Dataset 3's targets.

### 01_exploration — preprocessing / centrality features
`dataset_{1,2,3}/01_preprocessing.ipynb` → writes `classical_features/dataset_X/`.
(Dataset 3 skips **DebtRank**: it has no equity.)

### 02_contagion — targets
`dataset_{1,2,3}/02_contagion.ipynb` → contagion simulation → `targets/target.csv` (p0) +
`target_p5..p40.csv`. (Dataset 3's `02_contagion` is the ER default-cascade `Ki(p)` sweep.)

### 03_embeddings
- `dataset_1/` — `03_embeddings_v1` (link-pred GraphSAGE + Node2Vec), `03_embeddings_v2`
  (reconstruction GraphSAGE + Node2Vec q=2), `03_embeddings_split128` (128-dim → 128a/128b for space),
  `03_feature_embeddings_p_thresholds` (best feature embedding × p-targets).
- `dataset_2/` — `03_embeddings_v1`, `03_embeddings_v2` (GraphSAGE + Node2Vec at 32/64/128).
- `dataset_3/` — `03_embeddings_v1` (Node2Vec q=1), `03_embeddings_v2` (Node2Vec q=2). **No GraphSAGE**
  (no node features).

  > **Embedding versions:** `v1` = link-prediction objective, `v2` = reconstruction objective —
  > this distinction only affects **GraphSAGE**. Node2Vec is structure-only, so v1/v2 differ only by
  > its `q` (1 = unbiased, 2 = structural/BFS bias).

### 04_classical_ml — centrality models
- `dataset_X/04_centrality_selection_p0.ipynb` — **p0 model selection**: the full candidate set
  (Linear, Ridge, MLP, RF, GBM, XGB) + `RandomizedSearchCV` tuning; saves best to `models/dataset_X/04_a`.
- `dataset_{2,3}/04_centrality_threshold.ipynb` — apply the selected best model across `p0…p40`
  (clone + refit per threshold) → `predictions/dataset_X/centrality_threshold/`.
- `combined/04b_*`, `04c_*` — older Dataset 1+2 threshold notebooks (kept; superseded by per-dataset).

### 05_embedding_ml — embedding models
- `dataset_1/05_embedding_selection_p0_v1.ipynb`, `_v2.ipynb` — selection per embedding version.
- `dataset_{2,3}/05_embedding_selection_p0.ipynb` — selection across all embedding candidates
  (d2: GraphSAGE+Node2Vec v1/v2 × 32/64/128; d3: Node2Vec v1/v2 × 32/64/128); saves best per
  embedding to `models/dataset_X/05_a/<embedding>/`.
- `dataset_{2,3}/05_embedding_threshold.ipynb` — apply selected best across `p` →
  `predictions/dataset_X/embedding_threshold/`.
- `combined/05d_*`, `05e_*` — older Dataset 1+2 threshold notebooks.

### 06_ensemble — stacking (per dataset)
`dataset_{1,2,3}/06_ensemble.ipynb` — combine centrality + embedding predictions per `p`:
**average** ensemble and a **stacked** meta-learner (LinearRegression on the two base predictions,
trained on validation, evaluated on test). Outputs → `predictions/dataset_X/ensemble/`.
(Dataset 1's reads the combined `04b`/`05d` predictions; d2/d3 read their per-dataset threshold predictions.)

### 07_analysis — cross-approach analysis
`07_Embedding_Model_Comparison.ipynb`, `07_target_dataset_analysis.ipynb`.

### 08_comparison — the headline comparison
`08_p_threshold_comparison_centrality_vs_embeddings.ipynb` — unified test-error comparison of the
**four approaches** (Centrality, Embeddings, Average, Stacked) across `p`, for **all three datasets**:
metric line panels, grouped-bin bars, all-metric line charts, and summary tables. Figures →
`predictions/07_p_comparison/`.

### utils
`split_large_predictions.ipynb` — splits oversized `predictions.csv` into `predictions_parts/`
(≤45 MB) so they fit under GitHub's 100 MB limit; the full CSVs are git-ignored.

## 7. End-to-end methodology

For each dataset: **01** features → **02** targets (per `p`) → **03** embeddings →
**04/05** p0 model selection (pick best centrality model + best embedding+model) →
**04/05 threshold** (apply the selected best across `p5…p40`) → **06** stacking →
**08** comparison.

Splits: Dataset 1 = **temporal**; Datasets 2 & 3 = **stratified random 70/15/15** (same split used
by centrality and embeddings, so per-bank stacking is valid).

## 8. Key results (current run)

- **Dataset 2:** centrality ≈ embeddings; combining gives a small lift.
- **Datasets 1 & 3:** centrality clearly beats embeddings; the **stacked** meta-learner recovers
  centrality-level performance by down-weighting the weak embedding, while the **naive average hurts**.

## 9. Conventions / gotchas

- Targets: `systemic_risk_label` + `log_systemic_risk_label` (= `log1p`); modeling uses the log column.
- Embedding parquets store `bank_id + emb_* + log_systemic_risk_label`.
- 128-dim embeddings are stored as `128a`/`128b` row-halves (space); the full `*_128_*` file is intentionally absent.
- Large prediction CSVs are git-ignored and split into `predictions_parts/`.
- The R-replica notebooks read/write the (git-ignored) embedded repo `financial-contagion-in-R/`.
