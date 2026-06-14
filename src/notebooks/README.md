# Notebooks

Organised **by stage, then by dataset**. Notebooks discover the project root
automatically, so paths resolve regardless of the folder depth.

Datasets:
- **dataset_1** — AI4Risk interbank network, temporal (2016Q1–2023Q4, 32 quarters)
- **dataset_2** — synthetic European network, static (1 444 banks)
- **dataset_3** — ER synthetic network (avg degree 0.8), static; default-cascade p-threshold targets

Two embedding approaches (kept in separate folders under `src/data/embeddings/`):
- **feature_based** — GraphSAGE (uses node financial features + structure)
- **network_based** — Node2Vec (structure only, no node features)

Embedding versions: **v1** = link-prediction objective, **v2** = feature-reconstruction
objective. (v3 / no-log and the old base prototype were removed — each dataset uses v1 + v2 only.)

## 01 Exploration / preprocessing
- `01_exploration/dataset_1/01_preprocessing.ipynb`
- `01_exploration/dataset_2/01_preprocessing.ipynb`
- `01_exploration/dataset_3/01_preprocessing.ipynb` — ER centrality features (no DebtRank: no equity)

## 02 Contagion simulations
- `02_contagion/dataset_1/02_contagion.ipynb`
- `02_contagion/dataset_2/02_contagion.ipynb`
- `02_contagion/dataset_3/02_contagion.ipynb` — ER default-cascade p-threshold sweep

## 03 Embeddings
- `03_embeddings/dataset_1/03_embeddings_v1.ipynb` — GraphSAGE v1 + Node2Vec v1 (32/64/128)
- `03_embeddings/dataset_1/03_embeddings_v2.ipynb` — GraphSAGE v2 + Node2Vec v2 (32/64/128)
- `03_embeddings/dataset_1/03_embeddings_split128.ipynb` — split 128-dim files into 128a/128b
- `03_embeddings/dataset_1/03_feature_embeddings_p_thresholds.ipynb` — best feature embedding × p-threshold targets
- `03_embeddings/dataset_2/03_embeddings_v1.ipynb`
- `03_embeddings/dataset_2/03_embeddings_v2.ipynb`
- `03_embeddings/dataset_3/03_embeddings_v1.ipynb` — Node2Vec v1 (unbiased, q=1); no GraphSAGE (structure-only network)
- `03_embeddings/dataset_3/03_embeddings_v2.ipynb` — Node2Vec v2 (structural, q=2)

## 04 Models — centrality / classical
- `04_classical_ml/dataset_1/04_centrality_selection_p0.ipynb` — model selection at p=0 (temporal split)
- `04_classical_ml/dataset_2/04_centrality_selection_p0.ipynb` — same logic for Dataset 2 (random split)
- `04_classical_ml/combined/04b_centrality_threshold_apply_saved.ipynb` — apply saved model across p-thresholds (Dataset 1 + 2)
- `04_classical_ml/combined/04c_centrality_threshold_retrain.ipynb` — retrain across p-thresholds (Dataset 1 + 2)

## 05 Models — embeddings
- `05_embedding_ml/dataset_1/05_embedding_selection_p0_v1.ipynb` — model selection at p=0 (v1 embeddings, temporal split)
- `05_embedding_ml/dataset_1/05_embedding_selection_p0_v2.ipynb` — model selection at p=0 (v2 embeddings, temporal split)
- `05_embedding_ml/dataset_2/05_embedding_selection_p0.ipynb` — same logic for Dataset 2 (all v1+v2 candidates, random split)
- `05_embedding_ml/combined/05d_embedding_threshold_apply_saved.ipynb` — apply saved model across p-thresholds (Dataset 1 + 2)
- `05_embedding_ml/combined/05e_embedding_threshold_retrain.ipynb` — retrain across p-thresholds (Dataset 1 + 2)

> The `combined/` notebooks cover Dataset 1 **and** Dataset 2 in one notebook (their
> stored results were kept intact). They are not split per-dataset because that would
> require re-running them.

## 06–08 Comparisons (cross-dataset / cross-approach)
- `06_analysis/06_Embedding_Model_Comparison.ipynb`
- `06_analysis/06_target_dataset_analysis.ipynb`
- `07_comparison/07_p_threshold_comparison_centrality_vs_embeddings.ipynb`
- `08_ensemble/08_ensemble_centrality_embeddings.ipynb`

## Data / artifact layout (under `src/`)
```
datasets/dataset_{1,2,3}/{edges,nodes,targets,sim_*}   # raw data + contagion sims
data/embeddings/dataset_{1,2}/{feature_based,network_based}/
data/classical_features/dataset_{1,2,3}/
data/predictions/                                       # model outputs + comparison figures
models/dataset_1/{04_a,05_a,05_b}/                      # saved best models
```

## Notes
- The 128-dim embedding datasets are stored as two row-halves (`128a` / `128b`) rather than
  one full `*_128_srisk_dataset.parquet` file, **to save space** (`03_embeddings_split128.ipynb`
  produces them; concatenating the two halves reconstructs the full 128-dim dataset). The `05`
  selection notebooks still reference the full `*_128_*` file, so re-running their 128-dim cells
  requires concatenating the halves (or regenerating). Stored outputs are intact.
