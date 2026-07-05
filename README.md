# Using Graph Neural Networks to Minimize Financial Contagion in Dynamic Interbank Networks

Repo for the master's thesis: *"Using Graph Neural Networks to Minimize Financial Contagion"*.

Code for systemic risk prediction on interbank networks. It simulates default contagion to build a systemic-risk target (cascade size), learns node representations with Graph Neural Networks (GraphSAGE) and Node2Vec, and predicts the target with classical, embedding-based, and hybrid models across a sweep of default thresholds (`p`).

## Installation

Clone this repository (Python 3.12):

```bash
git clone https://github.com/RubenCMarques/Using-GNNs-to-minimize-systemic-risk.git
```

Install [PyTorch](https://pytorch.org/) and [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/).

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

`torch-scatter`, `torch-sparse`, and `torch-cluster` must be installed manually (see the notes in `requirements.txt`).

## Datasets

Three interbank networks, under `src/datasets/`:

- **dataset_1** — AI4Risk interbank network, temporal (2016 Q1 – 2023 Q4, 32 quarters).
- **dataset_2** — synthetic European network, static (1 444 banks).
- **dataset_3** — Erdős–Rényi synthetic network (avg degree 0.8), static.

Each network stores `edges`, `nodes`, and `targets`. The target `systemic_risk_label` is the cascade size from the default-contagion simulation, with `log_systemic_risk_label = log1p(...)`. A default-threshold sweep (`p0`, `p5` … `p40`) shrinks the failure buffer to produce a target per threshold.

## Code Organization

### Notebooks

Inside the `src/notebooks/` folder, organized **by stage, then by dataset** (each notebook auto-discovers the project root). See `src/notebooks/README.md` for the full index.

- **01_exploration** — preprocessing and classical / centrality features (degree, betweenness, closeness, eigenvector, PageRank, DebtRank).
- **02_contagion** — default-contagion simulations that generate the systemic-risk targets and the `p`-threshold sweep.
- **03_embeddings** — node embeddings: GraphSAGE (`feature_based`) and Node2Vec (`network_based`), in two variants — **v1** (link-prediction objective) and **v2** (feature-reconstruction objective), at 32 / 64 / 128 dims.
- **04_classical_ml** — model selection and threshold sweep on classical / centrality features.
- **05_embedding_ml** — model selection and threshold sweep on the embeddings.
- **06_hybrid_approach** — blends the centrality and embedding predictors (average and validation-fitted weighted average).
- **07_analysis** — embedding-model comparison, Node2Vec vs GraphSAGE, target EDA.
- **08_comparison** — centrality vs embeddings across the `p`-thresholds.
- **00_notes** — network summary tables/figures and R-replica contagion references.

### Models

The models and training code live in `src/models/`:

- `contagion.py` — default-contagion simulation and systemic-importance ranking.
- `classical_algorithms.py` — centrality measures and DebtRank.
- `embeddings.py` / `embedding_pipeline.py` — GraphSAGE and Node2Vec embedding extraction and pooling.
- `fix_embeddings.py` — reconstruction-objective (v2) embeddings.
- `ml_train_and_store.py` — `ModelTrainer`, dataset loaders, and model persistence.

Data loading is in `src/data/` (`data_loader.py`, `simulation_store.py`). The best trained model per dataset is saved under `src/models/dataset_{1,2,3}/`.

### Data & Results

Generated artifacts are written under `src/data/`:

```
data/classical_features/dataset_{1,2,3}/     # centrality features
data/embeddings/dataset_{1,2,3}/{feature_based,network_based}/   # dataset_3: network_based only
data/predictions/                            # model outputs + comparison figures
data/network_summary/                        # summary tables and network figures
```
