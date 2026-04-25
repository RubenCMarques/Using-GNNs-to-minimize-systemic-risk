"""Models for contagion simulation, classical algorithms, and graph embeddings."""

from .embeddings import (
    GNNConfig,
    Node2VecConfig,
    extract_embeddings_for_period,
)
from .embedding_pipeline import build_pooled_dataset
from .fix_embeddings import FixedGNNConfig, build_fixed_pooled_dataset
from .ml_train_and_store import (
    ModelTrainer,
    load_classical_dataset,
    load_gnn_dataset,
    make_pipeline,
    time_split,
)

try:
    from .contagion import (
        simulate_failure,
        run_default_contagion_analysis,
        run_default_contagion_analysis_for_quarter,
    )
except ModuleNotFoundError:
    pass

try:
    from .classical_algorithms import (
        degree_centrality,
        betweenness_centrality,
        closeness_centrality,
        eigenvector_centrality,
        weighted_degree,
        debtrank,
        pagerank_centrality,
    )
except ModuleNotFoundError:
    pass
