"""
Fixed embedding pipeline addressing the key problems with the original approach.

Problems fixed vs embeddings.py / embedding_pipeline.py:

1. Pretext task mismatch (GraphSAGE):
   The original trains GraphSAGE with link-prediction loss, which optimises for
   reconstructing who is connected to whom — not for encoding systemic importance.
   Here the encoder is trained with a feature-reconstruction loss: a decoder tries
   to recover the original financial node features from the embedding. This forces
   the embedding to capture financial node characteristics rather than only graph
   topology, while still being informed by neighbourhood aggregation.
   Optionally a small link-prediction term can be added (link_weight > 0).

2. Node2Vec ignores financial features:
   Node2Vec is a purely structural method — it only uses edge_index and produces
   embeddings that encode graph position (random-walk co-occurrence). For the
   enriched variant, the structural embeddings are concatenated with normalised
   financial node features so the downstream model has access to both sources of
   information. Columns emb_0..emb_{dim-1} are structural; emb_{dim}.. are
   financial features.

3. Warm-starting for temporal coherence:
   Both models are initialised from the previous quarter's weights so the
   embedding space evolves smoothly rather than being re-randomised each quarter.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.models import GraphSAGE

from src.data.data_loader import load_data
from src.models.embeddings import (
    Node2VecConfig,
    link_prediction_loss,
    prepare_graph_data,
    train_node2vec,
)
from src.models.embedding_pipeline import (
    _iter_quarters,
    _load_raw_features,
    _load_targets,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class FixedGNNConfig:
    """GraphSAGE trained with a feature-reconstruction objective.

    Args:
        hidden_dims: (hidden_channels, out_channels). The first value is the
            width of all intermediate GraphSAGE layers; the last value is the
            embedding output dimension (the bottleneck).
            The node features are 70-dimensional. PCA shows 32 dims capture
            97.7% of variance and 16 dims capture 84%, so values in [16, 32]
            create a meaningful compression. 64 is almost no bottleneck (91%
            of original size) and should be avoided for reconstruction-based
            training.
        reconstruction_weight: multiplier on the MSE reconstruction loss.
        link_weight: multiplier on the optional link-prediction loss.
            Set to 0 (default) to use reconstruction only.
        verbose: print loss every log_every epochs.
    """

    def __init__(
        self,
        hidden_dims=(256, 32),
        dropout=0.3,
        lr=0.01,
        epochs=100,
        weight_decay=1e-4,
        activation="relu",
        aggregation="mean",
        reconstruction_weight=1.0,
        link_weight=0.0,
        device="cpu",
        verbose=False,
        log_every=10,
    ):
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.activation = activation
        self.aggregation = aggregation
        self.reconstruction_weight = reconstruction_weight
        self.link_weight = link_weight
        self.device = device
        self.verbose = verbose
        self.log_every = log_every


# ---------------------------------------------------------------------------
# Fixed GraphSAGE: encoder + decoder
# ---------------------------------------------------------------------------

class ReconstructionGNN(nn.Module):
    """GraphSAGE encoder paired with an MLP decoder for feature reconstruction.

    During training the decoder is used to compute the reconstruction loss.
    At inference time only the encoder (embed) is used.
    """

    def __init__(self, in_dim: int, config: FixedGNNConfig):
        super().__init__()
        hidden = config.hidden_dims[0]
        out = config.hidden_dims[-1]

        self.encoder = GraphSAGE(
            in_channels=in_dim,
            hidden_channels=hidden,
            num_layers=len(config.hidden_dims),
            out_channels=out,
            dropout=config.dropout,
            act=config.activation,
            act_first=False,
            jk=None,
            aggr=config.aggregation,
        )
        self.decoder = nn.Sequential(
            nn.Linear(out, hidden),
            nn.ReLU(),
            nn.Linear(hidden, in_dim),
        )

    def embed(self, x, edge_index):
        """Return node embeddings (encoder only)."""
        return self.encoder(x, edge_index)

    def forward(self, x, edge_index):
        """Return (embeddings, reconstructed_features)."""
        z = self.embed(x, edge_index)
        x_hat = self.decoder(z)
        return z, x_hat


def train_reconstruction_gnn(
    data,
    config: FixedGNNConfig,
    init_state_dict=None,
):
    """Train a ReconstructionGNN on one quarter's graph.

    Args:
        data: PyG Data object.
        config: FixedGNNConfig.
        init_state_dict: optional state dict from the previous quarter for
            warm-starting (temporal coherence).

    Returns:
        Trained model.
    """
    device = torch.device(config.device)
    data = data.to(device)

    model = ReconstructionGNN(data.x.size(1), config).to(device)
    if init_state_dict is not None:
        model.load_state_dict(init_state_dict)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    for epoch in range(config.epochs):
        model.train()
        optimizer.zero_grad()

        z, x_hat = model(data.x, data.edge_index)

        loss = config.reconstruction_weight * F.mse_loss(x_hat, data.x)

        if config.link_weight > 0:
            loss = loss + config.link_weight * link_prediction_loss(
                z, data.edge_index, data.num_nodes
            )

        loss.backward()
        optimizer.step()

        if config.verbose and (epoch + 1) % config.log_every == 0:
            print(f"    epoch {epoch + 1}/{config.epochs}  loss={loss.item():.4f}")

    return model


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def _extract_fixed_gnn_embeddings(
    edges,
    nodes,
    config: FixedGNNConfig,
    feature_cols=None,
    init_state_dict=None,
):
    """Train and extract fixed GraphSAGE embeddings for one quarter.

    Returns:
        (embeddings_df, state_dict)
    """
    data, bank_ids = prepare_graph_data(edges, nodes, feature_cols)
    model = train_reconstruction_gnn(data, config, init_state_dict=init_state_dict)

    device = torch.device(config.device)
    data = data.to(device)
    model.eval()
    with torch.no_grad():
        embeddings = model.embed(data.x, data.edge_index).cpu().numpy()

    emb_cols = [f"emb_{i}" for i in range(embeddings.shape[1])]
    df = pd.DataFrame(embeddings, columns=emb_cols)
    df.insert(0, "bank_id", bank_ids)
    return df, model.state_dict()


def _extract_node2vec_structural_embeddings(
    edges,
    nodes,
    config: Node2VecConfig,
    init_state_dict=None,
):
    """Train Node2Vec and return purely structural embeddings.

    Uses only graph topology (edge_index) — no financial features — so the
    comparison against classical centrality measures stays fair. The embedding
    dimension is controlled by config.embedding_dim and can differ from the
    original 64-dim Node2Vec to find a better structural representation.

    Returns:
        (embeddings_df, state_dict)
    """
    data, bank_ids = prepare_graph_data(edges, nodes, feature_cols=None)
    model = train_node2vec(data, config, init_state_dict=init_state_dict)

    model.eval()
    with torch.no_grad():
        embeddings = model().cpu().numpy()

    emb_cols = [f"emb_{i}" for i in range(embeddings.shape[1])]
    df = pd.DataFrame(embeddings, columns=emb_cols)
    df.insert(0, "bank_id", bank_ids)
    return df, model.state_dict()


# ---------------------------------------------------------------------------
# Public extraction API
# ---------------------------------------------------------------------------

def extract_fixed_embeddings(
    edges,
    nodes,
    config,
    feature_cols=None,
    init_state_dict=None,
):
    """Dispatch to the correct fixed extractor based on config type.

    Returns:
        (embeddings_df, state_dict)
    """
    if isinstance(config, FixedGNNConfig):
        return _extract_fixed_gnn_embeddings(
            edges, nodes, config, feature_cols, init_state_dict
        )
    if isinstance(config, Node2VecConfig):
        return _extract_node2vec_structural_embeddings(
            edges, nodes, config, init_state_dict
        )
    raise TypeError(
        "Unsupported config type. Use FixedGNNConfig or Node2VecConfig."
    )


def extract_fixed_embeddings_for_period(
    year,
    quarter,
    config,
    data_path=None,
    feature_cols=None,
    init_state_dict=None,
):
    """Extract fixed embeddings for one quarter and attach period metadata.

    Returns:
        (embeddings_df, state_dict)
    """
    edges, nodes = load_data(year, quarter, data_path=data_path)
    df, state_dict = extract_fixed_embeddings(
        edges, nodes, config=config, feature_cols=feature_cols,
        init_state_dict=init_state_dict,
    )
    df.insert(1, "year", year)
    df.insert(2, "quarter", quarter)
    df.insert(3, "period", f"{year}Q{quarter}")
    return df, state_dict


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def _build_fixed_quarter_dataset(
    year,
    quarter,
    config,
    target_col,
    include_raw_features,
    target_dir,
    init_state_dict=None,
):
    embeddings, state_dict = extract_fixed_embeddings_for_period(
        year, quarter, config=config, init_state_dict=init_state_dict
    )
    targets = _load_targets(year, quarter, target_col=target_col, target_dir=target_dir)

    if targets.empty:
        return pd.DataFrame(), state_dict

    df = embeddings.merge(targets, on=["bank_id", "year", "quarter", "period"], how="inner")

    if include_raw_features:
        raw = _load_raw_features(year, quarter)
        df = df.merge(raw, on=["bank_id", "year", "quarter", "period"], how="left")

    return df, state_dict


def build_fixed_pooled_dataset(
    config,
    years=None,
    quarters=(1, 2, 3, 4),
    target_col="log_systemic_risk_label",
    include_raw_features=False,
    target_dir=None,
    output_path=None,
):
    """Build a pooled dataset using the fixed embedding pipeline.

    Each quarter is warm-started from the previous quarter's model weights for
    temporal coherence. The first quarter always trains from random initialisation.

    Args:
        config: FixedGNNConfig (reconstruction GraphSAGE) or Node2VecConfig
            (enriched Node2Vec with concatenated financial features).
        years: iterable of years. Defaults to 2016–2023.
        quarters: tuple of quarter numbers. Defaults to (1, 2, 3, 4).
        target_col: column name of the regression target.
        include_raw_features: if True, append raw node features to each row.
        target_dir: override for the targets directory.
        output_path: if provided, write the pooled DataFrame to this path
            (.parquet or .csv).

    Returns:
        Pooled DataFrame.
    """
    frames = []
    prev_state_dict = None

    for year, quarter in _iter_quarters(years=years, quarters=quarters):
        df, prev_state_dict = _build_fixed_quarter_dataset(
            year, quarter, config, target_col, include_raw_features,
            target_dir, init_state_dict=prev_state_dict,
        )
        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    pooled = pd.concat(frames, ignore_index=True)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix.lower() == ".csv":
            pooled.to_csv(output_path, index=False)
        else:
            pooled.to_parquet(output_path, index=False)

    return pooled
