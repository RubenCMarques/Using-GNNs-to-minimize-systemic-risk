"""
Dynamic graph embedding backends for interbank networks.

This module exposes a shared embedding API while keeping GraphSAGE and
Node2Vec training logic separate internally.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn.models import GraphSAGE, Node2Vec
from torch_geometric.utils import negative_sampling

from src.data.data_loader import load_data


class GNNConfig:
    def __init__(
        self,
        hidden_dims=(64, 32),
        dropout=0.3,
        lr=0.01,
        epochs=200,
        weight_decay=1e-4,
        activation="relu",
        aggregation="mean",
        negative_sampling_ratio=1.0,
        device="cpu",
    ):
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.activation = activation
        self.aggregation = aggregation
        self.negative_sampling_ratio = negative_sampling_ratio
        self.device = device


class Node2VecConfig:
    def __init__(
        self,
        embedding_dim=64,
        walk_length=20,
        context_size=10,
        walks_per_node=10,
        num_negative_samples=1,
        p=1.0,
        q=1.0,
        batch_size=128,
        lr=0.01,
        epochs=100,
        num_workers=0,
        sparse=True,
        device="cpu",
    ):
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.context_size = context_size
        self.walks_per_node = walks_per_node
        self.num_negative_samples = num_negative_samples
        self.p = p
        self.q = q
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.num_workers = num_workers
        self.sparse = sparse
        self.device = device


def prepare_graph_data(edges, nodes, feature_cols=None):
    """Convert pandas DataFrames to a PyTorch Geometric Data object."""
    bank_ids = nodes["index"].tolist()
    id_to_idx = {bank_id: idx for idx, bank_id in enumerate(bank_ids)}

    if feature_cols is None:
        exclude = {"index", "rank_next_quarter", "srisk_ratio", "srisk_value"}
        feature_cols = [
            col
            for col in nodes.columns
            if col not in exclude and nodes[col].dtype in ["float64", "int64", "float32"]
        ]

    x = nodes[feature_cols].values.astype(np.float32)
    x = StandardScaler().fit_transform(x)
    x = torch.tensor(x, dtype=torch.float)

    valid_edges = edges[
        edges["Sourceid"].isin(id_to_idx)
        & edges["Targetid"].isin(id_to_idx)
    ]

    src = torch.tensor([id_to_idx[s] for s in valid_edges["Sourceid"]], dtype=torch.long)
    dst = torch.tensor([id_to_idx[t] for t in valid_edges["Targetid"]], dtype=torch.long)
    edge_index = torch.stack([src, dst], dim=0)

    weights = valid_edges["Weights"].values.astype(np.float32)
    if weights.max() > 0:
        weights = weights / weights.max()
    edge_attr = torch.tensor(weights, dtype=torch.float).unsqueeze(1)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=len(bank_ids),
    )
    return data, bank_ids


class FlexibleGNN(nn.Module):
    """Compatibility wrapper around the GraphSAGE backend."""

    def __init__(self, in_dim, config: GNNConfig):
        super().__init__()
        self.graphsage_model = GraphSAGE(
            in_channels=in_dim,
            hidden_channels=config.hidden_dims[0],
            num_layers=len(config.hidden_dims),
            out_channels=config.hidden_dims[-1],
            dropout=config.dropout,
            act=config.activation,
            act_first=False,
            jk=None,
            aggr=config.aggregation,
        )

    def forward(self, x, edge_index):
        return self.graphsage_model(x, edge_index)


def link_prediction_loss(embeddings, edge_index, num_nodes, negative_ratio=1.0):
    src, dst = edge_index

    pos_score = (embeddings[src] * embeddings[dst]).sum(dim=1)
    pos_loss = F.binary_cross_entropy_with_logits(
        pos_score, torch.ones_like(pos_score)
    )

    num_neg = int(src.size(0) * negative_ratio)
    neg_edges = negative_sampling(
        edge_index,
        num_nodes=num_nodes,
        num_neg_samples=num_neg,
    )
    neg_src, neg_dst = neg_edges

    neg_score = (embeddings[neg_src] * embeddings[neg_dst]).sum(dim=1)
    neg_loss = F.binary_cross_entropy_with_logits(
        neg_score, torch.zeros_like(neg_score)
    )

    return pos_loss + neg_loss


def train_gnn(data, config: GNNConfig):
    """Train a GraphSAGE model on one graph."""
    device = torch.device(config.device)
    data = data.to(device)
    model = FlexibleGNN(data.x.size(1), config).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    for _ in range(config.epochs):
        model.train()
        optimizer.zero_grad()
        embeddings = model(data.x, data.edge_index)
        loss = link_prediction_loss(
            embeddings,
            data.edge_index,
            data.num_nodes,
            config.negative_sampling_ratio,
        )
        loss.backward()
        optimizer.step()

    return model


def train_node2vec(data, config: Node2VecConfig):
    """Train a Node2Vec model on one graph."""
    device = torch.device(config.device)

    model = Node2Vec(
        edge_index=data.edge_index.to(device),
        embedding_dim=config.embedding_dim,
        walk_length=config.walk_length,
        context_size=config.context_size,
        walks_per_node=config.walks_per_node,
        p=config.p,
        q=config.q,
        num_negative_samples=config.num_negative_samples,
        num_nodes=data.num_nodes,
        sparse=config.sparse,
    ).to(device)

    loader = model.loader(
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )

    optimizer_cls = torch.optim.SparseAdam if config.sparse else torch.optim.Adam
    optimizer = optimizer_cls(model.parameters(), lr=config.lr)

    for _ in range(config.epochs):
        model.train()
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()

    return model


def _extract_graphsage_embeddings(edges, nodes, config: GNNConfig, feature_cols=None):
    data, bank_ids = prepare_graph_data(edges, nodes, feature_cols)
    model = train_gnn(data, config)

    device = torch.device(config.device)
    data = data.to(device)
    model.eval()
    with torch.no_grad():
        embeddings = model(data.x, data.edge_index).cpu().numpy()

    emb_cols = [f"emb_{i}" for i in range(embeddings.shape[1])]
    df = pd.DataFrame(embeddings, columns=emb_cols)
    df.insert(0, "bank_id", bank_ids)
    return df


def _extract_node2vec_embeddings(edges, nodes, config: Node2VecConfig):
    data, bank_ids = prepare_graph_data(edges, nodes, feature_cols=None)
    model = train_node2vec(data, config)

    model.eval()
    with torch.no_grad():
        embeddings = model().cpu().numpy()

    emb_cols = [f"emb_{i}" for i in range(embeddings.shape[1])]
    df = pd.DataFrame(embeddings, columns=emb_cols)
    df.insert(0, "bank_id", bank_ids)
    return df


def extract_embeddings(edges, nodes, config, feature_cols=None):
    """Extract embeddings using the backend implied by the config type."""
    if isinstance(config, GNNConfig):
        return _extract_graphsage_embeddings(edges, nodes, config, feature_cols)

    if isinstance(config, Node2VecConfig):
        return _extract_node2vec_embeddings(edges, nodes, config)

    raise TypeError(
        "Unsupported embedding config. Use GNNConfig for GraphSAGE or "
        "Node2VecConfig for Node2Vec."
    )


def extract_embeddings_for_period(
    year,
    quarter,
    config,
    data_path=None,
    feature_cols=None,
):
    """Train one embedding model for a single quarter and return embeddings."""
    edges, nodes = load_data(year, quarter, data_path=data_path)
    df = extract_embeddings(edges, nodes, config=config, feature_cols=feature_cols)
    df.insert(1, "year", year)
    df.insert(2, "quarter", quarter)
    df.insert(3, "period", f"{year}Q{quarter}")
    return df


def extract_temporal_embeddings(
    config,
    years=None,
    quarters=(1, 2, 3, 4),
    data_path=None,
    feature_cols=None,
    output_path=None,
):
    """Train one embedding model per quarter and concatenate embeddings."""
    if years is None:
        years = range(2016, 2024)

    frames = []
    for year in years:
        for quarter in quarters:
            frame = extract_embeddings_for_period(
                year=year,
                quarter=quarter,
                config=config,
                data_path=data_path,
                feature_cols=feature_cols,
            )
            frames.append(frame)

    embeddings_df = pd.concat(frames, ignore_index=True)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix.lower() == ".csv":
            embeddings_df.to_csv(output_path, index=False)
        else:
            embeddings_df.to_parquet(output_path, index=False)

    return embeddings_df
