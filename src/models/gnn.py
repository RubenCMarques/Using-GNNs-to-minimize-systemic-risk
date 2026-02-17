"""
Modular Graph Neural Network Embedding Framework

This module implements flexible GNN architectures for extracting node embeddings
from interbank networks.

Supports:
- GCN
- GAT
- GraphSAGE

Includes:
- Configurable architecture
- Flexible training
- Edge weighting
- Modular experimentation
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.utils import negative_sampling

from sklearn.preprocessing import StandardScaler



# =====================================================
# CONFIGURATION
# =====================================================

class GNNConfig:
    def __init__(
        self,
        model_type="GCN",
        hidden_dims=(64, 32),
        dropout=0.3,
        lr=0.01,
        epochs=200,
        weight_decay=1e-4,
        activation="relu",
        use_edge_weights=False,
        aggregation="mean",
        negative_sampling_ratio=1.0,
        device="cpu",
    ):
        self.model_type = model_type
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.activation = activation
        self.use_edge_weights = use_edge_weights
        self.aggregation = aggregation
        self.negative_sampling_ratio = negative_sampling_ratio
        self.device = device


# =====================================================
# ACTIVATION
# =====================================================

def get_activation(name):
    return {
        "relu": F.relu,
        "elu": F.elu,
        "gelu": F.gelu,
        "tanh": torch.tanh,
    }.get(name, F.relu)


# =====================================================
# DATA PREPARATION
# =====================================================

def prepare_graph_data(edges, nodes, feature_cols=None):
    """
    Convert pandas DataFrames to PyTorch Geometric Data object.
    """

    bank_ids = nodes["index"].tolist()
    id_to_idx = {b: i for i, b in enumerate(bank_ids)}

    # Node features
    if feature_cols is None:
        exclude = {"index", "rank_next_quarter", "srisk_ratio", "srisk_value"}
        feature_cols = [
            c for c in nodes.columns
            if c not in exclude and nodes[c].dtype in ["float64", "int64", "float32"]
        ]

    x = nodes[feature_cols].values.astype(np.float32)

    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    x = torch.tensor(x, dtype=torch.float)

    # Edge index
    valid_edges = edges[
        edges["Sourceid"].isin(id_to_idx)
        & edges["Targetid"].isin(id_to_idx)
    ]

    src = torch.tensor(
        [id_to_idx[s] for s in valid_edges["Sourceid"]],
        dtype=torch.long,
    )
    dst = torch.tensor(
        [id_to_idx[t] for t in valid_edges["Targetid"]],
        dtype=torch.long,
    )

    edge_index = torch.stack([src, dst], dim=0)

    # Edge weights (normalized to avoid numerical instability)
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


# =====================================================
# FLEXIBLE GNN
# =====================================================

class FlexibleGNN(nn.Module):
    def __init__(self, in_dim, config: GNNConfig):
        super().__init__()

        self.config = config
        self.activation = get_activation(config.activation)

        dims = (in_dim,) + tuple(config.hidden_dims)

        layers = []
        for i in range(len(dims) - 1):
            layers.append(self._build_layer(dims[i], dims[i + 1]))

        self.layers = nn.ModuleList(layers)

    def _build_layer(self, in_dim, out_dim):

        if self.config.model_type == "GCN":
            return GCNConv(in_dim, out_dim)

        elif self.config.model_type == "GAT":
            return GATConv(in_dim, out_dim, heads=1, concat=False)

        elif self.config.model_type == "GraphSAGE":
            return SAGEConv(in_dim, out_dim, aggr=self.config.aggregation)

        else:
            raise ValueError("Unknown model type")

    def forward(self, x, edge_index, edge_weight=None):

        for layer in self.layers:

            if isinstance(layer, GCNConv) and self.config.use_edge_weights:
                x = layer(x, edge_index, edge_weight)
            else:
                x = layer(x, edge_index)

            x = self.activation(x)
            x = F.dropout(x, p=self.config.dropout, training=self.training)

        return x


# =====================================================
# LOSS FUNCTION
# =====================================================

def link_prediction_loss(embeddings, edge_index, num_nodes, negative_ratio=1.0):

    src, dst = edge_index

    # Positive edges
    pos_score = (embeddings[src] * embeddings[dst]).sum(dim=1)
    pos_loss = F.binary_cross_entropy_with_logits(
        pos_score, torch.ones_like(pos_score)
    )

    # Negative edges
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


# =====================================================
# TRAINING
# =====================================================

def train_gnn(data, config: GNNConfig):

    device = torch.device(config.device)

    data = data.to(device)

    model = FlexibleGNN(data.x.size(1), config).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    for epoch in range(config.epochs):

        model.train()
        optimizer.zero_grad()

        edge_weight = (
            data.edge_attr.squeeze()
            if config.use_edge_weights and data.edge_attr is not None
            else None
        )

        embeddings = model(data.x, data.edge_index, edge_weight)

        loss = link_prediction_loss(
            embeddings,
            data.edge_index,
            data.num_nodes,
            config.negative_sampling_ratio,
        )

        loss.backward()
        optimizer.step()

    return model


# =====================================================
# EMBEDDING EXTRACTION
# =====================================================

def extract_embeddings(edges, nodes, config: GNNConfig, feature_cols=None):

    data, bank_ids = prepare_graph_data(edges, nodes, feature_cols)

    model = train_gnn(data, config)

    device = torch.device(config.device)
    data = data.to(device)

    model.eval()
    with torch.no_grad():

        edge_weight = (
            data.edge_attr.squeeze()
            if config.use_edge_weights and data.edge_attr is not None
            else None
        )

        embeddings = model(
            data.x,
            data.edge_index,
            edge_weight,
        ).cpu().numpy()

    emb_cols = [f"emb_{i}" for i in range(embeddings.shape[1])]
    df = pd.DataFrame(embeddings, columns=emb_cols)
    df.insert(0, "bank_id", bank_ids)

    return df
