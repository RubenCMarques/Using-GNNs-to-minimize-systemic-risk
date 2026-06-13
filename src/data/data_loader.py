"""
Data Loader for the AI4Risk Dataset

Loads quarterly interbank network data from CSV files.
Available quarters: 2016 Q1 to 2023 Q4 (32 quarters)
"""

import os
import pandas as pd
from pathlib import Path


def load_data(year, quarter, data_path=None):
    """
    Load edge and node data for one quarter.

    Args:
        year (int): Year (2016-2023)
        quarter (int): Quarter (1-4)
        data_path (str): Path to a dataset folder. Default: root/src/datasets/dataset_1

    Returns:
        tuple: (edges, nodes)
        
    Example:
        edges, nodes = load_data(2023, 1)
    """
    if data_path is None:
        project_root = Path(__file__).parent.parent.parent
        data_path = project_root / "src" / "datasets" / "dataset_1"

    edge_file = data_path / "edges" / f"edge_{year}Q{quarter}.csv"
    node_file = data_path / "nodes" / f"{year}Q{quarter}.csv"

    edges = pd.read_csv(edge_file)
    nodes = pd.read_csv(node_file)

    print(f"Loaded {year} Q{quarter}: {len(nodes)} banks, {len(edges)} edges")

    return edges, nodes
