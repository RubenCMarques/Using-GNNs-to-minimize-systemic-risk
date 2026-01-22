"""
Simple Data Loader for Interbank Network Dataset

This script loads quarterly interbank network data from the AI4Risk repository.

Data Structure:
- datasets/edges/edge_{YEAR}Q{QUARTER}.csv: Network connections between banks
  Columns: [Sourceid, Targetid, Weights]
  - Sourceid: Source bank identifier
  - Targetid: Target bank identifier  
  - Weights: Exposure/loan amount between banks
  
- datasets/nodes/{YEAR}Q{QUARTER}: Bank features and labels
  Columns: [index, 70+ financial features..., rank_next_quarter, srisk_ratio, srisk_value]
  - index: Bank identifier
  - Middle columns: 70+ financial features (Total_assets, Total_liabilities, Equity, etc.)
  - rank_next_quarter: Credit rating for next quarter (1=A, 2=B, 3=C, 4=D)
  - srisk_ratio: Systemic risk ratio
  - srisk_value: Systemic risk value
  
Available quarters: 2016 Q1 to 2023 Q4 (32 quarters total)
"""

import os
import pandas as pd
import numpy as np


def load_data(year, quarter, data_path="../datasets"):
    """
    Load edge and node data for one quarter.
    
    This is the main function to load the interbank network data.
    It reads two CSV files: one for network connections (edges) and one for bank information (nodes).
    
    Args:
        year (int): Year of the data to load (2016-2023)
        quarter (int): Quarter of the year (1-4)
        data_path (str): Path to the datasets folder relative to this file
                        Default assumes interbank repo is cloned two levels up
        
    Returns:
        tuple: (edges_df, nodes_df)
            - edges_df: DataFrame with network connections
              Columns: [Sourceid, Targetid, Weights]
            - nodes_df: DataFrame with bank features and labels
              Columns: [index, features..., rank_next_quarter, srisk_ratio, srisk_value]
    
    Example:
        edges, nodes = load_data(2023, 1)
        print(f"Loaded {len(nodes)} banks with {len(edges)} connections")
    """
    # Build the full file paths for this specific quarter
    # File naming: edges/edge_YYYYQX.csv and nodes/YYYYQX.csv
    edge_file = os.path.join(data_path, "edges", f"edge_{year}Q{quarter}.csv")
    node_file = os.path.join(data_path, "nodes", f"{year}Q{quarter}.csv")
    
    # Load the CSV files into pandas DataFrames
    # edges: who lends to whom and how much (Sourceid -> Targetid with Weights)
    # nodes: characteristics of each bank (index + 70 features + rating + SRISK)
    edges = pd.read_csv(edge_file)
    nodes = pd.read_csv(node_file)
    
    # Print summary for confirmation
    print(f"Loaded {year} Q{quarter}: {len(nodes)} banks, {len(edges)} connections")
    
    return edges, nodes


def get_network_info(edges, nodes):
    """
    Calculate basic statistics about the network structure.
    
    This helps understand the connectivity and density of the interbank network.
    
    Args:
        edges (DataFrame): Edge data from load_data()
        nodes (DataFrame): Node data from load_data()
        
    Returns:
        dict: Dictionary containing:
            - num_banks: Total number of banks in the network
            - num_connections: Total number of connections (edges) between banks
            - avg_connections_per_bank: Average number of connections each bank has
            - total_exposure: Sum of all edge weights (total interbank exposure)
    
    Example:
        edges, nodes = load_data(2023, 1)
        stats = get_network_info(edges, nodes)
        print(f"Network density: {stats['avg_connections_per_bank']:.2f} connections per bank")
    """
    # Count total number of banks (nodes in the network)
    num_banks = len(nodes)
    
    # Count total number of connections (edges in the network)
    num_connections = len(edges)
    
    # Calculate average degree (how many connections per bank on average)
    avg_connections = num_connections / num_banks if num_banks > 0 else 0
    
    # Calculate total exposure (sum of all edge weights)
    total_exposure = edges['Weights'].sum() if 'Weights' in edges.columns else 0
    
    # Return as a dictionary for easy access
    return {
        'num_banks': num_banks,
        'num_connections': num_connections,
        'avg_connections_per_bank': avg_connections,
        'total_exposure': total_exposure
    }


def get_bank_labels(nodes):
    """
    Extract just the bank IDs and their credit ratings.
    
    Credit ratings (rank_next_quarter):
    - 1: A rating (Best - lowest risk)
    - 2: B rating (Good)
    - 3: C rating (Fair)
    - 4: D rating (Poor - highest risk)
    - NaN: No rating available
    
    Args:
        nodes (DataFrame): Node data from load_data()
        
    Returns:
        DataFrame: Two columns:
            - bank_id: Unique identifier for each bank (from 'index' column)
            - rating: Credit rating (1-4 or NaN)
    
    Example:
        edges, nodes = load_data(2023, 1)
        labels = get_bank_labels(nodes)
        print(labels.head())
        
        # Count how many banks have each rating
        print(labels['rating'].value_counts())
    """
    # Extract bank ID (index column) and rating (rank_next_quarter column)
    return pd.DataFrame({
        'bank_id': nodes['index'],                    # Bank identifier
        'rating': nodes['rank_next_quarter']          # Credit rating (1=A, 2=B, 3=C, 4=D)
    })


# Example usage
if __name__ == "__main__":
    # Load one quarter
    edges, nodes = load_data(2016, 1)
    
    print("\n--- Edge data (first 5 rows) ---")
    print(edges.head())
    print(f"Shape: {edges.shape}")
    print(f"Columns: {edges.columns.tolist()}")
    
    print("\n--- Node data (first 5 rows, first 10 columns) ---")
    print(nodes.iloc[:5, :10])
    print(f"Shape: {nodes.shape}")
    print(f"Columns: {nodes.columns.tolist()}")
    
    print("\n--- Network info ---")
    info = get_network_info(edges, nodes)
    for key, val in info.items():
        print(f"{key}: {val}")
    
    print("\n--- Bank labels (first 10) ---")
    labels = get_bank_labels(nodes)
    print(labels.head(10))
    
    print("\n--- Rating distribution ---")
    print(labels['rating'].value_counts().sort_index())


# Example usage
if __name__ == "__main__":
    # Load one quarter
    edges, nodes = load_data(2016, 1)
    
    print("\n--- Edge data (first 5 rows) ---")
    print(edges.head())
    print(f"Shape: {edges.shape}")
    
    print("\n--- Node data (first 5 rows) ---")
    print(nodes.head())
    print(f"Shape: {nodes.shape}")
    
    print("\n--- Network info ---")
    info = get_network_info(edges, nodes)
    for key, val in info.items():
        print(f"{key}: {val}")
    
    print("\n--- Bank labels (first 10) ---")
    labels = get_bank_labels(nodes)
    print(labels.head(10))
    
    print("\n--- Rating distribution ---")
    print(labels['rating'].value_counts())