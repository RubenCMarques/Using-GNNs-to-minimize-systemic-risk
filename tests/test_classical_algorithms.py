"""
Test script for classical centrality algorithms

This script tests the classical network centrality measures.
"""

"""
Test script for classical_algorithms.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_loader import load_data
from src.models.classical_algorithms import (
    degree_centrality,
    betweenness_centrality,
    closeness_centrality,
    debtrank
)


def test_degree_centrality():
    """Test degree centrality."""
    edges, nodes = load_data(2023, 1)
    
    df = degree_centrality(edges, nodes)
    
    assert len(df) == len(nodes)
    assert 'bank_id' in df.columns
    assert 'degree_centrality_in' in df.columns
    assert 'degree_centrality_out' in df.columns
    assert 'degree_centrality_total' in df.columns
    assert df['degree_centrality_in'].between(0, 1).all()
    assert df['degree_centrality_out'].between(0, 1).all()
    assert df['degree_centrality_total'].between(0, 2).all()
    
    print(f"Degree centrality total: max={df['degree_centrality_total'].max():.3f}")


def test_betweenness_centrality():
    """Test betweenness centrality."""
    edges, nodes = load_data(2023, 1)
    
    df = betweenness_centrality(edges, nodes)
    
    assert len(df) == len(nodes)
    assert 'bank_id' in df.columns
    assert 'betweenness_centrality' in df.columns
    
    print(f"Betweenness centrality: max={df['betweenness_centrality'].max():.3f}")


def test_closeness_centrality():
    """Test closeness centrality."""
    edges, nodes = load_data(2023, 1)
    
    df = closeness_centrality(edges, nodes)
    
    assert len(df) == len(nodes)
    assert 'bank_id' in df.columns
    assert 'closeness_centrality' in df.columns
    
    print(f"Closeness centrality: max={df['closeness_centrality'].max():.3f}")


def test_debtrank():
    """Test DebtRank."""
    edges, nodes = load_data(2023, 1)
    
    df = debtrank(edges, nodes)
    
    assert len(df) == len(nodes)
    assert 'bank_id' in df.columns
    assert 'debtrank' in df.columns
    
    print(f"DebtRank: max={df['debtrank'].max():.3f}")


if __name__ == "__main__":
    test_degree_centrality()
    test_betweenness_centrality()
    test_closeness_centrality()
    test_debtrank()
