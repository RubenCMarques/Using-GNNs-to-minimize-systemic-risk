"""
Test script for data loader

This script tests if the data loader works correctly.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_loader import load_data


def test_load_data():
    """Test that we can load a quarter of data."""
    edges, nodes = load_data(2023, 1)

    # Check edges
    assert len(edges) > 0, "No edges loaded"
    assert "Sourceid" in edges.columns
    assert "Targetid" in edges.columns
    assert "Weights" in edges.columns

    # Check nodes
    assert len(nodes) > 0, "No nodes loaded"
    assert "index" in nodes.columns

    print(f"Loaded {len(nodes)} banks, {len(edges)} edges")


if __name__ == "__main__":
    test_load_data()