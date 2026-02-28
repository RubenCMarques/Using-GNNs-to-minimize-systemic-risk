"""
Test script for contagion model

This script tests the contagion model implementation.
Tests include shock simulations and multi-bank failures.
"""

"""
Test script for contagion.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_loader import load_data
from src.models.contagion import simulate_failure, compute_systemic_importance


def test_simulate_failure():
    """Test that simulate_failure works."""
    edges, nodes = load_data(2023, 1)
    
    # Pick first bank
    bank_id = nodes['index'].iloc[10]
    
    result = simulate_failure(
        bank_id,
        edges,
        nodes,
        mechanism="Exposure",
        initial_loss_mode="fixed",
        initial_loss_frac=1.0,
    )
    
    assert 'failed_banks' in result
    assert 'total_loss' in result
    assert 'num_failed' in result
    assert 'rounds' in result
    assert 'initial_shock_fraction' in result
    assert 'initial_default' in result
    assert bank_id in result['failed_banks']
    assert result['num_failed'] >= 1
    
    print(f"Bank {bank_id} failure: {result['num_failed']} failed, loss={result['total_loss']:.0f}")


def test_compute_systemic_importance():
    """Test that compute_systemic_importance works."""
    edges, nodes = load_data(2023, 1)
    
    df = compute_systemic_importance(edges, nodes, mechanism="Exposure")
    
    assert len(df) == len(nodes)
    assert 'bank_id' in df.columns
    assert 'total_loss' in df.columns
    assert 'num_failed' in df.columns
    assert 'is_systemically_important' in df.columns
    assert df['is_systemically_important'].isin([0, 1]).all()
    
    systemic_count = df['is_systemically_important'].sum()
    print(f"{systemic_count}/{len(df)} banks labeled as systemically important")


def test_spread_toggle_without_initial_default():
    """Test propagation toggle when the initial shock does not trigger default."""
    edges, nodes = load_data(2023, 1)
    bank_id = nodes['index'].iloc[10]

    no_spread = simulate_failure(
        bank_id,
        edges,
        nodes,
        mechanism="Exposure",
        spread_without_default=False,
        initial_loss_mode="fixed",
        initial_loss_frac=0.2,
    )
    spread = simulate_failure(
        bank_id,
        edges,
        nodes,
        mechanism="Exposure",
        spread_without_default=True,
        initial_loss_mode="fixed",
        initial_loss_frac=0.2,
    )

    assert no_spread["initial_default"] is False
    assert no_spread["rounds"] == 0
    assert no_spread["num_failed"] == 0
    assert spread["rounds"] >= 1


if __name__ == "__main__":
    test_simulate_failure()
    test_compute_systemic_importance()
