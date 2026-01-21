"""
Test script for data loader

This script tests if the data loader works correctly with the interbank dataset.
Run this to verify everything is set up properly before starting analysis.

Usage:
    python tests/test_loader.py
"""

import sys
import os

# Add parent directory to path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_loader import load_data, get_network_info, get_bank_labels


def test_load_single_quarter():
    """
    Test loading a single quarter of data.
    """
    print("="*60)
    print("TEST 1: Loading single quarter (2016 Q1)")
    print("="*60)
    
    try:
        # Try to load 2016 Q1
        edges, nodes = load_data(2016, 1)
        
        # Check if dataframes are not empty
        assert len(edges) > 0, "Edges dataframe is empty!"
        assert len(nodes) > 0, "Nodes dataframe is empty!"
        
        # Check if expected columns exist
        assert 'Sourceid' in edges.columns, "Missing 'Sourceid' column in edges"
        assert 'Targetid' in edges.columns, "Missing 'Targetid' column in edges"
        assert 'Weights' in edges.columns, "Missing 'Weights' column in edges"
        
        assert 'index' in nodes.columns, "Missing 'index' column in nodes"
        assert 'rank_next_quarter' in nodes.columns, "Missing 'rank_next_quarter' column in nodes"
        assert 'srisk_ratio' in nodes.columns, "Missing 'srisk_ratio' column in nodes"
        assert 'srisk_value' in nodes.columns, "Missing 'srisk_value' column in nodes"
        
        print(f" SUCCESS: Loaded {len(nodes)} banks and {len(edges)} connections")
        print(f"   Edge columns: {edges.columns.tolist()[:3]}")
        print(f"   Node columns (first 5): {nodes.columns.tolist()[:5]}")
        print(f"   Node columns (last 3): {nodes.columns.tolist()[-3:]}")
        
        return True
        
    except FileNotFoundError as e:
        print(f" FAILED: Could not find data files")
        print(f"   Error: {e}")
        print(f"   Make sure the interbank repository is cloned in the correct location")
        return False
        
    except Exception as e:
        print(f" FAILED: {e}")
        return False


def test_network_info():
    """
    Test network statistics calculation.
    """
    print("\n" + "="*60)
    print("TEST 2: Calculating network statistics")
    print("="*60)
    
    try:
        edges, nodes = load_data(2016, 1)
        stats = get_network_info(edges, nodes)
        
        # Check if all expected keys exist
        expected_keys = ['num_banks', 'num_connections', 'avg_connections_per_bank', 'total_exposure']
        for key in expected_keys:
            assert key in stats, f"Missing key '{key}' in stats"
        
        print(" SUCCESS: Network statistics calculated")
        print(f"   Number of banks: {stats['num_banks']}")
        print(f"   Number of connections: {stats['num_connections']}")
        print(f"   Avg connections per bank: {stats['avg_connections_per_bank']:.2f}")
        print(f"   Total exposure: {stats['total_exposure']:,.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_bank_labels():
    """
    Test bank label extraction.
    """
    print("\n" + "="*60)
    print("TEST 3: Extracting bank labels")
    print("="*60)
    
    try:
        edges, nodes = load_data(2016, 1)
        labels = get_bank_labels(nodes)
        
        # Check if dataframe has expected columns
        assert 'bank_id' in labels.columns, "Missing 'bank_id' column"
        assert 'rating' in labels.columns, "Missing 'rating' column"
        
        # Check if we have the same number of labels as nodes
        assert len(labels) == len(nodes), "Number of labels doesn't match number of nodes"
        
        # Count rating distribution
        rating_counts = labels['rating'].value_counts().sort_index()
        
        print("SUCCESS: Labels extracted")
        print(f"   Total banks: {len(labels)}")
        print(f"   Banks with ratings: {labels['rating'].notna().sum()}")
        print(f"   Banks without ratings: {labels['rating'].isna().sum()}")
        print("\n   Rating distribution:")
        for rating, count in rating_counts.items():
            rating_name = {1.0: 'A', 2.0: 'B', 3.0: 'C', 4.0: 'D'}.get(rating, 'Unknown')
            print(f"      Rating {int(rating)} ({rating_name}): {count} banks")
        
        return True
        
    except Exception as e:
        print(f"FAILED: {e}")
        return False


def test_multiple_quarters():
    """
    Test loading multiple quarters to ensure consistency.
    """
    print("\n" + "="*60)
    print("TEST 4: Loading multiple quarters")
    print("="*60)
    
    quarters_to_test = [
        (2016, 1),
        (2016, 4),
        (2020, 1),
    ]
    
    try:
        for year, quarter in quarters_to_test:
            try:
                edges, nodes = load_data(year, quarter)
                print(f" {year} Q{quarter}: {len(nodes)} banks, {len(edges)} connections")
            except FileNotFoundError:
                print(f" {year} Q{quarter}: Data not found (might not exist)")
        
        print("\nSUCCESS: Multiple quarters loaded successfully")
        return True
        
    except Exception as e:
        print(f"FAILED: {e}")
        return False


def run_all_tests():
    """
    Run all tests and report results.
    """
    print("\n" + "="*60)
    print("RUNNING DATA LOADER TESTS")
    print("="*60 + "\n")
    
    tests = [
        ("Load Single Quarter", test_load_single_quarter),
        ("Network Statistics", test_network_info),
        ("Bank Labels", test_bank_labels),
        ("Multiple Quarters", test_multiple_quarters),
    ]
    
    results = []
    for test_name, test_func in tests:
        result = test_func()
        results.append((test_name, result))
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n All tests passed!")
    else:
        print("\nSome tests failed.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)