#!/usr/bin/env python3
"""
Tests for the _get_entity_values method on BIDSReader.

This test verifies that:
1. _get_entity_values returns the input value when it's a string
2. _get_entity_values returns the input value when it's a list
3. _get_entity_values returns all existing values from BIDS dataset when input is None
4. _get_entity_values handles edge cases properly
"""

import sys
import tempfile
from pathlib import Path

# Find the repository root
repo_root = Path(__file__).parent.parent
src_dir = repo_root / "src"

# Add src to path
sys.path.insert(0, str(src_dir))


def create_mock_bids_dataset(bids_root):
    """Create a minimal mock BIDS dataset for testing."""
    bids_root = Path(bids_root)
    
    # Create subjects
    subjects = ['01', '02', '03']
    tasks = ['rest', 'task1']
    sessions = ['01', '02']
    
    for sub in subjects:
        sub_dir = bids_root / f'sub-{sub}'
        sub_dir.mkdir(parents=True, exist_ok=True)
        
        for ses in sessions:
            ses_dir = sub_dir / f'ses-{ses}'
            eeg_dir = ses_dir / 'eeg'
            eeg_dir.mkdir(parents=True, exist_ok=True)
            
            for task in tasks:
                # Create minimal BIDS files
                filename = f'sub-{sub}_ses-{ses}_task-{task}_eeg.vhdr'
                (eeg_dir / filename).touch()
    
    return bids_root


def test_get_entity_values_with_string():
    """Test that _get_entity_values returns a list when given a string."""
    try:
        from meegflow import MEEGFlowPipeline
        from meegflow.readers import BIDSReader
        
        with tempfile.TemporaryDirectory() as tmpdir:
            bids_root = Path(tmpdir)
            reader = BIDSReader(bids_root)
            pipeline = MEEGFlowPipeline(reader=reader, config={})
            
            result = reader._get_entity_values('subject', '01')
            assert result == ['01'], f"Expected ['01'], got {result}"
            
        print("✓ _get_entity_values correctly handles string input")
    except ImportError as e:
        print(f"⚠ Skipping test (missing dependencies): {e}")
        raise


def test_get_entity_values_with_list():
    """Test that _get_entity_values returns the list when given a list."""
    try:
        from meegflow import MEEGFlowPipeline
        from meegflow.readers import BIDSReader
        
        with tempfile.TemporaryDirectory() as tmpdir:
            bids_root = Path(tmpdir)
            reader = BIDSReader(bids_root)
            pipeline = MEEGFlowPipeline(reader=reader, config={})
            
            result = reader._get_entity_values('subject', ['01', '02', '03'])
            assert result == ['01', '02', '03'], f"Expected ['01', '02', '03'], got {result}"
            
        print("✓ _get_entity_values correctly handles list input")
    except ImportError as e:
        print(f"⚠ Skipping test (missing dependencies): {e}")
        raise


def test_get_entity_values_with_none_returns_all_subjects():
    """Test that _get_entity_values returns all subjects when input is None."""
    try:
        from meegflow import MEEGFlowPipeline
        from meegflow.readers import BIDSReader
        
        with tempfile.TemporaryDirectory() as tmpdir:
            bids_root = create_mock_bids_dataset(tmpdir)
            reader = BIDSReader(bids_root)
            pipeline = MEEGFlowPipeline(reader=reader, config={})
            
            result = reader._get_entity_values('subject', None)
            assert isinstance(result, list), f"Expected list, got {type(result)}"
            assert set(result) == {'01', '02', '03'}, f"Expected {{'01', '02', '03'}}, got {set(result)}"
            
        print("✓ _get_entity_values correctly finds all subjects when None")
    except ImportError as e:
        print(f"⚠ Skipping test (missing dependencies): {e}")
        raise


def test_get_entity_values_with_none_returns_all_tasks():
    """Test that _get_entity_values returns all tasks when input is None."""
    try:
        from meegflow import MEEGFlowPipeline
        from meegflow.readers import BIDSReader
        
        with tempfile.TemporaryDirectory() as tmpdir:
            bids_root = create_mock_bids_dataset(tmpdir)
            reader = BIDSReader(bids_root)
            pipeline = MEEGFlowPipeline(reader=reader, config={})
            
            result = reader._get_entity_values('task', None)
            assert isinstance(result, list), f"Expected list, got {type(result)}"
            assert set(result) == {'rest', 'task1'}, f"Expected {{'rest', 'task1'}}, got {set(result)}"
            
        print("✓ _get_entity_values correctly finds all tasks when None")
    except ImportError as e:
        print(f"⚠ Skipping test (missing dependencies): {e}")
        raise


def test_get_entity_values_with_none_returns_all_sessions():
    """Test that _get_entity_values returns all sessions when input is None."""
    try:
        from meegflow import MEEGFlowPipeline
        from meegflow.readers import BIDSReader
        
        with tempfile.TemporaryDirectory() as tmpdir:
            bids_root = create_mock_bids_dataset(tmpdir)
            reader = BIDSReader(bids_root)
            pipeline = MEEGFlowPipeline(reader=reader, config={})
            
            result = reader._get_entity_values('session', None)
            assert isinstance(result, list), f"Expected list, got {type(result)}"
            assert set(result) == {'01', '02'}, f"Expected {{'01', '02'}}, got {set(result)}"
            
        print("✓ _get_entity_values correctly finds all sessions when None")
    except ImportError as e:
        print(f"⚠ Skipping test (missing dependencies): {e}")
        raise


def test_get_entity_values_with_none_empty_dataset():
    """Test that _get_entity_values returns [None] when no values found."""
    try:
        from meegflow import MEEGFlowPipeline
        from meegflow.readers import BIDSReader
        
        with tempfile.TemporaryDirectory() as tmpdir:
            bids_root = Path(tmpdir)
            # Create empty BIDS root
            bids_root.mkdir(exist_ok=True)
            
            reader = BIDSReader(bids_root)
            pipeline = MEEGFlowPipeline(reader=reader, config={})
            
            result = reader._get_entity_values('subject', None)
            assert result == [None], f"Expected [None] for empty dataset, got {result}"
            
        print("✓ _get_entity_values correctly returns [None] for empty dataset")
    except ImportError as e:
        print(f"⚠ Skipping test (missing dependencies): {e}")
        raise


def test_get_entity_values_invalid_type():
    """Test that _get_entity_values raises ValueError for invalid input types."""
    try:
        from meegflow import MEEGFlowPipeline
        from meegflow.readers import BIDSReader
        
        with tempfile.TemporaryDirectory() as tmpdir:
            bids_root = Path(tmpdir)
            reader = BIDSReader(bids_root)
            pipeline = MEEGFlowPipeline(reader=reader, config={})
            
            try:
                result = reader._get_entity_values('subject', 123)
                raise AssertionError("Expected ValueError for invalid type")
            except ValueError as e:
                assert "Invalid type" in str(e), f"Expected 'Invalid type' in error message, got {e}"
            
        print("✓ _get_entity_values correctly raises ValueError for invalid types")
    except ImportError as e:
        print(f"⚠ Skipping test (missing dependencies): {e}")
        raise


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running _get_entity_values Tests")
    print("=" * 60)
    print()
    
    tests = [
        test_get_entity_values_with_string,
        test_get_entity_values_with_list,
        test_get_entity_values_with_none_returns_all_subjects,
        test_get_entity_values_with_none_returns_all_tasks,
        test_get_entity_values_with_none_returns_all_sessions,
        test_get_entity_values_with_none_empty_dataset,
        test_get_entity_values_invalid_type,
    ]
    
    failed_tests = []
    skipped_tests = 0
    
    for test in tests:
        try:
            test()
        except ImportError as e:
            skipped_tests += 1
            if skipped_tests == 1:  # Only print once
                print(f"\n⚠ Skipping remaining tests (missing dependencies): {e}")
                print("  Install dependencies with: pip install -r requirements.txt")
            break  # Skip remaining tests if dependencies missing
        except AssertionError as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed_tests.append(test.__name__)
        except Exception as e:
            print(f"✗ {test.__name__} error: {e}")
            failed_tests.append(test.__name__)
    
    print()
    print("=" * 60)
    if skipped_tests > 0:
        print(f"SKIPPED: Tests skipped due to missing dependencies")
        return 0  # Don't fail if dependencies not installed
    elif failed_tests:
        print(f"FAILED: {len(failed_tests)} test(s) failed")
        for test in failed_tests:
            print(f"  - {test}")
        return 1
    else:
        print("SUCCESS: All _get_entity_values tests passed!")
        return 0


if __name__ == '__main__':
    sys.exit(run_all_tests())
