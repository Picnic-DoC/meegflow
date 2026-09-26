#!/usr/bin/env python3
"""
Test the report module functions.
"""

import sys
from pathlib import Path
import mne

# Add src to path
repo_root = Path(__file__).parent.parent
src_dir = repo_root / "src"
sys.path.insert(0, str(src_dir))

from meegflow.report import (
    collect_bad_channels_from_steps,
    create_bad_channels_topoplot,
    create_preprocessing_steps_table
)


def create_mock_info_with_montage():
    """Create a mock Info object with montage for testing."""
    # Use standard channel names
    ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2']
    ch_types = ['eeg'] * len(ch_names)
    sfreq = 250
    
    # Create info
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    
    # Add standard montage
    montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage, match_case=False)
    
    return info


def test_collect_bad_channels_from_steps():
    """Test collecting bad channels from preprocessing steps."""
    print("Testing collect_bad_channels_from_steps...")
    
    # Test with empty steps
    bad_channels = collect_bad_channels_from_steps([])
    assert bad_channels == [], "Should return empty list for empty steps"
    
    # Test with steps containing bad channels
    steps = [
        {'step': 'find_flat_channels', 'bad_channels': ['F3', 'C4']},
        {'step': 'bandpass_filter'},
        {'step': 'find_bads_maxwell', 'bad_channels': ['P4']},
    ]
    
    bad_channels = collect_bad_channels_from_steps(steps)
    assert bad_channels == ['F3', 'C4', 'P4'], f"Expected ['F3', 'C4', 'P4'], got {bad_channels}"
    
    # Test with duplicate channels
    steps_with_dupes = [
        {'step': 'find_flat_channels', 'bad_channels': ['F3', 'C4']},
        {'step': 'find_bads_maxwell', 'bad_channels': ['F3', 'P4']},
    ]
    
    bad_channels = collect_bad_channels_from_steps(steps_with_dupes)
    assert bad_channels == ['F3', 'C4', 'P4'], f"Should remove duplicates, got {bad_channels}"
    
    print("✓ collect_bad_channels_from_steps test passed")


def test_create_bad_channels_topoplot():
    """Test creating bad channels topoplot."""
    print("Testing create_bad_channels_topoplot...")
    
    info = create_mock_info_with_montage()
    
    # Test with valid bad channels
    bad_channels = ['F3', 'P4']
    fig = create_bad_channels_topoplot(info, bad_channels)
    
    assert fig is not None, "Figure should be created for valid bad channels"
    
    # Close the figure
    import matplotlib.pyplot as plt
    plt.close(fig)
    
    # Test with empty bad channels
    fig = create_bad_channels_topoplot(info, [])
    assert fig is None, "Figure should be None for empty bad channels"
    
    # Test with non-existent channels
    bad_channels = ['NonExistent1', 'NonExistent2']
    fig = create_bad_channels_topoplot(info, bad_channels)
    assert fig is None, "Figure should be None when bad channels not in EEG channels"
    
    print("✓ create_bad_channels_topoplot test passed")


def test_create_preprocessing_steps_table():
    """Test creating preprocessing steps HTML table."""
    print("Testing create_preprocessing_steps_table...")
    
    # Test with empty steps
    html = create_preprocessing_steps_table([])
    assert html == "", "Should return empty string for empty steps"
    
    # Test with valid steps
    steps = [
        {'step': 'bandpass_filter', 'l_freq': 0.5, 'h_freq': 45.0},
        {'step': 'reference', 'ref_channels': 'average'},
    ]
    
    html = create_preprocessing_steps_table(steps)
    
    # Check that HTML contains expected elements
    assert '<table class="params-table table table-hover">' in html, "HTML should contain params table"
    assert 'bandpass_filter' in html, "HTML should contain step name"
    assert '0.5' in html, "HTML should contain parameter value"
    assert 'toggleStep' in html, "HTML should contain toggle function"
    assert 'step-details' in html, "HTML should contain details div"
    # Check for two-column table structure
    assert '<td>l_freq</td>' in html, "HTML should have key in first column"
    assert '<td>0.5</td>' in html, "HTML should have value in second column"
    
    print("✓ create_preprocessing_steps_table test passed")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running Report Module Tests")
    print("=" * 60)
    print()
    
    tests = [
        test_collect_bad_channels_from_steps,
        test_create_bad_channels_topoplot,
        test_create_preprocessing_steps_table,
    ]
    
    failed_tests = []
    
    for test in tests:
        try:
            test()
        except AssertionError as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed_tests.append(test.__name__)
        except Exception as e:
            print(f"✗ {test.__name__} error: {e}")
            import traceback
            traceback.print_exc()
            failed_tests.append(test.__name__)
    
    print()
    print("=" * 60)
    if failed_tests:
        print(f"FAILED: {len(failed_tests)} test(s) failed")
        for test in failed_tests:
            print(f"  - {test}")
        return 1
    else:
        print("SUCCESS: All report module tests passed!")
        return 0


if __name__ == '__main__':
    sys.exit(run_all_tests())
