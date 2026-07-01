#!/usr/bin/env python3
"""
Functional tests for _step_find_flat_channels with synthetic data.

These tests verify that the find_flat_channels step correctly identifies
flat channels based on variance threshold, handles excluded channels,
and works with different channel types.
"""

import sys
import numpy as np
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

try:
    import mne
    from mne.utils import logger
    from meegflow import MEEGFlowPipeline
    from meegflow.readers import BIDSReader
    MNE_AVAILABLE = True
except ImportError as e:
    MNE_AVAILABLE = False
    print(f"Warning: Could not import required modules: {e}")


def create_test_raw_with_flat_channels():
    """
    Create test raw data with some flat channels for testing.
    
    Returns
    -------
    raw : mne.io.RawArray
        Raw data with channels at different variance levels
    flat_channel_names : list
        Names of channels that should be detected as flat
    """
    n_channels = 10
    n_times = 1000
    sfreq = 100.0
    
    # Create data with normal variance for most channels
    data = np.random.randn(n_channels, n_times) * 1e-5  # Normal EEG variance
    
    # Make channels 2 and 5 completely flat (zero variance)
    data[2, :] = 1e-6  # Constant value (essentially flat)
    data[5, :] = -2e-6  # Another constant value
    
    # Make channel 7 very low variance (should be detected as flat)
    data[7, :] = np.random.randn(n_times) * 1e-13  # Very low variance
    
    info = mne.create_info(
        ch_names=[f'EEG{i:03d}' for i in range(n_channels)],
        sfreq=sfreq,
        ch_types='eeg'
    )
    
    raw = mne.io.RawArray(data, info, verbose=False)
    
    flat_channel_names = ['EEG002', 'EEG005', 'EEG007']
    
    return raw, flat_channel_names


def test_find_flat_channels_basic():
    """Test basic functionality of find_flat_channels."""
    if not MNE_AVAILABLE:
        print("✗ Skipped: MNE not available")
        return False
    
    print("Testing find_flat_channels basic functionality...")
    
    try:
        raw, expected_flat = create_test_raw_with_flat_channels()
        
        # Initialize pipeline
        pipeline = MEEGFlowPipeline(reader=BIDSReader('/tmp'), config={})
        
        # Prepare data dict
        data = {
            'raw': raw,
            'preprocessing_steps': []
        }
        
        # Run find_flat_channels step
        step_config = {
            'threshold': 1e-12  # Default threshold
        }
        
        result = pipeline.run_step("find_flat_channels", data, step_config)
        
        # Check that flat channels were detected
        detected_flat = result['preprocessing_steps'][-1]['bad_channels']
        
        print(f"  Expected flat channels: {expected_flat}")
        print(f"  Detected flat channels: {detected_flat}")
        
        # Verify all expected flat channels are detected
        for ch in expected_flat:
            if ch not in detected_flat:
                print(f"  ✗ Expected flat channel {ch} not detected")
                return False
        
        # Verify they were added to info['bads']
        for ch in expected_flat:
            if ch not in result['raw'].info['bads']:
                print(f"  ✗ Flat channel {ch} not added to info['bads']")
                return False
        
        # Verify preprocessing_steps was updated
        step_info = result['preprocessing_steps'][-1]
        assert step_info['step'] == 'find_flat_channels', "Step name incorrect"
        assert step_info['instance'] == 'raw', "Instance incorrect"
        assert step_info['threshold'] == 1e-12, "Threshold not recorded"
        assert step_info['n_bad_channels'] == len(detected_flat), "Bad channel count incorrect"
        
        print("  ✓ find_flat_channels basic functionality works correctly")
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_find_flat_channels_with_custom_threshold():
    """Test find_flat_channels with custom threshold."""
    if not MNE_AVAILABLE:
        print("✗ Skipped: MNE not available")
        return False
    
    print("Testing find_flat_channels with custom threshold...")
    
    try:
        raw, _ = create_test_raw_with_flat_channels()
        
        pipeline = MEEGFlowPipeline(reader=BIDSReader('/tmp'), config={})
        
        data = {
            'raw': raw,
            'preprocessing_steps': []
        }
        
        # Use a very high threshold - should detect more channels
        step_config = {
            'threshold': 1e-9  # Higher threshold
        }
        
        result = pipeline.run_step("find_flat_channels", data, step_config)
        detected_flat = result['preprocessing_steps'][-1]['bad_channels']
        
        print(f"  With high threshold (1e-9): {len(detected_flat)} flat channels detected")
        
        # Should detect at least the completely flat channels
        assert len(detected_flat) >= 2, "Should detect at least 2 flat channels"
        
        # Now test with very low threshold - should detect fewer channels
        raw2, _ = create_test_raw_with_flat_channels()
        data2 = {
            'raw': raw2,
            'preprocessing_steps': []
        }
        
        step_config2 = {
            'threshold': 1e-15  # Very low threshold
        }
        
        result2 = pipeline.run_step("find_flat_channels", data2, step_config2)
        detected_flat2 = result2['preprocessing_steps'][-1]['bad_channels']
        
        print(f"  With low threshold (1e-15): {len(detected_flat2)} flat channels detected")
        
        # With lower threshold, should detect fewer channels
        assert len(detected_flat2) <= len(detected_flat), "Lower threshold should detect fewer channels"
        
        print("  ✓ Custom threshold works correctly")
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_find_flat_channels_no_flat():
    """Test find_flat_channels when no channels are flat."""
    if not MNE_AVAILABLE:
        print("✗ Skipped: MNE not available")
        return False
    
    print("Testing find_flat_channels with no flat channels...")
    
    try:
        # Create data with all channels having normal variance
        n_channels = 5
        n_times = 1000
        sfreq = 100.0
        
        data = np.random.randn(n_channels, n_times) * 1e-5
        
        info = mne.create_info(
            ch_names=[f'EEG{i:03d}' for i in range(n_channels)],
            sfreq=sfreq,
            ch_types='eeg'
        )
        
        raw = mne.io.RawArray(data, info, verbose=False)
        
        pipeline = MEEGFlowPipeline(reader=BIDSReader('/tmp'), config={})
        
        data_dict = {
            'raw': raw,
            'preprocessing_steps': []
        }
        
        step_config = {'threshold': 1e-12}
        
        result = pipeline.run_step("find_flat_channels", data_dict, step_config)
        detected_flat = result['preprocessing_steps'][-1]['bad_channels']
        
        print(f"  Detected flat channels: {detected_flat}")
        
        assert len(detected_flat) == 0, "Should not detect any flat channels"
        assert len(result['raw'].info['bads']) == 0, "info['bads'] should be empty"
        
        print("  ✓ Correctly handles data with no flat channels")
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_find_flat_channels_all_flat():
    """Test find_flat_channels when all channels are flat."""
    if not MNE_AVAILABLE:
        print("✗ Skipped: MNE not available")
        return False
    
    print("Testing find_flat_channels with all channels flat...")
    
    try:
        n_channels = 5
        n_times = 1000
        sfreq = 100.0
        
        # Create all flat channels
        data = np.ones((n_channels, n_times)) * 1e-6
        
        info = mne.create_info(
            ch_names=[f'EEG{i:03d}' for i in range(n_channels)],
            sfreq=sfreq,
            ch_types='eeg'
        )
        
        raw = mne.io.RawArray(data, info, verbose=False)
        
        pipeline = MEEGFlowPipeline(reader=BIDSReader('/tmp'), config={})
        
        data_dict = {
            'raw': raw,
            'preprocessing_steps': []
        }
        
        step_config = {'threshold': 1e-12}
        
        result = pipeline.run_step("find_flat_channels", data_dict, step_config)
        detected_flat = result['preprocessing_steps'][-1]['bad_channels']
        
        print(f"  Detected flat channels: {detected_flat}")
        
        assert len(detected_flat) == n_channels, "Should detect all channels as flat"
        assert len(result['raw'].info['bads']) == n_channels, "All channels should be in info['bads']"
        
        print("  ✓ Correctly handles all flat channels")
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_find_flat_channels_with_excluded_channels():
    """Test find_flat_channels with excluded channels."""
    if not MNE_AVAILABLE:
        print("✗ Skipped: MNE not available")
        return False
    
    print("Testing find_flat_channels with excluded channels...")
    
    try:
        raw, expected_flat = create_test_raw_with_flat_channels()
        
        pipeline = MEEGFlowPipeline(reader=BIDSReader('/tmp'), config={})
        
        data = {
            'raw': raw,
            'preprocessing_steps': []
        }
        
        # Exclude one of the flat channels
        excluded = ['EEG002']
        
        step_config = {
            'threshold': 1e-12,
            'excluded_channels': excluded
        }
        
        result = pipeline.run_step("find_flat_channels", data, step_config)
        detected_flat = result['preprocessing_steps'][-1]['bad_channels']
        
        print(f"  Excluded channels: {excluded}")
        print(f"  Detected flat channels: {detected_flat}")
        
        # EEG002 should not be in detected flat channels (it was excluded)
        assert 'EEG002' not in detected_flat, "Excluded channel should not be detected"
        
        # But other flat channels should still be detected
        assert 'EEG005' in detected_flat or 'EEG007' in detected_flat, \
            "Other flat channels should still be detected"
        
        # Verify excluded_channels is recorded in preprocessing_steps
        step_info = result['preprocessing_steps'][-1]
        assert step_info['excluded_channels'] == excluded, "excluded_channels not recorded correctly"
        
        print("  ✓ Excluded channels work correctly")
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_find_flat_channels_with_picks():
    """Test find_flat_channels with specific channel picks."""
    if not MNE_AVAILABLE:
        print("✗ Skipped: MNE not available")
        return False
    
    print("Testing find_flat_channels with channel picks...")
    
    try:
        n_channels = 10
        n_times = 1000
        sfreq = 100.0
        
        # Create mixed channel types
        ch_names = [f'EEG{i:03d}' for i in range(8)] + ['EOG001', 'EOG002']
        ch_types = ['eeg'] * 8 + ['eog'] * 2
        
        data = np.random.randn(n_channels, n_times) * 1e-5
        
        # Make one EEG channel flat and one EOG channel flat
        data[2, :] = 1e-6  # EEG002 flat
        data[8, :] = 1e-6  # EOG001 flat
        
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        raw = mne.io.RawArray(data, info, verbose=False)
        
        pipeline = MEEGFlowPipeline(reader=BIDSReader('/tmp'), config={})
        
        data_dict = {
            'raw': raw,
            'preprocessing_steps': []
        }
        
        # Only check EEG channels
        step_config = {
            'threshold': 1e-12,
            'picks': ['eeg']
        }
        
        result = pipeline.run_step("find_flat_channels", data_dict, step_config)
        detected_flat = result['preprocessing_steps'][-1]['bad_channels']
        
        print(f"  Detected flat channels (EEG only): {detected_flat}")
        
        # Should detect flat EEG channel but not EOG
        assert 'EEG002' in detected_flat, "Flat EEG channel should be detected"
        assert 'EOG001' not in detected_flat, "EOG channel should not be checked"
        
        # Verify picks is recorded in preprocessing_steps
        step_info = result['preprocessing_steps'][-1]
        assert step_info['picks'] == ['eeg'], "picks not recorded correctly"
        
        print("  ✓ Channel picks work correctly")
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_find_flat_channels_no_duplicate_bads():
    """Test that find_flat_channels doesn't add duplicates to info['bads']."""
    if not MNE_AVAILABLE:
        print("✗ Skipped: MNE not available")
        return False
    
    print("Testing find_flat_channels doesn't add duplicate bads...")
    
    try:
        raw, expected_flat = create_test_raw_with_flat_channels()
        
        # Pre-mark one channel as bad
        raw.info['bads'] = ['EEG002']
        
        pipeline = MEEGFlowPipeline(reader=BIDSReader('/tmp'), config={})
        
        data = {
            'raw': raw,
            'preprocessing_steps': []
        }
        
        step_config = {'threshold': 1e-12}
        
        result = pipeline.run_step("find_flat_channels", data, step_config)
        
        # Check that EEG002 appears only once in bads
        bads_count = result['raw'].info['bads'].count('EEG002')
        
        print(f"  info['bads']: {result['raw'].info['bads']}")
        print(f"  EEG002 count in bads: {bads_count}")
        
        assert bads_count == 1, "Channel should not be duplicated in info['bads']"
        
        print("  ✓ No duplicate bads added")
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_find_flat_channels_missing_raw():
    """Test that find_flat_channels raises error when 'raw' is missing."""
    if not MNE_AVAILABLE:
        print("✗ Skipped: MNE not available")
        return False
    
    print("Testing find_flat_channels error handling for missing 'raw'...")
    
    try:
        pipeline = MEEGFlowPipeline(reader=BIDSReader('/tmp'), config={})
        
        # Data without 'raw'
        data = {
            'preprocessing_steps': []
        }
        
        step_config = {'threshold': 1e-12}
        
        try:
            pipeline.run_step("find_flat_channels", data, step_config)
            print("  ✗ Should have raised ValueError")
            return False
        except ValueError as e:
            if "requires 'raw'" in str(e):
                print(f"  ✓ Correctly raises ValueError: {e}")
                return True
            else:
                print(f"  ✗ Wrong error message: {e}")
                return False
        
    except Exception as e:
        print(f"  ✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all functional tests for find_flat_channels."""
    print("=" * 70)
    print("Running find_flat_channels Functional Tests with Synthetic Data")
    print("=" * 70)
    print()
    
    if not MNE_AVAILABLE:
        print("MNE is not available. Please install requirements.")
        return 1
    
    print(f"MNE version: {mne.__version__}")
    print()
    
    tests = [
        test_find_flat_channels_basic,
        test_find_flat_channels_with_custom_threshold,
        test_find_flat_channels_no_flat,
        test_find_flat_channels_all_flat,
        test_find_flat_channels_with_excluded_channels,
        test_find_flat_channels_with_picks,
        test_find_flat_channels_no_duplicate_bads,
        test_find_flat_channels_missing_raw,
    ]
    
    failed_tests = []
    
    for test in tests:
        try:
            if not test():
                failed_tests.append(test.__name__)
        except Exception as e:
            print(f"✗ {test.__name__} raised exception: {e}")
            import traceback
            traceback.print_exc()
            failed_tests.append(test.__name__)
        print()
    
    print("=" * 70)
    if failed_tests:
        print(f"FAILED: {len(failed_tests)} test(s) failed")
        for test in failed_tests:
            print(f"  - {test}")
        return 1
    else:
        print("SUCCESS: All find_flat_channels functional tests passed!")
        print()
        print("Summary:")
        print("  ✓ Basic flat channel detection works correctly")
        print("  ✓ Custom thresholds are respected")
        print("  ✓ Edge cases (no flat, all flat) handled properly")
        print("  ✓ Excluded channels feature works")
        print("  ✓ Channel picks (e.g., EEG only) work")
        print("  ✓ No duplicate bads added")
        print("  ✓ Error handling for missing data works")
        return 0


if __name__ == '__main__':
    sys.exit(run_all_tests())
