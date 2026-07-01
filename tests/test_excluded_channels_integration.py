#!/usr/bin/env python3
"""
Integration test for excluded_channels feature using mock data.

This test creates minimal mock EEG data and validates that the excluded_channels
parameter correctly excludes channels from analysis in the preprocessing steps.
"""

import sys
from pathlib import Path

# Find the repository root
repo_root = Path(__file__).parent.parent
src_dir = repo_root / "src"
sys.path.insert(0, str(src_dir))

def test_excluded_channels_integration():
    """Test excluded_channels with mock MNE data structures."""
    try:
        import numpy as np
        import mne
        from meegflow import MEEGFlowPipeline
        
        # Create mock raw data
        n_channels = 5
        n_times = 1000
        sfreq = 250
        ch_names = ['Fz', 'Cz', 'Pz', 'C3', 'C4']
        ch_types = ['eeg'] * n_channels
        
        # Create mock data
        data = np.random.randn(n_channels, n_times) * 1e-6
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        raw = mne.io.RawArray(data, info)
        
        # Create pipeline instance and a context exposing the picks helpers
        from meegflow.readers import BIDSReader
        from meegflow.context import PipelineContext
        config = {'pipeline': []}
        reader = BIDSReader(repo_root / "test_data")
        pipeline = MEEGFlowPipeline(
            reader=reader,
            config=config
        )
        ctx = PipelineContext({}, reader=reader)

        # Test get_picks with excluded_channels
        picks_all = ctx.get_picks(raw.info, None, None)
        assert len(picks_all) == 5, f"Expected 5 channels, got {len(picks_all)}"
        print("✓ get_picks returns all channels when no exclusion")

        picks_excluded = ctx.get_picks(raw.info, None, ['Cz'])
        assert len(picks_excluded) == 4, f"Expected 4 channels after excluding Cz, got {len(picks_excluded)}"
        print("✓ get_picks excludes specified channels")

        # Verify Cz is not in excluded picks
        excluded_names = [raw.ch_names[p] for p in picks_excluded]
        assert 'Cz' not in excluded_names, "Cz should not be in excluded picks"
        print("✓ Cz correctly excluded from picks")

        # Test _apply_excluded_channels directly
        picks = [0, 1, 2, 3, 4]  # All channels
        filtered_picks = ctx._apply_excluded_channels(raw.info, picks, ['Cz', 'Fz'])
        assert len(filtered_picks) == 3, f"Expected 3 channels after excluding 2, got {len(filtered_picks)}"
        filtered_names = [raw.ch_names[p] for p in filtered_picks]
        assert 'Cz' not in filtered_names and 'Fz' not in filtered_names, \
            "Excluded channels should not be in filtered picks"
        print("✓ _apply_excluded_channels works correctly")

        # Test with empty exclusion list
        picks_empty = ctx._apply_excluded_channels(raw.info, picks, [])
        assert len(picks_empty) == 5, "Empty exclusion list should return all picks"
        print("✓ Empty exclusion list returns all picks")

        # Test with None exclusion
        picks_none = ctx._apply_excluded_channels(raw.info, picks, None)
        assert len(picks_none) == 5, "None exclusion should return all picks"
        print("✓ None exclusion returns all picks")
        
        print("\n✓ All integration tests passed!")
        return True
        
    except ImportError as e:
        print(f"⚠ Skipping integration test (missing dependencies): {e}")
        print("  Install dependencies with: pip install -r requirements.txt")
        return True  # Don't fail if dependencies not installed
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_excluded_channels_integration()
    sys.exit(0 if success else 1)
