#!/usr/bin/env python3
"""
Test the HTML report enhancements: bad channels topoplot and preprocessing steps table.
"""

import sys
import tempfile
from pathlib import Path
import numpy as np
import mne
from mne_bids import BIDSPath

# Add src to path
repo_root = Path(__file__).parent.parent
src_dir = repo_root / "src"
sys.path.insert(0, str(src_dir))

from meegflow import MEEGFlowPipeline
from meegflow.readers import BIDSReader


def create_mock_raw_with_montage():
    """Create a mock Raw object with montage for testing."""
    # Create mock raw data with standard 10-20 montage
    n_channels = 10
    sfreq = 250
    n_times = 1000
    
    # Use standard channel names
    ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2']
    ch_types = ['eeg'] * n_channels
    
    # Create random data
    data = np.random.randn(n_channels, n_times) * 1e-6
    
    # Create info
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    
    # Create raw
    raw = mne.io.RawArray(data, info)
    
    # Add standard montage
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, match_case=False)
    
    # Mark some channels as bad
    raw.info['bads'] = ['F3', 'P4']
    
    return raw


def test_bad_channels_topoplot_generation():
    """Test that bad channels topoplot section is generated correctly."""
    print("Testing bad channels topoplot generation...")
    
    # Create mock data
    raw = create_mock_raw_with_montage()
    
    # Create a temporary directory for BIDS root
    with tempfile.TemporaryDirectory() as tmpdir:
        bids_root = Path(tmpdir)
        
        # Initialize pipeline
        reader = BIDSReader(bids_root)
        pipeline = MEEGFlowPipeline(reader=reader)
        
        # Create mock data dictionary with preprocessing steps that have bad channels
        data = {
            'subject': '01',
            'task': 'test',
            'session': None,
            'acquisition': None,
            'run': None,
            'raw': raw,
            'preprocessing_steps': [
                {
                    'step': 'find_bads_channels_threshold',
                    'bad_channels': ['F3', 'P4']
                }
            ]
        }
        
        # Call the HTML report generation step
        result = pipeline.run_step("generate_html_report", data, {})
        
        # Verify HTML report was created
        assert 'html_report' in result
        html_path = Path(result['html_report'])
        assert html_path.exists(), f"HTML report not created at {html_path}"
        
        # Read HTML content
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Check that bad channels section exists
        assert 'Bad Channels' in html_content, "Bad channels section not found in HTML"
        assert 'F3' in html_content or 'P4' in html_content, "Bad channel names not in HTML"
        
        print("✓ Bad channels topoplot generation test passed")


def test_preprocessing_steps_table_generation():
    """Test that preprocessing steps table is generated correctly."""
    print("Testing preprocessing steps table generation...")
    
    # Create mock data
    raw = create_mock_raw_with_montage()
    
    # Create a temporary directory for BIDS root
    with tempfile.TemporaryDirectory() as tmpdir:
        bids_root = Path(tmpdir)
        
        # Initialize pipeline
        reader = BIDSReader(bids_root)
        pipeline = MEEGFlowPipeline(reader=reader)
        
        # Create mock data dictionary with preprocessing steps
        data = {
            'subject': '01',
            'task': 'test',
            'session': None,
            'acquisition': None,
            'run': None,
            'raw': raw,
            'preprocessing_steps': [
                {
                    'step': 'load_data',
                },
                {
                    'step': 'bandpass_filter',
                    'l_freq': 0.5,
                    'h_freq': 45.0,
                    'picks': None
                },
                {
                    'step': 'reference',
                    'ref_channels': 'average'
                },
                {
                    'step': 'find_bads_channels_threshold',
                    'reject': {'eeg': 150e-6},
                    'bad_channels': ['F3', 'P4'],
                    'n_bad_channels': 2
                }
            ]
        }
        
        # Call the HTML report generation step
        result = pipeline.run_step("generate_html_report", data, {})
        
        # Verify HTML report was created
        assert 'html_report' in result
        html_path = Path(result['html_report'])
        assert html_path.exists(), f"HTML report not created at {html_path}"
        
        # Read HTML content
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Check that preprocessing steps section exists
        assert 'Preprocessing Steps' in html_content, "Preprocessing steps section not found"
        assert 'params-table' in html_content, "Params table not found"
        
        # Check for collapsible functionality
        assert 'toggleStep' in html_content, "Toggle function not found"
        assert 'step-details' in html_content, "Step details div not found"
        
        # Check that step names are present
        assert 'bandpass_filter' in html_content, "bandpass_filter step not found"
        assert 'reference' in html_content, "reference step not found"
        
        # Check that parameters are present
        assert '0.5' in html_content, "Filter parameter not found"
        assert 'average' in html_content, "Reference parameter not found"
        
        print("✓ Preprocessing steps table generation test passed")


def test_html_report_without_bad_channels():
    """Test HTML report generation when there are no bad channels."""
    print("Testing HTML report without bad channels...")
    
    # Create mock data
    raw = create_mock_raw_with_montage()
    raw.info['bads'] = []  # No bad channels
    
    # Create a temporary directory for BIDS root
    with tempfile.TemporaryDirectory() as tmpdir:
        bids_root = Path(tmpdir)
        
        # Initialize pipeline
        reader = BIDSReader(bids_root)
        pipeline = MEEGFlowPipeline(reader=reader)
        
        # Create mock data dictionary
        data = {
            'subject': '01',
            'task': 'test',
            'session': None,
            'acquisition': None,
            'run': None,
            'raw': raw,
            'preprocessing_steps': []
        }
        
        # Call the HTML report generation step (should not fail)
        result = pipeline.run_step("generate_html_report", data, {})
        
        # Verify HTML report was created
        assert 'html_report' in result
        html_path = Path(result['html_report'])
        assert html_path.exists(), f"HTML report not created at {html_path}"
        
        print("✓ HTML report without bad channels test passed")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running HTML Report Enhancement Tests")
    print("=" * 60)
    print()
    
    tests = [
        test_bad_channels_topoplot_generation,
        test_preprocessing_steps_table_generation,
        test_html_report_without_bad_channels,
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
        print("SUCCESS: All HTML report enhancement tests passed!")
        return 0


if __name__ == '__main__':
    sys.exit(run_all_tests())
