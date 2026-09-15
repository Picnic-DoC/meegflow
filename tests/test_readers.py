#!/usr/bin/env python3
"""
Tests for readers module.

This test verifies that:
1. BIDSReader works correctly and maintains backward compatibility
2. GlobReader correctly extracts variables from patterns
3. Both readers return consistent data structures
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


def create_mock_bids_dataset_with_runs(bids_root):
    """Create a minimal mock BIDS dataset that includes run entities."""
    bids_root = Path(bids_root)

    subjects = ['01', '02']
    sessions = ['01']
    tasks = ['rest']
    runs = ['01', '02']

    for sub in subjects:
        for ses in sessions:
            eeg_dir = bids_root / f'sub-{sub}' / f'ses-{ses}' / 'eeg'
            eeg_dir.mkdir(parents=True, exist_ok=True)

            for task in tasks:
                for run in runs:
                    filename = f'sub-{sub}_ses-{ses}_task-{task}_run-{run}_eeg.vhdr'
                    (eeg_dir / filename).touch()

    return bids_root


def create_mock_bids_dataset_meg(bids_root):
    """Create a minimal mock BIDS dataset holding MEG recordings."""
    bids_root = Path(bids_root)

    subjects = ['01', '02']
    sessions = ['01']
    tasks = ['rest']

    for sub in subjects:
        for ses in sessions:
            meg_dir = bids_root / f'sub-{sub}' / f'ses-{ses}' / 'meg'
            meg_dir.mkdir(parents=True, exist_ok=True)

            for task in tasks:
                filename = f'sub-{sub}_ses-{ses}_task-{task}_meg.fif'
                (meg_dir / filename).touch()

    return bids_root


def create_mock_glob_dataset(data_root):
    """Create a minimal dataset for glob pattern testing."""
    data_root = Path(data_root)

    subjects = ['01', '02', '03']
    sessions = ['01', '02']
    tasks = ['rest', 'task1']

    for sub in subjects:
        for ses in sessions:
            for task in tasks:
                file_dir = data_root / 'data' / f'sub-{sub}' / f'ses-{ses}' / 'eeg'
                file_dir.mkdir(parents=True, exist_ok=True)

                filename = f'sub-{sub}_ses-{ses}_task-{task}_eeg.vhdr'
                (file_dir / filename).touch()

    return data_root


def create_mock_glob_dataset_with_runs(data_root):
    """Create a minimal dataset for glob pattern testing that includes run entities."""
    data_root = Path(data_root)

    subjects = ['01', '02']
    sessions = ['01']
    tasks = ['rest']
    runs = ['01', '02']

    for sub in subjects:
        for ses in sessions:
            for task in tasks:
                for run in runs:
                    file_dir = data_root / 'data' / f'sub-{sub}' / f'ses-{ses}' / 'eeg'
                    file_dir.mkdir(parents=True, exist_ok=True)

                    filename = f'sub-{sub}_ses-{ses}_task-{task}_run-{run}_eeg.vhdr'
                    (file_dir / filename).touch()

    return data_root


def test_bids_reader_basic():
    """Test BIDSReader basic functionality."""
    try:
        from meegflow.readers import BIDSReader
        
        with tempfile.TemporaryDirectory() as tmpdir:
            bids_root = create_mock_bids_dataset(tmpdir)
            reader = BIDSReader(bids_root)
            
            # Test finding all recordings
            recordings = reader.find_recordings()
            
            assert len(recordings) > 0, f"Expected recordings, got {len(recordings)}"
            
            # Check structure
            for recording in recordings:
                assert 'paths' in recording, "Recording should have 'paths'"
                assert 'metadata' in recording, "Recording should have 'metadata'"
                assert 'recording_name' in recording, "Recording should have 'recording_name'"
                
                metadata = recording['metadata']
                assert 'subject' in metadata, "Metadata should have 'subject'"
                assert 'task' in metadata, "Metadata should have 'task'"
                
        print("✓ BIDSReader basic functionality works")
    except ImportError as e:
        print(f"⚠ Skipping test (missing dependencies): {e}")
        raise


def test_bids_reader_filtering():
    """Test BIDSReader filtering by subject and task."""
    try:
        from meegflow.readers import BIDSReader
        
        with tempfile.TemporaryDirectory() as tmpdir:
            bids_root = create_mock_bids_dataset(tmpdir)
            reader = BIDSReader(bids_root)
            
            # Test filtering by subject
            recordings = reader.find_recordings(subjects='01')
            
            for recording in recordings:
                assert recording['metadata']['subject'] == '01', \
                    f"Expected subject '01', got {recording['metadata']['subject']}"
            
            # Test filtering by task
            recordings = reader.find_recordings(tasks='rest')
            
            for recording in recordings:
                assert recording['metadata']['task'] == 'rest', \
                    f"Expected task 'rest', got {recording['metadata']['task']}"
                
        print("✓ BIDSReader filtering works correctly")
    except ImportError as e:
        print(f"⚠ Skipping test (missing dependencies): {e}")
        raise


def test_bids_reader_meg_dataset():
    """Test BIDSReader discovers MEG recordings without being told the datatype.

    Both the datatype and the suffix must follow the dataset: the files are
    named ``sub-01_ses-01_task-rest_meg.fif``, so a hard-coded 'eeg' suffix
    would match nothing.
    """
    try:
        from meegflow.readers import BIDSReader

        with tempfile.TemporaryDirectory() as tmpdir:
            bids_root = create_mock_bids_dataset_meg(tmpdir)
            reader = BIDSReader(bids_root)

            recordings = reader.find_recordings(extension='.fif')

            assert len(recordings) > 0, f"Expected MEG recordings, got {len(recordings)}"

            for recording in recordings:
                for path in recording['paths']:
                    assert path.datatype == 'meg', \
                        f"Expected datatype 'meg', got {path.datatype}"
                    assert path.suffix == 'meg', \
                        f"Expected suffix 'meg', got {path.suffix}"

        print("✓ BIDSReader discovers MEG recordings")
    except ImportError as e:
        print(f"⚠ Skipping test (missing dependencies): {e}")
        raise


def test_bids_reader_datatype_selection():
    """Test datatype selection on a dataset holding both EEG and MEG."""
    try:
        from meegflow.readers import BIDSReader

        with tempfile.TemporaryDirectory() as tmpdir:
            bids_root = create_mock_bids_dataset(tmpdir)
            create_mock_bids_dataset_meg(bids_root)

            # Without an explicit datatype, 'eeg' is still preferred: datasets
            # processed before this option existed keep resolving to 'eeg'.
            reader = BIDSReader(bids_root)
            recordings = reader.find_recordings(subjects='01', tasks='rest')

            assert len(recordings) > 0, "Should find EEG recordings"
            for recording in recordings:
                for path in recording['paths']:
                    assert path.datatype == 'eeg', \
                        f"Expected datatype 'eeg', got {path.datatype}"

            # An explicit datatype selects the MEG recordings instead.
            meg_reader = BIDSReader(bids_root, datatype='meg')
            meg_recordings = meg_reader.find_recordings(
                subjects='01', tasks='rest', extension='.fif'
            )

            assert len(meg_recordings) > 0, "Should find MEG recordings"
            for recording in meg_recordings:
                for path in recording['paths']:
                    assert path.datatype == 'meg', \
                        f"Expected datatype 'meg', got {path.datatype}"

        print("✓ BIDSReader datatype selection works correctly")
    except ImportError as e:
        print(f"⚠ Skipping test (missing dependencies): {e}")
        raise


def test_bids_reader_ambiguous_datatype():
    """Test that an ambiguous dataset with no EEG asks for an explicit datatype."""
    try:
        from meegflow.readers import BIDSReader

        with tempfile.TemporaryDirectory() as tmpdir:
            bids_root = create_mock_bids_dataset_meg(tmpdir)

            ieeg_dir = Path(bids_root) / 'sub-01' / 'ses-01' / 'ieeg'
            ieeg_dir.mkdir(parents=True, exist_ok=True)
            (ieeg_dir / 'sub-01_ses-01_task-rest_ieeg.vhdr').touch()

            reader = BIDSReader(bids_root)
            try:
                reader.find_recordings(subjects='01', tasks='rest', extension='.fif')
            except ValueError as e:
                assert 'datatype' in str(e), \
                    f"Error should mention the datatype argument, got: {e}"
            else:
                raise AssertionError(
                    "Expected a ValueError for a dataset with several datatypes"
                )

            # Naming the datatype resolves the ambiguity
            reader = BIDSReader(bids_root, datatype='meg')
            recordings = reader.find_recordings(
                subjects='01', tasks='rest', extension='.fif'
            )
            assert len(recordings) > 0, "Should find MEG recordings once told the datatype"

        print("✓ BIDSReader reports ambiguous datatypes")
    except ImportError as e:
        print(f"⚠ Skipping test (missing dependencies): {e}")
        raise


def test_glob_reader_variable_extraction():
    """Test GlobReader extracts variables correctly."""
    try:
        from meegflow.readers import GlobReader
        
        with tempfile.TemporaryDirectory() as tmpdir:
            data_root = create_mock_glob_dataset(tmpdir)
            
            pattern = "data/sub-{subject}/ses-{session}/eeg/sub-{subject}_ses-{session}_task-{task}_eeg.vhdr"
            reader = GlobReader(data_root, pattern)
            
            # Check pattern parsing
            assert 'subject' in reader.variable_names, "Should extract 'subject' variable"
            assert 'session' in reader.variable_names, "Should extract 'session' variable"
            assert 'task' in reader.variable_names, "Should extract 'task' variable"
            
        print("✓ GlobReader variable extraction works")
    except ImportError as e:
        print(f"⚠ Skipping test (missing dependencies): {e}")
        raise


def test_glob_reader_find_recordings():
    """Test GlobReader finds recordings correctly."""
    try:
        from meegflow.readers import GlobReader
        
        with tempfile.TemporaryDirectory() as tmpdir:
            data_root = create_mock_glob_dataset(tmpdir)
            
            pattern = "data/sub-{subject}/ses-{session}/eeg/sub-{subject}_ses-{session}_task-{task}_eeg.vhdr"
            reader = GlobReader(data_root, pattern)
            
            # Find all recordings
            recordings = reader.find_recordings()
            
            assert len(recordings) > 0, f"Expected recordings, got {len(recordings)}"
            
            # Check structure matches BIDSReader
            for recording in recordings:
                assert 'paths' in recording, "Recording should have 'paths'"
                assert 'metadata' in recording, "Recording should have 'metadata'"
                assert 'recording_name' in recording, "Recording should have 'recording_name'"
                
                metadata = recording['metadata']
                assert 'subject' in metadata, "Metadata should have 'subject'"
                assert 'task' in metadata, "Metadata should have 'task'"
                assert 'session' in metadata, "Metadata should have 'session'"
                
        print("✓ GlobReader finds recordings correctly")
    except ImportError as e:
        print(f"⚠ Skipping test (missing dependencies): {e}")
        raise


def test_glob_reader_filtering():
    """Test GlobReader filtering by criteria."""
    try:
        from meegflow.readers import GlobReader
        
        with tempfile.TemporaryDirectory() as tmpdir:
            data_root = create_mock_glob_dataset(tmpdir)
            
            pattern = "data/sub-{subject}/ses-{session}/eeg/sub-{subject}_ses-{session}_task-{task}_eeg.vhdr"
            reader = GlobReader(data_root, pattern)
            
            # Test filtering by subject
            recordings = reader.find_recordings(subjects='01')
            
            assert len(recordings) > 0, "Should find recordings for subject 01"
            
            for recording in recordings:
                assert recording['metadata']['subject'] == '01', \
                    f"Expected subject '01', got {recording['metadata']['subject']}"
            
            # Test filtering by task
            recordings = reader.find_recordings(tasks='rest')
            
            assert len(recordings) > 0, "Should find recordings for task rest"
            
            for recording in recordings:
                assert recording['metadata']['task'] == 'rest', \
                    f"Expected task 'rest', got {recording['metadata']['task']}"
                
        print("✓ GlobReader filtering works correctly")
    except ImportError as e:
        print(f"⚠ Skipping test (missing dependencies): {e}")
        raise


def test_glob_reader_regex_pattern():
    """Test GlobReader creates correct regex pattern, including raw * wildcards."""
    try:
        from meegflow.readers import GlobReader

        # Named variables only
        pattern = "data/sub-{subject}/task-{task}.vhdr"
        reader = GlobReader("/tmp", pattern)
        match = reader.regex_pattern.match("data/sub-01/task-rest.vhdr")
        assert match is not None, "Regex should match named-variable pattern"
        assert match.group('subject') == '01'
        assert match.group('task') == 'rest'

        # Mixed: named variable + raw * wildcard (e.g. session position not a grouping key)
        pattern_wild = "data/sub-{subject}/ses-*/eeg/sub-{subject}_task-{task}_eeg.vhdr"
        reader_wild = GlobReader("/tmp", pattern_wild)
        assert 'subject' in reader_wild.variable_names
        assert 'task' in reader_wild.variable_names

        match2 = reader_wild.regex_pattern.match("data/sub-01/ses-02/eeg/sub-01_task-rest_eeg.vhdr")
        assert match2 is not None, "Regex should match path with * wildcard"
        assert match2.group('subject') == '01'
        assert match2.group('task') == 'rest'

        print("✓ GlobReader regex pattern works correctly")
    except ImportError as e:
        print(f"⚠ Skipping test (missing dependencies): {e}")
        raise


def test_readers_consistent_interface():
    """Test that BIDSReader and GlobReader return consistent data structures."""
    try:
        from meegflow.readers import BIDSReader, GlobReader
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test BIDSReader
            bids_root = create_mock_bids_dataset(tmpdir)
            bids_reader = BIDSReader(bids_root)
            bids_recordings = bids_reader.find_recordings(subjects='01', tasks='rest')
            
            # Test GlobReader with a different dataset
            glob_root = Path(tmpdir) / 'glob_test'
            glob_root.mkdir()
            create_mock_glob_dataset(glob_root)
            
            pattern = "data/sub-{subject}/ses-{session}/eeg/sub-{subject}_ses-{session}_task-{task}_eeg.vhdr"
            glob_reader = GlobReader(glob_root, pattern)
            glob_recordings = glob_reader.find_recordings(subjects='01', tasks='rest')
            
            # Both should return non-empty lists
            assert len(bids_recordings) > 0, "BIDSReader should find recordings"
            assert len(glob_recordings) > 0, "GlobReader should find recordings"
        
            # Check both have the same structure
            for recording in bids_recordings + glob_recordings:
                assert isinstance(recording, dict), "Recording should be a dict"
                assert 'paths' in recording
                assert 'metadata' in recording
                assert 'recording_name' in recording
                assert isinstance(recording['paths'], list)
                assert isinstance(recording['metadata'], dict)
                assert isinstance(recording['recording_name'], str)
        
        print("✓ Readers have consistent interface")
    except ImportError as e:
        print(f"⚠ Skipping test (missing dependencies): {e}")
        raise


def test_bids_reader_run_filtering():
    """Test BIDSReader run filtering: selected runs are concatenated into one recording."""
    try:
        from meegflow.readers import BIDSReader

        with tempfile.TemporaryDirectory() as tmpdir:
            bids_root = create_mock_bids_dataset_with_runs(tmpdir)
            reader = BIDSReader(bids_root)

            # No filter: all runs included, each (subject, session, task) is one recording
            all_recordings = reader.find_recordings()
            assert len(all_recordings) > 0, "Expected recordings"
            # Each recording should contain paths for both runs
            for recording in all_recordings:
                assert len(recording['paths']) == 2, \
                    f"Expected 2 run paths per recording, got {len(recording['paths'])}"
                assert 'run' not in recording['metadata'], \
                    "run should not appear in metadata — runs are concatenated"

            # Filter to one run: each (subject, session, task) still one recording, but with 1 path
            recordings_run01 = reader.find_recordings(runs='01')
            assert len(recordings_run01) == len(all_recordings), \
                "Number of recordings should be the same regardless of run filter"
            for recording in recordings_run01:
                assert len(recording['paths']) == 1, \
                    f"Expected 1 path when filtering to run 01, got {len(recording['paths'])}"
                assert all(p.run == '01' for p in recording['paths']), \
                    "All paths should be from run 01"

            # Filter to both runs: equivalent to no filter
            recordings_both = reader.find_recordings(runs=['01', '02'])
            for recording in recordings_both:
                assert len(recording['paths']) == 2, \
                    f"Expected 2 paths when selecting both runs, got {len(recording['paths'])}"

        print("✓ BIDSReader run filtering works correctly")
    except ImportError as e:
        print(f"⚠ Skipping test (missing dependencies): {e}")
        raise


def test_glob_reader_wildcard_concatenation():
    """Test that * wildcards concatenate files into one recording per named-variable combination."""
    try:
        from meegflow.readers import GlobReader

        with tempfile.TemporaryDirectory() as tmpdir:
            data_root = create_mock_glob_dataset_with_runs(tmpdir)

            # {subject} and {task} are grouping keys; run position uses * so all runs concatenate
            pattern = "data/sub-{subject}/ses-{session}/eeg/sub-{subject}_ses-{session}_task-{task}_run-*_eeg.vhdr"
            reader = GlobReader(data_root, pattern)

            assert 'run' not in reader.variable_names, "'run' should not be extracted (it's a * wildcard)"

            recordings = reader.find_recordings()
            assert len(recordings) > 0, "Expected recordings"

            # Each (subject, session, task) is one recording containing both run files
            for recording in recordings:
                assert len(recording['paths']) == 2, \
                    f"Expected both run files concatenated into one recording, got {len(recording['paths'])}"

        print("✓ GlobReader * wildcard concatenation works correctly")
    except ImportError as e:
        print(f"⚠ Skipping test (missing dependencies): {e}")
        raise


def test_glob_reader_named_run_filtering():
    """Test that {run} as a named variable creates separate recordings, filterable with --runs."""
    try:
        from meegflow.readers import GlobReader

        with tempfile.TemporaryDirectory() as tmpdir:
            data_root = create_mock_glob_dataset_with_runs(tmpdir)

            # {run} is a named variable → each run is a separate recording
            pattern = "data/sub-{subject}/ses-{session}/eeg/sub-{subject}_ses-{session}_task-{task}_run-{run}_eeg.vhdr"
            reader = GlobReader(data_root, pattern)

            assert 'run' in reader.variable_names, "Should extract 'run' as a named variable"

            # No filter: one recording per (subject, session, task, run) combination
            all_recordings = reader.find_recordings()
            assert len(all_recordings) > 0, "Expected recordings"
            for recording in all_recordings:
                assert len(recording['paths']) == 1, \
                    f"Each named-run recording should have exactly 1 path, got {len(recording['paths'])}"

            # Filter to run '01': only run-01 recordings remain
            recordings_run01 = reader.find_recordings(runs='01')
            assert len(recordings_run01) == len(all_recordings) // 2, \
                "Filtering to one run should halve the number of recordings"
            for recording in recordings_run01:
                assert recording['metadata']['run'] == '01', \
                    f"Expected run '01', got {recording['metadata']['run']}"

        print("✓ GlobReader named {run} variable filtering works correctly")
    except ImportError as e:
        print(f"⚠ Skipping test (missing dependencies): {e}")
        raise


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running Readers Tests")
    print("=" * 60)
    print()
    
    tests = [
        test_bids_reader_basic,
        test_bids_reader_filtering,
        test_bids_reader_run_filtering,
        test_bids_reader_meg_dataset,
        test_bids_reader_datatype_selection,
        test_bids_reader_ambiguous_datatype,
        test_glob_reader_variable_extraction,
        test_glob_reader_find_recordings,
        test_glob_reader_filtering,
        test_glob_reader_wildcard_concatenation,
        test_glob_reader_named_run_filtering,
        test_glob_reader_regex_pattern,
        test_readers_consistent_interface,
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
            import traceback
            traceback.print_exc()
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
        print("SUCCESS: All readers tests passed!")
        return 0


if __name__ == '__main__':
    sys.exit(run_all_tests())
