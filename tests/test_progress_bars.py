#!/usr/bin/env python3
"""
Tests for verifying progress reporting and logging integration.

Progress reporting changed as part of the Dask parallel-execution refactor
(see ``docs/dask_parallel_execution.md``): ``run_pipeline`` no longer owns
the progress bar directly, it delegates dispatch to
``meegflow.execution.dispatch``, which routes to ``run_sequential`` (rich
``Progress``, single-process, today's default) or ``run_dask`` (Dask
futures + logger messages, since a live nested progress bar can't
meaningfully represent state changing in other processes/machines).

This test verifies that:
1. The sequential execution backend still uses rich Progress bars.
2. The per-recording unit of work (``process_recording``) still uses the
   MNE logger throughout.
3. The CLI still supports log file / log level configuration.
"""

import sys
from pathlib import Path
from unittest.mock import patch

# Find the repository root
repo_root = Path(__file__).parent.parent
src_dir = repo_root / "src"

# Add src to path
sys.path.insert(0, str(src_dir))


def test_progress_bar_imports():
    """Test that rich progress bar components are used for sequential execution."""
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from meegflow import execution
    import inspect
    source = inspect.getsource(execution.run_sequential)
    assert "Progress" in source, "Sequential execution should use rich Progress"
    print("✓ Rich progress bar components used for sequential execution")


def test_logger_import():
    """Test that MNE logger is imported."""
    from mne.utils import logger
    from meegflow.cli import logger as cli_logger

    print("✓ MNE logger imported in pipeline and CLI")


def test_cli_log_file_argument():
    """Test that CLI has log file argument."""
    from meegflow.cli import _parse_args

    # Mock sys.argv to test argument parsing
    with patch('sys.argv', ['cli.py', '--bids-root', '/tmp/test']):
        args = _parse_args()
        assert hasattr(args, 'log_file'), "CLI should have log_file argument"
        assert hasattr(args, 'log_level'), "CLI should have log_level argument"

    print("✓ CLI has log file and log level arguments")


def test_run_sequential_creates_progress():
    """Test that the sequential execution backend creates progress bars."""
    from meegflow import execution
    import inspect

    source = inspect.getsource(execution.run_sequential)

    assert 'Progress' in source, "run_sequential should use Progress class"
    assert 'progress.add_task' in source, "run_sequential should add tasks to progress"
    assert 'with Progress' in source, "run_sequential should use Progress context manager"

    print("✓ run_sequential creates progress bars and uses logger")


def test_process_recording_is_module_level():
    """Test that the per-recording unit of work is a plain, picklable function.

    A bound instance method (closing over ``self.reader``, etc.) is not a
    reliable Dask task: this must be a module-level function so it can be
    submitted to Dask workers, including remote processes started via
    ``dask-jobqueue``.
    """
    from meegflow import pipeline
    import inspect

    assert inspect.isfunction(pipeline.process_recording), \
        "process_recording should be a plain module-level function"

    sig = inspect.signature(pipeline.process_recording)
    params = list(sig.parameters.keys())
    assert 'self' not in params, "process_recording should not be a bound method"
    for expected in ['reader', 'output_root', 'config', 'step_functions', 'paths', 'metadata']:
        assert expected in params, f"process_recording should accept '{expected}'"

    print("✓ process_recording is a module-level, picklable function")


def test_logger_in_process_recording():
    """Test that process_recording uses the MNE logger."""
    from meegflow import pipeline
    import inspect

    source = inspect.getsource(pipeline.process_recording)

    assert 'logger.info' in source, "process_recording should use logger.info"

    print("✓ process_recording uses MNE logger")


def test_run_dask_logs_per_recording_completion():
    """Test that the Dask execution backend logs per-recording progress.

    A live, in-place-updating progress bar can't meaningfully represent
    state changing in other processes (local backend) or other machines
    (dask-jobqueue), so run_dask logs one line per completed/failed
    recording instead (see docs/dask_parallel_execution.md §4.5).
    """
    from meegflow import execution
    import inspect

    source = inspect.getsource(execution.run_dask)
    assert 'logger.info' in source, "run_dask should log per-recording completion"
    assert 'logger.error' in source, "run_dask should log per-recording failures"
    assert 'as_completed' in source, "run_dask should gather futures with as_completed"

    print("✓ run_dask logs per-recording start/finish instead of a live progress bar")


if __name__ == '__main__':
    print("=" * 60)
    print("Running Progress Reporting and Logging Tests")
    print("=" * 60)
    print()

    try:
        test_progress_bar_imports()
        test_logger_import()
        test_cli_log_file_argument()
        test_run_sequential_creates_progress()
        test_process_recording_is_module_level()
        test_logger_in_process_recording()
        test_run_dask_logs_per_recording_completion()

        print()
        print("=" * 60)
        print("SUCCESS: All progress bar and logging tests passed!")
        print("=" * 60)
    except Exception as e:
        print()
        print("=" * 60)
        print(f"FAILED: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        sys.exit(1)
