#!/usr/bin/env python3
"""
Tests for excluded_channels feature in the MEEG preprocessing pipeline.

This test file validates that the excluded_channels parameter works correctly
across all steps that support it, allowing channels (like Cz) to be excluded
from analysis to avoid reference problems.

After the steps/context refactor:
- ``get_picks`` / ``_apply_excluded_channels`` live on PipelineContext
  (``src/meegflow/context.py``).
- each step is a module-level ``def <name>(data, step_config)`` function in
  ``src/meegflow/steps/`` and calls ``data.get_picks(...)``.
"""

import sys
import ast
from pathlib import Path

# Find the repository root
repo_root = Path(__file__).parent.parent
src_dir = repo_root / "src"

CONTEXT_FILE = src_dir / "meegflow" / "context.py"
STEPS_DIR = src_dir / "meegflow" / "steps"


def _context_code():
    return CONTEXT_FILE.read_text()


def _slice_context_method(name):
    """Return the source of a 4-space-indented method ``name`` in context.py."""
    code = _context_code()
    pattern = f"def {name}"
    start = code.find(pattern)
    assert start != -1, f"{name} not found in context.py"
    nxt = code.find("\n    def ", start + len(pattern))
    return code[start:(nxt if nxt != -1 else len(code))]


def _slice_step(name):
    """Return the source of the module-level step function ``name``."""
    pat = f"def {name}(data, step_config):"
    for p in sorted(STEPS_DIR.glob("*.py")):
        code = p.read_text()
        i = code.find(pat)
        if i == -1:
            continue
        ends = [x for x in (code.find("\n@register(", i + len(pat)),
                             code.find("\ndef ", i + len(pat))) if x != -1]
        return code[i:(min(ends) if ends else len(code))]
    raise AssertionError(f"step function {name} not found in steps/")


def test_apply_excluded_channels_exists():
    """Test that the _apply_excluded_channels helper exists on the context."""
    assert "def _apply_excluded_channels" in _context_code(), \
        "_apply_excluded_channels function not found"
    print("✓ _apply_excluded_channels helper function exists")


def test_get_picks_has_excluded_channels_param():
    """Test that get_picks has an excluded_channels parameter."""
    tree = ast.parse(_context_code())

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == 'get_picks':
            arg_names = [arg.arg for arg in node.args.args]
            assert 'excluded_channels' in arg_names, "get_picks missing excluded_channels parameter"
            print("✓ get_picks has excluded_channels parameter")
            return

    raise AssertionError("get_picks function not found")


def test_steps_support_excluded_channels():
    """Test that appropriate steps support excluded_channels parameter."""
    steps_with_exclusion = [
        'bandpass_filter',
        'notch_filter',
        'interpolate_bad_channels',
        'drop_bad_channels',
        'ica',
        'find_flat_channels',
        'find_bads_channels_threshold',
        'find_bads_channels_variance',
        'find_bads_channels_high_frequency',
        'find_bads_epochs_threshold',
    ]

    for step_name in steps_with_exclusion:
        function_code = _slice_step(step_name)
        assert "excluded_channels" in function_code, \
            f"Step {step_name} does not handle excluded_channels parameter"
        print(f"✓ {step_name} supports excluded_channels")


def test_steps_pass_excluded_channels_to_get_picks():
    """Test that steps pass excluded_channels to get_picks."""
    steps_using_get_picks = [
        'bandpass_filter',
        'notch_filter',
        'ica',
        'find_flat_channels',
        'find_bads_channels_threshold',
        'find_bads_channels_variance',
        'find_bads_channels_high_frequency',
        'find_bads_epochs_threshold',
    ]

    for step_name in steps_using_get_picks:
        function_code = _slice_step(step_name)

        assert "get_picks(" in function_code, \
            f"Step {step_name} does not call get_picks"

        get_picks_calls = [line for line in function_code.split('\n')
                           if 'get_picks(' in line]
        has_excluded = any('excluded_channels' in call for call in get_picks_calls)
        assert has_excluded, \
            f"Step {step_name} does not pass excluded_channels to get_picks"

        print(f"✓ {step_name} passes excluded_channels to get_picks")


def test_preprocessing_steps_report_excluded_channels():
    """Test that steps include excluded_channels in their preprocessing_steps report."""
    steps_to_check = [
        'bandpass_filter',
        'notch_filter',
        'interpolate_bad_channels',
        'ica',
        'find_flat_channels',
        'find_bads_channels_threshold',
        'find_bads_channels_variance',
        'find_bads_channels_high_frequency',
        'find_bads_epochs_threshold',
    ]

    for step_name in steps_to_check:
        function_code = _slice_step(step_name)
        if "preprocessing_steps" in function_code:
            assert "'excluded_channels'" in function_code, \
                f"Step {step_name} does not report excluded_channels in preprocessing_steps"
        print(f"✓ {step_name} reports excluded_channels in preprocessing_steps")


def test_apply_excluded_channels_implementation():
    """Test the implementation of _apply_excluded_channels."""
    function_code = _slice_context_method("_apply_excluded_channels")

    assert "if excluded_channels is None" in function_code, \
        "_apply_excluded_channels should handle None case"
    assert "return picks" in function_code, \
        "_apply_excluded_channels should return picks"
    assert "filtered_picks" in function_code or "filter" in function_code.lower(), \
        "_apply_excluded_channels should filter picks"

    print("✓ _apply_excluded_channels implementation looks correct")


def test_excluded_channels_documentation():
    """Test that excluded_channels is documented in docstrings."""
    function_code = _slice_context_method("_apply_excluded_channels")

    assert '"""' in function_code or "'''" in function_code, \
        "_apply_excluded_channels should have a docstring"

    print("✓ _apply_excluded_channels has documentation")


def test_steps_without_excluded_channels():
    """Test that steps where exclusion doesn't make sense are not modified."""
    steps_without_exclusion = [
        'reference',            # Reference computation, handled differently
        'resample',             # Resamples all data
        'set_montage',          # Sets positions for all channels
        'drop_unused_channels',  # Explicit drop, not exclusion
    ]

    for step_name in steps_without_exclusion:
        function_code = _slice_step(step_name)
        config_get_lines = [line for line in function_code.split('\n')
                            if 'step_config.get' in line]
        has_excluded_param = any("'excluded_channels'" in line
                                 for line in config_get_lines)
        assert not has_excluded_param, \
            f"Step {step_name} should not support excluded_channels"
        print(f"✓ {step_name} correctly does not support excluded_channels")


def run_all_tests():
    """Run all tests in this file."""
    print("=" * 60)
    print("Running Excluded Channels Feature Tests")
    print("=" * 60)
    print()

    test_functions = [
        test_apply_excluded_channels_exists,
        test_get_picks_has_excluded_channels_param,
        test_steps_support_excluded_channels,
        test_steps_pass_excluded_channels_to_get_picks,
        test_preprocessing_steps_report_excluded_channels,
        test_apply_excluded_channels_implementation,
        test_excluded_channels_documentation,
        test_steps_without_excluded_channels,
    ]

    failed_tests = []

    for test_func in test_functions:
        try:
            test_func()
        except AssertionError as e:
            failed_tests.append((test_func.__name__, str(e)))
            print(f"✗ {test_func.__name__}: {e}")
        except Exception as e:
            failed_tests.append((test_func.__name__, f"Unexpected error: {e}"))
            print(f"✗ {test_func.__name__}: Unexpected error: {e}")

    print()
    print("=" * 60)
    if failed_tests:
        print(f"FAILED: {len(failed_tests)} test(s) failed")
        for test_name, error in failed_tests:
            print(f"  - {test_name}: {error}")
    else:
        print("SUCCESS: All excluded_channels feature tests passed!")
    print("=" * 60)

    return len(failed_tests) == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
