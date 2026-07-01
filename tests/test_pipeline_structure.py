#!/usr/bin/env python3
"""
Basic structure tests for the MEEGFlow preprocessing pipeline.

This file tests the pipeline structure without requiring actual MEEG data
or full MNE installation.
"""

import sys
import ast
import json
from pathlib import Path

# Find the repository root
repo_root = Path(__file__).parent.parent
src_dir = repo_root / "src"
configs_dir = repo_root / "configs"


def test_pipeline_file_exists():
    """Test that the main pipeline file exists."""
    pipeline_file = src_dir / "meegflow" / "pipeline.py"
    assert pipeline_file.exists(), "Pipeline file does not exist"
    print("✓ Pipeline file exists")


def test_pipeline_syntax():
    """Test that the pipeline file has valid Python syntax."""
    pipeline_file = src_dir / "meegflow" / "pipeline.py"
    with open(pipeline_file, 'r') as f:
        code = f.read()
    
    try:
        ast.parse(code)
        print("✓ Pipeline file has valid syntax")
    except SyntaxError as e:
        raise AssertionError(f"Syntax error in pipeline: {e}")


def test_pipeline_has_required_classes():
    """Test that the pipeline file contains required classes."""
    pipeline_file = src_dir / "meegflow" / "pipeline.py"
    with open(pipeline_file, 'r') as f:
        code = f.read()
    
    assert "class MEEGFlowPipeline" in code, "MEEGFlowPipeline class not found"
    print("✓ Required class MEEGFlowPipeline found")


def test_pipeline_has_required_methods():
    """Test that the pipeline exposes its orchestration methods."""
    pipeline_file = src_dir / "meegflow" / "pipeline.py"
    with open(pipeline_file, 'r') as f:
        code = f.read()

    # Orchestration methods remain on the pipeline class.
    for method in ["run_pipeline", "run_step", "_load_custom_steps",
                   "_get_pipeline_steps", "_process_single_recording"]:
        assert f"def {method}" in code, f"Method {method} not found"
        print(f"✓ Method {method} found")


def test_all_builtin_steps_registered():
    """Every built-in step is registered in the steps package registry."""
    from meegflow.steps import STEP_REGISTRY

    required_steps = [
        "strip_recording", "concatenate_recordings", "copy_instance",
        "call_module", "set_montage", "drop_unused_channels",
        "bandpass_filter", "notch_filter", "resample", "reference",
        "interpolate_bad_channels", "drop_bad_channels", "ica",
        "find_events", "epoch", "chunk_in_epoch", "find_flat_channels",
        "find_bads_channels_threshold", "find_bads_channels_variance",
        "find_bads_channels_high_frequency", "find_bads_epochs_threshold",
        "save_clean_instance", "generate_json_report", "generate_html_report",
    ]
    for step in required_steps:
        assert step in STEP_REGISTRY, f"Step {step} not registered"
        assert callable(STEP_REGISTRY[step]), f"Step {step} is not callable"
        print(f"✓ Step {step} registered")


def test_config_example_valid_yaml():
    """Test that the example config is valid YAML."""
    config_file = configs_dir / "config_example.yaml"
    assert config_file.exists(), "Config example file does not exist"
    
    import yaml
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check for pipeline configuration structure (new config format)
    assert "pipeline" in config, "Config must have 'pipeline' key"
    assert isinstance(config["pipeline"], list), "Pipeline must be a list of steps"
    assert len(config["pipeline"]) > 0, "Pipeline must have at least one step"
    
    # Check that steps have names
    for step in config["pipeline"]:
        assert "name" in step, "Each step must have a 'name' key"
    
    print("✓ Config example is valid YAML with pipeline structure")


def test_requirements_in_setup():
    """Test that setup.py declares the necessary packages."""
    setup_file = repo_root / "setup.py"
    assert setup_file.exists(), "setup.py does not exist"

    with open(setup_file, 'r') as f:
        setup_content = f.read()

    required_packages = ["mne", "mne-bids", "numpy", "scipy", "PyYAML"]

    for package in required_packages:
        assert package in setup_content, f"Required package {package} not in setup.py"

    print("✓ setup.py exists with required packages")


def test_output_directories_structure():
    """Test that the pipeline creates correct output directory structure."""
    pipeline_file = src_dir / "meegflow" / "pipeline.py"
    with open(pipeline_file, 'r') as f:
        code = f.read()
    
    # Check that the output directories are mentioned (reports and epochs)
    assert "epochs" in code, "epochs directory not mentioned"
    assert "reports" in code, "reports directory not mentioned"
    
    print("✓ All output directories are configured")


def test_readme_exists():
    """Test that README exists and mentions key features."""
    readme_file = repo_root / "README.md"
    assert readme_file.exists(), "README.md does not exist"
    
    with open(readme_file, 'r') as f:
        readme = f.read()
    
    required_sections = [
        "MNE-BIDS",
        "epochs",
        "reports",
        "Installation",
        "Usage",
        "YAML"
    ]
    
    for section in required_sections:
        assert section in readme, f"Required section '{section}' not in README"
    
    print("✓ README.md exists with required sections")


def test_batch_processing_support():
    """Test that the pipeline supports batch processing."""
    pipeline_file = src_dir / "meegflow" / "pipeline.py"
    with open(pipeline_file, 'r') as f:
        code = f.read()
    
    # Check that the pipeline supports multiple subjects
    assert "--subjects" in code or "subjects" in code.lower(), "Batch processing support not found"
    assert "run_pipeline" in code, "run_pipeline method not found"
    
    print("✓ Pipeline supports batch processing of multiple subjects")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running MEEGFlow Preprocessing Pipeline Structure Tests")
    print("=" * 60)
    print()
    
    tests = [
        test_pipeline_file_exists,
        test_pipeline_syntax,
        test_pipeline_has_required_classes,
        test_pipeline_has_required_methods,
        test_config_example_valid_yaml,
        test_requirements_file_exists,
        test_output_directories_structure,
        test_readme_exists,
        test_batch_processing_support,
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
            failed_tests.append(test.__name__)
    
    print()
    print("=" * 60)
    if failed_tests:
        print(f"FAILED: {len(failed_tests)} test(s) failed")
        for test in failed_tests:
            print(f"  - {test}")
        return 1
    else:
        print("SUCCESS: All structure tests passed!")
        print()
        print("Note: Full functionality testing requires:")
        print("  - Installing dependencies (pip install -r requirements.txt)")
        print("  - BIDS-formatted MEEG data")
        return 0


if __name__ == '__main__':
    sys.exit(run_all_tests())
