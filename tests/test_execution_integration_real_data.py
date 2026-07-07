#!/usr/bin/env python3
"""
Integration tests exercising all three execution backends (sequential,
local, slurm) against two real M/EEG datasets present on this development
machine.

These are opt-in fixtures meant for a human to review and run manually --
they are NOT part of the regular automated test suite and must never run
by accident:

1. The whole module is skipped unless the environment variable
   ``MEEGFLOW_RUN_REAL_DATA_TESTS`` is set to ``"1"`` (mirrors the
   ``pytestmark = pytest.mark.skipif(...)`` idiom already used elsewhere in
   this test suite, e.g. ``test_excluded_channels_integration.py``).
2. Each individual test additionally skips if its dataset root doesn't
   exist on disk, so this file is harmless on any machine other than the
   one it was authored on (or a partial checkout with only one of the two
   datasets present).

Datasets (both external to this repo, not created by this test): point the
``TEST_DATASETS_ROOT`` environment variable at a local folder containing
both of the following (any developer can have their own such folder; if
the variable isn't set, every case below is skipped as "dataset not
found"):
- ``$TEST_DATASETS_ROOT/ssvep``
- ``$TEST_DATASETS_ROOT/decoding_csp_eeg``

Configs: configs/integration/{ssvep,decoding_csp}_{sequential,local,slurm}.yaml
(pipeline steps copied verbatim from each dataset's known-good config; only
the top-level 'execution' block varies across the three backend variants).

IMPORTANT: the *_slurm.yaml configs' cluster_kwargs (queue/cores/memory/
walltime) are generic placeholders -- there is no Slurm scheduler on this
development machine, and the slurm-backend tests below WILL fail until
those are edited to match a real cluster the user actually has access to.
"""
import os
import sys
from pathlib import Path

import pytest
import yaml

# Find the repository root
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))

CONFIGS_DIR = repo_root / "configs" / "integration"

RUN_REAL_DATA_TESTS = os.environ.get("MEEGFLOW_RUN_REAL_DATA_TESTS") == "1"

pytestmark = pytest.mark.skipif(
    not RUN_REAL_DATA_TESTS,
    reason=(
        "Real-data integration tests are opt-in only: set "
        "MEEGFLOW_RUN_REAL_DATA_TESTS=1 to run them. They process real EEG "
        "recordings on disk (and, for the 'slurm' variants, require an "
        "actual Slurm cluster reachable from this machine) and are meant "
        "to be run manually, not automatically in CI."
    ),
)

# --------------------------------------------------------------------- #
# Dataset definitions                                                    #
# --------------------------------------------------------------------- #

_datasets_root_env = os.environ.get("TEST_DATASETS_ROOT")
if _datasets_root_env:
    _DATASETS_ROOT = Path(_datasets_root_env)
    SSVEP_ROOT = _DATASETS_ROOT / "ssvep"
    DECODING_CSP_ROOT = _DATASETS_ROOT / "decoding_csp_eeg"
else:
    # Nothing configured on this machine -- every case's missing-dataset
    # skip below will fire, so the module simply reports "no tests run"
    # rather than pointing at one specific developer's disk layout.
    SSVEP_ROOT = Path("/dev/null/ssvep")
    DECODING_CSP_ROOT = Path("/dev/null/decoding_csp_eeg")

# (dataset_id, config_name, backend) -> everything a test needs to run and
# check one (dataset, backend) combination.
CASES = [
    # -- ssvep: subjects 01/02, session 01, task 'ssvep' -----
    dict(
        dataset_id="ssvep",
        backend="sequential",
        config_name="ssvep_sequential.yaml",
        dataset_root=SSVEP_ROOT,
        subjects=["01", "02"],
        sessions="01",
        tasks="ssvep",
        extension=".vhdr",
    ),
    dict(
        dataset_id="ssvep",
        backend="local",
        config_name="ssvep_local.yaml",
        dataset_root=SSVEP_ROOT,
        subjects=["01", "02"],
        sessions="01",
        tasks="ssvep",
        extension=".vhdr",
    ),
    dict(
        dataset_id="ssvep",
        backend="slurm",
        config_name="ssvep_slurm.yaml",
        dataset_root=SSVEP_ROOT,
        subjects=["01", "02"],
        sessions="01",
        tasks="ssvep",
        extension=".vhdr",
    ),
    # -- decoding_csp_eeg: subjects 001/002, no session, task 'eegbci' ----
    # (the source config.yaml's header comment says 'motorimagery', but
    # that's a stale comment -- the real files on disk are named
    # 'sub-001_task-eegbci_run-XX_...', so the real task label is 'eegbci')
    dict(
        dataset_id="decoding_csp",
        backend="sequential",
        config_name="decoding_csp_sequential.yaml",
        dataset_root=DECODING_CSP_ROOT,
        subjects=["001", "002"],
        sessions=None,
        tasks="eegbci",
        extension=".vhdr",
    ),
    dict(
        dataset_id="decoding_csp",
        backend="local",
        config_name="decoding_csp_local.yaml",
        dataset_root=DECODING_CSP_ROOT,
        subjects=["001", "002"],
        sessions=None,
        tasks="eegbci",
        extension=".vhdr",
    ),
    dict(
        dataset_id="decoding_csp",
        backend="slurm",
        config_name="decoding_csp_slurm.yaml",
        dataset_root=DECODING_CSP_ROOT,
        subjects=["001", "002"],
        sessions=None,
        tasks="eegbci",
        extension=".vhdr",
    ),
]


def _case_id(case):
    return f"{case['dataset_id']}-{case['backend']}"


def _make_param(case):
    """Wrap a case in pytest.param, additionally skipping if its dataset
    root isn't present on this machine (so a partial checkout -- e.g. only
    one of the two example datasets -- still runs whatever it can)."""
    missing_dataset = not case["dataset_root"].exists()
    return pytest.param(
        case,
        id=_case_id(case),
        marks=pytest.mark.skipif(
            missing_dataset,
            reason=f"Dataset root not found on this machine: {case['dataset_root']}",
        ),
    )


# --------------------------------------------------------------------- #
# Expected output path helpers                                          #
# --------------------------------------------------------------------- #

def _expected_save_clean_instance_path(
    deriv_root, subject, task, session, instance,
    datatype=None, processing=None, description=None, fmt="fif",
):
    """Mirror steps.output.save_clean_instance's BIDSPath construction, so
    expectations stay derived from the same rules as the production code
    (default suffix by instance, default extension by format) rather than
    hardcoded filename strings."""
    from mne_bids import BIDSPath
    from meegflow.savers import FORMAT_EXTENSIONS

    suffix = "epo" if instance == "epochs" else "eeg"
    extension = FORMAT_EXTENSIONS[fmt]
    bids_path = BIDSPath(
        subject=subject,
        task=task,
        session=session,
        datatype=datatype,
        root=deriv_root / instance,
        suffix=suffix,
        extension=extension,
        processing=processing,
        description=description,
        check=False,
    )
    return Path(bids_path.fpath)


def _expected_report_paths(deriv_root, subject, task, session):
    """report.py hardcodes datatype='eeg', processing='clean',
    description='cleaned' for both JSON and HTML reports, regardless of
    step_config -- mirrored here rather than duplicated as literal strings."""
    from mne_bids import BIDSPath

    paths = []
    for suffix_extension in (("report", ".json"), ("report", ".html")):
        suffix, extension = suffix_extension
        bids_path = BIDSPath(
            subject=subject,
            task=task,
            session=session,
            datatype="eeg",
            root=deriv_root / "reports",
            suffix=suffix,
            extension=extension,
            processing="clean",
            description="cleaned",
            check=False,
        )
        paths.append(Path(bids_path.fpath))
    return paths


def _expected_output_paths(case, subject):
    """All files a successful run of this case's pipeline should produce
    for one subject."""
    deriv_root = case["dataset_root"] / "derivatives" / "meegflow"
    session = case["sessions"] if isinstance(case["sessions"], str) else None
    task = case["tasks"]

    paths = []
    if case["dataset_id"] == "ssvep":
        # ssvep_*.yaml: two save_clean_instance calls (epochs, raw), neither
        # sets datatype/processing/description.
        paths.append(_expected_save_clean_instance_path(
            deriv_root, subject, task, session, instance="epochs",
        ))
        paths.append(_expected_save_clean_instance_path(
            deriv_root, subject, task, session, instance="raw",
        ))
    elif case["dataset_id"] == "decoding_csp":
        # decoding_csp_*.yaml: one save_clean_instance call (epochs), with
        # datatype/processing/description set explicitly.
        paths.append(_expected_save_clean_instance_path(
            deriv_root, subject, task, session, instance="epochs",
            datatype="eeg", processing="clean", description="cleaned",
        ))
    else:
        raise AssertionError(f"Unknown dataset_id: {case['dataset_id']}")

    paths.extend(_expected_report_paths(deriv_root, subject, task, session))
    return paths


# --------------------------------------------------------------------- #
# The test                                                               #
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("case", [_make_param(c) for c in CASES])
def test_execution_backend_against_real_dataset(case):
    """Run the full pipeline (subjects x steps) for one (dataset, backend)
    combination and check it completed cleanly, with the expected output
    files on disk.

    NOTE: the 'slurm' backend cases require a real, reachable Slurm cluster
    -- there isn't one on this development machine, and
    configs/integration/*_slurm.yaml's cluster_kwargs are placeholders that
    must be edited before this will succeed against a real cluster.
    """
    from meegflow import MEEGFlowPipeline
    from meegflow.readers import BIDSReader

    config_path = CONFIGS_DIR / case["config_name"]
    assert config_path.exists(), f"Missing integration config: {config_path}"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    assert config["execution"]["backend"] == case["backend"], (
        f"{config_path} execution.backend should be '{case['backend']}'"
    )

    reader = BIDSReader(case["dataset_root"])
    pipeline = MEEGFlowPipeline(reader=reader, config=config)

    all_results = pipeline.run_pipeline(
        subjects=case["subjects"],
        sessions=case["sessions"],
        tasks=case["tasks"],
        extension=case["extension"],
    )

    assert set(all_results.keys()) == set(case["subjects"]), (
        f"Expected results for subjects {case['subjects']}, got {list(all_results.keys())}"
    )

    for subject in case["subjects"]:
        subject_results = all_results[subject]
        assert len(subject_results) > 0, f"No results recorded for subject {subject}"

        for result in subject_results:
            assert "error" not in result, (
                f"[{_case_id(case)}] subject {subject} failed: {result.get('error')}"
            )

        for expected_path in _expected_output_paths(case, subject):
            assert expected_path.exists(), (
                f"[{_case_id(case)}] expected output missing for subject "
                f"{subject}: {expected_path}"
            )
