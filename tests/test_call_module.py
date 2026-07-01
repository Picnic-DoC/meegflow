"""Tests for the call_module pipeline step."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))


def _make_pipeline():
    from meegflow import MEEGFlowPipeline
    reader = MagicMock()
    return MEEGFlowPipeline(reader=reader, config={})


def _base_data():
    return {"preprocessing_steps": []}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

class TestCallModuleRegistered:
    def test_in_step_functions(self):
        pipeline = _make_pipeline()
        assert "call_module" in pipeline.step_functions

    def test_maps_to_correct_method(self):
        from meegflow.steps import STEP_REGISTRY
        pipeline = _make_pipeline()
        assert pipeline.step_functions["call_module"] is STEP_REGISTRY["call_module"]


# ---------------------------------------------------------------------------
# Basic invocation
# ---------------------------------------------------------------------------

class TestCallModuleBasic:
    def test_plain_kwargs_forwarded(self):
        """Call a function with plain keyword arguments."""
        pipeline = _make_pipeline()
        data = _base_data()
        # dict(**kwargs) accepts arbitrary keyword args
        result = pipeline.run_step("call_module", data, {
            "module": "builtins.dict",
            "var_name": "out",
            "name": "test",
            "value": 42,
        })
        assert result["out"] == {"name": "test", "value": 42}

    def test_plain_positional_args_forwarded(self):
        """Call a positional-only function via args list."""
        pipeline = _make_pipeline()
        data = _base_data()
        result = pipeline.run_step("call_module", data, {
            "module": "os.path.join",
            "var_name": "joined",
            "args": ["/tmp", "subdir", "file.txt"],
        })
        assert result["joined"] == "/tmp/subdir/file.txt"

    def test_mixed_args_and_kwargs(self):
        """Positional args and keyword args can be combined."""
        pipeline = _make_pipeline()
        data = _base_data()
        # sorted(iterable, reverse=True) — iterable is positional-only
        result = pipeline.run_step("call_module", data, {
            "module": "builtins.sorted",
            "var_name": "out",
            "args": [[3, 1, 2]],
            "reverse": True,
        })
        assert result["out"] == [3, 2, 1]

    def test_result_stored_under_var_name(self):
        pipeline = _make_pipeline()
        data = _base_data()
        result = pipeline.run_step("call_module", data, {
            "module": "os.path.basename",
            "var_name": "filename",
            "args": ["/some/path/data.fif"],
        })
        assert result["filename"] == "data.fif"

    def test_var_name_none_discards_result(self):
        """When var_name is None the result is not stored."""
        pipeline = _make_pipeline()
        data = _base_data()
        keys_before = set(data.keys())
        pipeline.run_step("call_module", data, {
            "module": "os.path.basename",
            "var_name": None,
            "args": ["/some/path/data.fif"],
        })
        new_keys = set(data.keys()) - keys_before - {"preprocessing_steps"}
        assert not new_keys, f"Unexpected keys added: {new_keys}"

    def test_returns_data_dict(self):
        pipeline = _make_pipeline()
        data = _base_data()
        result = pipeline.run_step("call_module", data, {
            "module": "os.path.basename",
            "var_name": "x",
            "args": ["/a/b"],
        })
        assert result is data


# ---------------------------------------------------------------------------
# data__ value resolution
# ---------------------------------------------------------------------------

class TestCallModuleDataRef:
    def test_top_level_data_ref_in_kwargs(self):
        """data__key in a kwarg value resolves to data[key]."""
        pipeline = _make_pipeline()
        data = _base_data()
        data["label"] = "hello"
        result = pipeline.run_step("call_module", data, {
            "module": "builtins.dict",
            "var_name": "out",
            "the_label": "data__label",
        })
        assert result["out"]["the_label"] == "hello"

    def test_top_level_data_ref_in_args(self):
        """data__key in the args list resolves to data[key]."""
        pipeline = _make_pipeline()
        data = _base_data()
        data["items"] = [3, 1, 2]
        result = pipeline.run_step("call_module", data, {
            "module": "builtins.sorted",
            "var_name": "out",
            "args": ["data__items"],
        })
        assert result["out"] == [1, 2, 3]

    def test_nested_data_ref(self):
        """data__a__b resolves to data['a']['b']."""
        pipeline = _make_pipeline()
        data = _base_data()
        data["container"] = {"value": 42}
        result = pipeline.run_step("call_module", data, {
            "module": "builtins.str",
            "var_name": "stringified",
            "args": ["data__container__value"],
        })
        assert result["stringified"] == "42"

    def test_mixed_plain_and_data_ref(self):
        """Mixing plain values and data__ references in the same call."""
        pipeline = _make_pipeline()
        data = _base_data()
        data["nums"] = [3, 1, 2]
        result = pipeline.run_step("call_module", data, {
            "module": "builtins.sorted",
            "var_name": "out",
            "args": ["data__nums"],
            "reverse": True,
        })
        assert result["out"] == [3, 2, 1]

    def test_data_ref_passes_object_identity(self):
        """The exact object from data is passed, not a copy."""
        pipeline = _make_pipeline()
        sentinel = object()
        data = _base_data()
        data["obj"] = sentinel
        captured = {}

        def capture(thing):
            captured["thing"] = thing
            return None

        import sys
        sys.modules["_test_capture"] = MagicMock()
        sys.modules["_test_capture"].capture = capture

        pipeline.run_step("call_module", data, {
            "module": "_test_capture.capture",
            "var_name": None,
            "thing": "data__obj",
        })
        assert captured["thing"] is sentinel


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestCallModuleErrors:
    def test_missing_module_key_raises(self):
        pipeline = _make_pipeline()
        with pytest.raises(ValueError, match="module"):
            pipeline.run_step("call_module", _base_data(), {"var_name": "x"})

    def test_bad_data_path_raises_value_error(self):
        pipeline = _make_pipeline()
        data = _base_data()
        with pytest.raises(ValueError, match="data__missing_key"):
            pipeline.run_step("call_module", data, {
                "module": "builtins.str",
                "var_name": "x",
                "object": "data__missing_key",
            })

    def test_bad_nested_data_path_raises_value_error(self):
        pipeline = _make_pipeline()
        data = _base_data()
        data["top"] = {"exists": 1}
        with pytest.raises(ValueError, match="data__top__missing"):
            pipeline.run_step("call_module", data, {
                "module": "builtins.str",
                "var_name": "x",
                "object": "data__top__missing",
            })

    def test_invalid_module_path_raises(self):
        pipeline = _make_pipeline()
        with pytest.raises(Exception):
            pipeline.run_step("call_module", _base_data(), {
                "module": "nonexistent_package.some_function",
                "var_name": "x",
            })


# ---------------------------------------------------------------------------
# preprocessing_steps recording
# ---------------------------------------------------------------------------

class TestCallModuleStepRecording:
    def test_step_appended(self):
        pipeline = _make_pipeline()
        data = _base_data()
        pipeline.run_step("call_module", data, {
            "module": "os.path.basename",
            "var_name": "x",
            "p": "/a/b",
        })
        assert len(data["preprocessing_steps"]) == 1

    def test_step_name_is_call_module(self):
        pipeline = _make_pipeline()
        data = _base_data()
        pipeline.run_step("call_module", data, {
            "module": "os.path.basename",
            "var_name": "x",
            "p": "/a/b",
        })
        assert data["preprocessing_steps"][0]["step"] == "call_module"

    def test_module_recorded(self):
        pipeline = _make_pipeline()
        data = _base_data()
        pipeline.run_step("call_module", data, {
            "module": "os.path.basename",
            "var_name": "x",
            "p": "/a/b",
        })
        assert data["preprocessing_steps"][0]["module"] == "os.path.basename"

    def test_var_name_recorded(self):
        pipeline = _make_pipeline()
        data = _base_data()
        pipeline.run_step("call_module", data, {
            "module": "os.path.basename",
            "var_name": "my_result",
            "p": "/a/b",
        })
        assert data["preprocessing_steps"][0]["var_name"] == "my_result"

    def test_kwargs_recorded_as_config_strings(self):
        """data__ values must be recorded as the original string, not the resolved object."""
        pipeline = _make_pipeline()
        data = _base_data()
        data["nums"] = [1, 2, 3]
        pipeline.run_step("call_module", data, {
            "module": "builtins.sorted",
            "var_name": "out",
            "args": ["data__nums"],
            "reverse": True,
        })
        step_record = data["preprocessing_steps"][0]
        # args are recorded separately, preserving the original data__ strings
        assert step_record["args"] == ["data__nums"]
        assert step_record["kwargs"]["reverse"] is True

    def test_module_and_var_name_not_in_kwargs(self):
        pipeline = _make_pipeline()
        data = _base_data()
        pipeline.run_step("call_module", data, {
            "module": "os.path.basename",
            "var_name": "x",
            "p": "/a/b",
        })
        recorded_kwargs = data["preprocessing_steps"][0]["kwargs"]
        assert "module" not in recorded_kwargs
        assert "var_name" not in recorded_kwargs

    def test_target_and_unpack_as_not_in_kwargs(self):
        pipeline = _make_pipeline()
        data = _base_data()
        data["container"] = {"items": [3, 1, 2]}
        pipeline.run_step("call_module", data, {
            "module": "builtins.dict",
            "var_name": "out",
            "x": 1,
        })
        recorded_kwargs = data["preprocessing_steps"][0]["kwargs"]
        assert "target" not in recorded_kwargs
        assert "unpack_as" not in recorded_kwargs

    def test_target_recorded_in_step(self):
        pipeline = _make_pipeline()
        data = _base_data()
        data["my_list"] = [3, 1, 2]
        pipeline.run_step("call_module", data, {
            "module": "sort",
            "target": "data__my_list",
            "var_name": None,
        })
        assert data["preprocessing_steps"][0]["target"] == "data__my_list"

    def test_unpack_as_recorded_in_step(self):
        pipeline = _make_pipeline()
        data = _base_data()
        pipeline.run_step("call_module", data, {
            "module": "builtins.divmod",
            "unpack_as": ["quotient", "remainder"],
            "args": [10, 3],
        })
        assert data["preprocessing_steps"][0]["unpack_as"] == ["quotient", "remainder"]


# ---------------------------------------------------------------------------
# target: method calls on data objects
# ---------------------------------------------------------------------------

class TestCallModuleTarget:
    def test_method_called_on_target(self):
        """target resolves to a data object and module is the method name."""
        pipeline = _make_pipeline()
        data = _base_data()
        data["my_list"] = [3, 1, 2]
        pipeline.run_step("call_module", data, {
            "module": "sort",
            "target": "data__my_list",
            "var_name": None,
        })
        assert data["my_list"] == [1, 2, 3]

    def test_method_return_stored(self):
        """Return value of a method call is stored under var_name."""
        pipeline = _make_pipeline()
        data = _base_data()
        data["my_str"] = "hello world"
        result = pipeline.run_step("call_module", data, {
            "module": "upper",
            "target": "data__my_str",
            "var_name": "uppercased",
        })
        assert result["uppercased"] == "HELLO WORLD"

    def test_method_with_kwargs(self):
        """Keyword arguments are forwarded to the method."""
        pipeline = _make_pipeline()
        data = _base_data()
        data["my_list"] = [3, 1, 2]
        pipeline.run_step("call_module", data, {
            "module": "sort",
            "target": "data__my_list",
            "var_name": None,
            "reverse": True,
        })
        assert data["my_list"] == [3, 2, 1]

    def test_method_arg_from_data(self):
        """Method arguments can also reference data via data__."""
        pipeline = _make_pipeline()
        data = _base_data()
        data["base"] = "hello"
        data["suffix"] = " world"
        result = pipeline.run_step("call_module", data, {
            "module": "__add__",
            "target": "data__base",
            "var_name": "combined",
            "args": ["data__suffix"],
        })
        assert result["combined"] == "hello world"

    def test_target_bad_path_raises(self):
        pipeline = _make_pipeline()
        data = _base_data()
        with pytest.raises(ValueError, match="data__missing"):
            pipeline.run_step("call_module", data, {
                "module": "upper",
                "target": "data__missing",
                "var_name": None,
            })

    def test_mne_set_montage_via_target(self):
        """The canonical MNE use-case: create a montage then apply it as a method."""
        import mne
        pipeline = _make_pipeline()
        n_times = 200
        ch_names = ["Fp1", "Fp2", "Fz", "Cz", "Pz"]
        info = mne.create_info(ch_names=ch_names, sfreq=100.0, ch_types="eeg")
        raw = mne.io.RawArray(np.random.randn(5, n_times) * 1e-6, info)

        data = _base_data()
        data["raw"] = raw

        pipeline.run_step("call_module", data, {
            "module": "mne.channels.make_standard_montage",
            "var_name": "montage",
            "kind": "standard_1020",
        })
        pipeline.run_step("call_module", data, {
            "module": "set_montage",
            "target": "data__raw",
            "var_name": None,
            "montage": "data__montage",
            "on_missing": "ignore",
        })

        assert data["raw"].get_montage() is not None


# ---------------------------------------------------------------------------
# unpack_as: multi-value returns
# ---------------------------------------------------------------------------

class TestCallModuleUnpackAs:
    def test_tuple_unpacked_into_separate_keys(self):
        pipeline = _make_pipeline()
        data = _base_data()
        pipeline.run_step("call_module", data, {
            "module": "builtins.divmod",
            "unpack_as": ["quotient", "remainder"],
            "args": [10, 3],
        })
        assert data["quotient"] == 3
        assert data["remainder"] == 1

    def test_unpack_as_overrides_var_name_error(self):
        """Specifying both var_name and unpack_as raises ValueError."""
        pipeline = _make_pipeline()
        data = _base_data()
        with pytest.raises(ValueError, match="mutually exclusive"):
            pipeline.run_step("call_module", data, {
                "module": "builtins.divmod",
                "var_name": "result",
                "unpack_as": ["q", "r"],
                "args": [10, 3],
            })

    def test_partial_unpack(self):
        """Fewer names than returned values stores only the first N items (zip semantics)."""
        pipeline = _make_pipeline()
        data = _base_data()
        pipeline.run_step("call_module", data, {
            "module": "builtins.divmod",
            "unpack_as": ["quotient"],
            "args": [10, 3],
        })
        assert data["quotient"] == 3
        assert "remainder" not in data

    def test_unpack_does_not_store_under_var_name(self):
        pipeline = _make_pipeline()
        data = _base_data()
        pipeline.run_step("call_module", data, {
            "module": "builtins.divmod",
            "unpack_as": ["q", "r"],
            "args": [10, 3],
        })
        assert "None" not in data
        assert "var_name" not in data

    def test_mne_events_from_annotations_unpacked(self):
        """Smoke test: unpack a real MNE function that returns a tuple."""
        import mne
        pipeline = _make_pipeline()
        n_times = 500
        ch_names = ["Fp1", "Fp2"]
        info = mne.create_info(ch_names=ch_names, sfreq=100.0, ch_types="eeg")
        raw = mne.io.RawArray(np.random.randn(2, n_times) * 1e-6, info)
        raw.set_annotations(mne.Annotations(onset=[1.0, 2.0], duration=[0.1, 0.1], description=["stim", "stim"]))

        data = _base_data()
        data["raw"] = raw

        pipeline.run_step("call_module", data, {
            "module": "mne.events_from_annotations",
            "unpack_as": ["events", "event_id"],
            "args": ["data__raw"],
        })

        assert "events" in data
        assert "event_id" in data
        assert data["events"].shape[1] == 3


# ---------------------------------------------------------------------------
# MNE integration smoke test
# ---------------------------------------------------------------------------

class TestCallModuleMNE:
    def test_make_standard_montage(self):
        """Smoke test: call an actual MNE function."""
        import mne
        pipeline = _make_pipeline()
        data = _base_data()
        result = pipeline.run_step("call_module", data, {
            "module": "mne.channels.make_standard_montage",
            "var_name": "montage",
            "kind": "standard_1020",
        })
        assert "montage" in result
        assert isinstance(result["montage"], mne.channels.DigMontage)

    def test_mne_result_usable_in_next_step(self):
        """Result stored in data is accessible by subsequent config-driven steps."""
        import mne
        pipeline = _make_pipeline()

        n_times = 200
        ch_names = ["Fp1", "Fp2", "Fz", "Cz", "Pz"]
        info = mne.create_info(ch_names=ch_names, sfreq=100.0, ch_types="eeg")
        raw = mne.io.RawArray(np.random.randn(5, n_times) * 1e-6, info)

        data = _base_data()
        data["raw"] = raw

        pipeline.run_step("call_module", data, {
            "module": "mne.channels.make_standard_montage",
            "var_name": "montage",
            "kind": "standard_1020",
        })

        assert "montage" in data
        data["raw"].set_montage(data["montage"], on_missing="ignore")
