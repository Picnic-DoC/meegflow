"""Built-in pipeline steps.

Importing this package populates ``STEP_REGISTRY`` with every built-in step.
"""
from .registry import STEP_REGISTRY, register  # noqa: F401
from . import (  # noqa: F401  (imported for their registration side effects)
    recording,
    channels,
    filtering,
    transforms,
    ica,
    bad_detection,
    epoching,
    output,
)

__all__ = ["STEP_REGISTRY", "register"]
