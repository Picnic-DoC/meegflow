"""Registry mapping step names to their implementations."""

STEP_REGISTRY = {}


def register(name):
    """Decorator that registers a step function under ``name``."""
    def _deco(fn):
        STEP_REGISTRY[name] = fn
        return fn
    return _deco
