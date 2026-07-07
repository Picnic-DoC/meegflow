"""Shared test helpers."""

from meegflow.context import PipelineContext


def run_step(pipeline, name, data, step_config=None):
    """Dispatch a single registered step the same way ``process_recording`` does.

    Wraps ``data`` in a :class:`~meegflow.context.PipelineContext` using
    ``pipeline``'s reader/output_root/config, calls the registered step
    function, and returns the updated data mapping. For tests exercising one
    step in isolation.
    """
    ctx = PipelineContext(
        data,
        reader=pipeline.reader,
        output_root=pipeline.output_root,
        config=pipeline.config,
    )
    result = pipeline.step_functions[name](ctx, step_config or {})
    if isinstance(result, PipelineContext):
        ctx = result
    elif isinstance(result, dict):
        ctx.data = result
    return ctx.data
