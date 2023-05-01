import tensorflow as tf
import pytest
import importlib
import logging

log = logging.getLogger(__name__)


@pytest.fixture(scope="session", autouse=True)
def profile():
    from transformers_gradients import config

    if config.run_with_profiler:
        options = tf.profiler.experimental.ProfilerOptions(
            host_tracer_level=3, python_tracer_level=1, device_tracer_level=1
        )
        tf.profiler.experimental.start("profile_logs", options=options)
        log.info("Started profiler session.")

    yield
    if config.run_with_profiler:
        tf.profiler.experimental.stop()
        log.info("Stopped profiler session.")


def importer(name, globals=None, locals=None, fromlist=(), level=0):
    if name != "numpy":
        log.debug(name, fromlist, level)

    # not exactly a good verification layer
    frommodule = globals["__name__"] if globals else None
    if name == "numpy" and "transformers_gradients" in frommodule:
        raise ImportError("module '%s' is restricted." % name)

    return importlib.__import__(name, globals, locals, fromlist, level)


@pytest.fixture(
    scope="session",
    # autouse=True
)
def prevent_numpy_functions():
    # We want to use TF only, so we can avoid overhead copying tensors from device to host.
    old = __builtins__["__import__"]  # noqa
    __builtins__["__import__"] = importer  # noqa
    yield
    __builtins__["__import__"] = old  # noqa
