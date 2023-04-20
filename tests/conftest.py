import tensorflow as tf
import pytest

# @pytest.fixture(scope="session", autouse=True)
# def profile():
#    options = tf.profiler.experimental.ProfilerOptions(
#        host_tracer_level=3, python_tracer_level=1, device_tracer_level=1
#    )
#    tf.profiler.experimental.start("profile_logs", options=options)
#
#    yield
#
#    tf.profiler.experimental.stop()
