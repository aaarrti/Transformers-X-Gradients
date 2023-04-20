from os import environ
import pytest

skip_in_ci = pytest.mark.skipif("CI" in environ, reason="OOM in GitHub action.")
