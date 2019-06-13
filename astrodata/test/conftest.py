
import pytest
import os


@pytest.fixture
def input_test_path():

    try:
        path = os.environ['DRAGONS_TEST_IN_PATH']
    except KeyError:
        pytest.skip(
            "Could not find environment variable: $DRAGONS_TEST_IN_PATH")

    if not os.path.exists(path):
        pytest.skip(
            "Could not access path stored in $DRAGONS_TEST_IN_PATH: "
            "{}".format(path)
        )

    return path
