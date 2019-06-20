#!/usr/bin/env python
"""
Fixtures to be used in tests in DRAGONS
"""

import os
import pytest


@pytest.fixture
def path_to_inputs():

    try:
        path = os.environ['DRAGONS_TEST_INPUTS']
    except KeyError:
        pytest.skip(
            "Could not find environment variable: $DRAGONS_TEST_INPUTS")

    if not os.path.exists(path):
        pytest.skip(
            "Could not access path stored in $DRAGONS_TEST_INPUTS: "
            "{}".format(path)
        )

    return path
