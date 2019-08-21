#!/usr/bin/env python
"""
Fixtures to be used in tests in DRAGONS
"""

import os
import pytest
import warnings


@pytest.fixture
def path_to_inputs():

    try:
        path = os.path.expanduser(os.environ['DRAGONS_TEST_INPUTS'])
        path = path.strip()
    except KeyError:
        pytest.skip(
            "Could not find environment variable: $DRAGONS_TEST_INPUTS")

    if not os.path.exists(path):
        pytest.skip(
            "Could not access path stored in $DRAGONS_TEST_INPUTS: "
            "{}".format(path)
        )

    return path


@pytest.fixture
def path_to_refs():

    try:
        path = os.path.expanduser(os.environ['DRAGONS_TEST_REFS'])
    except KeyError:
        pytest.skip(
            "Could not find environment variable: $DRAGONS_TEST_REFS")

    if not os.path.exists(path):
        pytest.skip(
            "Could not access path stored in $DRAGONS_TEST_REFS: "
            "{}".format(path)
        )

    return path


@pytest.fixture
def path_to_outputs():

    try:
        path = os.path.expanduser(os.environ['DRAGONS_TEST_OUTPUTS'])
    except KeyError:
        warnings.warn("Could not find environment variable: $DRAGONS_TEST_REFS"
                      "\n Using current working directory")
        path = os.getcwd()

    if not os.path.exists(path):
        pytest.skip(
            "Could not access path stored in $DRAGONS_TEST_REFS: "
            "{}".format(path) +
            "\n Using current working directory"
        )
        path = os.getcwd()

    return path
