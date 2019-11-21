#!/usr/bin/env python
"""
Configuration for tests that will propagate inside DRAGONS.
"""

import pytest

from astrodata import testing

path_to_inputs = testing.path_to_inputs
path_to_outputs = testing.path_to_outputs
path_to_refs = testing.path_to_refs


def pytest_addoption(parser):

    try:
        parser.addoption(
            "--dragons-remote-data",
            action="store_true",
            default=False,
            help="run only tests marked with `dragons_remote_data`"
        )
    # This file is imported several times and might bring conflict
    except ValueError:
        pass


def pytest_configure(config):
    config.addinivalue_line("markers", "dragons_remote_data: tests with this "
                                       "mark will download a large volume of "
                                       "data and run")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--dragons-remote-data"):
        skip_dragons_remote_data = pytest.mark.skip(reason="need --dragons-remote-data to run")
        for item in items:
            if "dragons_remote_data" in item.keywords:
                item.add_marker(skip_dragons_remote_data)
