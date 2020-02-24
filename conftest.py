#!/usr/bin/env python
"""
Configuration for tests that will propagate inside DRAGONS.
"""

import pytest

from astrodata import testing

path_to_inputs = testing.path_to_inputs
path_to_outputs = testing.path_to_outputs
path_to_refs = testing.path_to_refs
path_to_test_data = testing.path_to_test_data
new_path_to_inputs = testing.new_path_to_inputs


def pytest_addoption(parser):

    try:
        parser.addoption(
            "--dragons-remote-data",
            action="store_true",
            default=False,
            help="run only tests marked with `dragons_remote_data`"
        )
        parser.addoption(
            "--force-preprocess-data",
            action="store_true",
            default=False,
            help="Force preprocessing data as part of the tests."
        )
        parser.addoption(
            "--do-plots",
            action="store_true",
            default=False,
            help="Plot results of each test after running them."
        )
    # This file is imported several times and might bring conflict
    except ValueError:
        pass


def pytest_configure(config):
    config.addinivalue_line("markers", "dragons_remote_data: tests with this "
                                       "mark will download a large volume of "
                                       "data and run")
    config.addinivalue_line("markers", "preprocessed_data: tests with this "
                                       "download anr preprocess the data if it "
                                       "does not exist in the cache folder.")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--dragons-remote-data"):
        skip_dragons_remote_data = pytest.mark.skip(reason="need --dragons-remote-data to run")
        for item in items:
            if "dragons_remote_data" in item.keywords:
                item.add_marker(skip_dragons_remote_data)
