#!/usr/bin/env python
"""
Configuration for tests that will propagate inside DRAGONS.
"""

import pytest

# noinspection PyUnresolvedReferences
from astrodata.testing import (
    change_working_dir,
    path_to_inputs,
    path_to_outputs,
    path_to_refs)


def pytest_addoption(parser):
    try:
        parser.addoption(
            "--dragons-remote-data",
            action="store_true",
            default=False,
            help="enable tests that use the cache_file_from_archive fixture"
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
            if 'cache_file_from_archive' in item.fixturenames or "dragons_remote_data" in item.keywords:
                item.add_marker(skip_dragons_remote_data)
