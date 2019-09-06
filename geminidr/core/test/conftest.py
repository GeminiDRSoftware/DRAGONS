import pytest
import os.path

from astrodata import testing

path_to_inputs = testing.path_to_inputs
path_to_refs = testing.path_to_refs
path_to_outputs = testing.path_to_outputs

#
# def pytest_runtest_setup(item):
#
#     envnames = [mark.args[0] for mark in item.iter_markers(name="env")]
#
#     if envnames:
#         if item.config.getoption("-E") not in envnames:
#             pytest.skip("test requires env in {!r}".format(envnames))
