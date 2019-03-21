#!/usr/bin/env python
import pytest
import os
import sys

from contextlib import contextmanager
from io import StringIO

from recipe_system import cal_service


on_travis = pytest.mark.skipif(
    'TRAVIS' in os.environ, reason="Test won't run on Travis.")

print('TRAVIS' in os.environ)


@pytest.fixture
def caldb():
    return cal_service.CalibrationService()


@contextmanager
def captured_output():

    new_out, new_err = StringIO(), StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err


@on_travis
def test_can_call_caldb_config(caldb):

    caldb.config()


@on_travis
def test_can_call_caldb_init(caldb):

    caldb.config()
    caldb.init()


@on_travis
def test_can_change_caldb_config_file_path(caldb):

    test_directory = os.path.dirname(__file__)
    config_file = os.path.join(test_directory, 'my_caldb.cfg')

    with captured_output() as (out, err):
        caldb.config(config_file=config_file)

    output_lines = [s.strip() for s in out.getvalue().split('\n')]

    expected_string = "Using configuration file: {}".format(config_file)
    assert expected_string in output_lines[0]


@pytest.mark.skip('This method should fail when calling init twice but it is '
                  'not. But it fails if you call "caldb init" from the command '
                  "line. I don't know how to solve this!")
def test_fail_init_if_called_twice(caldb):

    os.remove(os.path.expanduser("~/.geminidr/cal_manager.db"))

    caldb.config()
    caldb.init()
    caldb.init()


if __name__ == '__main__':
    pytest.main()
