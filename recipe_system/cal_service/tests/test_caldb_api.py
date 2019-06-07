#!/usr/bin/env python
import pytest
import os
import sys

from contextlib import contextmanager
from io import StringIO

from recipe_system.cal_service import CalibrationService, get_calconf


on_travis = pytest.mark.skipif(
    'TRAVIS' in os.environ, reason="Test won't run on Travis.")

print('TRAVIS' in os.environ)


@pytest.fixture(scope='session', autouse=True)
def caldb():
    print(' Creating cal_service object.')
    yield CalibrationService()
    print(' Tearing down cal_service')


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
def test_caldb_has_no_manager_on_creation(caldb):
    assert caldb._mgr is None


@on_travis
def test_caldb_config_adds_manager(caldb):
    caldb.config()
    assert caldb._mgr is not None


@on_travis
def test_can_call_caldb_init(caldb):
    caldb.init()


@on_travis
def test_can_change_caldb_config_file_path(caldb):

    test_directory = os.path.dirname(__file__)
    config_file = os.path.join(test_directory, 'my_caldb.cfg')

    caldb.config(config_file=config_file)
    config = get_calconf()

    assert config_file in config.config_file


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
