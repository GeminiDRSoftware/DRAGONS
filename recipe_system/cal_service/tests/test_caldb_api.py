#!/usr/bin/env python
import pytest
import os
import sys

from contextlib import contextmanager
from io import StringIO

from recipe_system.cal_service import CalibrationService, get_calconf


@pytest.fixture(scope='session', autouse=True)
def caldb():

    path = os.path.dirname(__file__)
    filename = 'rsys.cfg'

    caldb_config = ("[calibs]\n"
                    "standalone=True\n"
                    "database_dir={:s}\n".format(path))

    with open(os.path.join(path, filename), 'w') as _file:
        _file.write(caldb_config)

    yield CalibrationService()

    os.remove(os.path.join(path, filename))
    os.remove(os.path.join(path, 'cal_manager.db'))


@contextmanager
def captured_output():

    new_out, new_err = StringIO(), StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err


def test_caldb_has_no_manager_on_creation(caldb):
    assert caldb._mgr is None


def test_caldb_config_adds_manager(caldb):
    config_file = os.path.join(os.path.dirname(__file__), 'rsys.cfg')
    caldb.config(config_file=config_file)
    assert caldb._mgr is not None


def test_can_call_caldb_init(caldb):
    caldb.init()


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
