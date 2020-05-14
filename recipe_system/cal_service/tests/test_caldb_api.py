#!/usr/bin/env python
import pytest
import os

from recipe_system.cal_service import CalibrationService, get_calconf


@pytest.fixture(scope='module')
def caldb(path_to_outputs):

    filename = 'rsys.cfg'

    caldb_config = ("[calibs]\n"
                    "standalone=True\n"
                    "database_dir={:s}\n".format(path_to_outputs))

    with open(os.path.join(path_to_outputs, filename), 'w') as _file:
        _file.write(caldb_config)

    yield CalibrationService()

    try:
        os.remove(os.path.join(path_to_outputs, filename))
        os.remove(os.path.join(path_to_outputs, 'cal_manager.db'))
    except FileNotFoundError:
        pass


def test_caldb_has_no_manager_on_creation(caldb):
    assert caldb._mgr is None


def test_caldb_config_adds_manager(caldb, path_to_outputs):
    config_file = os.path.join(path_to_outputs, 'rsys.cfg')
    caldb.config(config_file=config_file)
    assert caldb._mgr is not None


def test_can_call_caldb_init(caldb, path_to_outputs):
    caldb.init()


def test_can_change_caldb_config_file_path(caldb, path_to_outputs):

    test_directory = os.path.dirname(__file__)
    config_file = os.path.join(test_directory, 'my_caldb.cfg')

    caldb.config(config_file=config_file)
    config = get_calconf()

    assert config_file in config.config_file


@pytest.mark.xfail(
    reason='This method should fail when calling init twice but it does '
           'not. But it fails if you call "caldb init" from the command '
           "line. I don't know how to solve this!", run=False)
def test_fail_init_if_called_twice(caldb, path_to_outputs):

    os.remove(os.path.expanduser("~/.geminidr/cal_manager.db"))

    caldb.config()
    caldb.init()
    caldb.init()


if __name__ == '__main__':
    pytest.main()
