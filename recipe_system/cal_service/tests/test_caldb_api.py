#!python
import pytest
import os

from recipe_system.cal_service import CalibrationService

on_travis = pytest.mark.skipif('TRAVIS' in os.environ,
                               reason="Test won't run on Travis.")
print('TRAVIS' in os.environ)


@pytest.fixture
def caldb():
    return CalibrationService()


@on_travis
def test_can_call_caldb_config(caldb):

    caldb.config()


@on_travis
def test_can_call_caldb_init(caldb):

    caldb.config()
    caldb.init()


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
