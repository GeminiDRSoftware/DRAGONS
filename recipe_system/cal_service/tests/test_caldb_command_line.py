
import pytest

from recipe_system.cal_service import CalibrationService


def test_can_call_caldb_config():

    caldb = CalibrationService()
    caldb.config()


def test_can_call_caldb_init():

    caldb = CalibrationService()
    caldb.init()


if __name__ == '__main__':
    pytest.main()
