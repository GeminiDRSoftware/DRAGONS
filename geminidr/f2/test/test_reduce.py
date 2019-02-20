#!/usr/bin/env python

import glob
import pytest
import os

from gempy.adlibrary import dataselect
from recipe_system.reduction.coreReduce import Reduce


@pytest.fixture
def test_path():

    try:
        path = os.environ['TEST_PATH']
    except KeyError:
        pytest.skip("Could not find environment variable: $TEST_PATH")

    if not os.path.exists(path):
        pytest.skip("Could not find path stored in $TEST_PATH: {}".format(path))

    return path


@pytest.fixture(scope='module')
def caldb():

    from recipe_system.cal_service import set_calservice, CalibrationService

    calibration_service = CalibrationService()
    calibration_service.config()
    calibration_service.init(wipe=True)

    set_calservice()

    yield calibration_service


def test_reduce_image(test_path, caldb):

    all_files = glob.glob(
        os.path.join(test_path, 'F2/test_reduce/', '*.fits'))
    assert len(all_files) > 1

    darks_3s = dataselect.select_data(
        all_files, ['F2', 'DARK', 'RAW'], [],
        dataselect.expr_parser('exposure_time==3'))

    darks_20s = dataselect.select_data(
        all_files, ['F2', 'DARK', 'RAW'], [],
        dataselect.expr_parser('exposure_time==20'))

    darks_120s = dataselect.select_data(
        all_files, ['F2', 'DARK', 'RAW'], [],
        dataselect.expr_parser('exposure_time==120'))

    flats = dataselect.select_data(
        all_files, ['F2', 'FLAT', 'RAW'], [],
        dataselect.expr_parser('filter_name=="Y"'))

    science = dataselect.select_data(
        all_files, ['F2', 'RAW'], ['CAL'],
        dataselect.expr_parser('filter_name=="Y"'))

    for darks in [darks_3s, darks_20s, darks_120s]:

        reduce_darks = Reduce()
        assert len(reduce_darks.files) == 0

        reduce_darks.files.extend(darks)
        assert len(reduce_darks.files) == len(darks)

        reduce_darks.runr()

    reduce_bpm = Reduce()
    reduce_bpm.files.extend(flats)
    reduce_bpm.files.extend(darks_3s)
    reduce_bpm.recipename = 'makeProcessedBPM'
    reduce_bpm.runr()

    bpm_filename = reduce_bpm.output_filenames[0]

    reduce_flats = Reduce()
    reduce_flats.files.extend(flats)
    reduce_flats.uparms = [('addDQ:user_bpm', bpm_filename)]
    reduce_flats.runr()

    reduce_target = Reduce()
    reduce_target.files.extend(science)
    reduce_target.uparms = [('addDQ:user_bpm', bpm_filename)]
    reduce_target.runr()


if __name__ == '__main__':
    pytest.main()

