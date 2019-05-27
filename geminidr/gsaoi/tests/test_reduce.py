#!/usr/bin/env python

import glob
import pytest
import os

from gempy.adlibrary import dataselect
from recipe_system.reduction.coreReduce import Reduce

from gempy.utils import logutils


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

    from recipe_system.cal_service import CalibrationService

    caldb_folder = os.path.dirname(__file__)
    caldb_conf_file = os.path.join(caldb_folder, 'rsys.cfg')
    caldb_database_file = os.path.join(caldb_folder, 'cal_manager.db')

    with open(caldb_conf_file, 'w') as buffer:

        buffer.write(
            "[calibs]\n"
            "standalone = True\n"
            "database_dir = {:s}".format(caldb_folder)
        )

    print(' Test file path: {}'.format(caldb_conf_file))

    calibration_service = CalibrationService()
    calibration_service.config(config_file=caldb_conf_file)
    calibration_service.init(wipe=True)

    yield calibration_service

    os.remove(caldb_conf_file)
    os.remove(caldb_database_file)


def test_reduce_image(test_path, caldb):

    logutils.config(file_name='gsaoi_test_reduce_image.log')

    caldb.init(wipe=True)

    all_files = glob.glob(
        os.path.join(test_path, 'GSAOI/test_reduce/', '*.fits'))

    all_files.sort()

    assert len(all_files) > 1

    list_of_darks = dataselect.select_data(
        all_files, ['DARK'], [])
    list_of_darks.sort()

    list_of_kshort_flats = dataselect.select_data(
        all_files, ['FLAT'], [],
        dataselect.expr_parser('filter_name=="Kshort"'))
    list_of_kshort_flats.sort()

    list_of_h_flats = dataselect.select_data(
        all_files, ['FLAT'], [],
        dataselect.expr_parser('filter_name=="H"'))
    list_of_h_flats.sort()

    list_of_std_LHS_2026 = dataselect.select_data(
        all_files, [], [],
        dataselect.expr_parser('object=="LHS 2026"'))
    list_of_std_LHS_2026.sort()

    list_of_std_cskd8 = dataselect.select_data(
        all_files, [], [],
        dataselect.expr_parser('object=="cskd-8"'))
    list_of_std_cskd8.sort()

    list_of_science_files = dataselect.select_data(
        all_files, [], [],
        dataselect.expr_parser('observation_class=="science" and exposure_time==60.'))
    list_of_science_files.sort()

    for darks in [list_of_darks]:

        reduce_darks = Reduce()
        assert len(reduce_darks.files) == 0

        reduce_darks.files.extend(darks)
        assert len(reduce_darks.files) == len(darks)

        reduce_darks.runr()

        caldb.add_cal(reduce_darks.output_filenames[0])

    reduce_bpm = Reduce()
    reduce_bpm.files.extend(list_of_h_flats)
    reduce_bpm.files.extend(list_of_darks)
    reduce_bpm.recipename = 'makeProcessedBPM'
    reduce_bpm.runr()

    bpm_filename = reduce_bpm.output_filenames[0]

    reduce_flats = Reduce()
    reduce_flats.files.extend(list_of_kshort_flats)
    reduce_flats.uparms = [('addDQ:user_bpm', bpm_filename)]
    reduce_flats.runr()

    caldb.add_cal(reduce_flats.output_filenames[0])

    reduce_target = Reduce()
    reduce_target.files.extend(list_of_science_files)
    reduce_target.uparms = [('addDQ:user_bpm', bpm_filename)]
    reduce_target.runr()

    for f in caldb.list_files():
        print(f)


if __name__ == '__main__':
    pytest.main()
