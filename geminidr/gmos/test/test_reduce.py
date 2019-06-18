#!/usr/bin/env python

import glob
import pytest
import os

from gempy.adlibrary import dataselect
from recipe_system.reduction.coreReduce import Reduce

from gempy.utils import logutils


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


def test_reduce_image_GN_HAM_2x2_z(path_to_raw_files, caldb):

    logutils.config(file_name='gmos_test_reduce_image_GN_HAM_2x2_z.log')

    caldb.init(wipe=True)

    raw_subdir = 'GMOS/GN-2017B-LP-15'

    all_files = glob.glob(
        os.path.join(path_to_raw_files, raw_subdir, '*.fits'))
    assert len(all_files) > 1

    list_of_bias = dataselect.select_data(
        all_files,
        ['BIAS'],
        []
    )

    list_of_z_flats = dataselect.select_data(
        all_files,
        ['TWILIGHT'],
        [],
        dataselect.expr_parser('filter_name=="z"')
    )

    list_of_science_files = dataselect.select_data(
        all_files, [],
        ['CAL'],
        dataselect.expr_parser(
            'observation_class=="science" and filter_name=="z"'
        )
    )

    reduce_bias = Reduce()
    assert len(reduce_bias.files) == 0

    reduce_bias.files.extend(list_of_bias)
    assert len(reduce_bias.files) == len(list_of_bias)

    reduce_bias.runr()

    caldb.add_cal(reduce_bias.output_filenames[0])

    reduce_flats = Reduce()
    reduce_flats.files.extend(list_of_z_flats)
    reduce_flats.runr()

    caldb.add_cal(reduce_flats.output_filenames[0])

    reduce_target = Reduce()
    reduce_target.files.extend(list_of_science_files)
    reduce_target.runr()

    for f in caldb.list_files():
        print(f)


if __name__ == '__main__':
    pytest.main()
