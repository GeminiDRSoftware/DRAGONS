#!/usr/bin/env python

import glob
import pytest
import os

from gempy.adlibrary import dataselect
from recipe_system.reduction.coreReduce import Reduce
from recipe_system.utils.reduce_utils import normalize_ucals

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


# These tests need refactoring to reduce the replication of API boilerplate

def test_reduce_image_GN_HAM_2x2_z(path_to_inputs):

    logutils.config(file_name='gmos_test_reduce_image_GN_HAM_2x2_z.log')

    calib_files = []

    raw_subdir = 'GMOS/GN-2017B-LP-15'

    all_files = sorted(glob.glob(
        os.path.join(path_to_inputs, raw_subdir, '*.fits')))
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

    calib_files.append(
        'processed_bias:{}'.format(reduce_bias.output_filenames[0])
    )

    reduce_flats = Reduce()
    reduce_flats.files.extend(list_of_z_flats)
    reduce_flats.ucals = normalize_ucals(reduce_flats.files, calib_files)
    reduce_flats.runr()

    calib_files.append(
        'processed_flat:{}'.format(reduce_flats.output_filenames[0])
    )

    # If makeFringe is included in the science recipe, this can be omitted:
    reduce_fringe = Reduce()
    reduce_fringe.files.extend(list_of_science_files)
    reduce_fringe.ucals = normalize_ucals(reduce_fringe.files, calib_files)
    reduce_fringe.recipename = 'makeProcessedFringe'
    reduce_fringe.runr()

    calib_files.append(
        'processed_fringe:{}'.format(reduce_fringe.output_filenames[0])
    )

    reduce_target = Reduce()
    reduce_target.files.extend(list_of_science_files)
    reduce_target.ucals = normalize_ucals(reduce_target.files, calib_files)
    reduce_target.runr()


def test_reduce_image_GN_EEV_2x2_g(path_to_inputs, caldb):

    logutils.config(file_name='gmos_test_reduce_image_GN_EEV_2x2_g.log')

    raw_subdir = 'GMOS/GN-2002A-Q-89'

    caldb.init(wipe=True)

    all_files = sorted(glob.glob(
        os.path.join(path_to_inputs, raw_subdir, '*.fits')))
    assert len(all_files) > 1

    list_of_bias = dataselect.select_data(
        all_files,
        ['BIAS'],
        []
    )

    list_of_flats = dataselect.select_data(
        all_files,
        ['IMAGE', 'FLAT'],
        [],
        dataselect.expr_parser('filter_name=="g"')
    )

    # These old data don't have an OBSCLASS keyword:
    list_of_science_files = dataselect.select_data(
        all_files, [],
        ['CAL'],
        dataselect.expr_parser(
            'object=="PerseusField4" and filter_name=="g"'
        )
    )

    reduce_bias = Reduce()
    assert len(reduce_bias.files) == 0

    reduce_bias.files.extend(list_of_bias)
    assert len(reduce_bias.files) == len(list_of_bias)

    reduce_bias.runr()

    caldb.add_cal(reduce_bias.output_filenames[0])

    reduce_flats = Reduce()
    reduce_flats.files.extend(list_of_flats)
    reduce_flats.runr()

    caldb.add_cal(reduce_flats.output_filenames[0])

    reduce_target = Reduce()
    reduce_target.files.extend(list_of_science_files)
    reduce_target.runr()

    for f in caldb.list_files():
        print(f)


def test_reduce_image_GS_HAM_1x1_i(path_to_inputs, caldb):

    logutils.config(file_name='gmos_test_reduce_image_GS_HAM_1x1_i.log')

    caldb.init(wipe=True)

    raw_subdir = 'GMOS/GS-2017B-Q-6'

    all_files = sorted(glob.glob(
        os.path.join(path_to_inputs, raw_subdir, '*.fits')))
    assert len(all_files) > 1

    list_of_sci_bias = dataselect.select_data(
        all_files,
        ['BIAS'],
        [],
        dataselect.expr_parser('detector_x_bin==1 and detector_y_bin==1')
    )

    list_of_sci_flats = dataselect.select_data(
        all_files,
        ['TWILIGHT'],
        [],
        dataselect.expr_parser(
            'filter_name=="i" and detector_x_bin==1 and detector_y_bin==1'
        )
    )

    list_of_science_files = dataselect.select_data(
        all_files, [],
        ['CAL'],
        dataselect.expr_parser(
            'observation_class=="science" and filter_name=="i"'
        )
    )

    reduce_bias = Reduce()
    assert len(reduce_bias.files) == 0

    reduce_bias.files.extend(list_of_sci_bias)
    assert len(reduce_bias.files) == len(list_of_sci_bias)

    reduce_bias.runr()

    caldb.add_cal(reduce_bias.output_filenames[0])

    reduce_flats = Reduce()
    reduce_flats.files.extend(list_of_sci_flats)
    # reduce_flats.uparms = [('addDQ:user_bpm', 'fixed_bpm_1x1_FullFrame.fits')]
    reduce_flats.runr()

    caldb.add_cal(reduce_flats.output_filenames[0])

    reduce_target = Reduce()
    reduce_target.files.extend(list_of_science_files)
    reduce_target.uparms = [
        ('stackFrames:memory', 1),
        # ('addDQ:user_bpm', 'fixed_bpm_1x1_FullFrame.fits'),
        ('adjustWCSToReference:rotate', True),
        ('adjustWCSToReference:scale', True),
        ('resampleToCommonFrame:interpolator', 'spline3')
    ]
    reduce_target.runr()

    for f in caldb.list_files():
        print(f)


def test_reduce_image_GS_HAM_2x2_i_std(path_to_inputs, caldb):

    logutils.config(file_name='gmos_test_reduce_image_GS_HAM_1x1_i.log')

    caldb.init(wipe=True)

    raw_subdir = 'GMOS/GS-2017B-Q-6'

    all_files = sorted(glob.glob(
        os.path.join(path_to_inputs, raw_subdir, '*.fits')))
    assert len(all_files) > 1

    list_of_sci_bias = dataselect.select_data(
        all_files,
        ['BIAS'],
        [],
        dataselect.expr_parser('detector_x_bin==2 and detector_y_bin==2')
    )

    list_of_sci_flats = dataselect.select_data(
        all_files,
        ['TWILIGHT'],
        [],
        dataselect.expr_parser(
            'filter_name=="i" and detector_x_bin==2 and detector_y_bin==2'
        )
    )

    list_of_science_files = dataselect.select_data(
        all_files, [],
        [],
        dataselect.expr_parser(
            'observation_class=="partnerCal" and filter_name=="i"'
        )
    )

    reduce_bias = Reduce()
    assert len(reduce_bias.files) == 0

    reduce_bias.files.extend(list_of_sci_bias)
    assert len(reduce_bias.files) == len(list_of_sci_bias)

    reduce_bias.runr()

    caldb.add_cal(reduce_bias.output_filenames[0])

    reduce_flats = Reduce()
    reduce_flats.files.extend(list_of_sci_flats)
    # reduce_flats.uparms = [('addDQ:user_bpm', 'fixed_bpm_2x2_FullFrame.fits')]
    reduce_flats.runr()

    caldb.add_cal(reduce_flats.output_filenames[0])

    reduce_target = Reduce()
    reduce_target.files.extend(list_of_science_files)
    reduce_target.uparms = [
        ('stackFrames:memory', 1),
        # ('addDQ:user_bpm', 'fixed_bpm_2x2_FullFrame.fits'),
        ('resampleToCommonFrame:interpolator', 'spline3')
    ]
    reduce_target.runr()

    for f in caldb.list_files():
        print(f)


if __name__ == '__main__':
    pytest.main()
