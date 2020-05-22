#!/usr/bin/env python

import glob
import pytest
import os

from gempy.adlibrary import dataselect
from recipe_system.reduction.coreReduce import Reduce
from recipe_system.utils.reduce_utils import normalize_ucals
from recipe_system.testing import reduce_bias, reduce_flat

from gempy.utils import logutils

datasets = [

    # GN HAM 2x2 z-band
    ("GN-2017B-LP-15", [
        "N20170912S0295.fits",
        "N20170912S0296.fits",
        "N20170912S0297.fits",
        "N20170912S0298.fits",
        "N20170912S0299.fits",
        "N20170913S0153.fits",
        "N20170913S0154.fits",
        "N20170913S0155.fits",
        "N20170913S0156.fits",
        "N20170913S0157.fits",
        "N20170913S0158.fits",
        "N20170914S0481.fits",
        "N20170914S0482.fits",
        "N20170914S0483.fits",
        "N20170914S0484.fits",
        "N20170914S0485.fits",
        "N20170915S0274.fits",
        "N20170915S0275.fits",
        "N20170915S0276.fits",
        "N20170915S0277.fits",
        "N20170915S0278.fits",
        "N20170915S0279.fits",
        "N20170915S0280.fits",
        "N20170915S0281.fits",
        "N20170915S0282.fits",
        "N20170915S0283.fits",
        "N20170915S0284.fits",
        "N20170915S0285.fits",
        "N20170915S0286.fits",
        "N20170915S0287.fits",
        "N20170915S0337.fits",
        "N20170915S0338.fits",
        "N20170915S0339.fits",
        "N20170915S0340.fits",
        "N20170915S0341.fits",
    ]),

]


# These tests need refactoring to reduce the replication of API boilerplate
@pytest.mark.skip('TODO: Reactivate me')
@pytest.mark.parametrize("program_id,list_of_files", datasets)
def test_reduce_image(cache_file_from_archive, change_working_dir, list_of_files,
                      program_id, reduce_bias, reduce_flat):
    with change_working_dir():
        os.mkdir(program_id)
        os.chdir(program_id)
        paths = [cache_file_from_archive(f) for f in list_of_files]

        master_bias = reduce_bias(program_id,
                                  dataselect.select_data(paths, tags=['BIAS']))

        master_flat = reduce_flat(program_id,
                                  dataselect.select_data(paths, tags=['FLAT']))


# noinspection PyPep8Naming
@pytest.mark.skip('TODO: Reactivate me')
@pytest.mark.integtest
def test_reduce_image_GN_HAM_2x2_z(path_to_inputs):
    objgraph = pytest.importorskip("objgraph")

    logutils.config(file_name='gmos_test_reduce_image_GN_HAM_2x2_z.log')

    calib_files = []

    raw_subdir = 'GMOS/GN-2017B-LP-15'
    all_files = sorted(glob.glob(
        os.path.join(path_to_inputs, raw_subdir, '*.fits')))
    assert len(all_files) > 1

    list_of_bias = dataselect.select_data(all_files, ['BIAS'], [])

    expr = dataselect.expr_parser('filter_name=="z"')
    list_of_z_flats = dataselect.select_data(all_files, ['TWILIGHT'], [], expr)

    expr = dataselect.expr_parser(
        'observation_class=="science" and filter_name=="z"'
    )
    list_of_science = dataselect.select_data(all_files, [], ['CAL'], expr)

    def reduce(filelist, saveto=None, label='', calib_files=None,
               recipename=None):
        red = Reduce()
        assert len(red.files) == 0
        red.files.extend(filelist)
        assert len(red.files) == len(filelist)
        if calib_files:
            red.ucals = normalize_ucals(red.files, calib_files)
        if recipename:
            red.recipename = recipename
        red.runr()
        if saveto:
            calib_files.append(f'{saveto}:{red.output_filenames[0]}')

        # check that we are not leaking objects
        assert len(objgraph.by_type('NDAstroData')) == 0

    reduce(list_of_bias, saveto='processed_bias', label='bias',
           calib_files=calib_files)
    reduce(list_of_z_flats, saveto='processed_flat', label='flat',
           calib_files=calib_files)

    # If makeFringe is included in the science recipe, this can be omitted:
    reduce(list_of_science, saveto='processed_fringe', label='fringe',
           calib_files=calib_files, recipename='makeProcessedFringe')

    reduce(list_of_science, label='science', calib_files=calib_files)


# noinspection PyPep8Naming
@pytest.mark.skip(reason="Investigate MemoryError")
@pytest.mark.integtest
def test_reduce_image_GN_EEV_2x2_g(path_to_inputs):
    logutils.config(file_name='gmos_test_reduce_image_GN_EEV_2x2_g.log')

    calib_files = []

    raw_subdir = 'GMOS/GN-2002A-Q-89'

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

    calib_files.append(
        'processed_bias:{}'.format(reduce_bias.output_filenames[0])
    )

    reduce_flats = Reduce()
    reduce_flats.files.extend(list_of_flats)
    reduce_flats.ucals = normalize_ucals(reduce_flats.files, calib_files)
    reduce_flats.runr()

    calib_files.append(
        'processed_flat:{}'.format(reduce_flats.output_filenames[0])
    )

    reduce_target = Reduce()
    reduce_target.files.extend(list_of_science_files)
    reduce_target.ucals = normalize_ucals(reduce_target.files, calib_files)
    reduce_target.runr()


# noinspection PyPep8Naming
@pytest.mark.skip(reason="Investigate MemoryError")
@pytest.mark.integtest
def test_reduce_image_GS_HAM_1x1_i(path_to_inputs):
    logutils.config(file_name='gmos_test_reduce_image_GS_HAM_1x1_i.log')

    calib_files = []

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

    calib_files.append(
        'processed_bias:{}'.format(reduce_bias.output_filenames[0])
    )

    reduce_flats = Reduce()
    reduce_flats.files.extend(list_of_sci_flats)
    # reduce_flats.uparms = [('addDQ:user_bpm', 'fixed_bpm_1x1_FullFrame.fits')]
    reduce_flats.ucals = normalize_ucals(reduce_flats.files, calib_files)
    reduce_flats.runr()

    calib_files.append(
        'processed_flat:{}'.format(reduce_flats.output_filenames[0])
    )

    reduce_target = Reduce()
    reduce_target.files.extend(list_of_science_files)
    reduce_target.ucals = normalize_ucals(reduce_target.files, calib_files)
    reduce_target.uparms = [
        ('stackFrames:memory', 1),
        # ('addDQ:user_bpm', 'fixed_bpm_1x1_FullFrame.fits'),
        ('adjustWCSToReference:rotate', True),
        ('adjustWCSToReference:scale', True),
        ('resampleToCommonFrame:interpolator', 'spline3')
    ]
    reduce_target.runr()


# noinspection PyPep8Naming
@pytest.mark.skip(reason="Investigate MemoryError")
@pytest.mark.integtest
def test_reduce_image_GS_HAM_2x2_i_std(path_to_inputs):
    logutils.config(file_name='gmos_test_reduce_image_GS_HAM_1x1_i.log')

    calib_files = []

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

    calib_files.append(
        'processed_bias:{}'.format(reduce_bias.output_filenames[0])
    )

    reduce_flats = Reduce()
    reduce_flats.files.extend(list_of_sci_flats)
    # reduce_flats.uparms = [('addDQ:user_bpm', 'fixed_bpm_2x2_FullFrame.fits')]
    reduce_flats.ucals = normalize_ucals(reduce_flats.files, calib_files)
    reduce_flats.runr()

    calib_files.append(
        'processed_flat:{}'.format(reduce_flats.output_filenames[0])
    )

    reduce_target = Reduce()
    reduce_target.files.extend(list_of_science_files)
    reduce_target.ucals = normalize_ucals(reduce_target.files, calib_files)
    reduce_target.uparms = [
        ('stackFrames:memory', 1),
        # ('addDQ:user_bpm', 'fixed_bpm_2x2_FullFrame.fits'),
        ('resampleToCommonFrame:interpolator', 'spline3')
    ]
    reduce_target.runr()


if __name__ == '__main__':
    pytest.main()
