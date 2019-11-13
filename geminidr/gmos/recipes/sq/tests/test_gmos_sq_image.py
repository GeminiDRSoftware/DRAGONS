#!/usr/bin/env python

import glob
import pytest
import os

from gempy.adlibrary import dataselect
from recipe_system.reduction.coreReduce import Reduce
from recipe_system.utils.reduce_utils import normalize_ucals

from gempy.utils import logutils

import datetime, psutil, objgraph


class MemoryLogger:
    def __init__(self, filename):
        self.filename = filename
        self.process = psutil.Process()
        self.log(f'start {datetime.datetime.now()}')

    def log(self, msg):
        nfitsfiles = len([f for f in self.process.open_files() if
                          f.path.endswith('fits')])
        mem = self.process.memory_info().rss / 1024**2
        refs = {name: len(objgraph.by_type(name)) for name in
                ('ImageHDU', 'NDAstroData', 'HDUList')}

        with open(self.filename, 'a') as f:
            f.write(f'{mem:.1f} : {msg}, {refs}, {nfitsfiles} FITS\n')

# These tests need refactoring to reduce the replication of API boilerplate

# noinspection PyPep8Naming
@pytest.mark.integtest
def test_reduce_image_GN_HAM_2x2_z(path_to_inputs):
    logutils.config(file_name='gmos_test_reduce_image_GN_HAM_2x2_z.log')
    mem = MemoryLogger('gmos_test_reduce_image_GN_HAM_2x2_z_mem.log')

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
        mem.log(f'start {label}')
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
        mem.log(f'{label} done')

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
