#!/usr/bin/env python

import glob
import pytest
import os

from gempy.adlibrary import dataselect
from gempy.utils import logutils
from recipe_system.reduction.coreReduce import Reduce
from recipe_system.utils.reduce_utils import normalize_ucals


@pytest.fixture
def test_path():

    try:
        path = os.environ['TEST_PATH']
    except KeyError:
        pytest.skip("Could not find environment variable: $TEST_PATH")

    if not os.path.exists(path):
        pytest.skip("Could not find path stored in $TEST_PATH: {}".format(path))

    return path


def test_reduce_image(test_path):

    calib_files = []

    all_files = glob.glob(
        os.path.join(test_path, 'F2/test_reduce/', '*.fits'))

    all_files.sort()

    assert len(all_files) > 1

    darks_3s = dataselect.select_data(
        all_files, ['F2', 'DARK', 'RAW'], [],
        dataselect.expr_parser('exposure_time==3'))
    darks_3s.sort()

    darks_20s = dataselect.select_data(
        all_files, ['F2', 'DARK', 'RAW'], [],
        dataselect.expr_parser('exposure_time==20'))
    darks_20s.sort()

    darks_120s = dataselect.select_data(
        all_files, ['F2', 'DARK', 'RAW'], [],
        dataselect.expr_parser('exposure_time==120'))
    darks_120s.sort()

    flats = dataselect.select_data(
        all_files, ['F2', 'FLAT', 'RAW'], [],
        dataselect.expr_parser('filter_name=="Y"'))
    flats.sort()

    science = dataselect.select_data(
        all_files, ['F2', 'RAW'], ['CAL'],
        dataselect.expr_parser('filter_name=="Y"'))
    science.sort()

    for darks in [darks_3s, darks_20s, darks_120s]:

        reduce_darks = Reduce()
        assert len(reduce_darks.files) == 0

        reduce_darks.files.extend(darks)
        assert len(reduce_darks.files) == len(darks)

        logutils.config(file_name='f2_test_reduce_darks.log', mode='quiet')
        reduce_darks.runr()

        calib_files.append(
            'processed_dark:{}'.format(reduce_darks.output_filenames[0])
        )

    logutils.config(file_name='f2_test_reduce_bpm.log', mode='quiet')
    reduce_bpm = Reduce()
    reduce_bpm.files.extend(flats)
    reduce_bpm.files.extend(darks_3s)
    reduce_bpm.recipename = 'makeProcessedBPM'
    reduce_bpm.runr()

    bpm_filename = reduce_bpm.output_filenames[0]

    logutils.config(file_name='f2_test_reduce_flats.log', mode='quiet')
    reduce_flats = Reduce()
    reduce_flats.files.extend(flats)
    reduce_flats.uparms = [('addDQ:user_bpm', bpm_filename)]
    reduce_flats.runr()

    calib_files.append(
        'processed_flat:{}'.format(reduce_flats.output_filenames[0])
    )

    logutils.config(file_name='f2_test_reduce_science.log', mode='quiet')
    reduce_target = Reduce()
    reduce_target.files.extend(science)
    reduce_target.uparms = [('addDQ:user_bpm', bpm_filename)]
    reduce_target.ucals = normalize_ucals(reduce_target.files, calib_files)
    reduce_target.runr()


if __name__ == '__main__':
    pytest.main()
