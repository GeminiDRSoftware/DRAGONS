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

    list_of_science_files = dataselect.select_data(
        all_files, [], [],
        dataselect.expr_parser(
            'observation_class=="science" and exposure_time==60.'))
    list_of_science_files.sort()

    for darks in [list_of_darks]:

        reduce_darks = Reduce()
        assert len(reduce_darks.files) == 0

        reduce_darks.files.extend(darks)
        assert len(reduce_darks.files) == len(darks)

        logutils.config(file_name='gsaoi_test_reduce_dark.log', mode='quiet')
        reduce_darks.runr()

    logutils.config(file_name='gsaoi_test_reduce_bpm.log', mode='quiet')
    reduce_bpm = Reduce()
    reduce_bpm.files.extend(list_of_h_flats)
    reduce_bpm.files.extend(list_of_darks)
    reduce_bpm.recipename = 'makeProcessedBPM'
    reduce_bpm.runr()

    bpm_filename = reduce_bpm.output_filenames[0]

    logutils.config(file_name='gsaoi_test_reduce_flats.log', mode='quiet')
    reduce_flats = Reduce()
    reduce_flats.files.extend(list_of_kshort_flats)
    reduce_flats.uparms = [('addDQ:user_bpm', bpm_filename)]
    reduce_flats.runr()

    calib_files.append(
        'processed_flat:{}'.format(reduce_flats.output_filenames[0])
    )

    logutils.config(file_name='gsaoi_test_reduce_science.log', mode='quiet')
    reduce_target = Reduce()
    reduce_target.files.extend(list_of_science_files)
    reduce_target.uparms = [('addDQ:user_bpm', bpm_filename)]
    reduce_target.ucals = normalize_ucals(reduce_target.files, calib_files)
    reduce_target.runr()


if __name__ == '__main__':
    pytest.main()
