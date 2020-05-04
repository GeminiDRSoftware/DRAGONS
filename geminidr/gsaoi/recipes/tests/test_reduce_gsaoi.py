#!/usr/bin/env python

import glob
import os

import pytest

from gempy.adlibrary import dataselect
from gempy.utils import logutils
from recipe_system.reduction.coreReduce import Reduce
from recipe_system.utils.reduce_utils import normalize_ucals

test_files = [
    "S20170505S0095.fits",
    "S20170505S0096.fits",
    "S20170505S0030.fits",
    "S20170505S0031.fits",
    "S20170505S0073.fits",
    "S20170505S0074.fits",
    "S20171208S0053.fits",
    "S20171208S0054.fits",
    "S20171208S0055.fits",
    "S20150609S0022.fits",
    "S20150609S0023.fits",
]


@pytest.mark.integtest
def test_reduce_image(cache_file_from_archive, change_working_dir):
    with change_working_dir():
        calib_files = []
        all_files = [cache_file_from_archive(f) for f in test_files]
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

            del reduce_darks

        logutils.config(file_name='gsaoi_test_reduce_bpm.log', mode='quiet')
        reduce_bpm = Reduce()
        reduce_bpm.files.extend(list_of_h_flats)
        reduce_bpm.files.extend(list_of_darks)
        reduce_bpm.recipename = 'makeProcessedBPM'
        reduce_bpm.runr()

        bpm_filename = reduce_bpm.output_filenames[0]

        del reduce_bpm

        logutils.config(file_name='gsaoi_test_reduce_flats.log', mode='quiet')
        reduce_flats = Reduce()
        reduce_flats.files.extend(list_of_kshort_flats)
        reduce_flats.uparms = [('addDQ:user_bpm', bpm_filename)]
        reduce_flats.runr()

        calib_files.append(
            'processed_flat:{}'.format(reduce_flats.output_filenames[0])
        )

        del reduce_flats

        logutils.config(file_name='gsaoi_test_reduce_science.log', mode='quiet')
        reduce_target = Reduce()
        reduce_target.files.extend(list_of_science_files)
        reduce_target.uparms = [('addDQ:user_bpm', bpm_filename)]
        reduce_target.ucals = normalize_ucals(reduce_target.files, calib_files)
        reduce_target.runr()

        del reduce_target


if __name__ == '__main__':
    pytest.main()
