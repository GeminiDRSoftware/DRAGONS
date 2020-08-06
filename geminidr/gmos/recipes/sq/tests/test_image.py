#!/usr/bin/env python

import glob
import pytest
import os

import astrodata
import gemini_instruments

from astrodata.testing import download_from_archive
from gempy.adlibrary import dataselect
from recipe_system.reduction.coreReduce import Reduce
from recipe_system.utils.reduce_utils import normalize_ucals

from gempy.utils import logutils

test_cases = [
    # "GN HAM 2x2 z-band",
    "GN EEV 2x2 g-band",
]

datasets = {

    "GN HAM 2x2 z-band": {
        "bias": [f"N20170912S{n:04d}.fits" for n in range(295, 300)] +
                [f"N20170914S{n:04d}.fits" for n in range(481, 486)] +
                [f"N20170915S{n:04d}.fits" for n in range(337, 342)],
        "flat": [f"N20170915S{n:04d}.fits" for n in range(274, 288)],
        "sci": [f"N20170913S{n:04d}.fits" for n in range(153, 159)],
    },

    "GN EEV 2x2 g-band": {
        # Only three files to avoid memory errors or to speed up the test
        "bias": [f"N20020214S{n:03d}.fits" for n in range(22, 27)][:3],
        "flat": [f"N20020211S{n:03d}.fits" for n in range(156, 160)][:3],
        "sci": [f"N20020214S{n:03d}.fits" for n in range(59, 64)][:3],
    },

    "GS HAM 1x1 i-band": {
        "bias": [f"S20171204S{n:04d}.fits" for n in range(22, 27)] +
                [f"S20171206S{n:04d}.fits" for n in range(128, 133)],
        "flat": [f"S20171206S{n:04d}.fits" for n in range(120, 128)],
        "sci": [f"S20171205S{n:04d}.fits" for n in range(62, 77)],
    },

    "GS HAM 2x2 i-band std": {
        "bias": [f"S20171204S{n:04d}.fits" for n in range(37, 42)],
        "flat": [f"S20171120S{n:04d}.fits" for n in range(129, 140)],
        "std": ["S20171205S0077.fits"],
    },

}


@pytest.mark.parametrize("test_case", datasets.keys())
def test_reduce_image(change_working_dir, test_case):
    print(test_case, '-' * 50)
    for key, filenames in datasets[test_case].items():
        for filename in filenames:
            path = download_from_archive(filename)
            ad = astrodata.open(path)
            print(ad.filename, ad.observation_class(),
                  ad.object(), ad.detector_x_bin(), ad.detector_y_bin())

    with change_working_dir():
        pass


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
