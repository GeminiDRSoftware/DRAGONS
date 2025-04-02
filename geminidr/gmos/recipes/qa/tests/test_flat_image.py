#!/usr/bin/env python
"""
Tests the GMOS Image Bias reduction for QA mode.
"""
import glob
import os

import pytest
import shutil

import astrodata
import gemini_instruments
import numpy as np

import geminidr
from astrodata.testing import download_from_archive
from geminidr.gmos.primitives_gmos_image import GMOSImage
from gempy.utils import logutils
from recipe_system.cal_service import UserDB
from recipe_system.cal_service.caldb import CalReturn
from recipe_system.reduction.coreReduce import Reduce
from recipe_system.utils.reduce_utils import normalize_ucals

test_data = [
    # GMOS-N HeII
    ("N20190101S0494_bias.fits",
     ["N20181201S{:04d}.fits".format(n) for n in range(283, 286)]),
]


@pytest.mark.gmosimage
@pytest.mark.integration_test
@pytest.mark.parametrize("master_bias, flat_fnames", test_data)
def test_make_processed_flat(
        change_working_dir, flat_fnames, master_bias, path_to_inputs):
    """
    Regression test for GMOS

    Parameters
    ----------
    change_working_dir : fixture
        Custom fixture defined astrodata.testing that changes temporarily the
        working dir.
    flat_fnames : list
        Contains the flat names that will be reduced.
    master_bias : str
        Contains the name of the master flat.
    path_to_inputs : fixture
        Custom fixture that defines where the input data is stored.
    """
    master_bias = os.path.join(path_to_inputs, master_bias)
    calibration_files = ['processed_bias:{}'.format(master_bias)]

    with change_working_dir():
        logutils.config(file_name=f"log_flat_{flat_fnames[0].split('.')[0]}.txt")
        r = Reduce()

        # TODO map actual bpms and update reference files as needed
        # just getting to a baseline passing test first
        def patch(clazz, name, replacement):
            def wrap_original(orig):
                # when called with the original function, a new function will be returned
                # this new function, the wrapper, replaces the original function in the class
                # and when called it will call the provided replacement function with the
                # original function as first argument and the remaining arguments filled in by Python

                def wrapper(*args, **kwargs):
                    return replacement(orig, *args, **kwargs)

                return wrapper

            orig = getattr(clazz, name)
            setattr(clazz, name, wrap_original(orig))

        def mock_get_processed_bpm(orig, self, adinputs, caltype, *args, **kwargs):
            if caltype != 'processed_bpm':
                return orig(self, adinputs, caltype, *args, **kwargs)
            bpmfiles = []
            for ad in adinputs:
                inst = ad.instrument()  # Could be GMOS-N or GMOS-S
                xbin = ad.detector_x_bin()
                ybin = ad.detector_y_bin()
                det = ad.detector_name(pretty=True)[:3]
                amps = '{}amp'.format(3 * ad.phu['NAMPS'])
                mos = '_mosaic' if (ad.phu.get(geminidr.gemini.lookups.timestamp_keywords.timestamp_keys['mosaicDetectors'])
                                    or ad.phu.get(geminidr.gemini.lookups.timestamp_keywords.timestamp_keys['tileArrays'])) else ''
                mode_key = '{}_{}_{}{}_{}'.format(inst, det, xbin, ybin, amps)

                db_matches = sorted((k, v) for k, v in geminidr.gmos.lookups.maskdb.bpm_dict.items() \
                                    if k.startswith(mode_key) and k.endswith(mos))

                # If BPM(s) matched, use the one with the latest version number suffix:
                if db_matches:
                    bpm = db_matches[-1][1]
                else:
                    bpm = None

                if bpm is None:
                    bpmfiles.append(bpm)
                else:
                    # Prepend standard path if the filename doesn't start with '/'
                    bpm_dir = os.path.join(os.path.dirname(geminidr.gmos.lookups.maskdb.__file__), 'BPM')
                    bpmfiles.append(bpm if bpm.startswith(os.path.sep) else os.path.join(bpm_dir, bpm))

            return CalReturn(bpmfiles, [None]*len(bpmfiles))

        # I can't get monkeypatch to do the right thing, we don't have a UserDB instance yet...
        # TODO remove all of this mocking and drive the test off of bpm cal matching with new bpms and new refs
        patch(UserDB, "_get_calibrations", mock_get_processed_bpm)

        r.files = [download_from_archive(f) for f in flat_fnames]
        r.mode = 'qa'
        r.ucals = normalize_ucals(calibration_files)
        r.runr()

        # Delete files that won't be used
        shutil.rmtree('calibrations/')
        [os.remove(f) for f in glob.glob('*_forStack.fits')]

        ad = astrodata.from_file(r.output_filenames[0])

    for ext in ad:
        data = np.ma.masked_array(ext.data, mask=ext.mask)

        if not data.mask.all():
            np.testing.assert_allclose(np.ma.median(data.ravel()), 1, atol=0.072)
            np.testing.assert_allclose(data[~data.mask], 1, atol=0.45)

    # plot(ad)


# -- Helper functions ----------------------------------------------------------
def plot(ad):
    """
    Displays the tiled arrays with the DQ mask for analysing the data.

    Parameters
    ----------
    ad : multi-extension data
    """
    import numpy as np
    import matplotlib.pyplot as plt

    from astropy.visualization import ImageNormalize, ZScaleInterval
    from copy import deepcopy
    from geminidr.gmos.primitives_gmos_image import GMOSImage

    p = GMOSImage([deepcopy(ad)])
    _ad = p.tileArrays().pop()

    fig, axs = plt.subplots(num=ad.filename, ncols=len(_ad), sharey=True)

    norm = ImageNormalize(
        np.concatenate([ext.data.ravel()[ext.mask.ravel() == 0] for ext in _ad]),
        interval=ZScaleInterval())

    vmin = norm.vmin
    vmax = norm.vmax

    for i, ext in enumerate(_ad):

        data = np.ma.masked_array(ext.data, mask=ext.mask)
        cmap = plt.get_cmap('viridis')
        cmap.set_bad('red', alpha='0.5')
        axs[i].imshow(data, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
        # axs[i].imshow(data.data, origin='lower', vmin=vmin, vmax=vmax)

    plt.show()


# -- Recipe to create pre-processed data ---------------------------------------
def create_master_bias_for_tests():
    """
    Creates input bias data for tests.

    The raw files will be downloaded and saved inside the path stored in the
    `$DRAGONS_TEST/raw_inputs` directory.

    Processed files will be stored inside a new folder called
    "./dragons_test_inputs". The sub-directory structure should reflect the one
    returned by the `path_to_inputs` fixture.
    """
    bias_datasets = [
        [f"N20190101S{n:04d}.fits" for n in range(494, 499)],
    ]

    root_path = os.path.join("./dragons_test_inputs/")
    module_path = "geminidr/gmos/recipes/qa/{:s}/inputs".format(__file__.split('.')[0])
    path = os.path.join(root_path, module_path)
    os.makedirs(path, exist_ok=True)
    os.chdir(path)

    print('\n  Current working directory:\n    {:s}'.format(os.getcwd()))

    for filenames in bias_datasets:

        print('  Downloading files...')
        paths = [download_from_archive(f) for f in filenames]

        f = paths[0]
        ad = astrodata.from_file(f)

        if not os.path.exists(f.replace('.fits', '_bias.fits')):
            print(f"  Creating input file:\n")
            print(f"    {os.path.basename(f).replace('.fits', '_bias.fits')}")
            logutils.config(file_name=f'log_arc_{ad.data_label()}.txt')
            r = Reduce()
            r.files.extend(paths)
            r.runr()
        else:
            print(f"  Input file already exists:")
            print(f"    {f.replace('.fits', '_bias.fits')}")


if __name__ == '__main__':
    from sys import argv

    if '--create-inputs' in argv[1:]:
        create_master_bias_for_tests()
    else:
        pytest.main()
