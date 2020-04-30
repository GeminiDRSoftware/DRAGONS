#!/usr/bin/env python
import os

import numpy as np
import pytest
import requests

import astrodata

from astrodata import testing as ad_testing
from geminidr.core import primitives_visualize
from recipe_system.testing import reduce_arc


def test_mosaic_detectors_gmos_binning(astrofaker):
    """
    Tests that the spacing between amplifier centres for NxN binned data
    is precisely N times smaller than for unbinned data when run through
    mosaicDetectors()
    """
    from geminidr.gmos.primitives_gmos_image import GMOSImage
    astrofaker = pytest.importorskip("astrofaker")

    for hemi in 'NS':
        for ccd in ('EEV', 'e2v', 'Ham'):
            for binning in (1, 2, 4):
                try:
                    ad = astrofaker.create('GMOS-{}'.format(hemi), ['IMAGE', ccd])
                except ValueError:  # No e2v for GMOS-S
                    continue
                ad.init_default_extensions(binning=binning, overscan=False)
                for ext in ad:
                    shape = ext.data.shape
                    ext.add_star(amplitude=10000, x=0.5 * (shape[1] - 1),
                                 y=0.5 * (shape[0] - 1), fwhm=0.5 * binning)
                p = GMOSImage([ad])
                ad = p.mosaicDetectors([ad])[0]
                ad = p.detectSources([ad])[0]
                x = np.array(sorted(ad[0].OBJCAT['X_IMAGE']))
                if binning == 1:
                    unbinned_positions = x
                else:
                    diffs = np.diff(unbinned_positions) - binning * np.diff(x)
                    assert np.max(abs(diffs)) < 0.01


@pytest.mark.parametrize("input_ad", ["N20180112S0209.fits"], indirect=True)
@pytest.mark.usefixtures("check_adcc")
def test_plot_spectra_for_qa_single_frame(input_ad):
    p = primitives_visualize.Visualize([])
    p.plotSpectraForQA(adinputs=[input_ad])
    assert True


@pytest.fixture(scope='module')
def check_adcc():
    try:
        _ = requests.get(url="http://localhost:8777/rqsite.json")
        print("ADCC is up and running!")
    except requests.exceptions.ConnectionError:
        pytest.skip("ADCC is not running.")


@pytest.fixture(scope='module')
def input_ad(cache_file_from_archive, path_to_inputs, reduce_arc,
             reduce_data, request):

    basename = request.param
    should_preprocess = request.config.getoption("--force-preprocess-data")

    input_fname = basename.replace('.fits', '_flatCorrected.fits')
    input_path = os.path.join(path_to_inputs, input_fname)

    if os.path.exists(input_path):
        input_data = astrodata.open(input_path)

    elif should_preprocess:
        filename = cache_file_from_archive(basename)
        ad = astrodata.open(filename)
        cals = ad_testing.get_associated_calibrations(basename)

        arcs = [cache_file_from_archive(a)
                for a in cals[cals.caltype == 'arc'].filename.values]

        master_arc = reduce_arc(ad.data_label(), arcs)
        input_data = reduce_data(ad, master_arc)

    else:
        raise IOError(
            'Could not find input file:\n' +
            '  {:s}\n'.format(input_path) +
            '  Run pytest with "--force-preprocess-data" to get it')

    return input_data


@pytest.fixture(scope='module')
def reduce_data(enter_path_to_outputs):
    from geminidr.gmos.primitives_gmos_longslit import GMOSLongslit
    from gempy.utils import logutils

    def _reduce_data(ad, arc):
        with enter_path_to_outputs():
            print('Current working directory:\n    {:s}'.format(os.getcwd()))
            logutils.config(file_name='log_{}.txt'.format(ad.data_label()))

            p = GMOSLongslit([ad])
            p.prepare()
            p.addDQ(static_bpm=None)
            p.addVAR(read_noise=True)
            p.overscanCorrect()
            # p.biasCorrect()
            p.ADUToElectrons()
            p.addVAR(poisson_noise=True)
            # p.flatCorrect()
            # p.applyQECorrection()
            p.distortionCorrect(arc=arc)
            p.findSourceApertures(max_apertures=1)
            p.skyCorrectFromSlit()
            p.traceApertures()
            p.extract1DSpectra()
            p.linearizeSpectra()
            p.calculateSensitivity()

            ad = p.writeOutputs().pop()
        return ad

    return _reduce_data
