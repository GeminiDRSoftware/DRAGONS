#!/usr/bin/env python
import os
import itertools
import time

import numpy as np
import pytest
import requests

import astrodata
from astrodata.testing import download_from_archive
from geminidr.core import primitives_visualize
from geminidr.gmos.primitives_gmos_image import GMOSImage


single_aperture_data = [
    # (Input Files, Associated Bias, Associated Flats, Associated Arc)
    (["N20180112S0209.fits"], [], [], ["N20180112S0353.fits"]),
    ([f"S20190103S{i:04d}.fits" for i in range(138, 141)], [], [],
     ["S20190103S0136.fits"]),
    (["N20180521S0101.fits"],
     [f"N20180521S{i:04d}.fits" for i in range(217, 222)],
     ["N20180521S0100.fits", "N20180521S0102.fits"], ["N20180521S0185.fits"]),
]

HEMI = 'NS'
CCD = ('EEV', 'e2v', 'Ham')


@pytest.mark.parametrize('hemi, ccd', list(itertools.product(HEMI, CCD)))
def test_mosaic_detectors_gmos_binning(astrofaker, hemi, ccd):
    """
    Tests that the spacing between amplifier centres for NxN binned data
    is precisely N times smaller than for unbinned data when run through
    mosaicDetectors()
    """
    for binning in (1, 2, 4):
        try:
            ad = astrofaker.create('GMOS-{}'.format(hemi), ['IMAGE', ccd])
        except ValueError:  # No e2v for GMOS-S
            pytest.skip()

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

@pytest.mark.skip(reason='Relies on reading duplicate warning messages')
def test_mosaic_detectors_raises_warning_with_different_gains(astrofaker, caplog):
    ad = astrofaker.create('GMOS-N', ['IMAGE'])
    ad.init_default_extensions(overscan=False)
    p = GMOSImage([ad])
    p.mosaicDetectors()
    assert sum(["have different gains" in rec.msg for rec in caplog.records]) == 1

@pytest.mark.skip(reason='Relies on reading duplicate warning messages')
def test_tile_arrays_raises_warning_with_different_gains(astrofaker, caplog):
    ad = astrofaker.create('GMOS-N', ['IMAGE'])
    ad.init_default_extensions(overscan=False)
    p = GMOSImage([ad])
    p.tileArrays(tile_all=True)
    assert sum(["have different gains" in rec.msg for rec in caplog.records]) == 1
    caplog.clear()
    p = GMOSImage([ad])
    p.tileArrays(tile_all=False)
    assert sum(["have different gains" in rec.msg for rec in caplog.records]) == 3
    caplog.clear()
    p = GMOSImage([ad])
    p.prepare()
    p.ADUToElectrons()  # should set gain=1
    p.tileArrays(tile_all=False)
    assert sum(["have different gains" in rec.msg for rec in caplog.records]) == 0


def test_tile_arrays_does_not_raise_different_gain_warning_from_display(astrofaker, caplog):
    ad = astrofaker.create('GMOS-N', ['IMAGE'])
    ad.init_default_extensions(overscan=False)
    p = GMOSImage([ad])
    p.display()
    assert sum(["have different gains" in rec.msg for rec in caplog.records]) == 0


def test_tile_arrays_creates_average_read_noise(astrofaker):
    ad = astrofaker.create('GMOS-N', ['IMAGE'])
    ad.init_default_extensions(overscan=False)
    p = GMOSImage([ad])
    p.prepare()
    rn = ad.read_noise()
    ad = p.tileArrays(tile_all=True).pop()
    assert ad.read_noise()[0] == np.mean(rn)


def test_mosaicking_unaffected_by_tiling(astrofaker):
    """Confirm that tiling the amps on each CCD before mosaicking
    produces the same result as mosaicking the 12 extensions"""
    ad = astrofaker.create('GMOS-N', ['IMAGE'])
    ad.init_default_extensions(overscan=False)
    ad.hdr['GAIN'] = 1  # avoid wanings when tiling/mosaicking
    ad.phu['GPREPARE'] = "YES"  # ...continued
    ad.add_read_noise()
    p = GMOSImage([ad])
    p.tileArrays(tile_all=False, outstream="tiled")
    p.mosaicDetectors()
    p.mosaicDetectors(stream="tiled")
    np.testing.assert_allclose(p.streams['main'][0][0].data,
                               p.streams['tiled'][0][0].data)


@pytest.mark.preprocessed_data
@pytest.mark.parametrize("input_ads", single_aperture_data, indirect=True)
@pytest.mark.usefixtures("check_adcc")
def test_plot_spectra_for_qa(input_ads):
    for i, ad in enumerate(input_ads):

        # Plot single frame
        p = primitives_visualize.Visualize([])
        p.plotSpectraForQA(adinputs=[ad])

        # Gives some time to page refresh
        time.sleep(10)

        # Plot Stack
        if i >= 1:
            print('Reducing stack')
            stack_ad = GMOSImage([]).stackFrames(adinputs=input_ads[:i + 1])[0]
            p.plotSpectraForQA(adinputs=[stack_ad])

        # Gives some time to page refresh
        time.sleep(10)


# -- Fixtures -----------------------------------------------------------------
@pytest.fixture(scope='module')
def check_adcc():
    try:
        _ = requests.get(url="http://localhost:8777/rqsite.json")
        print("ADCC is up and running!")
    except requests.exceptions.ConnectionError:
        pytest.skip("ADCC is not running.")


@pytest.fixture(scope='module')
def input_ads(path_to_inputs, request):
    basenames = request.param[0]
    input_fnames = [b.replace('.fits', '_linearized.fits') for b in basenames]
    input_paths = [os.path.join(path_to_inputs, f) for f in input_fnames]

    input_data_list = []
    for p in input_paths:
        if os.path.exists(p):
            input_data_list.append(astrodata.from_file(p))
        else:
            raise FileNotFoundError(p)

    return input_data_list


# -- Input creation functions -------------------------------------------------
def create_inputs():
    """
    Create inputs for `test_plot_spectra_for_qa_single_frame`.

    The raw files will be downloaded and saved inside the path stored in the
    `$DRAGONS_TEST/raw_inputs` directory. Processed files will be stored inside
    a new folder called "dragons_test_inputs". The sub-directory structure
    should reflect the one returned by the `path_to_inputs` fixture.
    """
    import glob
    import os
    from geminidr.gmos.primitives_gmos_longslit import GMOSLongslit
    from gempy.utils import logutils
    from recipe_system.reduction.coreReduce import Reduce
    from recipe_system.utils.reduce_utils import normalize_ucals

    cwd = os.getcwd()
    path = f"./dragons_test_inputs/geminidr/core/{__file__.split('.')[0]}/"
    os.makedirs(path, exist_ok=True)
    os.chdir(path)

    os.makedirs("inputs/", exist_ok=True)

    for raw_list, bias_list, quartz_list, arc_list in single_aperture_data:

        if all([os.path.exists(f"inputs/{s.split('.')[0]}_extracted.fits")
                for s in raw_list]):
            print("Skipping already created input.")
            continue

        raw_paths = [download_from_archive(f) for f in raw_list]
        bias_paths = [download_from_archive(f) for f in bias_list]
        quartz_paths = [download_from_archive(f) for f in quartz_list]
        arc_paths = [download_from_archive(f) for f in arc_list]

        cals = []
        raw_ads = [astrodata.from_file(p) for p in raw_paths]
        data_label = raw_ads[0].data_label()
        print('Current working directory:\n    {:s}'.format(os.getcwd()))

        if len(bias_paths):
            logutils.config(file_name='log_bias_{}.txt'.format(data_label))
            r = Reduce()
            r.files.extend(bias_paths)
            r.runr()
            master_bias = r.output_filenames.pop()
            cals.append(f"processed_bias:{master_bias}")
            del r
        else:
            master_bias = None

        if len(quartz_paths):
            logutils.config(file_name='log_quartz_{}.txt'.format(data_label))
            r = Reduce()
            r.files.extend(quartz_paths)
            r.ucals = normalize_ucals(cals)
            r.runr()
            master_quartz = r.output_filenames.pop()
            cals.append(f"processed_flat:{master_quartz}")
            del r
        else:
            master_quartz = None

        logutils.config(file_name='log_arc_{}.txt'.format(data_label))
        r = Reduce()
        r.files.extend(arc_paths)
        r.ucals = normalize_ucals(cals)
        r.runr()
        master_arc = r.output_filenames.pop()


        do_cal_bias = 'skip' if master_bias is None else 'procmode'
        do_cal_flat = 'skip' if master_quartz is None else 'procmode'

        logutils.config(file_name='log_{}.txt'.format(data_label))
        p = GMOSLongslit(raw_ads)
        p.prepare()
        p.addDQ(static_bpm=None)
        p.addVAR(read_noise=True)
        p.overscanCorrect()
        p.biasCorrect(do_cal=do_cal_bias, bias=master_bias)
        p.ADUToElectrons()
        p.addVAR(poisson_noise=True)
        p.flatCorrect(do_cal=do_cal_flat, flat=master_quartz)
        p.QECorrect(arc=master_arc)
        p.distortionCorrect(arc=master_arc)
        p.findApertures(max_apertures=3)
        p.skyCorrectFromSlit()
        p.traceApertures()
        p.extractSpectra()
        p.linearizeSpectra()

        [os.remove(s) for s in glob.glob("*_arc.fits")]
        [os.remove(s) for s in glob.glob("*_bias.fits")]
        [os.remove(s) for s in glob.glob("*_flat.fits")]
        [os.remove(s) for s in glob.glob("*_mosaic.fits")]

        os.chdir("inputs/")
        print("\n\n    Writing processed files for tests into:\n"
              "    {:s}\n\n".format(os.getcwd()))
        _ = p.writeOutputs()
        os.chdir("../")

    os.chdir(cwd)


if __name__ == "__main__":
    import sys

    if '--create-inputs' in sys.argv[1:]:
        create_inputs()
    else:
        pytest.main()
