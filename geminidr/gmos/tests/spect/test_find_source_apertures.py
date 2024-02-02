#!/usr/bin/env python
"""
Tests for GMOS Spect findApertures.
"""
import glob
import os
import numpy as np
import pytest

from itertools import product as cart_product

import astrodata
import gemini_instruments

from astropy.modeling.models import Gaussian1D
from geminidr.gmos.primitives_gmos_spect import GMOSSpect

test_data = [
    ("N20180109S0287_distortionCorrected.fits", 256),  # GN-2017B-FT-20-13-001 B600 0.505um
    ("N20190302S0089_distortionCorrected.fits", 255),  # GN-2019A-Q-203-7-001 B600 0.550um
    ("N20190313S0114_distortionCorrected.fits", 256),  # GN-2019A-Q-325-13-001 B600 0.482um
    ("N20190427S0126_distortionCorrected.fits", 258),  # GN-2019A-FT-206-7-004 R400 0.625um
    ("N20190910S0028_distortionCorrected.fits", 252),  # GN-2019B-Q-313-5-001 B600 0.550um
    ("S20180919S0139_distortionCorrected.fits", 257),  # GS-2018B-Q-209-13-003 B600 0.45um
    ("S20191005S0051_distortionCorrected.fits", 261),  # GS-2019B-Q-132-35-001 R400 0.73um
]


# More real-world test cases with target aperture and snr parameter
# Filename, aperture location, tolerance, snr, # expected results
extra_test_data = [
    ("S20210219S0076_align.fits", [1169], [[1154, 1184]], 5, None),  # GS-2021A-DD-102 Supernova within a galaxy bg
    ("N20210730S0037_mosaic.fits", [1068], [[1063, 1073]], 4, 1)                   # Kathleen test data via Chris, expecting 1 ap
]
extra_test_data.clear()

#################################
# Fake Background Aperture Tests
#################################

# Edit these to provide iterables to iterate over
BACKGROUNDS = (0, 5,)  # overall background level
PEAKS = (50, 75)  # signal in galaxy peak
CONTRASTS = (0.25, 0.3)  # ratio of SN peak to galaxy peak
SEPARATIONS = (8, 12)  # pixel separation between galaxy/SN peaks
GAL_FWHMS = (40,)  # FWHM (pixels) of galaxy
SN_FWHMS = (3, 4)  # FWHM (pixels) of SN

extra_test_data.extend([
    (f"fake_bkgd{bkgd:04.0f}_peak{peak:03.0f}_con{contrast:4.2f}_"
                f"sep{sep:05.2f}_gal{gal_fwhm:5.2f}_sn{sn_fwhm:4.2f}.fits",
     [1024, 1024+sep], [[1024-gal_fwhm, 1023+sep], [1025-sep, 1039]], 3, None)
    for bkgd, peak, contrast, sep, gal_fwhm, sn_fwhm in cart_product(
            BACKGROUNDS, PEAKS, CONTRASTS, SEPARATIONS, GAL_FWHMS, SN_FWHMS)
])

# Some iterations are known to be failure cases
# we mark them accordingly so we can see if they start passing
_xfail_indices = (0, 1, 5, 9, 12, 13, 17, 20, 24, 25, 29)
for idx in _xfail_indices:
    args = extra_test_data[idx]
    extra_test_data[idx] = pytest.param(args, marks=pytest.mark.xfail)

##################################

# Parameters for test_find_apertures_with_fake_data(...)
# ToDo - Explore a bit more the parameter space (e.g., larger seeing)
seeing_pars = [0.5, 1.0, 1.25]
position_pars = [500]
value_pars = [50, 100, 500]

@pytest.mark.gmosls
@pytest.mark.parametrize("peak_position", position_pars)
@pytest.mark.parametrize("peak_value", value_pars)
@pytest.mark.parametrize("seeing", seeing_pars)
def test_find_apertures_with_fake_data(peak_position, peak_value, seeing, astrofaker):
    """
    Creates a fake AD object with a gaussian profile in spacial direction with a
    fwhm defined by the seeing variable in arcsec. Then add some noise, and
    test if p.findAperture can find its position.
    """
    np.random.seed(42)

    gmos_fake_noise = 4 # adu
    gmos_fake_noise *= peak_value / 25 # need to scale to avoid creating
                                       # unrealistically low peak-SNR data
    gmos_plate_scale = 0.0807  # arcsec . px-1
    fwhm_to_stddev = 2 * np.sqrt(2 * np.log(2))

    ad = astrofaker.create('GMOS-S', mode='SPECT')
    ad.init_default_extensions()

    fwhm = seeing / gmos_plate_scale
    stddev = fwhm / fwhm_to_stddev

    model = Gaussian1D(mean=peak_position, stddev=stddev, amplitude=peak_value)
    rows, cols = np.mgrid[:ad.shape[0][0], :ad.shape[0][1]]

    for ext in ad:
        ext.data = model(rows)
        ext.data += np.random.poisson(ext.data)
        ext.data += np.random.normal(scale=gmos_fake_noise, size=ext.shape)
        ext.mask = np.zeros_like(ext.data, dtype=np.uint)

    p = GMOSSpect([ad])
    _ad = p.findApertures(max_apertures=1)[0]

    for _ext in _ad:
        # ToDo - Could we improve the primitive to have atol=0.50 or less?
        np.testing.assert_allclose(_ext.APERTURE['c0'], peak_position, atol=0.6)


@pytest.mark.gmosls
@pytest.mark.preprocessed_data
@pytest.mark.parametrize("ad_and_center", test_data, indirect=True)
def test_find_apertures_using_standard_star(ad_and_center):
    """
    Test that p.findApertures can find apertures in Standard Star (which are
    normally bright) observations.
    """
    ad, expected_center = ad_and_center
    p = GMOSSpect([ad])
    _ad = p.findApertures(max_apertures=1).pop()

    assert hasattr(ad[0], 'APERTURE')
    assert len(ad[0].APERTURE) == 1
    np.testing.assert_allclose(ad[0].APERTURE['c0'], expected_center, 3)


@pytest.mark.skip("MUST WORK; temporary skip")
@pytest.mark.gmosls
@pytest.mark.preprocessed_data
@pytest.mark.parametrize("ad_center_tolerance_snr", extra_test_data, indirect=True)
def test_find_apertures_extra_cases(ad_center_tolerance_snr):
    """
    Test that p.findApertures can find apertures in special test cases, such as
    with galaxy background
    """
    ad, expected_centers, ranges, snr, count = ad_center_tolerance_snr
    args = dict()
    if snr is not None:
        args["min_snr"] = snr
    p = GMOSSpect([ad])
    _ad = p.findApertures(**args).pop()

    assert hasattr(ad[0], 'APERTURE')
    if count is not None:
        assert(len(ad[0].APERTURE) == count)
    if expected_centers is not None:
        apertures = ", ".join([str(ap) for ap in ad[0].APERTURE["c0"]])
        for expected_center, range in zip(expected_centers, ranges):
            range = (expected_center-2, expected_center+2)
            assert len([ap for ap in ad[0].APERTURE['c0'] if (ap <= range[1] and ap >= range[0])]) >= 1, \
                f'{ad.filename} check for aperture at {expected_center} not found within range {range} aps at ' \
                f'{apertures}'


# -- Fixtures -----------------------------------------------------------------
@pytest.fixture(scope='function')
def ad_and_center(path_to_inputs, request):
    """
    Returns the pre-processed spectrum file.

    Parameters
    ----------
    path_to_inputs : pytest.fixture
        Fixture defined in :mod:`astrodata.testing` with the path to the
        pre-processed input file.
    request : pytest.fixture
        PyTest built-in fixture containing information about parent test.

    Returns
    -------
    AstroData
        Input spectrum processed up to right before the `applyQECorrection`.
    """
    filename, center = request.param
    path = os.path.join(path_to_inputs, filename)

    if os.path.exists(path):
        ad = astrodata.open(path)
    else:
        raise FileNotFoundError(path)

    return ad, center


@pytest.fixture(scope='function')
def ad_center_tolerance_snr(path_to_inputs, request):
    """
    Returns the pre-processed spectrum file and some additional input/check parameters.

    Parameters
    ----------
    path_to_inputs : pytest.fixture
        Fixture defined in :mod:`astrodata.testing` with the path to the
        pre-processed input file.
    request : pytest.fixture
        PyTest built-in fixture containing information about parent test.

    Returns
    -------
    AstroData
        Input spectrum processed up to right before the `applyQECorrection`.
    center
        expected location of aperture center(s)
    tolerance
        valid range(s) for each aperture
    snr
        min_snr parameter for find_apertures
    count
        number of apertures expected
    """
    filename, center, range, snr, count = request.param
    path = os.path.join(path_to_inputs, filename)

    if os.path.exists(path):
        ad = astrodata.open(path)
    else:
        raise FileNotFoundError(path)

    return ad, center, range, snr, count


def create_inputs_recipe():
    """
    Creates input data for tests using pre-processed standard star and its
    calibration files.

    The raw files will be downloaded and saved inside the path stored in the
    `$DRAGONS_TEST/raw_inputs` directory. Processed files will be stored inside
    a new folder called "dragons_test_inputs". The sub-directory structure
    should reflect the one returned by the `path_to_inputs` fixture.
    """
    import os
    from astrodata.testing import download_from_archive
    from geminidr.gmos.tests.spect import CREATED_INPUTS_PATH_FOR_TESTS
    from gempy.utils import logutils
    from recipe_system.reduction.coreReduce import Reduce

    module_name, _ = os.path.splitext(os.path.basename(__file__))
    path = os.path.join(CREATED_INPUTS_PATH_FOR_TESTS, module_name)
    os.makedirs(path, exist_ok=True)
    os.chdir(path)
    os.makedirs("inputs", exist_ok=True)
    cwd = os.getcwd()

    associated_arcs = {
        "N20180109S0287.fits": "N20180109S0315.fits",  # GN-2017B-FT-20-13-001 B600 0.505um
        "N20190302S0089.fits": "N20190302S0274.fits",  # GN-2019A-Q-203-7-001 B600 0.550um
        "N20190313S0114.fits": "N20190313S0132.fits",  # GN-2019A-Q-325-13-001 B600 0.482um
        "N20190427S0126.fits": "N20190427S0267.fits",  # GN-2019A-FT-206-7-004 R400 0.625um
        "N20190910S0028.fits": "N20190910S0279.fits",  # GN-2019B-Q-313-5-001 B600 0.550um
        "S20180919S0139.fits": "S20180919S0141.fits",  # GS-2018B-Q-209-13-003 B600 0.45um
        "S20191005S0051.fits": "S20191005S0147.fits",  # GS-2019B-Q-132-35-001 R400 0.73um
    }

    for sci_fname, arc_fname in associated_arcs.items():

        sci_path = download_from_archive(sci_fname)
        arc_path = download_from_archive(arc_fname)

        sci_ad = astrodata.open(sci_path)
        data_label = sci_ad.data_label()

        logutils.config(file_name='log_arc_{}.txt'.format(data_label))
        arc_reduce = Reduce()
        arc_reduce.files.extend([arc_path])
        # arc_reduce.ucals = normalize_ucals(arc_reduce.files, calibration_files)

        os.chdir("inputs/")
        arc_reduce.runr()
        arc_ad = arc_reduce.output_filenames.pop()

        logutils.config(file_name='log_{}.txt'.format(data_label))
        p = GMOSSpect([sci_ad])
        p.prepare()
        p.addDQ(static_bpm=None)
        p.addVAR(read_noise=True)
        p.overscanCorrect()
        p.ADUToElectrons()
        p.addVAR(poisson_noise=True)
        p.attachWavelengthSolution(arc=arc_ad)
        p.distortionCorrect()
        p.writeOutputs()
        os.chdir("../")

    os.chdir(cwd)


def create_inputs_automated_recipe():
    from astropy.io import fits as pf
    import numpy as np
    from astropy.modeling.models import Gaussian1D
    from itertools import product as cart_product
    from geminidr.gmos.tests.spect import CREATED_INPUTS_PATH_FOR_TESTS

    module_name, _ = os.path.splitext(os.path.basename(__file__))
    path = os.path.join(CREATED_INPUTS_PATH_FOR_TESTS, module_name)
    os.makedirs(path, exist_ok=True)
    os.chdir(path)
    os.makedirs("inputs", exist_ok=True)
    cwd = os.getcwd()
    print(cwd)
    os.chdir("inputs/")

    SHAPE = (2048, 200)
    RDNOISE = 4

    # # Edit these to provide iterables to iterate over
    # BACKGROUNDS = (0,)  # overall background level
    # PEAKS = (50,)  # signal in galaxy peak
    # CONTRASTS = (0.25,)  # ratio of SN peak to galaxy peak
    # SEPARATIONS = (8,)  # pixel separation between galaxy/SN peaks
    # GAL_FWHMS = (40,)  # FWHM (pixels) of galaxy
    # SN_FWHMS = (3,)  # FWHM (pixels) of SN

    phu_dict = dict(
        INSTRUME='GMOS-N',
        OBJECT='FAKE',
        OBSTYPE='OBJECT',
    )
    hdr_dict = dict(
        WCSAXES=3,
        WCSDIM=3,
        CD1_1=-0.1035051453281131,
        CD2_1=0.0,
        CD1_2=0.0,
        CD2_2=-1.6608167576414E-05,
        CD3_2=4.17864941757412E-05,
        CD1_3=0.0,
        CD2_3=0.0,
        CD3_3=1.0,
        CRVAL2=76.3775200034826,
        CRVAL3=52.8303306863311,
        CRVAL1=495.0,
        CTYPE1='AWAV    ',
        CTYPE2='RA---TAN',
        CTYPE3='DEC--TAN',
        CRPIX1=1575.215466689882,
        CRPIX2=-555.7218408956066,
        CRPIX3=0.0,
        CUNIT1='nm      ',
        CUNIT2='deg     ',
        CUNIT3='deg     ',
        DATASEC='[1:{1},1:{0}]'.format(*SHAPE),
    )

    yc = 0.5 * SHAPE[0]
    for bkgd, peak, contrast, sep, gal_fwhm, sn_fwhm in cart_product(
            BACKGROUNDS, PEAKS, CONTRASTS, SEPARATIONS, GAL_FWHMS, SN_FWHMS):
        gal_std = 0.42466 * gal_fwhm
        sn_std = 0.42466 * sn_fwhm
        model = (Gaussian1D(amplitude=peak, mean=yc, stddev=gal_std) +
                 Gaussian1D(amplitude=peak * contrast, mean=yc + sep, stddev=sn_std))
        profile = model(np.arange(SHAPE[0]))
        data = np.zeros(SHAPE) + profile[:, np.newaxis]
        data += np.random.normal(scale=RDNOISE, size=data.size).reshape(data.shape)

        hdulist = pf.HDUList([pf.PrimaryHDU(header=pf.Header(phu_dict)),
                              pf.ImageHDU(data=data, header=pf.Header(hdr_dict))])
        filename = (f"fake_bkgd{bkgd:04.0f}_peak{peak:03.0f}_con{contrast:4.2f}_"
                    f"sep{sep:05.2f}_gal{gal_fwhm:5.2f}_sn{sn_fwhm:4.2f}.fits")
        hdulist.writeto(filename, overwrite=True)
    os.chdir(cwd)


if __name__ == '__main__':
    import sys
    if "--create-inputs" in sys.argv[1:]:
        create_inputs_recipe()
        create_inputs_automated_recipe()
    else:
        pytest.main()
