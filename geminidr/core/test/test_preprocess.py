# pytest suite
"""
Tests for primitives_preprocess.

This is a suite of tests to be run with pytest.

To run:
    1) Set the environment variable GEMPYTHON_TESTDATA to the path that
       contains the directories with the test data.
       Eg. /net/chara/data2/pub/gempython_testdata/
    2) From the ??? (location): pytest -v --capture=no
"""

# import astrodata, gemini_instruments, os, sys, AstroFaker
# import geminidr.core.test.__init__ as init
# af  = AstroFaker.create('NIRI','IMAGE')
# af2  = AstroFaker.create('NIRI','IMAGE')
# af3  = AstroFaker.create('F2','IMAGE')
# init.ad_compare(af, af2)

import os
import pytest
import numpy as np
from copy import deepcopy
import AstroFaker
import astrodata
import gemini_instruments
from gempy.utils import logutils

from . import ad_compare
from geminidr.niri.primitives_niri_image import NIRIImage
from geminidr.gmos.primitives_gmos_image import GMOSImage

TESTDATAPATH = os.getenv('GEMPYTHON_TESTDATA', '.')
logfilename = 'test_preprocess.log'

@pytest.fixture()
def niriImages():
    """Create two NIRI images, one all 1s, the other all 2s"""
    adinputs = []
    for i in (1, 2):
        ad = AstroFaker.create('NIRI', 'IMAGE')
        ad.init_default_extensions()
        ad[0].data += i
    adinputs.append(ad)
    return NIRIImage(adinputs)

def test_scaleByExposureTime(niriImages):
    ad1, ad2 = niriImages.streams['main']
    ad2.phu[ad2._keyword_for('exposure_time')] *= 0.5
    ad2_orig_value = ad2[0].data.mean()
    ad1, ad2 = niriImages.scaleByExposureTime(time=None)
    # Check that ad2 had its data doubled
    assert abs(ad2[0].data.mean() - ad2_orig_value * 2) < 0.001
    ad1, ad2 = niriImages.scaleByExposureTime(time=1)
    # Check that ad2 has been rescaled to 1-second
    print(ad2[0].data.mean(), ad2_orig_value, ad2.phu["ORIGTEXP"])
    assert abs(ad2[0].data.mean() - ad2_orig_value / ad2.phu["ORIGTEXP"]) < 0.001

def test_addObjectMaskToDQ():
    ad_orig = AstroFaker.create('F2','IMAGE')
        #astrodata.open(os.path.join(TESTDATAPATH, 'GMOS', 'N20150624S0106_refcatAdded.fits'))
    p = GMOSImage([deepcopy(ad_orig)])
    ad = p.addObjectMaskToDQ()[0]
    for ext, ext_orig in zip(ad, ad_orig):
        assert all(ext.mask[ext.OBJMASK==0] == ext_orig.mask[ext.OBJMASK==0])
        assert all(ext.mask[ext.OBJMASK==1] == ext_orig.mask[ext.OBJMASK==1] | 1)

def test_ADUTOElectrons():
    ad = AstroFaker.create("NIRI", "IMAGE")
        #astrodata.open(os.path.join(TESTDATAPATH, 'NIRI', 'N20070819S0104_dqAdded.fits'))
    p = NIRIImage([ad])
    ad = p.ADUToElectrons()[0]
    assert ad_compare(ad, os.path.join(TESTDATAPATH, 'NIRI',
                            'N20070819S0104_ADUToElectrons.fits'))

def test_applyDQPlane():
    ad = AstroFaker.create("NIRI","IMAGE")
        #astrodata.open(os.path.join(TESTDATAPATH, 'NIRI', 'N20070819S0104_nonlinearityCorrected.fits'))
    p = NIRIImage([ad])
    ad = p.applyDQPlane()[0]
    assert ad_compare(ad, os.path.join(TESTDATAPATH, 'NIRI',
                            'N20070819S0104_dqPlaneApplied.fits'))

def test_associateSky():
    filenames = ['N20070819S{:04d}_flatCorrected.fits'.format(i)
                 for i in range(104, 109)]
    adinputs = [astrodata.open(os.path.join(TESTDATAPATH, 'NIRI', f))
                for f in filenames]
    p = NIRIImage(adinputs)
    p.separateSky()  # Difficult to construct this by hand
    p.associateSky()
    filename_set = set([ad.phu['ORIGNAME'] for ad in adinputs])
    # Test here is that each science frame has all other frames as skies
    for k, v in p.sky_dict.items():
        v = [ad.phu['ORIGNAME'] for ad in v]
        assert len(v) == len(filenames) - 1
        assert set([k]+v) == filename_set

# def test_correctBackgroundToReference(self):
#     pass

# def test_darkCorrect(self):
#     ad = astrodata.open(os.path.join(TESTDATAPATH, 'NIRI',
#                             'N20070819S0104_nonlinearityCorrected.fits'))
#     p = NIRIImage([ad])
#     ad = p.darkCorrect()[0]
#     assert ad_compare(ad, os.path.join(TESTDATAPATH, 'NIRI',
#                             'N20070819S0104_darkCorrected.fits'))

# def test_darkCorrect_with_af(self):
#     science = AstroFaker.create('NIRI', 'IMAGE')
#     dark = AstroFaker.create('NIRI', 'IMAGE')
#     p = NIRIImage([science])
#     p.darkCorrect([science], dark=dark)
#     science.subtract(dark)
#     assert ad_compare(science, dark)




    # af.init_default_extensions()
    # af[0].mask = np.zeros_like(af[0].data, dtype=np.uint16)
# def test_flatCorrect(self):
#     ad = astrodata.open(os.path.join(TESTDATAPATH, 'NIRI',
#                             'N20070819S0104_darkCorrected.fits'))
#     p = NIRIImage([ad])
#     ad = p.flatCorrect()[0]
#     assert ad_compare(ad, os.path.join(TESTDATAPATH, 'NIRI',
#                             'N20070819S0104_flatCorrected.fits'))
#
# def test_makeSky(self):
#     pass
#
# def test_nonlinearityCorrect(self):
#     # Don't use NIRI data; NIRI has its own primitive
#     pass
#
# def test_normalizeFlat(self):
#     flat_file = os.path.join(TESTDATAPATH, 'NIRI',
#                             'N20070913S0220_flat.fits')
#     ad = astrodata.open(flat_file)
#     ad.multiply(10.0)
#     del ad.phu['NORMLIZE']  # Delete timestamp of previous processing
#     p = NIRIImage([ad])
#     ad = p.normalizeFlat(suffix='_flat', strip=True)[0]
#     assert ad_compare(ad, flat_file)
#
# def test_separateSky(self):
#     pass
#
# def test_skyCorrect(self):
#     pass
#
# def test_subtractSky(self):
#     pass
#
# def test_subtractSkyBackground(self):
#     ad = astrodata.open(os.path.join(TESTDATAPATH, 'NIRI',
#                             'N20070819S0104_flatCorrected.fits'))
#     ad.hdr['SKYLEVEL'] = 1000.0
#     orig_data = ad[0].data.copy()
#     p = NIRIImage([ad])
#     ad = p.subtractSkyBackground()[0]
#     assert (orig_data - ad[0].data).min() > 999.99
#     assert (orig_data - ad[0].data).max() < 1000.01
#
# def test_thresholdFlatfield(self):
#     ad = astrodata.open(os.path.join(TESTDATAPATH, 'NIRI',
#                                      'N20070913S0220_flat.fits'))
#     del ad.phu['TRHFLAT']  # Delete timestamp of previous processing
#     ad[0].data[100, 100] = 20.0
#     p = NIRIImage([ad])
#     ad = p.thresholdFlatfield()[0]
#     assert ad[0].mask[100, 100] == 64