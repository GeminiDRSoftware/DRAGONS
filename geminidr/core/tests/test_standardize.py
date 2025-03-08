# pytest suite
"""
Tests for primitives_standardize.

This is a suite of tests to be run with pytest.

To run:
    1) Set the environment variable GEMPYTHON_TESTDATA to the path that
       contains the directories with the test data.
       Eg. /net/chara/data2/pub/gempython_testdata/
    2) From the ??? (location): pytest -v --capture=no
"""
# TODO @bquint: clean up these tests

import os
import pytest

import astrodata
from gempy.utils import logutils
from astrodata.testing import ad_compare, download_from_archive

from geminidr.gmos.primitives_gmos_longslit import GMOSLongslit
from geminidr.f2.primitives_f2_longslit import F2Longslit
from geminidr.niri.primitives_niri_image import NIRIImage

logfilename = 'test_standardize.log'


class TestStandardize:
    """
    Suite of tests for the functions in the primitives_standardize module.
    """

    @classmethod
    def setup_class(cls):
        """Run once at the beginning."""
        if os.path.exists(logfilename):
            os.remove(logfilename)
        log = logutils.get_logger(__name__)
        log.root.handlers = []
        logutils.config(mode='standard',
                        file_name=logfilename)

    @classmethod
    def teardown_class(cls):
        """Run once at the end."""
        os.remove(logfilename)

    def setup_method(self, method):
        """Run once before every test."""
        pass

    def teardown_method(self, method):
        """Run once after every test."""
        pass

    @pytest.mark.niri
    @pytest.mark.regression
    @pytest.mark.preprocessed_data
    def test_addDQ(self, change_working_dir, path_to_refs,
                   path_to_common_inputs):

        with change_working_dir():
            ad = astrodata.open(os.path.join(path_to_refs,
                                             'N20070819S0104_prepared.fits'))
            bpmfile = os.path.join(path_to_common_inputs,
                                   'bpm_20010317_niri_niri_11_full_1amp.fits')
            p = NIRIImage([ad])
            adout = p.addDQ(static_bpm=bpmfile)[0]
        assert ad_compare(adout,
                          astrodata.open(os.path.join(path_to_refs,
                                           'N20070819S0104_dqAdded.fits')))

    @pytest.mark.niri
    @pytest.mark.regression
    @pytest.mark.preprocessed_data
    def test_addIllumMaskToDQ(self, change_working_dir, path_to_inputs,
                              path_to_refs):

        with change_working_dir():
            ad = astrodata.open(os.path.join(path_to_refs,
                                             'N20070819S0104_dqAdded.fits'))

            p = NIRIImage([ad])
            adout = p.addIllumMaskToDQ()[0]

        assert ad_compare(adout,
                          astrodata.open(os.path.join(
                              path_to_refs,
                              'N20070819S0104_illumMaskAdded.fits')))

    @pytest.mark.niri
    @pytest.mark.regression
    @pytest.mark.preprocessed_data
    def test_addVAR(self, change_working_dir, path_to_inputs, path_to_refs):

        with change_working_dir():
            ad = astrodata.open(os.path.join(path_to_inputs,
                                'N20070819S0104_ADUToElectrons.fits'))
            p = NIRIImage([ad])
            adout = p.addVAR(read_noise=True, poisson_noise=True)[0]
        assert ad_compare(adout, astrodata.open(os.path.join(path_to_refs,
                                             'N20070819S0104_varAdded.fits')))

    @pytest.mark.dragons_remote_data
    @pytest.mark.parametrize('filename,instrument,inst_class',
                             [('S20131002S0090.fits', 'F2', F2Longslit),
                              ('N20190501S0054.fits', 'GMOS', GMOSLongslit)])
    def test_makeIRAFCompatible(self, filename, instrument, inst_class):

        # Keywords from gempy/gemini/irafcompat.py.
        GMOS_keywords = ('GPREPARE', 'GGAIN', 'GAINMULT',
                         'CCDSUM')

        p = inst_class([astrodata.open(download_from_archive(filename))])
        p.prepare()
        p.ADUToElectrons()
        ad = p.makeIRAFCompatible()[0]

        if instrument == 'F2':
            assert ad.phu['NSCIEXT'] == 1
        elif instrument == 'GMOS':
            assert ad.phu['NSCIEXT'] == 12

            # Check that GMOS has had keywords added correctly.
            # TODO (DB): do a full GMOS reduction to check keywords related
            # to flat|sky|bias correction, mosaicing, etc..
            for kw in GMOS_keywords:
                assert ad.phu[kw]

    @pytest.mark.niri
    @pytest.mark.regression
    def test_prepare(self, change_working_dir, path_to_inputs,
                     path_to_refs):

        ad = astrodata.open(os.path.join(path_to_inputs,
                                         'N20070819S0104.fits'))
        with change_working_dir():
            logutils.config(file_name=f'log_regression_{ad.data_label()}.txt')
            p = NIRIImage([ad])
            p.prepare()
            prepared_ad = p.writeOutputs(
                outfilename='N20070819S0104_prepared.fits').pop()
            del prepared_ad.phu['SDZWCS']  # temporary fix

        ref_ad = astrodata.open(
            os.path.join(path_to_refs, 'N20070819S0104_prepared.fits'))

        assert ad_compare(prepared_ad, ref_ad)

    @pytest.mark.niri
    @pytest.mark.regression
    @pytest.mark.preprocessed_data
    def test_standardizeHeaders(self, change_working_dir, path_to_inputs):

        ad = astrodata.open(os.path.join(path_to_inputs,
                                         'N20070819S0104.fits'))

        with change_working_dir():
            p = NIRIImage([ad])
            adout = p.standardizeHeaders()[0]

            # These two are explicitly ignored in ad_compare() as otherwise we
            # would need to update all the reference files every release.
            for kw in ['PROCSOFT', 'PROCSVER']:
                assert kw in adout.phu
            assert adout.phu['PROCSOFT'] == 'DRAGONS'
