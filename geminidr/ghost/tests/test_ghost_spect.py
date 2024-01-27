# pytest suite
"""
Unit tests for :any:`geminidr.ghost.primitives_ghost_spect`.

This is a suite of tests to be run with pytest.
"""
from __future__ import division

import os
import pytest
from astropy.io import fits

import numpy as np
import astrodata, gemini_instruments
from gempy.utils import logutils
from six.moves import range

import copy
import datetime
import itertools

from geminidr.ghost.primitives_ghost_spect import GHOSTSpect



@pytest.mark.ghostspect
class TestGhost:
    """
    Suite of tests for the functions in the primitives_ghost_slit module

    In all tests, check that:
    - Correct keyword comment has been added
    """

    @staticmethod
    def generate_minimum_file():
        rawfilename = 'test_data.fits'

        # Create the AstroData object
        phu = fits.PrimaryHDU()
        phu.header.set('INSTRUME', 'GHOST')
        phu.header.set('DATALAB', 'test')
        phu.header.set('CAMERA', 'RED')
        phu.header.set('CCDNAME', 'E2V-CCD-231-C6')
        phu.header.set('CCDSUM', '1 1')

        # Create a simple data HDU with a zero BPM
        sci = fits.ImageHDU(data=np.ones((1024, 1024,)), name='SCI')
        sci.header.set('CCDSUM', '1 1')

        ad = astrodata.create(phu, [sci, ])
        ad.filename = rawfilename
        return ad

    @pytest.mark.skip(reason='Requires calibrators & polyfit-ing - do '
                             'in all-up testing')
    def test_attachWavelengthSolution(self, data_attachWavelengthSolution):
        """
        Checks to make:

        - Check for attached wavelength extension
        - Come up with some sanity checks for the solution
        - Check for wavelength unit in output
        """
        pass

    @pytest.fixture
    def data_barycentricCorrect(self):
        """
        Create data for the barycentric correction test.

        .. note::
            Fixture.
        """
        ad = self.generate_minimum_file()

        # Add a wavl extension - no need to be realistic
        ad[0].WAVL = np.random.rand(*ad[0].data.shape)
        return ad

    @pytest.mark.parametrize('ra,dec,dt,known_corr', [
        (90., -30., '2018-01-03 15:23:32', 0.999986388827),
        (180., -60., '2018-11-12 18:35:15', 1.00001645007),
        (270., -90., '2018-07-09 13:48:35', 0.999988565947),
        (0., -45., '2018-12-31 18:59:48', 0.99993510834),
        (101.1, 0., '2018-02-23 17:18:55', 0.999928361662),
    ])
    def test_barycentricCorrect(self, ra, dec, dt, known_corr,
                                data_barycentricCorrect):
        """
        Checks to make:

        - Make random checks that a non-1.0 correction factor works properly
        - Check before & after data shape

        Testing of the helper _compute_barycentric_correction is done
        separately.
        """
        ad = data_barycentricCorrect
        orig_wavl = copy.deepcopy(ad[0].WAVL)
        ad.phu.set('RA', ra)
        ad.phu.set('DEC', dec)
        # Assume a 10 min exposure
        exp_time_min = 10.
        dt_obs = datetime.datetime.strptime(dt, '%Y-%m-%d %H:%M:%S')
        dt_start = dt_obs - datetime.timedelta(minutes=exp_time_min)
        ad.phu.set('DATE-OBS', dt_start.date().strftime('%Y-%m-%d'))
        ad.phu.set('UTSTART', dt_start.time().strftime('%H:%M:%S.00'))
        ad.phu.set('EXPTIME', exp_time_min * 60.)

        gs = GHOSTSpect([])
        ad_out = gs.barycentricCorrect([ad]).pop()
        corr_fact = (ad_out[0].WAVL / orig_wavl).mean()
        assert np.allclose(ad_out[0].WAVL / known_corr,
                           orig_wavl), "barycentricCorrect appears not to " \
                                       "have made a valid correction " \
                                       f"(should be {known_corr}, " \
                                       f"apparent correction {corr_fact})"
        assert orig_wavl.shape == ad_out[0].WAVL.shape, \
            "barycentricCorrect has mangled the shape of the output " \
            "WAVL extension"

        assert ad_out.phu.get(
            gs.timestamp_keys['barycentricCorrect']), "barycentricCorrect did not " \
                                                      "timestamp-mark the " \
                                                      "output file"

    @pytest.mark.parametrize('xbin, ybin',
                             list(itertools.product(*[
                                 [1, 2, ],  # x binning
                                 [1, 2, 4, 8, ],  # y binning
                             ]))
                             )
    def test_darkCorrect_rebin(self, xbin, ybin):
        """
        Checks to make:

        - Ensure re-binning of darks is correctly triggered (i.e. deliberately
          pass science and dark data w/ different binning, ensure no failure)
        - Error mode check: see for error if length of dark list != length of
          science list
        - Check for DARKIM in output header
        - Check before & after data shape
        """
        ad = self.generate_minimum_file()
        dark = self.generate_minimum_file()
        dark.filename = 'dark.fits'

        # 'Re-bin' the data file
        ad[0].data = np.ones((int(1024 / ybin), int(1024 / xbin), ), dtype=np.float64)
        ad[0].hdr.set('CCDSUM', '{} {}'.format(xbin, ybin, ))

        gs = GHOSTSpect([])
        input_shape = ad[0].data.shape
        ad_out = gs.darkCorrect([ad], dark=dark, do_cal="force")[0]

        assert ad_out[0].data.shape == input_shape, "darkCorrect has mangled " \
                                                    "the shape of the input " \
                                                    "data"

    def test_darkCorrect_errors(self):
        ad = self.generate_minimum_file()
        dark = self.generate_minimum_file()
        dark.filename = 'dark.fits'

        gs = GHOSTSpect([])

        # Passing in data inputs with different binnings
        with pytest.raises(IOError):
            ad2 = copy.deepcopy(ad)
            ad2[0].hdr.set('CCDSUM', '2 2')
            gs.darkCorrect([ad, ad2, ], dark=[dark, dark, ], do_cal="force")

        # Mismatched list lengths
        with pytest.raises(Exception):
            gs.darkCorrect([ad, ad2, ad, ], dark=[dark, dark, ], do_cal="force")

    def test_darkCorrect(self):
        ad = self.generate_minimum_file()
        dark = self.generate_minimum_file()
        dark.filename = 'dark.fits'

        gs = GHOSTSpect([])
        ad_out = gs.darkCorrect([ad, ], dark=[dark, ], do_cal="force")

        # import pdb; pdb.set_trace()

        assert ad_out[0].phu.get('DARKIM') == dark.filename, \
            "darkCorrect failed to record the name of the dark " \
            "file used in the output header (expected {}, got {})".format(
                dark.filename, ad_out[0].phu.get('DARKIM'),
            )

        assert ad_out[0].phu.get(
            gs.timestamp_keys['darkCorrect']), "darkCorrect did not " \
                                               "timestamp-mark the " \
                                               "output file"

    @pytest.mark.skip(reason='Requires calibrators & polyfit-ing - save for '
                             'all-up testing')
    def test_extractSpectra(self):
        """
        Checks to make

        - Check functioning of writeResult kwarg
        - Check error catching for differing length of slit_list, objects and
          slitflat_list
        - Look for the order-by-order processed data (DATADESC)
        """
        pass

    def test_interpolateAndCombine(self):
        """
        Checks to make:

        - Error on invalid scale option
        - Correct functioning of 'skip' parameter

        Fuller testing needs to be done 'all-up' in a reduction sequence.
        """
        ad = self.generate_minimum_file()
        gs = GHOSTSpect([])

        # Make sure the correct error is thrown on an invalid scale
        with pytest.raises(ValueError):
            gs.interpolateAndCombine([ad, ], scale='not-a-scale')

        # Test the skip functionality
        #ad_out = gs.interpolateAndCombine([ad, ], skip=True)[0]
        #assert ad_out.phu.get(
        #    gs.timestamp_keys['interpolateAndCombine']
        #) is None, "interpolateAndCombine appears to have acted on a file " \
        #           "when skip=True"

    @pytest.mark.skip(reason='Requires calibrators & polyfit-ing - save for '
                             'all-up testing')
    def test_traceFibers(self):
        """
        Checks to make:

        - Look for XMOD in output
        - Ensure said XMOD is valid (i.e. polyfit will accept it)
        - Ensure skip_pixel_model flag works correctly
        """
        pass

    @pytest.mark.skip(reason='Requires calibrators & polyfit-ing - save for '
                             'all-up testing')
    def test_determineWavelengthSolution(self):
        """
        Checks to make:

        - Ensure only GHOST ARC frames are accepted
        - Ensure frame with primitive already applied can have it applied again
        - Make sure WFIT extension is present in output
        - Perform sanity checks on WFIT, e.g. will polyfit accept?
        - Ensure actual data is untouched
        """
        pass

    @pytest.mark.skip(reason='Needs a full reduction sequence to test')
    def test_formatOutput(self):
        """
        Checks to make:

        - Validate output of each layer of formatOutput
        - Ensure data proper is identical between outputs
        """
        pass

    def test_responseCorrect(self):
        """
        Checks to make:

        - Ensure parameters for no-op work correctly

        More complete testing to be made in 'all-up' reduction
        """
        ad = self.generate_minimum_file()
        gs = GHOSTSpect([])

        # Test the skip functionality
        ad_out = gs.responseCorrect([ad, ], standard=None)[0]
        assert ad_out.phu.get(
            gs.timestamp_keys['responseCorrect']
        ) is None, "responseCorrect appears to have acted on a file " \
                   "when standard=None"

    def test_standardizeStructure(self):
        """
        Checks to make:

        - This is a no-op primitive - ensure no change is made
        """
        ad = self.generate_minimum_file()
        ad_orig = copy.deepcopy(ad)
        gs = GHOSTSpect([])

        ad_out = gs.standardizeStructure([ad, ])[0]
        assert np.all([
            ad_orig.info() == ad_out.info(),
            ad_orig.phu == ad_out.phu,
            ad_orig[0].hdr == ad_out[0].hdr,
            len(ad_orig) == len(ad_out),
        ]), "standardizeStructure is no longer a no-op primitive"

    @pytest.mark.skip(reason='All-up testing required - needs full DATASEC, '
                             'CCDSEC, AMPSIZE, CCDSIZE etc. calculations')
    def test_tileArrays(self):
        """
        Checks to make:

        - Ensure single data extension after action
        - Check data still matches the input data (no interpolation/mods)
        """
        pass

    @pytest.mark.parametrize('arm,res,caltype', list(itertools.product(*[
        ['BLUE', 'RED'],  # Arm
        ['LO_ONLY', 'HI_ONLY'],  # Res. mode
        ['xmod', 'wavemod', 'spatmod', 'specmod', 'rotmod'],  # Cal. type
    ])))
    def test__get_polyfit_filename(self, arm, res, caltype):
        """
        Checks to make:

        - Provide a set of input (arm, res, ) arguments, see if a/ name is
          returned
        """
        ad = self.generate_minimum_file()
        ad.phu.set('SMPNAME', res)
        ad.phu.set('CAMERA', arm)
        ad.phu.set('UTSTART', datetime.datetime.now().time().strftime(
            '%H:%M:%S'))
        ad.phu.set('DATE-OBS', datetime.datetime.now().date().strftime(
            '%Y-%m-%d'))

        gs = GHOSTSpect([])
        polyfit_file = gs._get_polyfit_filename(ad, caltype)

        assert polyfit_file is not None, "Could not find polyfit file"

    def test__get_polyfit_filename_errors(self):
        """
        Check passing an invalid calib. type throws an error
        """
        ad = self.generate_minimum_file()
        ad.phu.set('SMPNAME', 'HI_ONLY')
        ad.phu.set('CAMERA', 'RED')
        ad.phu.set('UTSTART', datetime.datetime.now().time().strftime(
            '%H:%M:%S'))
        ad.phu.set('DATE-OBS', datetime.datetime.now().date().strftime(
            '%Y-%m-%d'))

        gs = GHOSTSpect([])
        polyfit_file = gs._get_polyfit_filename(ad, 'not-a-cal-type')
        assert polyfit_file is None, "_get_polyfit_filename didn't return " \
                                     "None when asked for a bogus " \
                                     "model type"

    @pytest.mark.skip(reason='Requires calibration system - '
                             'will need to be part of all-up testing')
    def test__request_bracket_arc(self):
        """
        Checks to make (will require correctly populated calib system):

        - Error handling for failing to define before kwarg
        - Ensure arcs returned are, indeed, before and after as requested
        """
        pass

    @pytest.fixture(scope='class')
    def data__interp_spect(self):
        """
        Create an interpolated wavelength scale.

        .. note::
            Fixture.
        """
        # Generate a wavelength scale
        wavl = np.arange(1000., 9000., 5.)
        # Form some data
        data = np.random.rand(len(wavl))
        # Put some random np.nan into the data
        for i in np.random.randint(0, len(data) - 1, 20):
            data[i] = np.nan
        # Generate a finer wavelength scale to interp. to
        # Make sure there are un-interpolable points
        new_wavl = np.arange(800., 9200., 1.)

        return wavl, data, new_wavl

    @pytest.mark.parametrize('interp',
                             ['linear', 'nearest', 'zero', 'slinear',
                              'quadratic', 'cubic', 'previous', 'next',])
    def test__interp_spect(self, interp, data__interp_spect):
        """
        Checks to make:

        - New wavelength array appears in output
        - Any point in output can be interpolated from surrounding points in
          input
        - Verify allowed values for kwarg 'interp'
        """
        wavl, data, new_wavl = data__interp_spect

        gs = GHOSTSpect([])
        new_data = gs._interp_spect(data, wavl, new_wavl, interp='linear')

        assert new_data.shape == new_wavl.shape, "Data was not successfully " \
                                                 "reshaped to the new " \
                                                 "wavelength scale " \
                                                 "(expected {}, " \
                                                 "have {})".format(
            new_wavl.shape, new_data.shape,
        )

    def test__interp_spect_invalid_type(self, data__interp_spect):
        wavl, data, new_wavl = data__interp_spect
        gs = GHOSTSpect([])
        with pytest.raises(NotImplementedError):
            new_data = gs._interp_spect(data, wavl, new_wavl,
                                        interp='no-such-method')

    def test__regrid_spect(self, data__interp_spect):
        """
        Checks to make:

        - Ensure new_wavl comes out correctly in output
        - Ensure no interp'd points above/below input points
        """
        wavl, data, new_wavl = data__interp_spect

        gs = GHOSTSpect([])
        new_data = gs._regrid_spect(data, wavl, new_wavl,
                                    waveunits='angstrom')

        assert new_data.shape == new_wavl.shape, "Data was not successfully " \
                                                 "reshaped to the new " \
                                                 "wavelength scale " \
                                                 "(expected {}, " \
                                                 "have {})".format(
            new_wavl.shape, new_data.shape,
        )

        max_wavl_spacing = np.max(wavl[1:] - wavl[:-1])
        assert np.sum(new_data[np.logical_or(
            new_wavl < np.min(wavl) - max_wavl_spacing,
            new_wavl > np.max(wavl) + max_wavl_spacing,
        )]) < 1e-6, "Non-zero interpolated data points " \
                    "have been identified outside the " \
                    "range of the original wavelength " \
                    "scale"
