import pytest
import os
import numpy as np
import itertools
import datetime

import astrodata
import pytest_dragons
import geminidr.ghost.polyfit as polyfit
from geminidr.ghost.lookups.polyfit_lookup import get_polyfit_filename

SEEING_ESTIMATES = (
        ("S20220915S0007_2x2_slit_blue006_slit.fits", {"blue": 0.56, "red": 0.53}),  # SR, 2-object
        ("HIP016085_20220913_b240r60_1x1_hr_2x2_slit_slit.fits", {"blue": 0.66, "red": 0.62}),  # HR
)


# Testing of the polyfit.slitview object, particularly the SlitView object

@pytest.mark.ghostslit
class TestSlitView():
    @pytest.fixture(scope='class',
                    params=['std', 'high'])
    def get_slitview_obj(self, request):
        res = request.param
        data_arr = np.ones((160, 160))
        flat_arr = np.zeros((160, 160))
        slitv_fn = get_polyfit_filename(
            None, 'slitv', res, datetime.date(2016, 11, 20), None, 'slitvmod')
        slitvpars = astrodata.open(slitv_fn).TABLE[0]
        sv = polyfit.slitview.SlitView(
            data_arr, flat_arr, slitvpars=slitvpars, mode=res)
        # import pdb; pdb.set_trace()
        return data_arr, flat_arr, sv

    @pytest.mark.skip("Not relevant any more due to SLITVMOD")
    def test_slitview_init(self, get_slitview_obj):
        """Test input checking on SlitView"""
        slitvpars = get_slitview_obj[-1]
        with pytest.raises(ValueError):
            _ = polyfit.slitview.SlitView(np.zeros((1, 1)), np.zeros((1, 1,)),
                                          slitvpars=slitvpars, mode='invalid')
            pytest.fail('slitview.SlitView failed to throw '
                        'ValueError when passed an invalid '
                        'instrument mode')

    @pytest.mark.skip("Not relevant any more due to SLITVMOD")
    def test_slitview_modes(self, get_slitview_obj):
        """Test instantiation for each mode"""
        _, _, sv = get_slitview_obj
        for attr, value in polyfit.slitview.SLITVIEW_PARAMETERS[
            sv.mode].items():
            assert getattr(sv, attr) == value, "SlitView object has " \
                                               "incorrect value for " \
                                               "attribute {} (expected {}, " \
                                               "found {})".format(
                attr, value, getattr(sv, attr),
            )

    def test_slitview_cutout_init(self, get_slitview_obj):
        """Test input checking on SlitView.cutout"""
        data_arr, flat_arr, sv = get_slitview_obj
        with pytest.raises(ValueError):
            sv.cutout(arm='invalid')
            pytest.fail('SlitView.cutout failed to throw '
                        'ValueError when given an invalid arm')

    @pytest.mark.parametrize('use_flat,arm', itertools.product(*[
        [True, False],
        ['red', 'blue'],
    ]))
    def test_slitview_cutout_useflat(self, use_flat, arm, get_slitview_obj):
        """Test use_flat option of SlitView.cutout"""
        data_arr, flat_arr, sv = get_slitview_obj
        assert int(np.median(sv.cutout(
            arm=arm, use_flat=use_flat))) != use_flat, "use_flat option of " \
                                                       "SlitView.cutout " \
                                                       "has not triggered " \
                                                       "correctly"

    @pytest.mark.parametrize('use_flat,arm', itertools.product(*[
        [True, False],
        ['red', 'blue'],
    ]))
    def test_slitview_cutout_dimensions(self, use_flat, arm, get_slitview_obj):
        """Test shape of output from Slitview.cutout"""
        data_arr, flat_arr, sv = get_slitview_obj
        assert sv.cutout(arm=arm, use_flat=use_flat).shape == (
            2*int(sv.slit_length/sv.microns_pix/2) + 1,
            2*sv.extract_half_width + 1,
        )

    @pytest.mark.parametrize('return_centroid,arm,use_flat',
                             itertools.product(*[
                                 [True, False],
                                 ['red', 'blue'],
                                 [True, False],
                             ]))
    def test_slitview_slit_profile_return_centroid(self, return_centroid, arm,
                                                   use_flat,
                                                   get_slitview_obj):
        """Test return_centroid option of SlitView.slit_profile method"""
        _, _, sv = get_slitview_obj
        func_returns = sv.slit_profile(
            arm=arm, return_centroid=return_centroid,
            use_flat=use_flat)
        if return_centroid:
            assert len(func_returns) == 2
        else:
            assert not isinstance(func_returns, tuple)

    @pytest.mark.parametrize('use_flat,arm',
                             itertools.product(*[
                                 [True, False],
                                 ['red', 'blue'],
                             ]))
    def test_slitview_slit_profile_shape(self, arm, use_flat, get_slitview_obj):
        """Test the shape of the return of SlitView.slit_profile"""
        _, _, sv = get_slitview_obj
        profile = sv.slit_profile(
            arm=arm, return_centroid=False,
            use_flat=use_flat)
        cutout = sv.cutout(arm, use_flat)
        assert len(profile.shape
                   ) == len(cutout.shape) - 1, "SlitView.slit_profile has " \
                                               "not reduced the dimensions " \
                                               "of the cutout by 1"
        assert profile.shape == cutout.shape[:1] + cutout.shape[2:]


@pytest.mark.ghostslit
@pytest.mark.parametrize("filename, results", SEEING_ESTIMATES)
def test_seeing_estimate(filename, results, path_to_inputs):
    ad = astrodata.open(os.path.join(path_to_inputs, filename))
    slitv_fn = get_polyfit_filename(
        None, 'slitv', ad.res_mode(), ad.ut_date(), None, 'slitvmod')
    sv = polyfit.slitview.SlitView(
        ad[0].data, None, slitvpars=astrodata.open(slitv_fn).TABLE[0],
        mode=ad.res_mode())
    m = sv.model_profile(ad[0].data)
    for k, v in m.items():
        assert results[k] == pytest.approx(v.estimate_seeing()[0], abs=0.05)
