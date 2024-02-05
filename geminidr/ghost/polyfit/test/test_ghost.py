import pytest
import geminidr.ghost.polyfit as polyfit
import astropy.io.fits as pyfits
import numpy as np
import os
import itertools

# Assert if all the correct attributes of the ghost class needed are there

# Note that we also test a lot of polyspect.PolySpect functionality here -
# this is because Polyspect requires a lot of input parameters to be
# initialized, so it's much easier to use a GhostArm instance. The only
# Polyspect method overwritten by GhostArm is slit_flat_convolve, which is
# a no-op function in Polyspect.


def idfn(fixture_value):
    return ','.join([str(_) for _ in fixture_value])


class TestGhostArmBasic():
    @pytest.fixture(scope='class',
                    params=list(itertools.product(*[
                        ['std', 'high', ],  # Resolution mode
                        ['red', 'blue', ],  # Spectrograph arm
                        [1, 2, ],  # x binning
                        [1, 2, 4, 8, ],  # y binning
                    ])), ids=idfn, )
    def make_ghostarm_basic(self, request):
        res, arm, xb, yb = request.param
        gen_ghost = polyfit.ghost.GhostArm(mode=res, arm=arm,
                                           detector_x_bin=xb,
                                           detector_y_bin=yb)
        return gen_ghost

    @pytest.mark.parametrize("attrib, tp", [
        ('m_ref', int),
        ('szx', int),
        ('szy', int),
        ('m_min', int),
        ('m_max', int),
        ('transpose', bool),
    ])
    def test_ghostarm_attributes(self, attrib, tp, make_ghostarm_basic):
        """
        Test if the ghost class contains all the needed attributes
        in the right format
        """
        gen_ghost = make_ghostarm_basic
        assert hasattr(gen_ghost, attrib), "GhostArm missing attribute " \
                                           "{}".format(attrib)
        assert isinstance(getattr(gen_ghost, attrib),
                          tp), "GhostArm attribute {} has incorrect type " \
                               "(expected {}, got {})".format(
            attrib, tp, type(getattr(gen_ghost, attrib)),
        )

    def test_evaluate_poly_inputs(self, make_ghostarm_basic):
        """ Function designed to test general input aspects of polyspect's
        evaluate_poly function"""
        gen_ghost = make_ghostarm_basic

        # Checking if function returns type errors upon input of wrong
        # params type.
        # This is required because a few operations are performed on params
        # in the function that need it to be a np.array
        with pytest.raises(TypeError):
            gen_ghost.evaluate_poly([[1., 2., ], [3., 4., ], ])
            pytest.fail('GhostArm.evaluate_poly '
                        'failed to raise a '
                        'TypeError when passed a list '
                        '(instead of np.array)')

    fit_resid_atts = ['params', 'orders', 'y_values', 'data', ]

    @pytest.mark.parametrize('attrib', fit_resid_atts)
    def test_fit_resid_inputs(self, attrib, make_ghostarm_basic):
        """Test the input checking of GhostArm.fit_resid"""

        gen_ghost = make_ghostarm_basic

        l = [[1, 2, ], [3, 4, ]]
        a = np.asarray(l)
        kw = {_: a for _ in self.fit_resid_atts}
        kw[attrib] = l
        with pytest.raises(TypeError):
            gen_ghost.fit_resid(**kw)
            pytest.fail('GhostArm.fit_resid failed to '
                        'raise TypeError when passed '
                        'a non-np.array for '
                        '{}'.format(attrib))

        with pytest.raises(UserWarning):
            gen_ghost.fit_resid(a, a, a[:1], a, ydeg=1, xdeg=1)
            pytest.fail('GhostArm.fit_resid failed to '
                        'raise UserWarning when params '
                        'orders and y_values have '
                        'different lengths')

    spectral_format_args = ['xparams', 'wparams']

    def test_spectral_format_inputs(self, make_ghostarm_basic):
        """Test the input checking of GhostArm.spectral_format"""
        gen_ghost = make_ghostarm_basic
        l = [[1, 2, ], [3, 4, ]]

        with pytest.raises(UserWarning):
            gen_ghost.spectral_format(wparams=None, xparams=None)
            pytest.fail('GhostArm.spectral_format '
                        'failed to raise '
                        'UserWarning when given no '
                        'xparams or wparams')
        with pytest.raises(UserWarning):
            gen_ghost.spectral_format(wparams=None, xparams=l)
            pytest.fail('GhostArm.spectral_format '
                        'failed to raise '
                        'UserWarning when given a'
                        'non-np.ndarray for xparams')
        with pytest.raises(UserWarning):
            gen_ghost.spectral_format(wparams=l, xparams=None)
            pytest.fail('GhostArm.spectral_format '
                        'failed to raise '
                        'UserWarning when given a'
                        'non-np.ndarray for wparams')

    def test_adjust_x_inputs(self, make_ghostarm_basic):
        """Test the input checking of GhostArm.adjust_x"""
        gen_ghost = make_ghostarm_basic
        l = [1, 2, 3, 4, ]
        a = np.asarray(l)
        with pytest.raises(TypeError):
            gen_ghost.adjust_x(l, a)
            pytest.fail('GhostArm.adjust_x failed to '
                        'raise TypeError when old_x '
                        'is not a np.ndarray')
        with pytest.raises(TypeError):
            gen_ghost.adjust_x(a, l)
            pytest.fail('GhostArm.adjust_x failed to '
                        'raise TypeError when image '
                        'is not a np.ndarray')
        with pytest.raises(UserWarning):
            gen_ghost.adjust_x(a, a)
            pytest.fail('GhostArm.adjust_x failed to '
                        'raise UserWarning when image '
                        'does not have dimensions '
                        '(2, 2)')

    def test_fit_x_to_image_inputs(self, make_ghostarm_basic):
        """Test the input checking of GhostArm.fit_x_to_image"""
        gen_ghost = make_ghostarm_basic
        img = np.zeros((3, 3, 3, ))
        pars = np.zeros((2, 2, ))
        with pytest.raises(UserWarning):
            gen_ghost.fit_x_to_image(img, pars, decrease_dim=4)
            pytest.fail('GhostArm.fit_x_to_image '
                        'failed to raise a '
                        'UserWarning when the image '
                        'could not be correctly '
                        'reduced by decrease_dim')

    def test_fit_to_x_inputs(self, make_ghostarm_basic):
        """Test the input checking of GhostArm.fit_to_x"""
        gen_ghost = make_ghostarm_basic
        xtf = np.zeros((3, 3, ))
        init_mod = np.zeros((2, 2, ))
        with pytest.raises(UserWarning):
            gen_ghost.fit_to_x(xtf.tolist(), init_mod)
            pytest.fail('GhostArm.fit_to_x failed to '
                        'raise a UserWarning when '
                        'x_to_fit is not a '
                        'np.ndarray')
        with pytest.raises(UserWarning):
            gen_ghost.fit_to_x(xtf, init_mod.tolist())
            pytest.fail('GhostArm.fit_to_x failed to '
                        'raise a UserWarning when '
                        'init_mod is not a '
                        'np.ndarray')
        with pytest.raises(UserWarning):
            gen_ghost.fit_to_x(xtf, init_mod, decrease_dim=4)
            pytest.fail('GhostArm.fit_to_x failed to '
                        'raise a UserWarning when '
                        'init_mod is not a '
                        'np.ndarray')

    def test_spectral_format_with_matrix_inputs(self, make_ghostarm_basic):
        """Test the input checking of spectral_format_with_matrix"""
        gen_ghost = make_ghostarm_basic
        a = np.zeros((2, 2, ))
        with pytest.raises(ValueError):
            gen_ghost.spectral_format_with_matrix(None, None)
            pytest.fail('GhostArm.spectral_format_with_matrix '
                        'failed to raise ValueError when given '
                        'no xmod or wavemod')
        with pytest.raises(ValueError):
            gen_ghost.spectral_format_with_matrix(a, a, spatmod=None,
                                                  specmod=None, rotmod=None)
            pytest.fail('GhostArm.spectral_format_with_matrix '
                        'failed to raise ValueError when given '
                        'no spatmod, specmod or rotmod')

    def test_manual_model_adjust_inputs(self, make_ghostarm_basic):
        """Test the input checking of manual_model_adjust"""
        gen_ghost = make_ghostarm_basic
        with pytest.raises(ValueError):
            gen_ghost.manual_model_adjust(np.zeros((2, 2)), None)
            pytest.fail('GhostArm.manual_model_adjust '
                        'failed to raise ValueError when given '
                        'no xparams')

    def test_bin_data_inputs(self, make_ghostarm_basic):
        """Test the input checking of bin_data"""
        gen_ghost = make_ghostarm_basic
        with pytest.raises(UserWarning):
            gen_ghost.bin_data(np.zeros((gen_ghost.szx + 1,
                                         gen_ghost.szy + 2)))
            pytest.fail('GhostArm.bin_data '
                        'failed to raise UserWarning when given '
                        'a data array not matching the GhostArm '
                        'CCD size parameters')

    # @pytest.mark.skip
    @pytest.mark.parametrize("xparams,wparams,img", [
        (None, None, None),
        ('test', None, None),
        (np.ones((3, 3,)), None, np.arange(10.0)),
        (np.ones((3, 3,)), None, 'test')
    ])
    def test_spectral_format(self, xparams, wparams, img, make_ghostarm_basic):
        """ Function to test the spectral_format method"""
        # Test that calling this function with the various combinations of
        # invalid inputs raises a UserWarning
        gen_ghost = make_ghostarm_basic
        with pytest.raises((UserWarning, TypeError)):
            gen_ghost.spectral_format(xparams, wparams, img)

    # @pytest.mark.skip
    @pytest.mark.parametrize("old_x,image", [
        ('test', np.ones((3, 3))),
        (np.ones((3, 3,)), 'test'),
        (np.ones((3, 3)), np.ones(10))
    ])
    def test_adjust_x(self, old_x, image, make_ghostarm_basic):
        """ Function to test the adjust_x function"""
        # Test the TypeError
        gen_ghost = make_ghostarm_basic
        with pytest.raises((TypeError, UserWarning)):
            gen_ghost.adjust_x(old_x, image)


# FIXME What was this originally meant to test? The relevant args have
# changed beyond recognition
# testing if output is the same shape as y_values input with dummy variables
# y_values = np.arange(10.0)
# assert gen_ghost.evaluate_poly(np.ones((3, 3)),
#                                33.,
#                                ).shape == y_values.shape

@pytest.mark.skip(reason='Requires non-existent test data; unsure of what '
                         'this is meant to do anyway')
@pytest.mark.parametrize("res, arm", [
    ('high', 'red'), ('std', 'red'), ('high', 'blue'), ('std', 'blue')])
def test_polyfit(res, arm):
    """ Function designed to test various aspects of polyfit in all modes"""

    ghost = polyfit.ghost.GhostArm(arm, res)

    yvalues = np.arange(ghost.szy)
    os.chdir(os.path.dirname(__file__))
    xparams = pyfits.getdata('Polyfit/' + arm +
                             '/' + res + '/161120/xmod.fits')
    xx, wave, blaze = ghost.spectral_format(xparams=xparams)
    # Start testing things
    assert xx.shape == wave.shape
    assert xx.shape == blaze.shape
    assert xx.dtype == 'float64'

    flat_data = pyfits.getdata(
        'tests/flat_' + ghost.mode + '_' + ghost.arm + '.fits')
    flat_conv = ghost.slit_flat_convolve(flat_data)
    assert flat_conv.shape == flat_data.shape
    fitted_params = ghost.fit_x_to_image(flat_conv, xparams=xparams,
                                         decrease_dim=8, inspect=False)
    assert fitted_params.shape == xparams.shape
