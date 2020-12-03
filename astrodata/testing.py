"""
Fixtures to be used in tests in DRAGONS
"""

import os
import shutil
import urllib
import xml.etree.ElementTree as et

import pytest
from astropy.utils.data import download_file

URL = 'https://archive.gemini.edu/file/'


def assert_most_close(actual, desired, max_miss, rtol=1e-7, atol=0,
                      equal_nan=True, verbose=True):
    """
    Raises an AssertionError if the number of elements in two objects that are
    not equal up to desired tolerance is greater than expected.

    See Also
    --------
    :func:`~numpy.testing.assert_allclose`

    Parameters
    ----------
    actual : array_like
        Array obtained.
    desired : array_like
        Array desired.
    max_miss : iny
        Maximum number of mismatched elements.
    rtol : float, optional
        Relative tolerance.
    atol : float, optional
        Absolute tolerance.
    equal_nan : bool, optional.
        If True, NaNs will compare equal.
    verbose : bool, optional
        If True, the conflicting values are appended to the error message.
    Raises
    ------
    AssertionError
        If actual and desired are not equal up to specified precision.
    """
    from numpy.testing import assert_allclose

    try:
        assert_allclose(actual, desired, atol=atol, equal_nan=equal_nan,
                        err_msg='', rtol=rtol, verbose=verbose)

    except AssertionError as e:
        n_miss = e.args[0].split('\n')[3].split(':')[-1].split('(')[0].split('/')[0]
        n_miss = int(n_miss.strip())

        if n_miss > max_miss:
            error_message = (
                    "%g mismatching elements are more than the " % n_miss +
                    "expected %g." % max_miss +
                    '\n'.join(e.args[0].split('\n')[3:]))

            raise AssertionError(error_message)


def assert_most_equal(actual, desired, max_miss, verbose=True):
    """
    Raises an AssertionError if more than `n` elements in two objects are not
    equal. For more information, check :func:`numpy.testing.assert_equal`.

    Parameters
    ----------
    actual : array_like
        The object to check.
    desired : array_like
        The expected object.
    max_miss : int
        Maximum number of mismatched elements.
    verbose : bool, optional
        If True, the conflicting values are appended to the error message.

    Raises
    ------
    AssertionError
        If actual and desired are not equal.
    """
    from numpy.testing import assert_equal

    try:
        assert_equal(actual, desired, err_msg='', verbose=verbose)
    except AssertionError as e:

        n_miss = e.args[0].split('\n')[3].split(':')[-1].split('(')[0].split('/')[0]
        n_miss = int(n_miss.strip())

        if n_miss > max_miss:
            error_message = (
                    "%g mismatching elements are more than the " % n_miss +
                    "expected %g." % max_miss +
                    '\n'.join(e.args[0].split('\n')[3:]))

            raise AssertionError(error_message)


def assert_same_class(ad, ad_ref):
    """
    Compare if two :class:`~astrodata.AstroData` (or any subclass) have the
    same class.

    Parameters
    ----------
        ad : :class:`astrodata.AstroData` or any subclass
            AstroData object to be checked.
        ad_ref : :class:`astrodata.AstroData` or any subclass
            AstroData object used as reference
    """
    from astrodata import AstroData

    assert isinstance(ad, AstroData)
    assert isinstance(ad_ref, AstroData)
    assert isinstance(ad, type(ad_ref))


def compare_models(model1, model2, rtol=1e-7, atol=0., check_inverse=True):
    """
    Check that any two models are the same, within some tolerance on parameters
    (using the same defaults as numpy.assert_allclose()).

    This is constructed like a test, rather than returning True/False, in order
    to provide more useful information as to how the models differ when a test
    fails (and with more concise syntax).

    If `check_inverse` is True (the default), only first-level inverses are
    compared, to avoid unending recursion, since the inverse of an inverse
    should be the supplied input model, if defined. The types of any inverses
    (and their inverses in turn) are required to match whether or not their
    parameters etc. are compared.

    This function might not completely guarantee that model1 & model2 are
    identical for some models whose evaluation depends on class-specific
    parameters controlling how the array of model `parameters` is interpreted
    (eg. the orders in SIP?), but it does cover our common use of compound
    models involving orthonormal polynomials etc.
    """

    from astropy.modeling import Model
    from numpy.testing import assert_allclose

    if not (isinstance(model1, Model) and isinstance(model2, Model)):
        raise TypeError('Inputs must be Model instances')

    if model1 is model2:
        return

    # Require each model to be composed of same number of constituent models:
    assert model1.n_submodels == model2.n_submodels

    # Treat everything like an iterable compound model:
    if model1.n_submodels == 1:
        model1 = [model1]
        model2 = [model2]

    # Compare the constituent model definitions:
    for m1, m2 in zip(model1, model2):
        assert type(m1) == type(m2)
        assert len(m1.parameters) == len(m2.parameters)
        # NB. For 1D models the degrees match if the numbers of parameters do
        if hasattr(m1, 'x_degree'):
            assert m1.x_degree == m2.x_degree
        if hasattr(m1, 'y_degree'):
            assert m1.y_degree == m2.y_degree
        if hasattr(m1, 'domain'):
            assert m1.domain == m2.domain
        if hasattr(m1, 'x_domain'):
            assert m1.x_domain == m2.x_domain
        if hasattr(m1, 'y_domain'):
            assert m1.y_domain == m2.y_domain

    # Compare the model parameters (coefficients):
    assert_allclose(model1.parameters, model2.parameters, rtol=rtol, atol=atol)

    # Now check for any inverse models and require them both to have the same
    # type or be undefined:
    try:
        inverse1 = model1.inverse
    except NotImplementedError:
        inverse1 = None
    try:
        inverse2 = model2.inverse
    except NotImplementedError:
        inverse2 = None

    assert type(inverse1) == type(inverse2)

    # Compare inverses only if they exist and are not the forward model itself:
    if inverse1 is None or (inverse1 is model1 and inverse2 is model2):
        check_inverse = False

    # Recurse over the inverse models (but not their inverses in turn):
    if check_inverse:
        compare_models(inverse1, inverse2, rtol=rtol, atol=atol,
                       check_inverse=False)


def download_from_archive(filename, sub_path='raw_files', env_var='DRAGONS_TEST'):
    """Download file from the archive and store it in the local cache.

    Parameters
    ----------
    filename : str
        The filename, e.g. N20160524S0119.fits
    sub_path : str
        By default the file is stored at the root of the cache directory, but
        using ``path`` allows to specify a sub-directory.
    env_var: str
        Environment variable containing the path to the cache directory.

    Returns
    -------
    str
        Name of the cached file with the path added to it.
    """
    # Find cache path and make sure it exists
    root_cache_path = os.getenv(env_var)

    if root_cache_path is None:
        raise ValueError('Environment variable not set: {:s}'.format(env_var))

    root_cache_path = os.path.expanduser(root_cache_path)

    if sub_path is not None:
        cache_path = os.path.join(root_cache_path, sub_path)

    if not os.path.exists(cache_path):
        os.makedirs(cache_path)

    # Now check if the local file exists and download if not
    local_path = os.path.join(cache_path, filename)
    if not os.path.exists(local_path):
        tmp_path = download_file(URL + filename, cache=False)
        shutil.move(tmp_path, local_path)

        # `download_file` ignores Access Control List - fixing it
        os.chmod(local_path, 0o664)

    return local_path


def get_associated_calibrations(filename, nbias=5):
    """
    Queries Gemini Observatory Archive for associated calibrations to reduce the
    data that will be used for testing.
    Parameters
    ----------
    filename : str
        Input file name
    """
    pd = pytest.importorskip("pandas", minversion='1.0.0')
    url = "https://archive.gemini.edu/calmgr/{}".format(filename)

    tree = et.parse(urllib.request.urlopen(url))
    root = tree.getroot()
    prefix = root.tag[:root.tag.rfind('}') + 1]

    def iter_nodes(node):
        cal_type = node.find(prefix + 'caltype').text
        cal_filename = node.find(prefix + 'filename').text
        return cal_filename, cal_type

    cals = pd.DataFrame(
        [iter_nodes(node) for node in tree.iter(prefix + 'calibration')],
        columns=['filename', 'caltype'])

    cals = cals.sort_values(by='filename')
    cals = cals[~cals.caltype.str.contains('processed_')]
    cals = cals[~cals.caltype.str.contains('specphot')]
    cals = cals.drop(cals[cals.caltype.str.contains('bias')][nbias:].index)

    return cals
