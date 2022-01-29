"""
Fixtures to be used in tests in DRAGONS
"""

import os
import shutil
import urllib
import inspect
import xml.etree.ElementTree as et

import numpy as np
import pytest
from astropy.table import Table
from astropy.utils.data import download_file
from geminidr.gemini.lookups.timestamp_keywords import timestamp_keys

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
    Queries Gemini Observatory Archive for associated calibrations to reduce
    the data that will be used for testing.

    Parameters
    ----------
    filename : str
        Input file name
    """
    url = f"https://archive.gemini.edu/calmgr/{filename}"
    tree = et.parse(urllib.request.urlopen(url))
    root = tree.getroot()
    prefix = root.tag[:root.tag.rfind('}') + 1]

    rows = []
    for node in tree.iter(prefix + 'calibration'):
        cal_type = node.find(prefix + 'caltype').text
        cal_filename = node.find(prefix + 'filename').text
        if not ('processed_' in cal_filename or 'specphot' in cal_filename):
            rows.append((cal_filename, cal_type))

    tbl = Table(rows=rows, names=['filename', 'caltype'])
    tbl.sort('filename')
    tbl.remove_rows(np.where(tbl['caltype'] == 'bias')[0][nbias:])
    return tbl


class ADCompare:
    """
    Compare two AstroData instances to determine whether they are basically
    the same. Various properties (both data and metadata) can be compared
    """
    def __init__(self, ad1, ad2):
        self.ad1 = ad1
        self.ad2 = ad2

    def run_comparison(self, max_miss=0, rtol=1e-7, atol=0, compare=None,
                       ignore=None, raise_exception=True):
        """
        Perform a comparison between the two AD objects in this instance.

        Parameters
        ----------
        max_miss: int
            maximum number of elements in each array that can disagree
        rtol: float
            relative tolerance allowed between array elements
        atol: float
            absolute tolerance allowed between array elements
        compare: list/None
            list of comparisons to perform
        ignore: list/None
            list of comparisons to ignore
        raise_exception: bool
            raise an AssertionError if the comparison fails? If False,
            the errordict is returned, which may be useful if a very
            specific mismatch is permitted

        Raises
        -------
        AssertionError if the AD objects do not agree.
        """
        self.max_miss = max_miss
        self.rtol = rtol
        self.atol = atol
        if compare is None:
            compare = ('filename', 'tags', 'numext', 'refcat', 'phu',
                           'hdr', 'attributes')
        if ignore is not None:
            compare = [c for c in compare if c not in ignore]

        errordict = {}
        for func_name in compare:
            errorlist = getattr(self, func_name)()
            if errorlist:
                errordict[func_name] = errorlist
        if errordict and raise_exception:
            raise AssertionError(self.format_errordict(errordict))
        return errordict

    def numext(self):
        """Check the number of extensions is equal"""
        numext1, numext2 = len(self.ad1), len(self.ad2)
        if numext1 != numext2:
            return [f'{numext1} vs {numext2}']

    def filename(self):
        """Check the filenames are equal"""
        fname1, fname2 = self.ad1.filename, self.ad2.filename
        if fname1 != fname2:
            return [f'{fname1} vs {fname2}']

    def tags(self):
        """Check the tags are equal"""
        tags1, tags2 = self.ad1.tags, self.ad2.tags
        if tags1 != tags2:
            return [f'{tags1}\n  vs: {tags2}']

    def phu(self):
        """Check the PHUs agree"""
        errorlist = self._header(self.ad1.phu, self.ad2.phu)
        if errorlist:
            return errorlist

    def hdr(self):
        """Check the extension headers agree"""
        errorlist = []
        for i, (hdr1, hdr2) in enumerate(zip(self.ad1.hdr, self.ad2.hdr)):
            elist = self._header(hdr1, hdr2)
            if elist:
                errorlist.extend([f'Slice {i} HDR mismatch'] + elist)
        return errorlist

    def _header(self, hdr1, hdr2):
        """General method for comparing headers, ignoring some keywords"""
        errorlist = []
        s1 = set(hdr1.keys()) - {'HISTORY', 'COMMENT'}
        s2 = set(hdr2.keys()) - {'HISTORY', 'COMMENT'}
        if s1 != s2:
            if s1 - s2:
                errorlist.append(f'Header 1 contains keywords {s1 - s2}')
            if s2 - s1:
                errorlist.append(f'Header 2 contains keywords {s2 - s1}')

        for kw in hdr1:
            # GEM-TLM is "time last modified"
            if kw not in timestamp_keys.values() and kw not in ['GEM-TLM',
                                                    'HISTORY', 'COMMENT', '']:
                try:
                    v1, v2 = hdr1[kw], hdr2[kw]
                except KeyError:  # Missing keyword in AD2
                    continue
                if not (isinstance(v1, float) and abs(v1 - v2) < 0.01 or v1 == v2):
                    errorlist.append('{} value mismatch: {} v {}'.
                                format(kw, v1, v2))
        return errorlist

    def refcat(self):
        """Check both ADs have REFCATs (or not) and that the lengths agree"""
        refcat1 = getattr(self.ad1, 'REFCAT', None)
        refcat2 = getattr(self.ad2, 'REFCAT', None)
        if (refcat1 is None) ^ (refcat2 is None):
            return [f'presence: {refcat1 is not None} vs {refcat2 is not None}']
        elif refcat1 is not None:  # and refcat2 must also exist
            len1, len2 = len(refcat1), len(refcat2)
            if len1 != len2:
                return [f'lengths: {len1} vs {len2}']

    def attributes(self):
        """Check extension-level attributes"""
        errorlist = []
        for i, (ext1, ext2) in enumerate(zip(self.ad1, self.ad2)):
            elist = self._attributes(ext1, ext2)
            if elist:
                errorlist.extend([f'Slice {i} attribute mismatch'] + elist)
        return errorlist

    def _attributes(self, ext1, ext2):
        """Helper method for checking attributes"""
        errorlist = []
        for attr in ['data', 'mask', 'variance', 'OBJMASK', 'OBJCAT']:
            attr1 = getattr(ext1, attr, None)
            attr2 = getattr(ext2, attr, None)
            if (attr1 is None) ^ (attr2 is None):
                errorlist.append(f'Attribute error for {attr}: '
                                 f'{attr1 is not None} v {attr2 is not None}')
            elif attr1 is not None:
                if isinstance(attr, Table):
                    if len(attr1) != len(attr2):
                        errorlist.append(f'attr lengths differ: '
                                         f'{len(attr1)} vs {len(attr2)}')
                else:  # everything else is pixel-like
                    if attr1.dtype.name != attr2.dtype.name:
                        errorlist.append(f'Datatype mismatch for {attr}: '
                                         f'{attr1.dtype} vs {attr2.dtype}')
                    if attr1.shape != attr2.shape:
                        errorlist.append(f'Shape mismatch for {attr}: '
                                         f'{attr1.shape} vs {attr2.shape}')
                    if 'int' in attr1.dtype.name:
                        try:
                            assert_most_equal(attr1, attr2, max_miss=self.max_miss)
                        except AssertionError as e:
                            errorlist.append(f'Inequality for {attr}: '+str(e))
                    else:
                        try:
                            assert_most_close(attr1, attr2, max_miss=self.max_miss,
                                              rtol=self.rtol, atol=self.atol)
                        except AssertionError as e:
                            errorlist.append(f'Mismatch for {attr}: '+str(e))
        return errorlist

    @staticmethod
    def format_errordict(errordict):
        """Format the errordict into a str for reporting"""
        errormsg = ''
        for k, v in errordict.items():
            errormsg += f'\nComparison failure in {k}'
            errormsg += '\n' + ('-' * (22 + len(k))) + '\n'
            errormsg += '\n  '.join(v)
        return errormsg

def ad_compare(ad1, ad2, **kwargs):
    """
    Compares the tags, headers, and pixel values of two images. This is simply
    a wrapper for ADCompare.run_comparison() for backward-compatibility.

    Parameters
    ----------
    ad1: AstroData
        first AD objects
    ad2: AstroData
        second AD object

    Returns
    -------
    bool: are the two AD instances basically the same?
    """
    compare = ADCompare(ad1, ad2).run_comparison(**kwargs)
    return compare == {}
