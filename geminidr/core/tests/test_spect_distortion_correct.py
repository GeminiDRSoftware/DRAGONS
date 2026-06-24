"""
Tests for the distortionCorrect primitive in Spect, using synthetic data.

These are based on a modified GNIRS LS file, but the AstroData object is
manipulated to behave differently for the various tests.

Note that, at present (24 Jun 2026, v4.2.2), the extraction along curved
slits for determineWavelengthSolution and determineDistortion can only
happen along the center of the slit.
"""
import pytest

import numpy as np

import astrodata, gemini_instruments
from astrodata.testing import download_from_archive
from geminidr.gnirs.primitives_gnirs_longslit import GNIRSLongslit
from geminidr.gnirs.primitives_gnirs_crossdispersed import GNIRSCrossDispersed
from gempy.library import astromodels as am

from astropy.modeling import models
from astropy.table import Table, vstack

rng = np.random.default_rng(42)
YSHIFT_PER_COL = 0.07

XD_EDGE_MODELS = [
    ((99.5,), (923.5,)),  # straight edges, centered on detector
    ((99.5,), (899.5,)),  # straight edges, offset from center
    ((99.5, 20), (899.5, 20)),
    ((99.5, -10), (899.5, -10))
]


@pytest.fixture
def ad_ls():
    ad = astrodata.open(download_from_archive("N20250508S0090.fits"))
    p = GNIRSLongslit([ad])
    p.prepare()  # to get a proper longslit WCS
    return p.adinputs.pop()


@pytest.fixture
def ad_xd():
    ad = astrodata.open(download_from_archive("N20250508S0090.fits"))
    p = GNIRSLongslit([ad])
    p.prepare()  # we keep it as LS here so there's only 1 slit in the MDF
    ad.phu['PRISM'] = 'SB+SXD_G5536'
    return p.adinputs.pop()


def create_distorted_data(ad, wavecal_column=None, subsampling=10):
    """
    Modify an AstroData object so that its data plane is a distorted image
    of a set of Gaussian lines, and optionally add a WAVECAL table.

    Parameters
    ----------
    ad: AstroData
    wavecal_column: float/None
        the column for the WAVECAL table, or None to not add a WAVECAL table
    subsampling: int
        number of subpixels to evaluate the Gaussian line profile before
        summing to output pixels

    Returns
    -------
    Model: the applied distortion model
    """
    def mkwavecal_table(lines, column=None):
        lines = np.asanyarray(lines)
        m_wave = models.Scale(0.5) | models.Shift(1000)
        wavelengths = m_wave(lines)
        names = ["column", "fwidth"] + [""] * (len(lines) - 2)
        coeffs = [column, 5] + [0] * (len(lines) - 2)
        t = Table([names, coeffs, lines + 1, wavelengths],
                  names=["name", "coefficients", "peaks", "wavelengths"])
        return t

    ad[0].mask = None
    ad[0].variance = None
    ny, nx = ad[0].shape
    mid_column = 0.5 * (nx-1)
    m2d = models.Chebyshev2D(x_degree=1, y_degree=1,
                             c0_0=0.5*(ny-1), c0_1=0.5*(ny-1),
                             c1_0=mid_column*YSHIFT_PER_COL,
                             x_domain=(0, nx-1), y_domain=(0, ny-1))
    m = models.Mapping((0, 0, 1)) | models.Identity(1) & m2d

    data = np.empty((nx, ny*subsampling), dtype=np.float32)
    ylines = np.arange(32, ny, 64)
    if wavecal_column is not None:
        ad[0].WAVECAL = mkwavecal_table(
            m(np.full_like(ylines, wavecal_column), ylines)[1],
            column=wavecal_column)

    y = np.arange(ny*subsampling)
    for x in range(nx):
        mcol = models.Const1D(0)
        for yy in m(np.full_like(ylines, x), ylines)[1]:
            ymean = (yy + 0.5) * subsampling - 0.5
            mcol += models.Gaussian1D(mean=ymean, amplitude=1000.,
                                      stddev=2.5*subsampling)
        data[x] = mcol(y)

    # Add noise to enable S/N calculations to work
    data = data.reshape(nx, ny, subsampling).sum(axis=-1)
    rng = np.random.default_rng(42)
    data += rng.normal(size=data.shape)
    ad[0].data = data.T
    return m


@pytest.mark.parametrize("wavecal_column", [None, 511.5, 720, 312.3])
def test_distortion_correct_ls(ad_ls, wavecal_column):
    p = GNIRSLongslit([ad_ls])
    m_ref = create_distorted_data(ad_ls, wavecal_column=wavecal_column)
    p.determineDistortion(max_shift=0.1)
    m_out = ad_ls[0].wcs.get_transform('pixels', 'distortion_corrected').inverse
    Y, X = np.mgrid[:ad_ls[0].shape[0], :ad_ls[0].shape[1]]
    real_yshift = m_ref(X, Y)[1] - Y
    if wavecal_column is not None:
        real_yshift -= YSHIFT_PER_COL * (wavecal_column - 511.5)
    xshift, yshift = [model_out - start
                      for model_out, start in zip(m_out(X, Y), (X, Y))]
    # A tolerance of 0.01 should be fine to spot problems
    np.testing.assert_allclose(xshift, 0, atol=0.01)
    np.testing.assert_allclose(yshift, real_yshift, atol=0.01)


@pytest.mark.parametrize("edges", XD_EDGE_MODELS)
def test_distortion_correct_xd(ad_xd, edges):
    p = GNIRSCrossDispersed([ad_xd])
    edge_models = [models.Chebyshev1D(degree=len(edge)-1, domain=(0, 1021),
                                      **{f"c{i}": coeff for i, coeff in enumerate(edge)})
                   for edge in edges]
    ad_xd[0].SLITEDGE = vstack([am.model_to_table(edge_model)
                                for edge_model in edge_models],
                               metadata_conflicts="silent")
    m_ref = create_distorted_data(ad_xd, wavecal_column=None)
    p.determineDistortion(max_shift=0.1)
    m_out = ad_xd[0].wcs.get_transform('pixels', 'distortion_corrected').inverse
    Y, X = np.mgrid[:ad_xd[0].shape[0], :ad_xd[0].shape[1]]
    real_yshift = m_ref(X, Y)[1] - Y
    # Since the "spectrum" is always extracted in the middle of the slit,
    # the y-shift is relative to the middle of the slit, not the middle of
    # the detector. So we need to adjust the expected y-shift.
    real_yshift -= (0.5 * (edge_models[0](Y) + edge_models[1](Y)) - 511.5) * YSHIFT_PER_COL
    xshift, yshift = [model_out - start
                      for model_out, start in zip(m_out(X, Y), (X, Y))]
    # A tolerance of 0.01 should be fine to spot problems
    np.testing.assert_allclose(xshift, 0, atol=0.01)
    np.testing.assert_allclose(yshift, real_yshift, atol=0.01)
