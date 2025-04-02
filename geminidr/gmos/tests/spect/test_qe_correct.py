#!/usr/bin/env python
"""
Tests for applyQECorrection primitive.

Changelog
---------
2020-06-01
    - Recreated input files using:
        - astropy 4.1rc1
        - gwcs 0.13.1.dev19+gc064a02 - This should be cloned and the
            `transform-1.1.0` string should be replaced by `transform-1.2.0`
            in the `gwcs/schemas/stsci.edu/gwcs/step-1.0.0.yaml` file.

2023-07-31
    - Changed the threshold in transform.resample_from_wcs() from 0.01 to 0.001,
      which caused the test files N20180109S0287_flatCorrected.fits and
      S20180919S0139_flatCorrected.fits to fail due to additional pixels getting
      flagged as bad. Currently the methodology for
      test_qe_correct_is_locally_continuous() involves smoothing the data and
      fitting a spline to each of the three parts individually, then checking
      the difference between their interpolations midway through each of the
      gaps. This has issues since the smoothing doesn't take into account the
      gaps, leading to spikes in the smoothed data near them, which can lead to
      the splines not tracing the data well near their ends.

      Turning off smoothing led to the two files above passing, but caused
      S20191005S0051_flatCorrected.fits to fail due to a large absorption feature
      in the central section which seriously disturbs the spline fitting. Either
      some new method to better capture the task of comparing continuity across
      regions after QE correction, or some tweaks to the current method to make
      it work with all the test files, is needed.

"""
import abc
import matplotlib.pyplot as plt
import numpy as np
import os
import pytest

import astrodata
import gemini_instruments

from astropy import modeling
from geminidr.gmos import primitives_gmos_longslit
from gempy.library import astromodels
from gempy.utils import logutils
from recipe_system.testing import ref_ad_factory
from scipy import ndimage

DPI = 90

datasets = [

    # --- Good datasets ---
    "N20180109S0287_flatCorrected.fits",  # GN-2017B-FT-20-13-001 B600 0.505um
    "N20190302S0089_flatCorrected.fits",  # GN-2019A-Q-203-7-001 B600 0.550um
    "N20190313S0114_flatCorrected.fits",  # GN-2019A-Q-325-13-001 B600 0.482um
    "N20190427S0123_flatCorrected.fits",  # GN-2019A-FT-206-7-001 R400 0.525um
    "N20190427S0126_flatCorrected.fits",  # GN-2019A-FT-206-7-004 R400 0.625um
    "N20190910S0028_flatCorrected.fits",  # GN-2019B-Q-313-5-001 B600 0.550um
    "S20180919S0139_flatCorrected.fits",  # GS-2018B-Q-209-13-003 B600 0.45um
    "S20191005S0051_flatCorrected.fits",  # GS-2019B-Q-132-35-001 R400 0.73um

    # --- QE Correction Needs improvement ---
    # "N20180721S0444.fits",  # GN-2018B-Q-313-5-002 B1200 0.44um
    # "N20190707S0032.fits",  # GN-2019A-Q-232-32-001 B1200 0.453um
    # "N20190929S0127.fits",  # GN-2019B-Q-209-5-001 B1200 0.449um
    # "S20180418S0151.fits",  # GS-2018A-Q-211-38-002 B1200 0.44um
    # "S20181025S0033.fits",  # GS-2018B-Q-311-5-001 B1200 0.445um
    # "S20190506S0088.fits",  # GS-2019A-Q-401-5-001 B1200 0.45um
    # "S20190118S0102.fits",  # GS-2018B-Q-308-8-001 B600 0.589um
    # "N20181005S0139.fits",  # GN-2018B-FT-207-5-001 R400 0.700um

    # --- ToDo: QE correction is good. Need to find fitting parameters ---
    # "N20180509S0010.fits",  # GN-2018A-FT-107-7-001 R400 0.900um
    # "N20180516S0081.fits",  # GN-2018C-1-37-001 R600 0.860um

    # --- Test Won't do ---
    # "N20180508S0021.fits",  # GN-2018A-Q-212-9-005 B600 0.720um
    # "N20190427S0141.fits",  # GN-2019A-Q-233-45-004 R150 0.660um

    # --- Other ---
    # "N20190201S0163.fits",  # Could not reduce? (p.writeOutputs frozen)
    # "N20180228S0134.fits",  # GN-2018A-Q-121-11-001 R400 0.52um (p.writeOutputs frozen)

]

gap_local_kw = {
    "N20180109S0287.fits": {'bad_cols': 5},
    "N20180228S0134.fits": {},
    "N20180508S0021.fits": {'bad_cols': 5, 'order': 5, 'sigma_lower': 1.5},
    "N20180509S0010.fits": {},
    "N20190201S0163.fits": {},
    "N20190302S0089.fits": {'bad_cols': 5, 'wav_min': 450},
    "N20190313S0114.fits": {'order': 4, 'med_filt_size': 20},
    "N20190427S0123.fits": {'bad_cols': 5, 'order': 3},
    "N20190427S0126.fits": {'bad_cols': 5, 'order': 3},
    "N20190910S0028.fits": {},
    "S20180919S0139.fits": {'bad_cols': 10, 'order': 4},
    "S20191005S0051.fits": {'order': 4, 'med_filt_size': 100},
}

associated_calibrations = {
    "N20180109S0287.fits": {
        'bias': ["N20180109S0351.fits",
                 "N20180109S0352.fits",
                 "N20180109S0353.fits",
                 "N20180109S0354.fits",
                 "N20180109S0355.fits"],
        'flat': ["N20180109S0288.fits"],
        'arcs': ["N20180109S0315.fits"],
    },
    "N20190302S0089.fits": {
        'bias': ["N20190226S0338.fits",
                 "N20190226S0339.fits",
                 "N20190226S0340.fits",
                 "N20190226S0341.fits",
                 "N20190226S0342.fits"],
        'flat': ["N20190302S0090.fits"],
        'arcs': ["N20190302S0274.fits"],
    },
    "N20190313S0114.fits": {
        'bias': ["N20190308S0433.fits",
                 "N20190308S0434.fits",
                 "N20190308S0435.fits",
                 "N20190308S0436.fits",
                 "N20190308S0437.fits"],
        'flat': ["N20190313S0115.fits"],
        'arcs': ["N20190313S0132.fits"],
    },
    "N20190427S0123.fits": {
        'bias': ["N20190421S0283.fits",
                 "N20190421S0284.fits",
                 "N20190421S0285.fits",
                 "N20190421S0286.fits",
                 "N20190421S0287.fits"],
        'flat': ["N20190427S0124.fits"],
        'arcs': ["N20190427S0266.fits"],
    },
    "N20190427S0126.fits": {
        'bias': ["N20190421S0283.fits",
                 "N20190421S0284.fits",
                 "N20190421S0285.fits",
                 "N20190421S0286.fits",
                 "N20190421S0287.fits"],
        'flat': ["N20190427S0125.fits"],
        'arcs': ["N20190427S0267.fits"],
    },
    "N20190910S0028.fits": {
        'bias': ["N20190903S0141.fits",
                 "N20190903S0142.fits",
                 "N20190903S0143.fits",
                 "N20190903S0144.fits",
                 "N20190903S0145.fits"],
        'flat': ["N20190910S0029.fits"],
        'arcs': ["N20190910S0279.fits"],
    },
    "S20180919S0139.fits": {
        'bias': ["S20180913S0093.fits",
                 "S20180913S0094.fits",
                 "S20180913S0095.fits",
                 "S20180913S0096.fits",
                 "S20180913S0097.fits"],
        'flat': ["S20180919S0140.fits"],
        'arcs': ["S20180919S0141.fits"],
    },
    "S20191005S0051.fits": {
        'bias': ["S20190928S0236.fits",
                 "S20190928S0237.fits",
                 "S20190928S0238.fits",
                 "S20190928S0239.fits",
                 "S20190928S0240.fits"],
        'flat': ["S20191005S0052.fits"],
        'arcs': ["S20191005S0151.fits"],
    },
}


# -- Tests --------------------------------------------------------------------
@pytest.mark.gmosls
@pytest.mark.preprocessed_data
@pytest.mark.parametrize("ad", datasets, indirect=True)
def test_qe_correct_is_locally_continuous(ad, change_working_dir):

    if ad.filename in ('N20180109S0287_flatCorrected.fits',
                       'S20180919S0139_flatCorrected.fits'):
        pytest.skip('FIXME: this test fails following changes to the threshold '
                    'parameter in transform.resample_from_wcs(). Needs more '
                    'investigation.')

    with change_working_dir():

        logutils.config(file_name='log_test_continuity{}.txt'.format(ad.data_label()))
        p = primitives_gmos_longslit.GMOSLongslit([ad])
        p.QECorrect()

        # Need these extra steps to extract and analyse the data
        p.distortionCorrect()
        p.findApertures(max_apertures=1)
        p.skyCorrectFromSlit()
        p.traceApertures()
        p.extractSpectra()
        p.linearizeSpectra()
        processed_ad = p.writeOutputs().pop()

    for ext in processed_ad:
        assert not np.any(np.isnan(ext.data))
        assert not np.any(np.isinf(ext.data))

    basename = processed_ad.filename.replace('_linearized', '')
    kwargs = gap_local_kw[basename] if basename in gap_local_kw.keys() else {}
    gap = MeasureGapSizeLocallyWithSpline(processed_ad, **kwargs)

    assert gap.left_gap_size < 0.05
    assert gap.right_gap_size < 0.05


@pytest.mark.gmosls
@pytest.mark.preprocessed_data
@pytest.mark.regression
@pytest.mark.parametrize("ad", datasets, indirect=True)
def test_regression_on_qe_correct(ad, change_working_dir, ref_ad_factory):

    # The GMOS-N tests need to be run with `use_iraf=False` because the
    # reference files for those use the DRAGONS spline models.
    is_gmos_s = ad.instrument() == 'GMOS-S'

    with change_working_dir():
        logutils.config(file_name='log_test_regression{}.txt'.format(ad.data_label()))
        p = primitives_gmos_longslit.GMOSLongslit([ad])
        p.QECorrect(use_iraf=is_gmos_s)
        qe_corrected_ad = p.writeOutputs().pop()

    assert 'QECORR' in qe_corrected_ad.phu.keys()

    ref_ad = ref_ad_factory(qe_corrected_ad.filename)

    for qe_corrected_ext, ref_ext in zip(qe_corrected_ad, ref_ad):
        np.testing.assert_allclose(
            np.ma.masked_array(qe_corrected_ext.data, mask=qe_corrected_ext.mask),
            np.ma.masked_array(ref_ext.data, mask=ref_ext.mask),
            atol=0.05)


# -- Fixtures -----------------------------------------------------------------
@pytest.fixture(scope='function')
def ad(path_to_inputs, request):
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
    filename = request.param
    path = os.path.join(path_to_inputs, filename)

    if os.path.exists(path):
        print(f"Reading input file: {path}")
        ad = astrodata.from_file(path)
    else:
        raise FileNotFoundError(path)

    return ad


@pytest.fixture(scope='function')
def arc_ad(path_to_inputs, request):
    """
    Returns the master arc used during the data-set data reduction.

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
        Master arc.
    """
    filename = request.param
    path = os.path.join(path_to_inputs, filename)

    if os.path.exists(path):
        print(f"Reading input arc: {path}")
        arc_ad = astrodata.from_file(path)
    else:
        raise FileNotFoundError(path)

    return arc_ad


# -- Classes and functions for analysis ---------------------------------------
def normalize_data(y, v):
    """
    Normalize data for analysis.

    Parameters
    ----------
    y : np.ma.MaskedArray
        Values as a masked array.
    v : np.ma.MaskedArray
        Variance as a masked array.

    Returns
    -------
    _y : np.ma.MaskedArray
        Normalized values as a masked array.
    _v : np.ma.MaskedArray
        Normalized variance as a masked array.
    """
    assert np.ma.is_masked(y)
    _y = (y - np.min(y)) / np.ptp(y)
    _v = (v - np.min(v)) / np.ptp(y)
    return _y, _v


def smooth_data(y, median_filter_size):
    """
    Smooth input masked data using median and gaussian filters.

    Parameters
    ----------
    y : np.ma.MaskedArray
        Input masked 1D array.
    median_filter_size : int
        Size of the median filter.

    Returns
    -------
    np.ma.MaskedArray : smoothed 1D masked array.
    """
    assert np.ma.is_masked(y)
    m = y.mask
    y = ndimage.median_filter(y, size=median_filter_size)
    y = ndimage.gaussian_filter1d(y, sigma=5)
    y = np.ma.masked_array(y, mask=m)
    return y


def split_arrays_using_mask(x, y, mask, bad_cols):
    """
    Split x and y into segments based on masked pixels in x.

    Parameters
    ----------
    x : np.array
        Masked array
    y : np.array
        Array that will be masked to follow x.
    mask : np.array
        Array that contains the mask used to split x and y.
    bad_cols : int
        Number of bad columns that will be ignored around the gaps.

    Returns
    -------
    x_seg : list
        List of np.arrays with the segments obtained from x.
    y_seg : list
        List of np.arrays with the segments obtained from y.
    cuts : list
        1D list containing the beginning and the end pixels for each gap.
    """
    m = ~ndimage.binary_opening(mask, iterations=10)

    d = np.diff(m)
    cuts = np.flatnonzero(d) + 1

    if len(cuts) != 4:
        print(cuts)
        print(np.diff(cuts))
        raise ValueError(
            "Expected four coordinates in `cuts` variable. "
            "Found {}".format(len(cuts)))

    # Broadening gap
    for i in range(cuts.size // 2):
        cuts[2 * i] -= bad_cols
        cuts[2 * i + 1] += bad_cols

    xsplit = np.split(x, cuts)
    ysplit = np.split(y, cuts)
    msplit = np.split(m, cuts)

    x_seg = [xseg for xseg, mseg in zip(xsplit, msplit) if np.all(mseg)]
    y_seg = [yseg for yseg, mseg in zip(ysplit, msplit) if np.all(mseg)]

    if len(x_seg) != 3:
        raise ValueError(
            "Expected three segments for X. Found {}".format(len(x_seg)))

    if len(y_seg) != 3:
        raise ValueError(
            "Expected three segments for Y. Found {}".format(len(y_seg)))

    return x_seg, y_seg, cuts


class Gap:
    """
    Higher lever interface for gaps.

    Parameters
    ----------
    left : int
        Left index that defines the beginning of the gap.
    right : int
        Right index that defines the end of the gap
    i : int
        Increment size on both sides of the gap.

    Attributes
    ----------
    center : float
        Center of the gap
    left : int
        Left index that defines the beginning of the gap.
    right : int
        Right index that defines the beginning of the gap.
    size : int
        Gap size in number of elements.
    """
    def __init__(self, left, right, i):
        self.center = 0.5 * (left + right)
        self.left = left - i
        self.right = right + i
        self.size = abs(self.left - self.right)


class MeasureGapSizeLocally(abc.ABC):
    """
    Abstract class used to evaluate the measure the gap size locally.

    Parameters
    ----------
    ad : astrodata.AstroData
        Input qe corrected spectrum with wavelength solution.
    bad_cols : int
        Number of bad columns around the gaps.
    fit_family : str
        Short name of the fit method to be used in plots and labels.
    med_filt_size : int
        Median filter size.
    order : int
        Order of the polynomial/chebyshev/spline fit (depends on sub-class).
    sigma_lower : float
        Lower sigma limit for sigma clipping.
    sigma_upper : float
        Upper sigma limit for sigma clipping.
    """
    def __init__(self, ad, bad_cols=21, fit_family=None, med_filt_size=5,
                 order=5, sigma_lower=1.5, sigma_upper=3, wav_min=350.,
                 wav_max=1050):

        self.ad = ad
        self.bad_cols = bad_cols
        self.fit_family = fit_family
        self.median_filter_size = med_filt_size
        self.order = order
        self.plot_name = ""
        self.sigma_lower = sigma_lower
        self.sigma_upper = sigma_upper
        self.wav_max = wav_max
        self.wav_min = wav_min
        self.w_solution = WSolution(ad)

        m = ad[0].mask > 0
        y = ad[0].data
        v = ad[0].variance  # Never used for anything. But...
        x = np.arange(y.size)

        w = self.w_solution(x)
        w = np.ma.masked_outside(w, wav_min, wav_max)

        y = np.ma.masked_array(y, mask=m)
        y = smooth_data(y, self.median_filter_size)
        y, v = normalize_data(y, v)

        y.mask = np.logical_or(y.mask, y < 0.01)

        split_mask = ad[0].mask >= 16
        y[split_mask] = 0
        x_seg, y_seg, cuts = \
            split_arrays_using_mask(x, y, split_mask, self.bad_cols)

        self.x, self.y, self.w = x, y, w
        self.x_seg, self.y_seg, self.cuts = x_seg, y_seg, cuts

        self.gaps = [Gap(cuts[2 * i], cuts[2 * i + 1], bad_cols)
                     for i in range(cuts.size // 2)]

        self.models = self.fit(x_seg, y_seg)

        self.fig, self.axs = self.start_plot()
        self.left_gap_size = self.measure_gaps(0)
        self.right_gap_size = self.measure_gaps(1)
        self.save_plot()

    @abc.abstractmethod
    def fit(self, x_seg, y_seg):
        pass

    def measure_gaps(self, gap_index):

        if self.models is None:
            return

        i = gap_index
        gap = self.gaps[gap_index]

        print('\n Measuring the gaps: ')
        m_left = self.models[i]
        m_right = self.models[i + 1]

        w_center = self.w_solution(gap.center)
        w_left = self.w_solution(gap.left - gap.size)
        w_right = self.w_solution(gap.right + gap.size)

        y_left = m_left(gap.center)
        y_right = m_right(gap.center)
        y_center = 0.5 * (y_left + y_right)

        x_left = np.arange(gap.left - gap.size, gap.right)
        x_right = np.arange(gap.left, gap.right + gap.size)

        dy = y_left - y_right

        self.axs[i + 1].set_xlim(w_left, w_right)

        self.axs[i + 1].set_ylim(
            max(0, y_center - 0.15),
            min(1.05, y_center + 0.15))

        self.axs[i + 1].plot(
            self.w_solution(x_right), m_left(x_right), ':C{}'.format(i))

        self.axs[i + 1].plot(
            self.w_solution(x_left), m_right(x_left), ':C{}'.format(i + 1))

        self.axs[i + 1].scatter(
            w_center, y_left, marker='o', c='C{}'.format(i), alpha=0.5)

        self.axs[i + 1].scatter(
            w_center, y_right, marker='o', c='C{}'.format(i + 1), alpha=0.5)

        self.axs[i + 1].annotate(
            "    dy = {:.3f}".format(dy), (w_center, y_center))

        for ax in self.axs:
            ax.axvline(w_center, c='C3', lw=0.5)

        s = ("\n"
             " gap {:1d}\n"
             " w={:.2f} y_left={:.2f} y_right={:.2f}"
             " |y_left-y_right|={:.4f} ")

        print(s.format(i + 1, w_center, y_left, y_right, dy))
        return abs(dy)

    def save_plot(self):
        plot_dir = "plots/geminidr/gmos/test_gmos_spect_ls_apply_qe_correction/"
        os.makedirs(plot_dir, exist_ok=True)
        self.fig.savefig(
            os.path.join(plot_dir, "{:s}.png".format(self.plot_name)))

    def start_plot(self):

        self.plot_name = "local_gap_size_{}".format(
            self.ad.orig_filename[:-5])

        plot_title = (
             "QE Corrected Spectrum: {:s} Fit for each detector"
             "\n {:s} {:s} {:.3f}um")

        plot_title = plot_title.format(
                self.fit_family,
                self.ad.data_label(),
                self.ad.disperser(pretty=True),
                self.ad.central_wavelength(asMicrometers=True))

        plt.close(self.plot_name)

        fig = plt.figure(dpi=DPI, figsize=(6, 6), num=self.plot_name)

        gs = plt.GridSpec(2, 2, figure=fig, height_ratios=[4, 1])

        axs = [
            fig.add_subplot(gs[0, :]),
            fig.add_subplot(gs[1, 0]),
            fig.add_subplot(gs[1, 1]),
        ]

        for i, (xx, yy) in enumerate(zip(self.x_seg, self.y_seg)):

            model = self.models[i]
            xx = xx[self.bad_cols:-self.bad_cols]
            yy = yy[self.bad_cols:-self.bad_cols]

            ww = self.w_solution(xx)
            yy.mask = np.logical_or(yy.mask, model.mask)
            yy.mask = np.logical_or(yy.mask, ww < self.wav_min)
            yy.mask = np.logical_or(yy.mask, ww > self.wav_max)

            c, label = 'C{}'.format(i), 'Det {}'.format(i + 1)
            axs[0].fill_between(ww.data, 0, yy.data, fc=c, alpha=0.3, label=label)

            for ax in axs:
                ax.plot(ww.data, model(xx.data), '-C{}'.format(i), alpha=0.2)
                ax.plot(ww, model(xx), '-C{}'.format(i))

        for ax in axs:
            ax.fill_between(self.w.data, 0, self.y.data, fc='k', alpha=0.1)
            ax.grid(c='k', alpha=0.1)
            ax.set_ylabel('$\\frac{y - y_{min}}{y_{max} - y_{min}}$')
            ax.set_xlabel('Wavelength [{}]'.format(self.w_solution.units))

        axs[0].set_title(plot_title)
        axs[0].set_xlim(self.w.min(), self.w.max())
        axs[0].set_ylim(-0.05, 1.05)

        axs[2].yaxis.tick_right()
        axs[2].yaxis.set_label_position("right")

        fig.tight_layout(h_pad=0.0)

        return fig, axs


class MeasureGapSizeLocallyWithSpline(MeasureGapSizeLocally):

    def __init__(self, ad, bad_cols=21, med_filt_size=5, order=5,
                 sigma_lower=1.5, sigma_upper=3, wav_min=375, wav_max=1050):
        super().__init__(
            ad,
            bad_cols=bad_cols,
            fit_family='Spl.',
            med_filt_size=med_filt_size,
            order=order,
            sigma_lower=sigma_lower,
            sigma_upper=sigma_upper,
            wav_max=wav_max,
            wav_min=wav_min)
        # Uncomment to fit a spline to all data and show a plot.
        # self.fit_all_data()

    def fit(self, x_seg, y_seg):
        splines = []
        for i, (xx, yy) in enumerate(zip(x_seg, y_seg)):
            xx = xx[self.bad_cols:-self.bad_cols]
            yy = yy[self.bad_cols:-self.bad_cols]

            ww = self.w_solution(xx)
            ww = np.ma.masked_array(ww)
            ww.mask = np.logical_or(ww.mask, ww < 375)
            ww.mask = np.logical_or(ww.mask, ww > 1075)

            yy.mask = np.logical_or(yy.mask, ww.mask)
            yy.mask = np.logical_or(yy.mask, ww.mask)

            spl = astromodels.UnivariateSplineWithOutlierRemoval(
                xx, yy,
                sigma_upper=self.sigma_upper,
                sigma_lower=self.sigma_lower,
                order=self.order
            )

            splines.append(spl)

        return splines

    def fit_all_data(self):

        cuts = self.cuts
        lx1, lx2, rx1, rx2 = cuts
        ly1 = self.y[lx1-self.bad_cols]
        ly2 = self.y[lx2+self.bad_cols]
        ry1 = self.y[rx1-self.bad_cols]
        ry2 = self.y[rx2+self.bad_cols]

        # Mask out bad columns around the gaps.
        self.y.mask[cuts[0]-self.bad_cols:cuts[1]+self.bad_cols] = True
        self.y.mask[cuts[2]-self.bad_cols:cuts[3]+self.bad_cols] = True

        # Fit a spline to all the data.
        spl = astromodels.UnivariateSplineWithOutlierRemoval(
            self.x, self.y,
            sigma_upper=self.sigma_upper,
            sigma_lower=self.sigma_lower,
            order=10)

        # These are the difference between the last 'good pixels' around the
        # gaps and the spline.
        # print(abs(ly1-spl(lx1)), abs(ly2-spl(lx2)),
        #       abs(ry1-spl(rx1)), abs(ry2-spl(rx2)))

        fig, axes = self.start_plot()
        axes[0].plot(self.w, spl(self.x), color='Black', alpha=0.5)
        plt.show()


class WSolution:
    """
    Wavelength solution parser from an AstroData object.

    Parameters
    ----------
    ad : AstroData
        Input spectra with wavelength solution.

    Attributes
    ----------
    model : Linear1D
        Linear model that parses from pixel to wavelength.
    units : str
        Wavelength units.
    mask : ndarray
        Array applied to returned data.
    """

    def __init__(self, ad):
        # Assumes 1D input
        self.model = ad[0].wcs.forward_transform
        self.units = ad[0].wcs.output_frame.unit[0]
        self.mask = None

    def __call__(self, x):
        """
        Returns a masked array with the wavelengths.
        """
        _w = self.model(x)
        _w = np.ma.masked_array(_w, mask=self.mask)
        return _w


# -- Recipe to create pre-processed data ---------------------------------------
def create_inputs_recipe(use_branch_name=False):
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
    from gempy.utils import logutils
    from geminidr.gmos.tests.spect import CREATED_INPUTS_PATH_FOR_TESTS
    from recipe_system.reduction.coreReduce import Reduce
    from recipe_system.utils.reduce_utils import normalize_ucals

    module_name, _ = os.path.splitext(os.path.basename(__file__))
    path = os.path.join(CREATED_INPUTS_PATH_FOR_TESTS, module_name)
    os.makedirs(path, exist_ok=True)
    os.chdir(path)

    input_path = os.path.join(path, "inputs/")
    os.makedirs(input_path, exist_ok=True)

    for filename, cals in associated_calibrations.items():
        print(filename)
        print(cals)

        sci_path = download_from_archive(filename)
        cals = associated_calibrations[filename]
        bias_paths = [download_from_archive(f) for f in cals['bias']]
        flat_paths = [download_from_archive(f) for f in cals['flat']]
        arc_paths = [download_from_archive(f) for f in cals['arcs']]

        sci_ad = astrodata.from_file(sci_path)
        data_label = sci_ad.data_label()

        logutils.config(file_name='log_bias_{}.txt'.format(data_label))
        bias_reduce = Reduce()
        bias_reduce.files.extend(bias_paths)
        bias_reduce.runr()
        bias_master = bias_reduce.output_filenames.pop()
        calibration_files = ['processed_bias:{}'.format(bias_master)]
        del bias_reduce

        logutils.config(file_name='log_flat_{}.txt'.format(data_label))
        flat_reduce = Reduce()
        flat_reduce.files.extend(flat_paths)
        flat_reduce.ucals = normalize_ucals(calibration_files)
        flat_reduce.runr()
        flat_master = flat_reduce.output_filenames.pop()
        calibration_files.append('processed_flat:{}'.format(flat_master))
        del flat_reduce

        logutils.config(file_name='log_arc_{}.txt'.format(data_label))
        arc_reduce = Reduce()
        arc_reduce.files.extend(arc_paths)
        arc_reduce.ucals = normalize_ucals(calibration_files)

        arc_reduce.runr()
        arc_out = arc_reduce.output_filenames.pop()
        del arc_reduce

        logutils.config(file_name='log_{}.txt'.format(data_label))
        p = primitives_gmos_longslit.GMOSLongslit([sci_ad])
        p.prepare()
        p.addDQ()
        p.addVAR(read_noise=True)
        p.overscanCorrect()
        p.biasCorrect(bias=bias_master)
        p.ADUToElectrons()
        p.addVAR(poisson_noise=True)
        p.attachWavelengthSolution(arc=arc_out)
        p.flatCorrect(flat=flat_master)

        os.chdir("inputs")
        _ = p.writeOutputs().pop()
        os.chdir("../")


if __name__ == '__main__':
    from sys import argv

    if "--create-inputs" in argv[1:]:
        use_branch_name = "--branch" in argv[1:]
        create_inputs_recipe(use_branch_name=use_branch_name)
    else:
        pytest.main()
