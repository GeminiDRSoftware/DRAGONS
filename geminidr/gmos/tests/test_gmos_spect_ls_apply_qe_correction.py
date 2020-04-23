#!/usr/bin/env python

import abc
import matplotlib.pyplot as plt
import numpy as np
import os
import pytest

import astrodata
import gemini_instruments

from astropy import modeling
from geminidr.gmos import primitives_gmos_longslit
from gempy.adlibrary import dataselect
from gempy.library import astromodels
from gempy.utils import logutils
from scipy import ndimage

from astrodata import testing

DPI = 90

datasets = {

    # --- Good datasets ---
    "N20180109S0287.fits",  # GN-2017B-FT-20-13-001 B600 0.505um
    "N20190302S0089.fits",  # GN-2019A-Q-203-7-001 B600 0.550um
    "N20190313S0114.fits",  # GN-2019A-Q-325-13-001 B600 0.482um
    "N20190427S0123.fits",  # GN-2019A-FT-206-7-001 R400 0.525um
    "N20190427S0126.fits",  # GN-2019A-FT-206-7-004 R400 0.625um
    "N20190910S0028.fits",  # GN-2019B-Q-313-5-001 B600 0.550um
    "S20180919S0139.fits",  # GS-2018B-Q-209-13-003 B600 0.45um
    "S20191005S0051.fits",  # GS-2019B-Q-132-35-001 R400 0.73um

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

}

gap_local_kw = {
    "N20180109S0287.fits": {'bad_cols': 5},
    "N20180228S0134.fits": {},
    "N20180508S0021.fits": {'bad_cols': 5, 'order': 5, 'sigma_lower': 1.5},
    "N20180509S0010.fits": {},
    "N20190201S0163.fits": {},
    "N20190302S0089.fits": {'bad_cols': 5, 'wav_min': 450},
    "N20190313S0114.fits": {},
    "N20190427S0123.fits": {'bad_cols': 5, 'order': 3},
    "N20190427S0126.fits": {'bad_cols': 5, 'order': 3},
    "N20190910S0028.fits": {},
    "S20180919S0139.fits": {'bad_cols': 10, 'order': 4},
    "S20191005S0051.fits": {'order': 8},
}


# -- Tests --------------------------------------------------------------------
@pytest.mark.dragons_remote_data
@pytest.mark.gmosls
@pytest.mark.preprocessed_data
def test_applied_qe_is_locally_continuous(preprocessed_data, output_path):

    input_ad, master_arc = preprocessed_data

    with output_path():

        p = primitives_gmos_longslit.GMOSLongslit([input_ad])
        p.applyQECorrection(arc=master_arc)

        # Need these extra steps to extract and analyse the data
        p.distortionCorrect(arc=master_arc)
        p.findSourceApertures(max_apertures=1)
        p.skyCorrectFromSlit()
        p.traceApertures()
        p.extract1DSpectra()
        p.linearizeSpectra()
        processed_ad = p.writeOutputs().pop()

    basename = processed_ad.filename.replace('_linearized', '')
    kwargs = gap_local_kw[basename] if basename in gap_local_kw.keys() else {}
    gap = MeasureGapSizeLocallyWithSpline(processed_ad, **kwargs)

    assert abs(gap.measure_gaps(0) < 0.05)
    assert abs(gap.measure_gaps(1) < 0.05)


@pytest.mark.dragons_remote_data
@pytest.mark.gmosls
@pytest.mark.preprocessed_data
def test_regression_on_apply_qe_correction(
        preprocessed_data, output_path, reference_ad):

    input_ad, master_arc = preprocessed_data

    with output_path():
        p = primitives_gmos_longslit.GMOSLongslit([input_ad])
        p.applyQECorrection(arc=master_arc)
        qe_corrected_ad = p.writeOutputs().pop()

    assert 'QECORR' in qe_corrected_ad.phu.keys()

    ref_ad = reference_ad(qe_corrected_ad.filename)

    for qe_corrected_ext, ref_ext in zip(qe_corrected_ad, ref_ad):
        np.testing.assert_allclose(
            np.ma.masked_array(qe_corrected_ext.data, mask=qe_corrected_ext.mask),
            np.ma.masked_array(ref_ext.data, mask=ref_ext.mask),
            atol=0.05)


# -- Fixtures -----------------------------------------------------------------
@pytest.fixture(scope='function', params=datasets)
def preprocessed_data(request, cache_path, get_master_arc, new_path_to_inputs,
                      reduce_arc, reduce_bias, reduce_data,  reduce_flat):
    """
    Returns the processed spectrum right after running `applyQECorrection`.

    Parameters
    ----------
    request : pytest.fixture
        Fixture that contains information this fixture's parent.
    cache_path : pytest.fixture
        Path to where the data will be temporarily cached.
    get_master_arc : pytest.fixture
        Fixture that reads the master flat either from the permanent input folder
        or from the temporary cache folder.
        Reads the input data or cache/process it in a temporary folder.
    new_path_to_inputs : pytest.fixture
        Path to the permanent local input files.
    reduce_arc : pytest.fixture
        Recipe to reduce the arc file.
    reduce_bias : pytest.fixture
        Recipe to reduce the bias files.
    reduce_data : pytest.fixture
        Recipe to reduce the data up to the step before `applyQECorrect`.
    reduce_flat : pytest.fixture
        Recipe to reduce the flat file.

    Returns
    -------
    AstroData
        QE Corrected astrodata.
    """
    basename = request.param
    should_preprocess = request.config.getoption("--force-preprocess-data")

    input_fname = basename.replace('.fits', '_flatCorrected.fits')
    input_path = os.path.join(new_path_to_inputs, input_fname)
    cals = testing.get_associated_calibrations(basename)

    if os.path.exists(input_path):
        input_data = astrodata.open(input_path)
        master_arc = get_master_arc(input_data, should_preprocess)

    elif should_preprocess:
        filename = cache_path(basename)
        ad = astrodata.open(filename)
        cals = [cache_path(c) for c in cals.filename.values]

        master_bias = reduce_bias(
            ad.data_label(), dataselect.select_data(cals, tags=['BIAS']))

        master_flat = reduce_flat(
            ad.data_label(), dataselect.select_data(cals, tags=['FLAT']), master_bias)

        master_arc = reduce_arc(
            ad.data_label(), dataselect.select_data(cals, tags=['ARC']))

        input_data = reduce_data(ad, master_bias, master_flat)

    else:
        raise IOError(
            'Could not find input file:\n' +
            '  {:s}\n'.format(input_path) +
            '  Run pytest with "--force-preprocess-data" to get it')

    return input_data, master_arc


@pytest.fixture(scope='module')
def reduce_data(output_path):
    """
    Factory for function for data reduction up to one step before
    `applyQECorrection`.

    Parameters
    ----------
    output_path : pytest.fixture
        Context manager used to write reduced data to a temporary folder.

    Returns
    -------
    function : A function that will read the standard star file, process them
    using a custom recipe and return an AstroData object.
    """
    def _reduce_data(ad, master_bias, master_flat):
        with output_path():
            # Use config to prevent outputs when running Reduce via API
            logutils.config(file_name='log_{}.txt'.format(ad.data_label()))

            p = primitives_gmos_longslit.GMOSLongslit([ad])
            p.prepare()
            p.addDQ(static_bpm=None)
            p.addVAR(read_noise=True)
            p.overscanCorrect()
            p.biasCorrect(bias=master_bias)
            p.ADUToElectrons()
            p.addVAR(poisson_noise=True)
            p.flatCorrect(flat=master_flat)

            processed_ad = p.writeOutputs().pop()

        return processed_ad
    return _reduce_data


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
    m = ~ndimage.morphology.binary_opening(mask, iterations=10)

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
        x = np.ma.masked_array(x, mask=y.mask)

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
        self.is_continuous_left_gap()
        self.is_continuous_right_gap()
        self.save_plot()

    @abc.abstractmethod
    def fit(self, x_seg, y_seg):
        pass

    def is_continuous_left_gap(self):
        gap_size = self.measure_gaps(0)
        return abs(gap_size < 0.05)

    def is_continuous_right_gap(self):
        gap_size = self.measure_gaps(1)
        return abs(gap_size < 0.05)

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
        # plt.show()
        self.fig.savefig("{:s}.png".format(self.plot_name))

    def start_plot(self):

        self.plot_name = "local_gap_size_{}_{}".format(
            self.fit_family.lower().replace('.', ''), self.ad.data_label())

        plot_title = (
             "QE Corrected Spectrum: {:s} Fit for each detector"
             "\n {:s} {:s} {:.3f}um")

        plot_title = plot_title.format(
                self.fit_family,
                self.ad.data_label(),
                self.ad.disperser(pretty=True),
                self.ad.central_wavelength(asMicrometers=True))

        plt.close(self.plot_name)

        fig = plt.figure(
            constrained_layout=True, dpi=DPI, figsize=(6, 6), num=self.plot_name)

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
            xx.mask = np.logical_or(xx.mask, model.mask)
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
                xx, yy, hsigma=self.sigma_upper, lsigma=self.sigma_lower,
                order=self.order)

            splines.append(spl)

        return splines


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
        self.model = modeling.models.Linear1D(
            slope=ad[0].hdr['CDELT1'],
            intercept=ad[0].hdr['CRVAL1'] - 1 * ad[0].hdr['CDELT1'])
        self.units = ad[0].hdr['CUNIT1']
        self.mask = None

    def __call__(self, x):
        """
        Returns a masked array with the wavelengths.
        """
        _w = self.model(x)
        _w = np.ma.masked_array(_w, mask=self.mask)
        return _w


if __name__ == '__main__':
    pytest.main()
