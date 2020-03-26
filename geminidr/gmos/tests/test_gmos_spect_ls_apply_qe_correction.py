#!/usr/bin/env python

import abc
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pytest
import shutil
import urllib
import xml.etree.ElementTree as et

import astrodata
import gemini_instruments

from astropy import modeling
from astropy.utils.data import download_file
from astropy.stats import sigma_clip
from contextlib import contextmanager
from geminidr.gmos import primitives_gmos_longslit
from gempy.adlibrary import dataselect
from gempy.library import astromodels
from gempy.utils import logutils
from recipe_system.reduction.coreReduce import Reduce
from recipe_system.utils.reduce_utils import normalize_ucals
from scipy import ndimage

DPI = 90
URL = 'https://archive.gemini.edu/file/'

datasets = {
    "N20190302S0089.fits",  # GN-2019A-Q-203-7-001 B600 0.550um L-Ok R-Ok G-Ok
}

gap_local_kw = {
    "N20180228S0134.fits": {'order': 3, 'sigma_upper': 1.5}
}

# -- Tests --------------------------------------------------------------------
@pytest.mark.gmosls
@pytest.mark.dragons_remote_data
def test_applied_qe_is_locally_continuous_at_left_gap(gap_local):
    assert gap_local.is_continuous_left_gap()


@pytest.mark.gmosls
@pytest.mark.dragons_remote_data
def test_applied_qe_is_locally_continuous_at_right_gap(gap_local):
    assert gap_local.is_continuous_right_gap()


# @pytest.mark.gmosls
# @pytest.mark.dragons_remote_data
# def test_applied_qe_is_globally_continuous(gap_global):
#     print(gap_global.x_out)
#     assert gap_global.is_continuous()
#
#
# @pytest.mark.gmosls
# @pytest.mark.dragons_remote_data
# def test_applied_qe_is_stable():
#     pass


# -- Fixtures -----------------------------------------------------------------
@pytest.fixture(scope='module')
def cache_path(new_path_to_inputs):
    """
    Factory as a fixture used to cache data and return its local path.

    Parameters
    ----------
    new_path_to_inputs : pytest.fixture
        Full path to cached folder.

    Returns
    -------
    function : Function used that downloads data from the archive and stores it
        locally.
    """
    def _cache_path(filename):
        """
        Download data from Gemini Observatory Archive and cache it locally.

        Parameters
        ----------
        filename : str
            The filename, e.g. N20160524S0119.fits

        Returns
        -------
        str : full path of the cached file.
        """
        local_path = os.path.join(new_path_to_inputs, filename)

        if not os.path.exists(local_path):
            tmp_path = download_file(URL + filename, cache=False)
            shutil.move(tmp_path, local_path)

            # `download_file` ignores Access Control List - fixing it
            os.chmod(local_path, 0o664)

        return local_path

    return _cache_path


@pytest.fixture(scope='module')
def gap_local(processed_ad, output_path):

    basename = processed_ad.filename.replace('_linearized', '')
    kwargs = gap_local_kw[basename] if basename in gap_local_kw.keys() else {}

    # Save plots in output folder
    with output_path():
        gap = MeasureGapSizeLocallyWithPolynomial(processed_ad, **kwargs)

    return gap


@pytest.fixture(scope='module')
def gap_global(processed_ad, output_path):
    # Save plots in output folder
    with output_path():
        gap = MeasureGapSizeGloballyWithSpline(processed_ad)
    return gap


def get_associated_calibrations(data_label):
    """
    Queries Gemini Observatory Archive for associated calibrations to reduce the
    data that will be used for testing.

    Parameters
    ----------
    data_label : str
        Input file datalabel.
    """
    url = "https://archive.gemini.edu/calmgr/{}".format(data_label)

    tree = et.parse(urllib.request.urlopen(url))
    root = tree.getroot()
    prefix = root.tag[:root.tag.rfind('}') + 1]

    def iter_nodes(node):
        cal_type = node.find(prefix + 'caltype').text
        filename = node.find(prefix + 'filename').text
        return filename, cal_type 

    cals = pd.DataFrame(
        [iter_nodes(node) for node in tree.iter(prefix + 'calibration')],
        columns=['filename', 'caltype'])

    cals = cals[~cals.caltype.str.contains('processed_')]
    cals = cals[~cals.caltype.str.contains('specphot')]
    cals = cals.drop(cals[cals.caltype.str.contains('bias')][5:].index)

    return cals.filename.values.tolist()


@pytest.fixture(scope='module')
def output_path(request, path_to_outputs):
    """
    Factory that returns the output path as a context manager object, allowing
    easy access to the path to where the processed data should be stored.

    Parameters
    ----------
    request : pytest.fixture
        Fixture that contains information this fixture's parent.
    path_to_outputs : pytest.fixture
        Fixture containing the root path to the output files.

    Returns
    -------
    contextmanager : A context manager function that allows easily changing
    folders.
    """
    module_path = request.module.__name__.split('.') + ["outputs"]
    module_path = [item for item in module_path if item not in "tests"]
    path = os.path.join(path_to_outputs, *module_path)

    os.makedirs(path, exist_ok=True)

    @contextmanager
    def _output_path():
        oldpwd = os.getcwd()
        os.chdir(path)
        try:
            yield
        finally:
            os.chdir(oldpwd)

    return _output_path


@pytest.fixture(scope='module', params=datasets)
def processed_ad(
        request, cache_path, reduce_arc, reduce_bias, reduce_data, reduce_flat):

    filename = cache_path(request.param)

    ad = astrodata.open(filename)
    cals = [cache_path(c) for c in get_associated_calibrations(ad.data_label())]

    master_bias = reduce_bias(
        ad.data_label(), dataselect.select_data(cals, tags=['BIAS']))

    master_flat = reduce_flat(
        ad.data_label(), dataselect.select_data(cals, tags=['FLAT']), master_bias)

    master_arc = reduce_arc(
        ad.data_label(), dataselect.select_data(cals, tags=['ARC']))

    processed_data = reduce_data(ad, master_arc, master_bias, master_flat)

    return processed_data.pop()


@pytest.fixture(scope='module')
def reduce_arc(output_path):
    """
    Factory for function for ARCS data reduction.

    Parameters
    ----------
    output_path : pytest.fixture
        Context manager used to write reduced data to a temporary folder.

    Returns
    -------
    function : A function that will read the arcs files, process them and
    return the name of the master arc.
    """
    def _reduce_arc(dlabel, arc_fnames):
        with output_path():
            # Use config to prevent duplicated outputs when running Reduce via API
            logutils.config(file_name='log_arc_{}.txt'.format(dlabel))

            reduce = Reduce()
            reduce.files.extend(arc_fnames)
            reduce.runr()

            master_arc = reduce.output_filenames.pop()
        return master_arc
    return _reduce_arc


@pytest.fixture(scope='module')
def reduce_bias(output_path):
    """
    Factory for function for BIAS data reduction.

    Parameters
    ----------
    output_path : pytest.fixture
        Context manager used to write reduced data to a temporary folder.

    Returns
    -------
    function : A function that will read the bias files, process them and
    return the name of the master bias.
    """
    def _reduce_bias(datalabel, bias_fnames):
        with output_path():
            logutils.config(file_name='log_bias_{}.txt'.format(datalabel))

            reduce = Reduce()
            reduce.files.extend(bias_fnames)
            reduce.runr()

            master_bias = reduce.output_filenames.pop()

        return master_bias
    return _reduce_bias


@pytest.fixture(scope='module')
def reduce_data(output_path):
    """
    Factory for function for FLAT data reduction.

    Parameters
    ----------
    output_path : pytest.fixture
        Context manager used to write reduced data to a temporary folder.

    Returns
    -------
    function : A function that will read the standard star file, process them
    using a custom recipe and return an AstroData object.
    """
    def _reduce_data(ad, master_arc, master_bias, master_flat):
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
            p.applyQECorrection(arc=master_arc)
            p.distortionCorrect(arc=master_arc)
            p.findSourceApertures(max_apertures=1)
            p.skyCorrectFromSlit()
            p.traceApertures()
            p.extract1DSpectra()
            p.linearizeSpectra()
            processed_ad = p.writeOutputs()

        return processed_ad
    return _reduce_data


@pytest.fixture(scope='module')
def reduce_flat(output_path):
    """
    Factory for function for FLAT data reduction.

    Parameters
    ----------
    output_path : pytest.fixture
        Context manager used to write reduced data to a temporary folder.

    Returns
    -------
    function : A function that will read the flat files, process them and
    return the name of the master flat.
    """
    def _reduce_flat(datalabel, flat_fnames, master_bias):
        with output_path():
            logutils.config(file_name='log_flat_{}.txt'.format(datalabel))

            calibration_files = ['processed_bias:{}'.format(master_bias)]

            reduce = Reduce()
            reduce.files.extend(flat_fnames)
            reduce.mode = 'ql'
            reduce.ucals = normalize_ucals(reduce.files, calibration_files)
            reduce.runr()

            master_flat = reduce.output_filenames.pop()
            master_flat_ad = astrodata.open(master_flat)

        return master_flat_ad
    return _reduce_flat


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
    l : int
        Left index that defines the beginning of the gap.
    r : int
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
    def __init__(self, l, r, i):
        self.center = 0.5 * (l + r)
        self.left = l - i
        self.right = r + i
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
        w = np.ma.masked_outside(w, 375, 1075)

        y = np.ma.masked_array(y, mask=m)
        y = smooth_data(y, self.median_filter_size)
        y, v = normalize_data(y, v)

        y.mask = np.logical_or(y.mask, y < 0.01)
        x = np.ma.masked_array(x, mask=y.mask)

        split_mask = ad[0].mask >= 16
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
        return abs(dy) < 0.5

    def save_plot(self):
        self.fig.savefig("{:s}.png".format(self.plot_name))

    def start_plot(self):

        self.plot_name = "local_gap_size_{}_{}".format(
            self.fit_family.lower().replace('.', ''), self.ad.data_label())

        plot_title = "QE Corrected Spectrum: {:s} Fit for each detector\n {:s}".format(
                self.fit_family, self.ad.data_label())

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
            ww = self.w_solution(xx)
            yy.mask = np.logical_or(yy.mask, ww < self.wav_min)
            yy.mask = np.logical_or(yy.mask, ww > self.wav_max)
            model = self.models[i]

            c, label = 'C{}'.format(i), 'Det {}'.format(i + 1)
            axs[0].fill_between(ww, 0, yy, fc=c, alpha=0.3, label=label)

            for ax in axs:
                ax.plot(ww.data, model(xx.data), '-C{}'.format(i), alpha=0.2)
                ax.plot(ww, model(xx), '-C{}'.format(i))

        for ax in axs:
            ax.fill_between(self.w, 0, self.y, fc='k', alpha=0.1)
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


class MeasureGapSizeLocallyWithPolynomial(MeasureGapSizeLocally):

    def __init__(self, ad, bad_cols=21, med_filt_size=5, order=5,
                 sigma_lower=1.5, sigma_upper=3, wav_min=375, wav_max=1050):

        super(MeasureGapSizeLocallyWithPolynomial, self).__init__(
            ad,
            bad_cols=bad_cols,
            fit_family='Pol.',
            med_filt_size=med_filt_size,
            order=order,
            sigma_lower=sigma_lower,
            sigma_upper=sigma_upper,
            wav_max=wav_max,
            wav_min=wav_min)

    def fit(self, x_seg, y_seg):

        fit = modeling.fitting.LinearLSQFitter()
        self.fit_family = "Pol."

        or_fit = modeling.fitting.FittingWithOutlierRemoval(
            fit, sigma_clip, niter=3,
            sigma_lower=self.sigma_lower, sigma_upper=self.sigma_upper)

        poly_init = modeling.models.Polynomial1D(degree=self.order)

        polynomials = []
        for i, (xx, yy) in enumerate(zip(x_seg, y_seg)):

            xx = xx[self.bad_cols:-self.bad_cols]
            yy = yy[self.bad_cols:-self.bad_cols]

            ww = self.w_solution(xx)
            ww = np.ma.masked_array(ww)
            ww.mask = np.logical_or(ww.mask, ww < 375)
            ww.mask = np.logical_or(ww.mask, ww > 1075)

            yy.mask = np.logical_or(yy.mask, ww.mask)
            yy.mask = np.logical_or(yy.mask, ww.mask)

            pp, rej_mask = or_fit(poly_init, xx, yy)
            ww.mask = np.logical_or(ww.mask, rej_mask)

            polynomials.append(pp)

        return polynomials


class MeasureGapSizeGlobally(abc.ABC):
    """
    Base class used to measure gap size globally.

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
                 order=5, sigma_lower=1.5, sigma_upper=3):

        self.ad = ad
        self.bad_cols = bad_cols
        self.fit_family = fit_family
        self.median_filter_size = med_filt_size
        self.obs_id = ad.observation_id()
        self.order = order
        self.sigma_lower = sigma_lower
        self.sigma_upper = sigma_upper

        self.w_solution = WSolution(ad)

        m = ad[0].mask > 0
        y = ad[0].data
        v = ad[0].variance  # Never used for anything. But...
        x = np.arange(y.size)

        w = self.w_solution(x)
        w = np.ma.masked_outside(w, 375, 1075)

        y = np.ma.masked_array(y, mask=m)
        y = smooth_data(y, self.median_filter_size)
        y, v = normalize_data(y, v)

        split_mask = ad[0].mask >= 16
        x_seg, y_seg, cuts = split_arrays_using_mask(x, y, split_mask, self.bad_cols)

        self.gaps = [Gap(cuts[2 * i], cuts[2 * i + 1], bad_cols)
                     for i in range(cuts.size // 2)]

        x_in, y_in, w_in = self.get_inner_data(x_seg, y_seg)
        x_out, y_out, w_out = self.get_outer_data(x_seg, y_seg)
        print(x_out)

        self.model_in, self.model_out = self.fit(x_in, y_in, x_out, y_out)

        w_in.mask = np.logical_or(w_in.mask, self.model_in.mask)
        w_out.mask = np.logical_or(w_out.mask, self.model_out.mask)

        # Saving variables to self to plot
        self.x, self.y, self.w = x, y, w
        self.x_seg, self.y_seg, self.cuts = x_seg, y_seg, cuts
        self.x_in, self.y_in, self.w_in = x_in, y_in, w_in
        self.x_out, self.y_out, self.w_out = x_out, y_out, w_out
        self.plot()

    @abc.abstractmethod
    def fit(self, x_in, y_in, x_out, y_out):
        pass

    def get_inner_data(self, x_seg, y_seg):
        x_in = x_seg[1][self.bad_cols:-self.bad_cols]
        y_in = y_seg[1][self.bad_cols:-self.bad_cols]
        w_in = self.w_solution(x_in)
        return x_in, y_in, w_in

    def get_outer_data(self, x_seg, y_seg):

        x_out = np.hstack((
            x_seg[0][self.bad_cols:-self.bad_cols],
            x_seg[2][self.bad_cols:-self.bad_cols]))

        y_out = np.hstack((
            y_seg[0][self.bad_cols:-self.bad_cols],
            y_seg[2][self.bad_cols:-self.bad_cols]))

        w_out = self.w_solution(x_out)
        w_out = np.ma.masked_array(w_out)
        w_out.mask = np.logical_or(w_out.mask, w_out < 375)
        w_out.mask = np.logical_or(w_out.mask, w_out > 1075)

        y_out.mask = np.logical_or(y_out.mask, w_out.mask)
        y_out.mask = np.logical_or(y_out.mask, y_out < 0.01)

        return x_out, y_out, w_out

    def is_continuous(self):
        diff = self.y_in - self.model_out(self.x_in)
        return np.all(np.logical_and(-0.05 < diff, diff < 0.05))

    def plot(self):

        plot_name = 'global_gap_{}_{}'.format(
            self.fit_family.lower().replace('.', ''), self.obs_id)

        plot_title = "QE Corrected Spectra: {:s} Bridge fit\n {:s}".format(
            self.fit_family, self.obs_id)

        plt.close(plot_name)

        fig, axs = plt.subplots(
            constrained_layout=True, dpi=DPI, figsize=(6, 6),
            gridspec_kw={'height_ratios': [4, 1]}, nrows=2,
            num=plot_name, sharex='all')

        axs[0].fill_between(self.w, 0, self.y, fc='k', alpha=0.1)

        for i, (xx, yy) in enumerate(zip(self.x_seg, self.y_seg)):
            ww = self.w_solution(xx)
            yy.mask = np.logical_or(yy.mask, ww < 375)
            yy.mask = np.logical_or(yy.mask, ww > 1075)

            c, label = 'C{}'.format(i), 'Det {}'.format(i + 1)

            axs[0].fill_between(ww, 0, yy, fc=c, alpha=0.3, label=label)

        axs[0].plot(self.w_out.data, self.model_out(self.x_out.data), 'C3', alpha=0.2)
        axs[0].plot(self.w_out, self.model_out(self.x_out), 'C3')
        axs[0].plot(self.w_in.data, self.model_in(self.x_in.data), 'C1', alpha=0.2)
        axs[0].plot(self.w_in, self.model_in(self.x_in), 'C1')

        axs[0].grid(c='k', alpha=0.1)
        axs[0].set_xlim(self.w.min(), self.w.max())
        axs[0].set_ylim(-0.05, 1.05)
        axs[0].set_ylabel('$\\frac{y - y_{min}}{y_{max} - y_{min}}$')
        axs[0].set_title(plot_title)

        axs[1].plot(self.w_in, self.y_in - self.model_out(self.x_in), 'C3')
        axs[1].axhspan(-0.05, 0.05, fc='k', alpha=0.1)

        axs[1].grid(c='k', alpha=0.1)
        axs[1].set_xlabel('Wavelength [{}]'.format(self.w_solution.units))
        axs[1].set_ylim(-0.075, 0.075)

        fig.tight_layout(h_pad=0.0)
        fig.savefig("{:s}.png".format(plot_name))


class MeasureGapSizeGloballyWithSpline(MeasureGapSizeGlobally):

    def __init__(self, ad, bad_cols=21, med_filt_size=5, order=5,
                 sigma_lower=1.5, sigma_upper=3):
        super(MeasureGapSizeGloballyWithSpline, self).__init__(
            ad,
            bad_cols=bad_cols,
            fit_family='Spl.',
            med_filt_size=med_filt_size,
            order=order,
            sigma_lower=sigma_lower,
            sigma_upper=sigma_upper)

    def fit(self, x_in, y_in, x_out, y_out):

        s_in = astromodels.UnivariateSplineWithOutlierRemoval(
            x_in, y_in, hsigma=self.sigma_upper, lsigma=self.sigma_lower,
            order=self.order)

        s_out = astromodels.UnivariateSplineWithOutlierRemoval(
            x_out, y_out, hsigma=self.sigma_upper, lsigma=self.sigma_lower,
            order=self.order)

        return s_in, s_out


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
