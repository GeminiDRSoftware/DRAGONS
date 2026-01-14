# Copyright(c) 2019-2024 Association of Universities for Research in Astronomy, Inc.

"""
tracing.py

This module contains classes and functions used to trace features in a 2D image.

Classes in this module:
Aperture:            defines an aperture on an image
Trace:               defines a trace on an image

Functions in this module:
trace_lines:         trace lines from a set of supplied starting positions
trace_aperture:      trace a single aperture after finding a starting point
"""
from collections import deque

import numpy as np
from astropy.modeling import fitting, models
from astropy.stats import sigma_clip, sigma_clipped_stats

from astrodata import NDAstroData
from geminidr.gemini.lookups import DQ_definitions as DQ
from gempy.library.fitting import fit_1D
from gempy.library.nddops import NDStacker, combine1d
from gempy.utils import logutils

from . import astrotools as at
from .peak_finding import cwt_ricker, pinpoint_peaks
from ..utils.decorators import unpack_nddata

log = logutils.get_logger(__name__)

###############################################################################
class Aperture:
    """
    A class describing an aperture. It has the following attributes:

    model: Model
        describes the pixel location of the aperture center as a function of
        pixel in the dispersion direction
    aper_lower: float
        location of the lower edge of the aperture relative to the center
        (i.e., lower edge is at center+aper_lower)
    aper_upper: float
        location of the upper edge of the aperture relative to the center
    last_extraction: tuple
        values of (aper_lower, aper_upper) used for the most recent extraction
    width: float
        property defining the total aperture width
    """
    def __init__(self, model, width=None, aper_lower=None, aper_upper=None):
        self.model = model
        if width is None:
            self.aper_lower = aper_lower
            self.aper_upper = aper_upper
        else:
            self.width = width
        self.last_extraction = None

    @property
    def width(self):
        """Aperture width in pixels. Since the model is pixel-based, it makes
        sense for this to be stored in pixels rather than arcseconds."""
        return abs(self.aper_upper - self.aper_lower)

    @width.setter
    def width(self, value):
        if value > 0:
            self.aper_upper = 0.5 * value
            self.aper_lower = -self.aper_upper
        else:
            raise ValueError("Width must be positive {}".format(value))

    @property
    def center(self):
        """Return value of the model in the middle of the domain"""
        return self.model(0.5 * np.sum(self.model.domain))

    def limits(self):
        """Return maximum and minimum values of the model across the domain"""
        pixels = np.arange(*self.model.domain)
        values = self.model(pixels)
        return np.min(values), np.max(values)

    def check_domain(self, npix):
        """Simple function to warn user if aperture model appears inconsistent
        with the array containing the data. Since resampleToCommonFrame() now
        modifies the domain, this will raise unnecessary warnings if left as-is"""
        pass
        #try:
        #    if self.model.domain != (0, npix - 1):
        #        log.warning("Model's domain is inconsistent with image size. "
        #                    "Results may be incorrect.")
        #except AttributeError:  # no "domain" attribute
        #    pass

    def aperture_mask(self, ext=None, width=None, aper_lower=None,
                      aper_upper=None, grow=None):
        """
        This method creates a boolean array indicating pixels that will be
        used to extract a spectrum from the parent Aperture object.

        Parameters
        ----------
        ext: single-slice AD object
            image containing aperture (needs .shape attribute and
            .dispersion_axis() descriptor)
        width: float/None
            total extraction width for symmetrical aperture (pixels)
        aper_lower: float/None
            lower extraction limit (pixels)
        aper_upper: float/None
            upper extraction limit (pixels)
        grow: float/None
            additional buffer zone around width

        Returns
        -------
        ndarray: boolean array, set to True if a pixel will be used in the
                 extraction
        """
        if width is not None:
            aper_upper = 0.5 * width
            aper_lower = -aper_upper
        if aper_lower is None:
            aper_lower = self.aper_lower
        if aper_upper is None:
            aper_upper = self.aper_upper
        if grow is not None:
            aper_lower -= grow
            aper_upper += grow

        dispaxis = 2 - ext.dispersion_axis()  # python sense
        npix = ext.shape[dispaxis]
        slitlength = ext.shape[1 - dispaxis]
        self.check_domain(npix)
        center_pixels = self.model(np.arange(npix))
        x1, x2 = center_pixels + aper_lower, center_pixels + aper_upper
        ix1 = np.where(x1 < -0.5, 0, (x1 + 0.5).astype(int))
        ix2 = np.where(x2 >= slitlength - 0.5, None, (x2 + 1.5).astype(int))
        apmask = np.zeros_like(ext.data if dispaxis == 0 else ext.data.T,
                               dtype=bool)
        for i, limits in enumerate(zip(ix1, ix2)):
            apmask[i, slice(*limits)] = True
        return apmask if dispaxis == 0 else apmask.T

    def aperture_extraction(self, data, mask, var, aper_lower, aper_upper):
        """Uniform extraction across an aperture of width pixels"""
        all_x1 = self._center_pixels + aper_lower
        all_x2 = self._center_pixels + aper_upper

        ext = NDAstroData(data, mask=mask, variance=var)
        results = [combine1d(ext[:, i], x1, x2)
                   for i, (x1, x2) in enumerate(zip(all_x1, all_x2))]
        self.data[:] = [result.data for result in results]
        if mask is not None:
            self.mask[:] = [result.mask for result in results]
        if var is not None:
            self.var[:] = [result.variance for result in results]

    def optimal_extraction(self, data, mask, var, aper_lower, aper_upper,
                           cr_rej=5, max_iters=None, degree=3):
        """Optimal extraction following Horne (1986, PASP 98, 609)"""
        BAD_BITS = DQ.bad_pixel | DQ.cosmic_ray | DQ.no_data | DQ.unilluminated

        slitlength, npix = data.shape
        pixels = np.arange(npix)

        all_x1 = self._center_pixels + aper_lower
        all_x2 = self._center_pixels + aper_upper
        ix1 = max(int(min(all_x1) + 0.5), 0)
        ix2 = min(int(max(all_x2) + 1.5), slitlength)

        fit_it = fitting.FittingWithOutlierRemoval(
            fitting.LinearLSQFitter(), outlier_func=sigma_clip, sigma_upper=3,
            sigma_lower=None)

        # If we don't have a VAR plane, assume uniform variance based
        # on the pixel-to-pixel variations in the data
        if var is None:
            var_model = models.Const1D(sigma_clipped_stats(data, mask=mask)[2] ** 2)
            var = np.full_like(data[ix1:ix2], var_model.amplitude)
            var_mask = np.zeros_like(var, dtype=bool)
        else:
            # straightening and resampling sky lines can create unmasked
            # VAR=0 pixels that shouldn't be used in making the variance model
            mvar_init = models.Polynomial1D(degree=1)
            var_model, var_mask = fit_it(
                mvar_init, np.ma.masked_where(np.logical_or(mask, var == 0).ravel(),
                                              abs(data).ravel()), var.ravel())
            var_mask = var_mask.reshape(var.shape)[ix1:ix2]
            var = np.where(var_mask, var[ix1:ix2],
                           var_model(data[ix1:ix2]).astype(np.float32))
        var[var < 0] = 0

        if mask is None:
            mask = np.zeros((ix2 - ix1, npix), dtype=DQ.datatype)
        else:
            mask = mask[ix1:ix2]

        data = data[ix1:ix2]
        # Step 4; first calculation of spectrum. We don't do any masking
        # here since we need all the flux
        spectrum = data.sum(axis=0)
        inv_var = at.divide0(1., var)
        unmask = np.ones_like(data, dtype=bool)

        iter = 0
        while True:
            # Step 5: construct spatial profile for each wavelength pixel
            profile = np.divide(data, spectrum,
                                out=np.zeros_like(data, dtype=np.float32),
                                where=spectrum > 0)
            profile_models = []
            for row, ivar_row in zip(profile, inv_var):
                m_init = models.Chebyshev1D(degree=degree, domain=[0, npix - 1])
                m_final, _ = fit_it(m_init, pixels, row,
                                    weights=np.sqrt(ivar_row) * np.where(spectrum > 0, spectrum, 0))
                profile_models.append(m_final(pixels).astype(np.float32))
            profile_model_spectrum = np.array([np.where(pm < 0, 0, pm) for pm in profile_models])
            sums = profile_model_spectrum.sum(axis=0)
            model_profile = at.divide0(profile_model_spectrum, sums)

            # Step 6: revise variance estimates
            var = np.where(
                var_mask | mask & BAD_BITS, var,
                var_model(abs(model_profile * spectrum)).astype(np.float32)
            )
            inv_var = at.divide0(1.0, var)

            # Step 7: identify cosmic ray hits: we're (probably) OK
            # to flag more than 1 per wavelength
            sigma_deviations = at.divide0(data - model_profile * spectrum, np.sqrt(var)) * unmask
            mask[sigma_deviations > cr_rej] |= DQ.cosmic_ray
            # most_deviant = np.argmax(sigma_deviations, axis=0)
            # for i, most in enumerate(most_deviant):
            #    if sigma_deviations[most, i] > cr_rej:
            #        mask[most, i] |= DQ.cosmic_ray

            last_unmask = unmask
            unmask = (mask & BAD_BITS) == 0
            spec_numerator = np.sum(unmask * model_profile * data * inv_var, axis=0)
            spec_denominator = np.sum(unmask * model_profile ** 2 * inv_var, axis=0)
            self.data = at.divide0(spec_numerator, spec_denominator)
            self.var = at.divide0(np.sum(unmask * model_profile, axis=0), spec_denominator)
            self.mask = np.bitwise_and.reduce(mask, axis=0)
            spectrum = self.data

            iter += 1
            # An unchanged mask means XORing always produces False
            if ((np.sum(unmask ^ last_unmask) == 0) or
                    (max_iters is not None and iter > max_iters)):
                break

    def extract(self, ext, width=None, aper_lower=None, aper_upper=None,
                method='aperture', dispaxis=None, viewer=None):
        """
        Extract a 1D spectrum by following the model trace and extracting in
        an aperture of a given number of pixels.

        Parameters
        ----------
        ext: single-slice AD object/ndarray
            spectral image from which spectrum is to be extracted
        width: float/None
            full width of extraction aperture (in pixels)
        aper_lower: float/None
            lower extraction limit (pixels)
        aper_upper: float/None
            upper extraction limit (pixels)
        method: str (standard|optimal)
            type of extraction
        dispaxis: int/None
            dispersion axis (python sense)
        viewer: Viewer/None
            Viewer object on which to display extraction apertures

        Returns
        -------
        NDAstroData: 1D spectrum
        """
        if width is not None:
            aper_upper = 0.5 * width
            aper_lower = -aper_upper
        if aper_lower is None:
            aper_lower = self.aper_lower
        if aper_upper is None:
            aper_upper = self.aper_upper

        if dispaxis is None:
            dispaxis = 2 - ext.dispersion_axis()  # python sense
        slitlength = ext.shape[1 - dispaxis]
        npix = ext.shape[dispaxis]
        direction = "row" if dispaxis == 0 else "column"

        self.check_domain(npix)

        if aper_lower > aper_upper:
            log.warning("Aperture lower limit is greater than upper limit.")
            aper_lower, aper_upper = aper_upper, aper_lower
        if aper_lower > 0:
            log.warning("Aperture lower limit is greater than zero.")
        if aper_upper < 0:
            log.warning("Aperture upper limit is less than zero.")

        # make data look like it's dispersed horizontally
        # (this is best for optimal extraction, but not standard)
        try:
            mask = ext.mask
        except AttributeError:  # ext is just an ndarray
            data = ext if dispaxis == 1 else ext.T
            mask = None
            var = None
        else:
            data = ext.data if dispaxis == 1 else ext.data.T
            if dispaxis == 0 and mask is not None:
                mask = mask.T
            var = ext.variance
            if dispaxis == 0 and var is not None:
                var = var.T

        # Avoid having to recalculate them
        self._center_pixels = self.model(np.arange(npix))
        all_x1 = self._center_pixels + aper_lower
        all_x2 = self._center_pixels + aper_upper
        if viewer is not None:
            # Display extraction edges on viewer, every 10 pixels (for speed)
            pixels = np.arange(npix)
            edge_coords = np.array([pixels, all_x1]).T
            viewer.polygon(edge_coords[::10], closed=False, xfirst=(dispaxis == 1), origin=0)
            edge_coords = np.array([pixels, all_x2]).T
            viewer.polygon(edge_coords[::10], closed=False, xfirst=(dispaxis == 1), origin=0)

        # Remember how pixel coordinates are defined!
        off_low = np.where(all_x1 < -0.5)[0]
        if len(off_low):
            log.warning("Aperture extends off {} of image between {}s {} and {}".
                        format(("left", "bottom")[dispaxis], direction,
                               min(off_low), max(off_low)))
        off_high = np.where(all_x2 >= slitlength - 0.5)[0]
        if len(off_high):
            log.warning("Aperture extends off {} of image between {}s {} and {}".
                        format(("right", "top")[dispaxis], direction,
                               min(off_high), max(off_high)))

        # Create the outputs here so the extraction function has access to them
        self.data = np.zeros((npix,), dtype=np.float32)
        self.mask = None if mask is None else np.zeros_like(self.data,
                                                            dtype=DQ.datatype)
        self.var = None if var is None else np.zeros_like(self.data)

        extraction_func = getattr(self, "{}_extraction".format(method))
        extraction_func(data, mask, var, aper_lower, aper_upper)

        del self._center_pixels
        self.last_extraction = (aper_lower, aper_upper)
        ndd = NDAstroData(self.data, mask=self.mask, variance=self.var)
        try:
            ndd.meta['header'] = ext.hdr.copy()
        except AttributeError:  # we only had an ndarray
            pass
        return ndd


class Trace:
    """
    A class describing a trace along columns. It has the following attributes:

    starting_point : len-2 iterable
        The starting point of the trace on the array, in (y, x) format.
    points : collections.deque
        A deque holding points (len-2 iterables) found for the trace.
    top_limit : float
        The highest y-value that the trace has reached.
    bottom_limit : float
        The lowest y-value that the trace have reached.

    Note that trace_lines() (which creates Trace objects) *always* traces in
    the vertical direction, regardless of the orientation of the image.
    """
    def __init__(self, starting_point, starting_weight=None, reverse_returned_coords=False):
        """
        Parameters
        ----------
        starting_point : len-2 iterable of numbers
            The point from which to start the trace, with the tracing axis as
            the first number.
        starting_weight : float/None
            A weight to assign to this point
        reverse_returned_coords : bool, optional
            Whether to reversed the coordinates when returning them. The default
            is False. This is because Trace keeps track of coordinates with the
            tracing axis first, which means (y, x) order if tracing vertically;
            but it may be preferable to get the output in (x, y) order.
        """
        self.starting_point = self._verify_point(starting_point)
        self.points = deque([self.starting_point + (starting_weight,)])
        self.last_point = self.starting_point
        self.steps_missed = 0
        self.active = True
        self.reversed = reverse_returned_coords

    def _as_list(self):
        return list(self.points)

    def __len__(self):
        return len(self.points)

    def __iter__(self):
        return iter(self.points)

    def __repr__(self):
        return str(self.points)

    def _verify_point(self, point):
        """Return a tuple from a len-2 iterable"""
        if len(point) != 2:
            raise RuntimeError(f"Point {point} should have 2 values, not "
                               f"{len(point)}")
        if isinstance(point, tuple):
            return point
        else:
            try:
                return(tuple(point))
            except:
                raise RuntimeError(f"Something went wrong with point {point}")

    @property
    def top_limit(self):
        return self.points[-1][0]

    @property
    def bottom_limit(self):
        return self.points[0][0]

    @property
    def start_coordinates(self, reverse=None):
        """Return the starting point in the same coordinate order as the
        input_coordinates() and reference_coordinates()"""
        return self.starting_point[1::-1] if (
                reverse or reverse is None and self.reversed) else self.starting_point[:2]

    def input_coordinates(self, reverse=None):
        if reverse or reverse is None and self.reversed:
            return [(x, y) for y, x, w in self.points]
        return [(y, x) for y, x, w in self.points]

    def reference_coordinates(self, reference_coord=None, reverse=None):
        xref = reference_coord or self.starting_point[1]
        if reverse or reverse is None and self.reversed:
            return [(xref, y) for y, _, w in self.points]
        return [(y, xref) for y, _, w in self.points]

    def weights(self):
        w = [p[2] for p in self.points]
        if w.count(None):  # all points must have weights
            return None
        return w

    def add_point(self, point, weight=None):
        """Add a point to the deque, at either end as appropriate"""
        point = self._verify_point(point)
        y = point[0]

        if y > self.top_limit:
            self.points.append(point + (weight,))
        elif y < self.bottom_limit:
            self.points.appendleft(point + (weight,))
        else:
            # Should only add points at ends of range
            raise RuntimeError("Trying to insert point in middle of trace,"
                               f"{point}, top: {self.top_limit}, "
                               f"bottom: {self.bottom_limit}")
        self.last_point = point

    def remove_point(self, point):
        """Remove a point from the deque"""
        index = [p[:2] for p in self.points].index(point)
        self.points.remove(self.points[index])

    def predict_location(self, row, lookback=4, order=1):
        """Predict where the next peak will be in the tracing direction.

        row : float
            Where to predict the location
        lookback : int
            The number of points in the trace (in addition to the final one) to
            include in the fit to predict where it's going.
        order: int
            order of fit function

        """
        # Save ourselves some trouble by quickly returning a dummy value if
        # this Trace is inactive
        if not self.active:
            return None

        # Make sure there are enough points for the requested lookback and that
        # it's a sensible number.
        assert lookback >= 0, "'lookback' must not be negative"
        lookback = min(lookback, len(self.points) - 1)
        order = min(order, lookback)

        # Get points to trace, from eithe end as appropriate. In either case,
        # `points` will be a list of points starting from one end and heading
        # towards the middle of the trace.
        if row > self.top_limit:
            points = [self.points[i] for i in range(-1, -(lookback+2), -1)]
        elif row < self.bottom_limit:
            points = [self.points[i] for i in range(0, lookback+1, 1)]
        else:
            raise RuntimeError("No prediction within existing trace")

        if order == 0:
            return points[0][1]

        # Set up model and fit to points
        _fit_1d = fit_1D([point[1] for point in points],
                         points=[point[0] for point in points],
                         domain=[self.bottom_limit, self.top_limit],
                         order=order)

        return _fit_1d.evaluate(row)[0]


################################################################################
# FUNCTIONS RELATED TO PEAK-TRACING

@unpack_nddata
def trace_lines(data, axis, mask=None, variance=None, start=None, initial=None,
                cwidth=5, rwidth=None, nsum=10, step=10, initial_tolerance=1.0,
                max_shift=0.05, max_missed=5, func=NDStacker.median, viewer=None,
                min_peak_value=None, min_line_length=0.):
    """
    This function traces features along one axis of a two-dimensional image.
    Initial peak locations are provided and then these are matched to peaks
    found a small distance away along the direction of tracing. In terms of
    its use to map the distortion from a 2D spectral image of an arc lamp,
    these lists of coordinates can then be used to determine a distortion map
    that will remove any curvature of lines of constant wavelength.

    For a horizontally-dispersed spectrum like GMOS, the reference y-coords
    will match the input y-coords, while the reference x-coords will all be
    equal to the initial x-coords of the peaks.

    Parameters
    ----------
    data : array/single-sliced AD object/NDData-like
        The extension within which to trace features.
    axis : int (0 or 1)
        Axis along which to trace (0=y-direction, 1=x-direction).
    mask : ndarray/None
        mask associated with the image
    variance: ndarray/None
        variance associated with the image
    start : int/None
        Row/column to start trace (None => middle).
    initial : sequence
        Coordinates of peaks
    cwidth : int
        Width of centroid box in pixels.
    rwidth : int/None
        width of Ricker filter to apply to each collapsed 1D slice
    nsum : int
        Number of rows/columns to combine at each step.
    step : int
        Step size along axis in pixels.
    initial_tolerance : float/None
        Maximum perpendicular shift (in pixels) between provided location and
        first calculation of peak.
    max_shift: float
        Maximum perpendicular shift (in pixels) from pixel to pixel.
    max_missed: int
        Maximum number of interactions without finding line before line is
        considered lost forever.
    func: callable
        function to use when collapsing to 1D. This takes the data, mask, and
        variance as arguments, and returns 1D versions of all three
    viewer: imexam viewer or None
        Viewer to draw lines on.
    min_peak_value: int or float
        Minimum amplitude of fit to be considered as a real detection. Peaks
        smaller than this value will be counted as a miss.
    min_line_length: float
        Minimum length of traced feature (as a fraction of the tracing
        dimension length) to be considered as a useful line.

    Returns
    -------
    list of Trace objects, or an empty list if no peaks were recoverable.
    These objects are *always* configured to return coordinates in (x, y) order.
    """
    log = logutils.get_logger(__name__)

    # Make life easier for the poor coder by transposing data if needed,
    # so that we're always tracing along columns
    if axis == 0:
        ext_data = data
        ext_mask = None if mask is None else mask & DQ.not_signal
        direction = "row"
    else:
        ext_data = data.T
        ext_mask = None if mask is None else mask.T & DQ.not_signal
        direction = "column"

    # Make a slice around a given row center. No bounds checking
    # as that needs to be done by the calling code.
    def _slice(center):
        return slice(center - nsum // 2, center + nsum - nsum // 2)

    # Create profile for centering. This could just be the data (maybe
    # Ricker-filtered) but we use the SNR
    def _profile_for_centering(data, var, rwidth):
        var[var <= 0] = np.inf
        if rwidth:
            return np.where(data / np.sqrt(var) > 0.5,
                            cwt_ricker(data, widths=[rwidth])[0], 0)
        return np.where(data / np.sqrt(var) > 0.5, data, 0)

    halfwidth = cwidth // 2
    if start is None:
        start = ext_data.shape[0] // 2
        log.stdinfo(f"Starting trace at {direction} {start}")
    else:  # just to be sure it's OK
        start = int(min(max(start, nsum // 2), ext_data.shape[0] - nsum / 2))

    # Get accurate starting positions for all peaks if requested
    if initial_tolerance is None:
        initial_peaks = initial
        initial_peak_values = [data[int(np.round(i))] for i in initial_peaks]
    else:
        data, mask, var = func(ext_data[_slice(start)],
                               mask=None if ext_mask is None
                               else ext_mask[_slice(start)], variance=None)
        data = _profile_for_centering(data, var, rwidth)

        peaks, peak_values = pinpoint_peaks(data, peaks=initial, mask=mask,
                                            halfwidth=halfwidth)
        if not peaks:
            return []
        initial_peaks, initial_peak_values = [], []
        for peak in initial:
            j = np.argmin(abs(np.array(peaks) - peak))
            new_peak = peaks[j]
            if abs(new_peak - peak) <= initial_tolerance:
                initial_peaks.append(new_peak)
                initial_peak_values.append(peak_values[j])
            else:
                log.debug(f"Cannot recenter peak at coordinate {peak}")

    # Allocate space for collapsed arrays of different sizes
    data = np.empty((max_missed + 1, ext_data.shape[1]))
    mask = np.zeros_like(data, dtype=DQ.datatype)
    var = np.empty_like(data)

    # We're going to make lists of valid step centers to help later
    step_centers_up = list(np.arange(start, ext_data.shape[0] - nsum / 2,
                                     step, dtype=int))
    step_centers_down = list(np.arange(start, nsum / 2, -step, dtype=int))

    # Eliminate blocks that are completely masked (e.g., chip gaps, bridges, amp5)
    # Also need to eliminate regions with only one valid column because NDStacker
    # can't compute the pixel-to-pixel variance and hence the S/N can't be calculated
    if ext_mask is not None:
        for i, s in reversed(list(enumerate(step_centers_up))):
            if np.bincount((ext_mask[_slice(s)] & DQ.not_signal).min(axis=1))[0] <= 1:
                step_centers_up[i] = None
        for i, s in reversed(list(enumerate(step_centers_down))):
            if np.bincount((ext_mask[_slice(s)] & DQ.not_signal).min(axis=1))[0] <= 1:
                step_centers_down[i] = None

    # If tracing vertically-dispersed data the coordinates in the Trace will
    # be in (y, x) order and since we need (x, y) elsewhere we reverse them here.
    traces = [Trace((start, peak), starting_weight=np.sqrt(value),
                    reverse_returned_coords=(axis == 0))
              for peak, value in zip(initial_peaks, initial_peak_values)]
    for direction, step_centers in zip((1, -1), (step_centers_up, step_centers_down)):
        for trace in traces:
            trace.last_point = trace.starting_point
            trace.active = True
            trace.steps_missed = 0
        step_index = 0
        latest_lookback_step = 0

        while any(t.active for t in traces):
            # Our first point is step_index=1. The point step_index=0 is the
            # start location, so we don't want to recompute that, but it's in
            # the step_centers so we can go back and bin up including it.
            step_index += 1

            # Reached the bottom or top?
            if step_index >= len(step_centers):
                break

            # Are we going across a "dead zone", which we don't bin across?
            if step_centers[step_index] is None:
                latest_lookback_step = None
                continue
            elif latest_lookback_step is None:  # recover from "dead zone"
                latest_lookback_step = step_index

            # This indicates we should start making profiles binned across
            # multiple steps because we have lost lines but they're not
            # completely lost yet.
            lookback = min(max(t.steps_missed for t in traces if t.active),
                           step_index - latest_lookback_step)

            # Make multiple arrays covering nsum to nsum*(largest_missed+1) rows
            # There's always at least one such array
            for i in range(lookback + 1):
                slices = [_slice(step_centers[step_index-j]) for j in range(i+1)]
                d, m, v = func(np.concatenate(list(ext_data[s] for s in slices)),
                               mask=None if ext_mask is None else np.concatenate(list(ext_mask[s] for s in slices)),
                               variance=None)
                data[i] = _profile_for_centering(d, v, rwidth)
                if m is not None:
                    mask[i] = m

            # The second piece of logic is to deal with situations where only
            # one valid row is in the slice, so NDStacker will return var=0
            # because it cannot derive pixel-to-pixel variations. This makes
            # the data array 0 as well, and we can't find any peaks
            if any(mask[0] == 0) and not all(np.isinf(var[0])):
                for trace in traces:
                    if not trace.active:
                        continue

                    for j in range(min(trace.steps_missed + 1, data.shape[0])):
                        effective_ypos = step_centers[step_index] - 0.5 * j * step
                        shift_tol = max_shift * abs(effective_ypos - trace.last_point[0])
                        predicted_peak = trace.predict_location(effective_ypos, order=1)
                        peak_tol = 5

                        peaks, peak_values = pinpoint_peaks(
                            data[j], peaks=[predicted_peak], mask=mask[j],
                            halfwidth=halfwidth)

                        if (not peaks or min_peak_value is not None and
                                peak_values[0] < min_peak_value or
                                abs(peaks[0] - trace.last_point[1]) > shift_tol or
                                abs(peaks[0] - predicted_peak) > peak_tol):
                            continue

                        # A valid peak has been found
                        break
                    else:  # no valid peak
                        trace.steps_missed += 1
                        if trace.steps_missed > max_missed:
                            trace.active = False
                        continue

                    # We found a peak. Is it too close to the edge?
                    if not (halfwidth < peaks[0] < ext_data.shape[1] -
                            halfwidth - 1):
                        trace.active = False
                        continue

                    if viewer:
                        kwargs = dict(zip(('y1', 'x1'), trace.last_point if axis == 0
                        else reversed(trace.last_point)))
                        kwargs.update(dict(zip(('y2', 'x2'), (effective_ypos, peaks[0]) if axis == 0
                        else (peaks[0], effective_ypos))))
                        viewer.line(origin=0, **kwargs)

                    trace.add_point((effective_ypos, peaks[0]), np.sqrt(peak_values[0]))
                    trace.steps_missed = 0
            else:  # We don't bin across completely dead regions
                # We really shouldn't get here as this should be handled by
                # the step_centers creation
                latest_lookback_step = None

        step *= -1

    # Remove short lines
    min_length_pixels = min_line_length * ext_data.shape[0]
    final_traces = [trace for trace in traces if (
            trace.points[-1][0] - trace.points[0][0] >= min_length_pixels)]

    return final_traces

def trace_aperture(ext, location, ui_params, viewer=None, apnum=None):
    """
    Trace a single aperture provided its location on the spatial axis

    Parameters
    ----------
    ext : AstroData.extension object
        The extension to trace in.
    location : float or int
        The pixel location along the spatial axis where the aperture is located.
    ui_params : :class:`~geminidr.interactive.interactive.UIParams`
        Fitting parameters to pass to `trace_lines()`. For interactive use,
        also contains UI parameters to use as inputs to generate the points.
    viewer: imexam viewer or None
        Viewer to draw lines on.
    apnum : int or None
        Aperture number (for when tracing multiple apertures in an image).

    Returns
    -------
    list of :class:`~gempy.library.tracing.Trace` objects.

    """
    dispaxis = 2 - ext.dispersion_axis()  # python sense

    c0 = int(location + 0.5)
    nsum = ui_params.values['nsum']
    spectrum = ext.data[c0, nsum:-nsum] if dispaxis == 1 else ext.data[nsum:-nsum, c0]
    if ext.mask is None:
        start = np.argmax(at.boxcar(spectrum, size=20)) + nsum
    else:
        good = ((ext.mask[c0, nsum:-nsum] if dispaxis == 1 else
                 ext.mask[nsum:-nsum, c0]) & DQ.not_signal) == 0

        start = nsum + np.arange(spectrum.size)[good][np.argmax(
            at.boxcar(spectrum[good], size=20))]
    if apnum is not None:
        log.stdinfo(f"{ext.filename}: Starting trace of "
                    f"aperture {apnum} at pixel {start+1}")

    # The coordinates are always returned as (x-coords, y-coords)
    return trace_lines(
        ext,
        axis=dispaxis,
        cwidth=5,
        initial=[location],
        initial_tolerance=None,
        max_missed=ui_params.values['max_missed'],
        max_shift=ui_params.values['max_shift'],
        nsum=ui_params.values['nsum'],
        rwidth=None,
        start=start,
        step=ui_params.values['step'],
        viewer=viewer
    )
