# Module for IGRINS-2 wavecal stuff
import warnings

import numpy as np

from astropy.modeling import fitting, models
from astropy.table import Table
from astropy import units as u
from astropy.utils.exceptions import AstropyWarning

from matplotlib import pyplot as plt, colors as mcolors

from gempy.library import astromodels as am


class IGRINSWavecal:
    def __init__(self, spectra, first_order=None, filename='', in_vacuo=True, log=None):
        """

        Parameters
        ----------
        spectra: iterable of single-slice AstroDataIgrins objects
        """
        self.npix = 2048
        self.pixels = np.arange(self.npix)
        self.spectra = spectra
        self.ref_waves = None
        self.first_order = first_order
        self.filename = filename
        self.in_vacuo = in_vacuo
        self.log = log
        self.fitted_lines ={}

    def create_2d_wavelength_solution(self, xdeg, ydeg, sigma_clip):
        orders, pix, waves = [np.asarray(self.fitted_lines[k])
                              for k in ('order', 'pixel', 'wavelength')]
        fit_it = fitting.FittingWithOutlierRemoval(fitting.LinearLSQFitter(),
                                                   outlier_func=sigma_clip)
        m_init = models.Chebyshev2D(x_degree=xdeg, y_degree=ydeg,
                                    x_domain=(0, self.npix-1),
                                    y_domain=(min(orders), max(orders)))
        m_final, mask = fit_it(m_init, pix, orders, waves * orders)
        self.fitted_model = m_final
        self.fitted_mask = mask

        # This code is comparable to update_wcs_with_solution()
        tbl_orders = orders[~mask].astype(int)
        tbl_pix = pix[~mask].astype(np.float32)
        tbl_waves = waves[~mask].astype(np.float32)
        # tbl_sigmas = sigmas[~mask].astype(np.float32)
        tbl_fitted = m_final(tbl_pix, tbl_orders) / tbl_orders
        rms = np.std(m_final(tbl_pix, tbl_orders) / tbl_orders - tbl_waves)
        self.fitted_rms = rms
        tbl_pix += 1  # use 1-based for output

        temptable = am.model_to_table(m_final, xunit=u.pixel, yunit=None)
        temptable.add_columns([[2], [xdeg], [ydeg], [0], [self.npix-1], [min(orders)], [max(orders)], [rms]],
                              names=("ndim", "xdegree", "ydegree", "xdomain_start",
                                     "xdomain_end", "ydomain_start", "ydomain_end", "rms"))

        pad_rows = tbl_orders.size - len(temptable.colnames)
        if pad_rows < 0:
            tbl_orders = list(tbl_orders) + [0] * (-pad_rows)
            tbl_pix = list(tbl_pix) + [0] * (-pad_rows)
            tbl_waves = list(tbl_waves) + [0] * (-pad_rows)
            pad_rows = 0

        fit_table = Table([temptable.colnames + [''] * pad_rows,
                           list(temptable[0].values()) + [0] * pad_rows,
                           tbl_orders, tbl_pix, tbl_waves,
                           tbl_fitted],
                          names=("name", "coefficients", "xdorder", "peaks",
                                 "wavelengths", "fitted"),
                          units=(None, None, None, u.pix, u.nm, u.nm),
                          meta=temptable.meta)
        medium = "vacuo" if self.in_vacuo else "air"
        fit_table.meta['comments'] = [
            'coefficients are based on 0-indexing',
            'peaks column is 1-indexed',
            f'calibrated with wavelengths in {medium}']
        return fit_table

    def get_lines_in_order(self, wcs, buffer=0):
        """
        Calculates which lines in the reference list fall within the
        wavelength range of this order.

        Parameters
        ----------
        wcs: gWCS object
            The WCS of the spectrum that maps from pixel to wavelength
        buffer: float
            minimum distance from edge of array

        Returns
        -------
        array: boolean array indicating which lines in the linelist are in this order
        """
        in_order = np.logical_and(self.ref_waves >= wcs(buffer),
                                  self.ref_waves <= wcs(self.npix - buffer - 1))
        return in_order

    def identify_sky_lines(self, idx, min_snr=3, ref_sigpx=None):
        def _create_fitting_groups(pixloc, sigpx):
            sigma = sigpx[0]
            groups = [[0]]
            for i, (x, s) in enumerate(zip(pixloc[1:], sigpx[1:]), start=1):
                sigma = max(sigma, s)
                if x - pixloc[groups[-1][-1]] < 3 * sigma:
                    groups[-1].append(i)
                else:
                    groups.append([i])
            return groups

        # The stddev for unresolved lines (empirically determined from fits)
        sigma_unres = 1.25
        line_id = len(self.fitted_lines.get('pixel', []))

        log = self.log
        fitted_lines = {'pixel': [], 'wavelength': [], 'sigma': [],
                        'debug': []}
        spec = self.spectra[idx]
        spec_order = self.first_order + idx

        if (in_order := self.get_lines_in_order(spec.wcs)).sum() == 0:
            # Nothing to see here
            return fitted_lines

        # Get the lines that appear in this order, and group them if
        # they're close together so that we can fit them simultaneously
        order_waves = self.ref_waves[in_order]
        order_sigma = ref_sigpx[in_order]
        pixloc = spec.wcs.invert(order_waves)
        groups = _create_fitting_groups(pixloc, order_sigma)
        log.stdinfo(f"Order {spec_order}: Fitting {len(order_waves)} "
                    f"lines in {len(groups)} groups")

        # Now do a combined fit of the the lines in each group
        for group in groups:
            linestr = "+".join(f"{order_waves[line]:.2f}" for line in group)
            x1 = int(pixloc[group[0]] - 5 * order_sigma[group[0]] - 5)
            x2 = int(pixloc[group[-1]] + 5 * order_sigma[group[-1]] + 6)
            if x1 < 0 or x2 >= self.npix:  # too close to edge, skip
                continue
            x = np.arange(x1, x2)
            y = np.ma.masked_array(spec.data[x1:x2], mask=spec.mask[x1:x2])
            if y.size - y.count() > 2:  # max number of masked pixels
                log.stdinfo(f"Rejecting line(s) at {linestr}nm in "
                            f"order {spec_order} due to excessive masking")
                continue
            m_init = models.Const1D()  # continuum
            for i, line in enumerate(group, start=1):
                g = models.Gaussian1D(amplitude=max(spec.data[int(pixloc[line])], 0),
                                      mean=pixloc[line], stddev=order_sigma[line])
                if i > 1:  # fix the separations of the lines
                    g.mean.tied = lambda m: m.mean_1 + (pixloc[line] - pixloc[group[0]])
                    g.stddev.tied = lambda m: m.stddev_1
                else:
                    g.mean.bounds = (pixloc[line] - 5, pixloc[line] + 5)
                g.amplitude.bounds = (0, np.inf)
                factor = 1.25 if order_sigma[line] == sigma_unres else 2.0
                g.stddev.bounds = (order_sigma[line] / factor,
                                   order_sigma[line] * factor)
                m_init += g

            fit_it = fitting.TRFLSQFitter()
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=AstropyWarning)
                m_final = fit_it(m_init, x, y, maxiter=1000)
            if not fit_it.fit_info.success:
                log.stdinfo(f"Rejecting line(s) at {linestr}nm in "
                            f"order {spec_order} due to fit failure: "
                            f"{fit_it.fit_info.message}")
                continue

            # Check that line passed min_snr requirement
            lines_only = m_final(x) - m_final[0](x)
            snr = lines_only.max() / np.sqrt(spec.variance[x[lines_only.argmax()]])
            if snr < min_snr:
                log.stdinfo(f"Rejecting line(s) at {linestr}nm in "
                            f"order {spec_order} with SNR={snr:.1f}")
                continue

            # Check that multiple lines actually *are* being fit, and
            # it's not fitting the whole thing with a single line and
            # adding low-amplitude features
            amplitudes = np.asarray([getattr(m_final, f"amplitude_{i}").value
                                     for i, _ in enumerate(group, start=1)])
            if amplitudes.min() < 0.5 * amplitudes.max():
                log.stdinfo(f"Rejecting line(s) at {linestr}nm in "
                            f"order {spec_order} due to large "
                            "amplitude differences")
                continue

            # Store order, pixel location, and wavelength. For a group
            # with multiple lines, we store the mean pixel location and
            # mean wavelength, since we've only fit a single parameter.
            pixel = np.mean([getattr(m_final, f"mean_{i + 1}").value
                             for i in range(len(group))])
            wavelength = np.mean([order_waves[line] for line in group])
            sigma = np.mean([getattr(m_final, f"stddev_{i + 1}").value
                             for i in range(len(group))])
            log.debug(f"Adding ({spec_order}, {pixel:.2f}, {linestr})"
                      " to fitted line list")
            fitted_lines['pixel'].append(pixel)
            fitted_lines['wavelength'].append(wavelength)
            fitted_lines['sigma'].append(sigma)

            # debug information
            for i, line in enumerate(group, start=1):
                _tuple = (spec_order,
                          order_waves[line],
                          getattr(m_final, f"mean_{i}").value,
                          getattr(m_final, f"amplitude_{i}").value,
                          getattr(m_final, f"stddev_{i}").value,
                          line_id)
                fitted_lines['debug'].append(_tuple)
            line_id += 1

        fitted_lines['order'] = [spec_order] * len(fitted_lines['pixel'])
        return fitted_lines

    def identify_telluric_lines(self, idx, min_snr=3):
        log = self.log
        fitted_lines = {'pixel': [], 'wavelength': []}
        spec = self.spectra[idx]
        spec_order = self.first_order + idx

        if (in_order := self.get_lines_in_order(spec.wcs, buffer=5)).sum() == 0:
            # Nothing to see here
            return fitted_lines

        order_waves = self.ref_waves[in_order]
        pixloc = spec.wcs.invert(order_waves)
        for x, wavelength in zip(pixloc, order_waves):
            pixel = centroid_frac(-spec.data, peaks=[x], mask=spec.mask, keep_bad=True)[0]
            if not np.isnan(pixel):
                log.stdinfo(f"Adding ({spec_order}, {pixel:.2f}, {wavelength:.2f})"
                            " to fitted line list")
                fitted_lines['pixel'].append(pixel)
                fitted_lines['wavelength'].append(wavelength)

        fitted_lines['order'] = [spec_order] * len(fitted_lines['pixel'])
        return fitted_lines

    def make_debug_plot(self, edge_func):
        fig, ax = plt.subplots()
        cmap = plt.get_cmap('jet')
        norm = mcolors.Normalize(vmin=-0.005, vmax=0.005)

        for idx, _ in enumerate(self.spectra):
            edge1, edge2 = edge_func(idx, self.pixels)
            xfill = list(self.pixels) + list(self.pixels[::-1])
            yfill = list(edge1) + list(edge2[::-1])
            ax.fill(xfill, yfill, color="gainsboro", edgecolor=None)

        for order, wave, x, mask in zip(
                self.fitted_lines['order'], self.fitted_lines['wavelength'],
                self.fitted_lines['pixel'], self.fitted_mask):
            yends = edge_func(order - self.first_order, x)
            linestyle = ':' if mask else '-'
            residual = self.fitted_model(x, order) / order - wave
            color = cmap(norm(residual))
            ax.plot([x, x], yends, color=color, ls=linestyle)

        ax.set_xlim(0, self.npix-1)
        if edge2[1024] > 200:  # edge2 is defined from the last order, this means it's sky
            ax.set_ylim(0, self.npix-1)
            ax.set_aspect('equal')
        else:
            ax.set_ylim(self.first_order-0.5, self.first_order+len(self.spectra)+0.5)
        ax.set_title(f"{self.filename}\n{self.fitted_mask.size} lines "
                     f"({self.fitted_mask.sum()} rejected) RMS = {self.fitted_rms:.4f}nm")
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label="Fitted residual (nm)")
        plt.tight_layout(pad=1)
        plt.savefig(self.filename.replace(".fits", "_wavecal.pdf"))

    def apply_global_shift(self):
        """
        Measure an overall horizontal shift for the wavelength solution and
        apply it to all spectra.

        Returns
        -------
        global_shift: float
            the overall shift
        """
        xcorr = np.zeros((2 * self.npix - 1,))
        for ext in self.spectra:
            if (lines := self.ref_waves[self.get_lines_in_order(ext.wcs)]).size:
                pixloc = ext.wcs.invert(lines)
                # Since we haven't matched the lines yet, we don't know their
                # widths, so just assume stddev=1.5 pixels for all
                expected = np.zeros_like(self.pixels) + np.sum([np.exp(-(cntr - self.pixels) ** 2 / 4.5)
                                                       for cntr in pixloc], axis=0)
                xcorr += np.correlate(expected, ext.data, mode='full')

        # -ve means line peaks are "ahead" of expected
        global_shift = xcorr.argmax() - (self.npix - 1)

        # Update the WCS of each spectrum
        for ext in self.spectra:
            ext.wcs.insert_transform(ext.wcs.input_frame, models.Shift(global_shift), after=True)

        return global_shift


def centroid_frac(data, peaks=None, mask=None, halfwidth=1.5, maxiter=50,
                  keep_bad=False):
    """

    Parameters
    ----------
    data: array
    peaks
    mask
    halfwidth
    maxiter
    keep_bad

    Returns
    -------

    """
    final_peaks = []
    for p in peaks:
        c_last = p
        for _ in range(maxiter):
            lo, hi = c_last - halfwidth, c_last + halfwidth
            i = np.arange(int(np.floor(lo)) - 1, int(np.ceil(hi)) + 2)
            i = i[(i >= 0) & (i < data.size)]
            w = np.clip(np.minimum(i + 0.5, hi) - np.maximum(i - 0.5, lo), 0, 1)
            xm = 0.5 * (np.minimum(i + 0.5, hi) + np.maximum(i - 0.5, lo))
            if mask is not None and any(mask[i]):
                if keep_bad:
                    final_peaks.append(np.nan)
                break
            d = (data[i] - data[i].min()) * w
            c = np.sum(xm * d) / d.sum()
            if abs(c - p) > 0.5 * halfwidth:  # it's moved too far
                if keep_bad:
                    final_peaks.append(np.nan)
                break
            if abs(c - c_last) < 1e-5:
                final_peaks.append(c_last)
                break
            c_last = c
        else:
            final_peaks.append(c)
    return np.array(final_peaks)
