#
#                                                                  gemini_python
#
#                                                          primtives_f2_spect.py
# ------------------------------------------------------------------------------
import numpy as np

from importlib import import_module
import os

from geminidr.core import Spect
from .primitives_f2 import F2
from . import parameters_f2_spect

#
#                                                                  gemini_python
#
#                                                        primtives_gnirs_spect.py
# ------------------------------------------------------------------------------
import os

import numpy as np
from copy import copy
from importlib import import_module
from datetime import datetime
from functools import reduce
from copy import deepcopy

from functools import partial

from gempy.library import astrotools as at

from astropy.table import Table

from astropy.modeling import models, Model
from astropy import units as u
from scipy.interpolate import UnivariateSpline

import geminidr.interactive.server
from geminidr.core import Spect
from gempy.library.fitting import fit_1D
from .primitives_f2 import F2
from . import parameters_f2_spect

from geminidr.gemini.lookups import DQ_definitions as DQ
from geminidr.gmos.lookups import geometry_conf as geotable

from gempy.gemini import gemini_tools as gt
from gempy.library import astromodels as am
from gempy.library import transform, wavecal

from recipe_system.utils.decorators import parameter_override, capture_provenance
from ..interactive.fit.wavecal import WavelengthSolutionVisualizer
from ..interactive.interactive import UIParameters
from ..core import NearIR

from recipe_system.utils.decorators import parameter_override, capture_provenance


# ------------------------------------------------------------------------------
@parameter_override
@capture_provenance
class F2Spect(F2, Spect):
    """
    This is the class containing all of the preprocessing primitives
    for the F2Spect level of the type hierarchy tree. It inherits all
    the primitives from the level above
    """
    tagset = {"GEMINI", "F2", "SPECT"}

    def _initialize(self, adinputs, **kwargs):
        super()._initialize(adinputs, **kwargs)
        self._param_update(parameters_f2_spect)

    def standardizeWCS(self, adinputs=None, suffix=None):
        """
        This primitive updates the WCS attribute of each NDAstroData extension
        in the input AstroData objects. For spectroscopic data, it means
        replacing an imaging WCS with an approximate spectroscopic WCS.

        Parameters
        ----------
        suffix: str/None
            suffix to be added to output files

        """

        log = self.log
        timestamp_key = self.timestamp_keys[self.myself()]
        log.debug(gt.log_message("primitive", self.myself(), "starting"))

        for ad in adinputs:
            log.stdinfo(f"Adding spectroscopic WCS to {ad.filename}")
            cenwave = ad.central_wavelength(asNanometers=True)
            print(f"central wvl as nano from descriptor:{ad.central_wavelength(asNanometers=True)}")
            print(f"dispersion as nano from descriptor:{ad.dispersion(asNanometers=True)}")
            transform.add_longslit_wcs(ad, central_wavelength=cenwave)

            # Timestamp and update filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=suffix, strip=True)
        return adinputs

    def determineWavelengthSolution(self, adinputs=None, **params):
        """
        Determines the wavelength solution for an ARC and updates the wcs
        with this solution. In addition, the solution and pixel/wavelength
        matches are stored as an attached `WAVECAL` :class:`~astropy.table.Table`.

        2D input images are converted to 1D by collapsing a slice of the image
        along the dispersion direction, and peaks are identified. These are then
        matched to an arc line list, using piecewise-fitting of (usually)
        linear functions to match peaks to arc lines, using the
        :class:`~gempy.library.matching.KDTreeFitter`.

        The `.WAVECAL` table contains four columns:
            ["name", "coefficients", "peaks", "wavelengths"]

        The `name` and the `coefficients` columns contain information to
        re-create an Chebyshev1D object, plus additional information about
        the way the spectrum was collapsed. The `peaks` column contains the
        (1-indexed) position of the lines that were matched to the catalogue,
        and the `wavelengths` column contains the matched wavelengths.

        Parameters
        ----------
        adinputs : list of :class:`~astrodata.AstroData`
             Mosaicked Arc data as 2D spectral images or 1D spectra.

        suffix : str/None
            Suffix to be added to output files

        order : int
            Order of Chebyshev fitting function.

        center : None or int
            Central row/column for 1D extraction (None => use middle).

        nsum : int, optional
            Number of rows/columns to average.

        min_snr : float
            Minimum S/N ratio in line peak to be used in fitting.

        weighting : {'natural', 'relative', 'none'}
            How to weight the detected peaks.

        fwidth : float/None
            Expected width of arc lines in pixels. It tells how far the
            KDTreeFitter should look for when matching detected peaks with
            reference arcs lines. If None, `fwidth` is determined using
            `tracing.estimate_peak_width`.

        min_sep : float
            Minimum separation (in pixels) for peaks to be considered distinct

        central_wavelength : float/None
            central wavelength in nm (if None, use the WCS or descriptor)

        dispersion : float/None
            dispersion in nm/pixel (if None, use the WCS or descriptor)

        linelist : str/None
            Name of file containing arc lines. If None, then a default look-up
            table will be used.

        alternative_centers : bool
            Identify alternative central wavelengths and try to fit them?

        nbright : int (or may not exist in certain class methods)
            Number of brightest lines to cull before fitting

        absorption : bool
            If feature type is absorption (default: "False")

        interactive : bool
            Use the interactive tool?

        debug : bool
            Enable plots for debugging.

        Returns
        -------
        list of :class:`~astrodata.AstroData`
            Updated objects with a `.WAVECAL` attribute and improved wcs for
            each slice

        See Also
        --------
        :class:`~geminidr.core.primitives_visualize.Visualize.mosaicDetectors`,
        :class:`~gempy.library.matching.KDTreeFitter`,
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        timestamp_key = self.timestamp_keys[self.myself()]
        sfx = params["suffix"]
        arc_file = params["linelist"]
        interactive = params["interactive"]
        absorption = params["absorption"]

        linelist = None
        if arc_file is not None:
            try:
                linelist = wavecal.LineList(arc_file)
            except OSError:
                log.warning(f"Cannot read file {arc_file} - "
                            "using default linelist")
            else:
                log.stdinfo(f"Read arc line list {arc_file}")

        # Pass the primitive configuration to the interactive object.
        config = copy(self.params[self.myself()])
        config.update(**params)

        for ad in adinputs:
            log.stdinfo(f"Determining wavelength solution for {ad.filename}")

            uiparams = UIParameters(
                config, reinit_params=["center", "nsum", "min_snr", "min_sep",
                                       "fwidth", "central_wavelength", "dispersion", "in_vacuo"])
            uiparams.fields["center"].max = min(
                ext.shape[ext.dispersion_axis() - 1] for ext in ad)

            if interactive:
                all_fp_init = [fit_1D.translate_params(
                    {**params, "function": "chebyshev"})] * len(ad)
                # This feels like I shouldn't have to do it here
                domains = []
                for ext in ad:
                    axis = 0 if ext.data.ndim == 1 else 2 - ext.dispersion_axis()
                    domains.append([0, ext.shape[axis] - 1])

                if absorption:
                    ad = deepcopy(ad)
                    for i, data in enumerate(ad.data):
                        ad[i].data = -data
                        # ad[i].data = np.reciprocal(data)
                        # # ad[i].data = at.divide0(1, data)
                reconstruct_points = partial(wavecal.create_interactive_inputs, ad, p=self,
                            linelist=linelist, bad_bits=DQ.not_signal)
                print(f"uiparams:{uiparams.values}")
                print(f"reconstruct points:{reconstruct_points.keywords}")

                visualizer = WavelengthSolutionVisualizer(
                    reconstruct_points, all_fp_init,
                    modal_message="Re-extracting 1D spectra",
                    tab_name_fmt="Slit {}",
                    xlabel="Fitted wavelength (nm)", ylabel="Non-linear component (nm)",
                    domains=domains,
                    absorption=absorption,
                    title="Wavelength Solution",
                    primitive_name=self.myself(),
                    filename_info=ad.filename,
                    enable_regions=False, plot_ratios=False, plot_height=350,
                    ui_params=uiparams)
                geminidr.interactive.server.interactive_fitter(visualizer)
                for ext, fit1d, image, other in zip(ad, visualizer.results(),
                                                    visualizer.image, visualizer.meta):
                    fit1d.image = image
               #     print(f"FINAL SPECTRA PARAMETERS: wvl_start={fit1d.evaluate(-0.5)}, "
                #          f"wlv_end={fit1d.evaluate(1021.5)}, cen_wvl={fit1d.evaluate(511)}, dw={(fit1d.evaluate(-0.5)-fit1d.evaluate(1021.5))/1022}")
                    wavecal.update_wcs_with_solution(ext, fit1d, other, config)
            else:
                for ext in ad:
                    if len(ad) > 1:
                        log.stdinfo(f"Determining solution for extension {ext.id}")

                    input_data, fit1d, acceptable_fit = wavecal.get_automated_fit(
                        ext, uiparams, p=self, linelist=linelist, bad_bits=DQ.not_signal)
                    if not acceptable_fit:
                        log.warning("No acceptable wavelength solution found "
                                    f"for {ext.id}")

                    wavecal.update_wcs_with_solution(ext, fit1d, input_data, config)
                    wavecal.save_fit_as_pdf(input_data["spectrum"], fit1d.points[~fit1d.mask],
                                             fit1d.image[~fit1d.mask], ad.filename)

            # Timestamp and update the filename
            gt.mark_history(ad, primname=self.myself(), keyword=timestamp_key)
            ad.update_filename(suffix=sfx, strip=True)

            print(f"FINAL SPECTRA PARAMETERS: wvl_start={fit1d.evaluate(-0.5)}, "
                f"wlv_end={fit1d.evaluate(2046.5)}, cen_wvl={fit1d.evaluate(1023)}, dw={(fit1d.evaluate(-0.5)-fit1d.evaluate(2046.5))/2047}")

        return adinputs

    def _get_arc_linelist(self, waves=None, ad=None):
        lookup_dir = os.path.dirname(import_module('.__init__',
                                                   self.inst_lookups).__file__)
        linelist = 'lowresargon.dat'

        filename = os.path.join(lookup_dir, linelist)
        return wavecal.LineList(filename)




