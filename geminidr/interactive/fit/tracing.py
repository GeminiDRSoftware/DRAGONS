"""
Interactive function and helper functions used to trace apertures.
"""
import numpy as np

from geminidr.interactive.fit import help as fit_help
from geminidr.interactive.interactive import UIParameters
from gempy.library import astrotools as at, tracing
from .fit1d import Fit1DVisualizer

from .. import server

__all__ = ["interactive_trace_apertures", ]


def interactive_trace_apertures(ad, tab_labels, fit1d_params, ui_params: UIParameters):
    """
    Run traceApertures() interactively.

    Parameters
    ----------
    ad : AstroData
        Containing APERTURE table(s) on one or more extensions
    config : :class:`geminidr.code.spect.traceAperturesConfig`
        Configuration object containing the parameters from traceApertures().
    fit1d_params : dict
        Dictionary containing initial parameters for fitting a model.

    Returns
    -------
    list of models describing the aperture curvature
    """
    domain_list = [
        [
            ext.APERTURE.meta["header"][kw]
            for kw in ("DOMAIN_START", "DOMAIN_END")
        ]
        for ext in ad
        for _ in (ext.APERTURE if hasattr(ext, "APERTURE") else [])
    ]

    fit_par_list = [{x: y for x, y in fit1d_params.items()}] * len(domain_list)

    dispaxes = set(ad.dispersion_axis())
    if len(dispaxes) > 1:
        xlabel = "Wavelength axis [px]"
        ylabel = "Spatial axis [px]"
    else:
        if dispaxes.pop() == 1:
            xlabel = "x / columns [px]"
            ylabel = "y / rows [px]"

        else:
            xlabel = "y / rows [px]"
            ylabel = "x / columns [px]"

    help_text = (
        fit_help.DEFAULT_HELP
        + fit_help.TRACE_APERTURES
        + fit_help.PLOT_TOOLS_WITH_SELECT_HELP_SUBTEXT
        + fit_help.REGION_EDITING_HELP_SUBTEXT
    )

    visualizer = Fit1DVisualizer(
        lambda ui_params: trace_apertures_data_provider(ad, ui_params),
        domains=domain_list,
        filename_info=ad.filename,
        fitting_parameters=fit_par_list,
        help_text=help_text,
        primitive_name="traceApertures",
        tab_name_fmt=lambda i: tab_labels[i],
        title="Interactive Trace Apertures",
        xlabel=xlabel,
        ylabel=ylabel,
        modal_button_label="Trace apertures",
        modal_message="Tracing apertures...",
        ui_params=ui_params,
        turbo_tabs=True
    )

    server.interactive_fitter(visualizer)
    models = visualizer.results()
    return [m.model for m in models]


# noinspection PyUnusedLocal
def trace_apertures_data_provider(ad, ui_params):
    """
    Function used by the interactive fitter to generate the a list with
    pairs of [x, y] data containing the knots used for tracing.

    Parameters
    ----------
    ad : AstroData
        Containing APERTURE table(s) on one or more extensions
    ui_params : :class:`~geminidr.interactive.interactive.UIParams`
        UI parameters to use as inputs to generate the points

    Returns
    -------
    dict : dictionary of x and y coordinates.
        Each is an array with a list of values for each aperture center.  The x
        coordinates have the spectral position of the knots, and y is the
        spacial position of the knots.
    """
    data = {"x": [], "y": []}
    for ext, dispaxis in zip(ad, ad.dispersion_axis()):
        if hasattr(ext, "APERTURE"):
            for row in ext.APERTURE:
                loc, apnum = row["c0"], row["number"]
                traces = tracing.trace_aperture(ext, loc, ui_params, apnum=apnum)

                # List of traced peak positions
                in_coords = np.array([coord for trace in traces for
                                      coord in trace.input_coordinates()]).T

                assert len(in_coords) == 2,\
                    f"No trace was found at {loc} in {ext.filename}."

                # dispaxis is NOT in python sense here
                data["x"].append(in_coords[dispaxis - 1])
                data["y"].append(in_coords[2 - dispaxis])

    return data
