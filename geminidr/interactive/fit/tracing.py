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


def interactive_trace_apertures(ext, fit1d_params, ui_params: UIParameters):
    """
    Run traceApertures() interactively.

    Parameters
    ----------
    ext : AstroData
        Single extension extracted from an AstroData object.
    config : :class:`geminidr.code.spect.traceAperturesConfig`
        Configuration object containing the parameters from traceApertures().
    fit1d_params : dict
        Dictionary containing initial parameters for fitting a model.

    Returns
    -------
    list of models describing the aperture curvature
    """
    ap_table = ext.APERTURE
    fit_par_list = list()
    for _ in range(len(ap_table)):
        fit_par_list.append({x: y for x, y in fit1d_params.items()})

    domain_list = [
        [
            ap_table.meta["header"][kw]
            for kw in ("DOMAIN_START", "DOMAIN_END")
        ]
        for _ in ap_table
    ]

    if (2 - ext.dispersion_axis()) == 1:
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
        lambda ui_params: trace_apertures_data_provider(ext, ui_params),
        domains=domain_list,
        filename_info=ext.filename,
        fitting_parameters=fit_par_list,
        help_text=help_text,
        primitive_name="traceApertures",
        tab_name_fmt=lambda i: f"Aperture {i+1}",
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
def trace_apertures_data_provider(ext, ui_params):
    """
    Function used by the interactive fitter to generate the a list with
    pairs of [x, y] data containing the knots used for tracing.

    Parameters
    ----------
    ext : AstroData
        Single extension of data containing an .APERTURE table.
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
    dispaxis = 2 - ext.dispersion_axis()  # python sense

    for loc in ext.APERTURE['c0'].data:
        c0 = int(loc + 0.5)
        spectrum = ext.data[c0] if dispaxis == 1 else ext.data[:, c0]
        start = np.argmax(at.boxcar(spectrum, size=20))

        # The coordinates are always returned as (x-coords, y-coords)
        traces = tracing.trace_lines(
            ext,
            axis=dispaxis,
            cwidth=5,
            initial=[loc],
            initial_tolerance=None,
            max_missed=ui_params.values['max_missed'],
            max_shift=ui_params.values['max_shift'],
            nsum=ui_params.values['nsum'],
            rwidth=None,
            start=start,
            step=ui_params.values['step'],
        )

        # List of traced peak positions
        in_coords = np.array([coord for trace in traces for
                              coord in trace.input_coordinates()]).T

        assert len(in_coords) == 2,\
            f"No trace was found at {loc} in {ext.filename}."

        data["x"].append(in_coords[1 - dispaxis])
        data["y"].append(in_coords[dispaxis])

    return data
