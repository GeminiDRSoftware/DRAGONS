import numpy as np
from bokeh.layouts import row, column
from bokeh.models import Div

from geminidr.interactive import server, interactive
from geminidr.interactive.interactive import \
    build_figure, build_text_slider, build_cds, connect_update_coords

__all__ = ["interactive_extract_spectra", ]


class ExtractSpectraModel:
    def __init__(self, apnum, aperture, ext,
                 method='standard', dispaxis=None):
        self.apnum = apnum
        self.aperture = aperture
        self.ext = ext
        self.width = aperture.width
        self.aper_lower = aperture.aper_lower
        self.aper_upper = aperture.aper_upper
        self.method = method
        self.dispaxis = dispaxis

        self.used_aper_lower = None
        self.used_aper_upper = None
        self.ndd_spec = None

        # These are the heart of the model.  The users of the model
        # register to listen to these two coordinate sets to get updates.
        # Whenever there is a call to recalc_spline, these coordinate
        # sets will update and will notify all registered listeners.
        self.points1_listeners = list()
        self.points2_listeners = list()

    def recalc_extract(self):
        """
        Recalculate the extract based on the currently set parameters.

        Whenever one of the parameters that goes into the extract function is
        changed, we come back in here to do the recalculation.  Additionally,
        the result is used to update the display

        Returns
        -------
        none
        """
        self.ndd_spec = self.aperture.extract(self.ext, width=self.width,
                                              method=self.method, viewer=None)
        # taken from tracing.py...
        if self.width is not None:
            aper_upper = 0.5 * self.width
            aper_lower = -aper_upper
        if aper_lower is None:
            aper_lower = self.aper_lower
        if aper_upper is None:
            aper_upper = self.aper_upper

        # for details display
        self.used_aper_upper = aper_upper
        self.used_aper_lower = aper_lower

        npix = self.ext.shape[self.dispaxis]

        center_pixels = self.aperture.model(np.arange(npix))
        all_x1 = center_pixels + aper_lower
        all_x2 = center_pixels + aper_upper
        # Display extraction edges on viewer, every 10 pixels (for speed)
        pixels = np.arange(npix)
        # edge_coords = np.array([pixels, all_x1]).T
        for fn in self.points1_listeners:
            fn(pixels, all_x1)
        # viewer.polygon(edge_coords[::10], closed=False, xfirst=(dispaxis == 1), origin=0)
        # edge_coords = np.array([pixels, all_x2]).T
        for fn in self.points2_listeners:
            fn(pixels, all_x2)
        # viewer.polygon(edge_coords[::10], closed=False, xfirst=(dispaxis == 1), origin=0)


class ExtractSpectraVisualizer(interactive.PrimitiveVisualizer):
    def __init__(self, model,
                 x_axis_label, y_axis_label):
        """
        Create a spectra extraction visualizer.

        This makes a visualizer for letting a user interactively set the
        extract parameters.  The class handles some common logic for setting up
        the web document and for holding the result of the interaction.  In
        future, other visualizers will follow a similar pattern.

        Parameters
        ----------
        model : :class:`~SpectraExtractModel`
            model holding the data, parameters and calculation logic
        """
        super().__init__()
        # Note that self._fields in the base class is setup with a dictionary mapping conveniently
        # from field name to the underlying config.Field entry, even though fields just comes in as
        # an iterable
        self.model = model
        self.p = None
        self.spline = None
        self.scatter_masked = None
        self.scatter_all = None
        self.line = None
        self.scatter_source = None
        self.line_source = None
        self.x_axis_label = x_axis_label
        self.y_axis_label = y_axis_label

        self.poly1 = None
        self.poly2 = None

        self.details = None

    def visualize(self, doc):
        """
        Build the visualization in bokeh in the given browser document.

        Parameters
        ----------
        doc
            Bokeh provided document to add visual elements to

        Returns
        -------
        none
        """
        super().visualize(doc)

        width_slider = build_text_slider("Width", self.model.width, 1, 0, self.model.width * 2,
                                         self.model, "width", self.model.recalc_extract, throttled=True)

        # Create a blank figure with labels
        self.p = build_figure(plot_width=600, plot_height=500,
                              title='Spectra 1D',
                              tools="pan,wheel_zoom,box_zoom,reset",
                              x_axis_label=self.x_axis_label, y_axis_label=self.y_axis_label,
                              x_range=(0, self.model.ext.shape[0]), y_range=(0, self.model.ext.shape[1]))

        # We can plot this here because it never changes
        # the overlay we plot later since it does change, giving
        # the illusion of "coloring" these points
        coords = build_cds()
        self.poly1 = self.p.patch(x='x', y='y', source=coords, color="blue")
        self.model.points1_listeners.append(connect_update_coords(coords))
        coords = build_cds()
        self.poly2 = self.p.patch(x='x', y='y', source=coords, color="blue")
        self.model.points2_listeners.append(connect_update_coords(coords))

        controls = column(width_slider, self.submit_button)

        self.details = Div(text="")
        self.model.points1_listeners.append(self.update_details)
        self.model.recalc_extract()

        col = column(self.p, self.details)
        layout = row(controls, col)

        doc.add_root(layout)

    def update_details(self, x, y):
        self.details.text = \
            """
            <b>Aperture ID:</b> %s<br/>
            <b>Aperture Upper:</b> %0.2f<br/>
            <b>Aperture Lower:</b> %0.2f<br/>
            """ % (self.model.apnum, self.model.used_aper_upper, self.model.used_aper_lower)

    def result(self):
        """
        Get the result of the user interaction.

        Returns
        -------
        :class:`~NDAstroData`
        """
        return self.model.ndd_spec


def interactive_extract_spectra(apnum, aperture, ext, method, dispaxis,
                                x_axis_label="X", y_axis_label="Y"):
    model = ExtractSpectraModel(apnum, aperture, ext, method, dispaxis)
    vis = ExtractSpectraVisualizer(model, x_axis_label, y_axis_label)
    server.set_visualizer(vis)

    server.start_server()

    return vis.result()
