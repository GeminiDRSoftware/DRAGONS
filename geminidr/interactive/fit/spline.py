
from bokeh.layouts import column, row
from bokeh.models import Tabs, Panel, ColumnDataSource, Button
from bokeh.plotting import figure

from gempy.library import astromodels

from geminidr.interactive import interactive


class SplineTab:
    def __init__(self, view, shape, pixels, masked_data, order, weights, grow=0, recalc_button=False,
                 **spline_kwargs):
        if isinstance(masked_data, list) and len(masked_data[0]) > 1:
            # multiplot, no select tools
            tools = "pan,wheel_zoom,box_zoom,reset"
        else:
            tools = "pan,wheel_zoom,box_zoom,reset,lasso_select,box_select,tap"
            masked_data = [masked_data]

        self.view = view
        self.recalc_button = recalc_button
        self.pixels = pixels
        self.masked_data = masked_data
        self.order = order
        self.grow=grow
        self.weights = weights
        self.spline_kwargs = spline_kwargs
        # self.model = SplineModel(shape, pixels, masked_data, weights, self.order, 1, 1)  #order, niter, grow)
        # Create a blank figure with labels
        self.p = figure(plot_width=600, plot_height=500,
                        title='Interactive Spline',
                        tools=tools,
                        x_axis_label='x', y_axis_label='y')

        self.scatter_sources = []
        self.scatters = []
        self.line_sources = []
        self.lines = []

        i = 0
        self.max_plots = 5
        self.mod = 1
        if len(masked_data) > self.max_plots:
            self.mod = int(len(masked_data) / self.max_plots)
        for md in masked_data:
            if (i % self.mod) == 0:
                # only do some of the lines to give a sense of it - if we did them all the ui falls over
                scatter_source = ColumnDataSource({'x': pixels, 'y': md})
                scatter = self.p.scatter(x='x', y='y', source=scatter_source, color="black", radius=5)
                self.scatter_sources.append(scatter_source)
                self.scatters.append(scatter)
                # we'll add data to the line sources whenever a fit is recalculated
                line_source = ColumnDataSource({'x': [], 'y': []})
                line = self.p.line(x='x', y='y', source=line_source, color='red', level='overlay')
                self.line_sources.append(line_source)
                self.lines.append(line)
            i = i+1

        # Setup Controls
        recalc = self.preview_fit
        if self.recalc_button:
            # if we have a recalc button, do not recalc live
            recalc = None
        order_slider = interactive.build_text_slider("Order", self.order, 1, 1, 50,
                                                     self, "order", recalc)
        grow_slider = interactive.build_text_slider("Growth radius", self.grow, 1, 0, 10,
                                                    self, "grow", recalc)
        if self.recalc_button:
            # we want a recalc button, so we make one and
            # wire it up with a modal wait message.  The
            # reason we would want a recalc button is
            # because the recalc is too slow to do live
            self.button = Button(label="Recalculate")
            controls = column(order_slider, grow_slider, self.button)

            self.view.make_modal(self.button, "<b>Recalculating Fit....</b>")

            # this nested fn allows the modal ui to display before the
            # expensive call to self.preview_fit()
            def recalculate_button_fn():
                self.button.disabled = True

                def fn():
                    self.preview_fit()
                    self.button.disabled=False
                self.view.do_later(fn)

            self.button.on_click(recalculate_button_fn)
        else:
            # just doing things live, no need for button
            controls = column(order_slider)

        # controls on left, plot on right
        self.component = row(controls, column(self.p))

        # do an initial fit
        self.preview_fit()

    def preview_fit(self):
        idx = 0
        i = 0
        self.spline_kwargs['grow'] = self.grow
        for md, w in zip(self.masked_data, self.weights):
            if (i % self.mod) == 0:
                scatter_source = ColumnDataSource({'x': self.pixels, 'y': md})
                scatter = self.p.scatter(x='x', y='y', source=scatter_source, color="black", radius=5)
                self.scatter_sources.append(scatter_source)
                self.scatters.append(scatter)

                spline = astromodels.UnivariateSplineWithOutlierRemoval(self.pixels, md,
                                                                        order=self.order, w=w,
                                                                        **self.spline_kwargs)
                self.line_sources[idx].data = {'x': self.pixels, 'y': spline(self.pixels)}
                idx = idx+1
            i = i+1

    def fitted_data(self):
        for spline in self.get_splines():
            yield spline(self.pixels)

    def get_splines(self):
        for md, w in zip(self.masked_data, self.weights):
            spline = astromodels.UnivariateSplineWithOutlierRemoval(self.pixels, md, order=self.order, w=w,
                                                                    **self.spline_kwargs)
            yield spline


class SplineVisualizer(interactive.PrimitiveVisualizer):
    def __init__(self, all_shapes, all_pixels, all_masked_data, all_orders, all_weights, config, recalc_button=False,
                 **spline_kwargs):
        super().__init__(config=config)

        self.recalc_button = recalc_button
        self.all_shapes = all_shapes
        self.all_pixels = all_pixels
        self.all_masked_data = all_masked_data
        self.all_orders = all_orders
        self.all_weights = all_weights
        self.spline_kwargs = spline_kwargs

        self.spline_tabs = []

    def visualize(self, doc):
        """
        Start the bokeh document using this visualizer.

        This call is responsible for filling in the bokeh document with
        the user interface.

        Parameters
        ----------
        doc : :class:`~bokeh.document.Document`
            bokeh document to draw the UI in
        """
        super().visualize(doc)
        tabs = Tabs()

        idx = 1
        for shape, pixels, masked_data, order, weights in zip(self.all_shapes, self.all_pixels, self.all_masked_data,
                                                              self.all_orders, self.all_weights):
            tab = SplineTab(self, shape, pixels, masked_data, order, weights, recalc_button=self.recalc_button,
                            **self.spline_kwargs)
            self.spline_tabs.append(tab)
            panel = Panel(child=tab.component, title='Spline %s' % idx)
            idx = idx+1
            tabs.tabs.append(panel)

        col = column(tabs, self.submit_button)
        col.sizing_mode = 'scale_width'
        layout = col
        doc.add_root(layout)

    def fitted_data(self):
        fd = []
        for st in self.spline_tabs:
            fd.append([fdi for fdi in st.fitted_data()])
        return fd

    def get_splines(self):
        for st in self.spline_tabs:
            for spline in st.get_splines():
                yield spline
