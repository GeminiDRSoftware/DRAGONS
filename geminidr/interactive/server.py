import sys

# DRAGONS
from astropy.table import Table

import astrodata

# bokeh basics
from bokeh.plotting import figure

# bokeh widgets
from bokeh.layouts import row
from bokeh.models import Slider, Column, Button
from bokeh.server.server import Server

# numpy
import numpy as np

# Offsets, may be updated by bokeh controls
from gempy.library import astromodels

# holder of the plot
p = None
lines = dict()
splines = dict()
scatters = dict()


def make_plot():
    global p, splinex
    print("In make_plot")
    # Create a blank figure with labels
    ht = 600
    p = figure(plot_width=600, plot_height=ht,
               title='Interactive Spline',
               x_axis_label='X', y_axis_label='Y')
    p.scatter(spline_wave, spline_zpt, color="blue", radius=50)
    recalc_spline()

    return p


def order_slider_handler(attr, old, new):
    global spline_order
    if old in lines and old != new:
        lines[old].visible = False
        scatters[old].visible = False
    spline_order = new
    recalc_spline()


def button_handler(stuff):
    server.io_loop.stop()


def bkapp(doc):
    global spline_order

    order_slider = Slider(start=1, end=10, value=spline_order, step=1, title="Order")
    order_slider.on_change("value", order_slider_handler)

    button = Button(label="Submit")
    button.on_click(button_handler)

    controls = Column(order_slider, button)

    p = make_plot()

    layout = row(controls, p)

    doc.add_root(layout)


spline = None
splinex = None
x = None

spline_wave = None
spline_zpt = None
spline_zpt_err = None
spline_order = None
spline_ext = None


def recalc_spline():
    global lines, p, spline, splinex, spline_wave, spline_zpt, spline_zpt_err, spline_order, spline_ext

    if spline_order in lines:
        lines[spline_order].visible = True
        scatters[spline_order].visible = True
        spline = splines[spline_order]
        return

    spline = astromodels.UnivariateSplineWithOutlierRemoval(spline_wave.value, spline_zpt.value,
                                                            w=1. / spline_zpt_err.value,
                                                            order=spline_order)
    # save it in case we come back to this order later
    splines[spline_order] = spline

    knots, coeffs, degree = spline.tck
    sensfunc = Table([knots * spline_wave.unit, coeffs * spline_zpt.unit],
                     names=('knots', 'coefficients'))
    spline_ext.SENSFUNC = sensfunc
    splinex = np.linspace(min(spline_wave), max(spline_wave), spline_ext.shape[0])

    scatters[spline_order] = p.scatter(spline_wave[spline.mask], spline_zpt[spline.mask], color="black", radius=50)

    lines[spline_order] = p.line(splinex, spline(splinex), color="red")


def interactive_spline(ext, wave, zpt, zpt_err, order):
    global spline, splinex, spline_wave, spline_zpt, spline_zpt_err, spline_order, spline_ext

    spline_ext = ext
    spline_wave = wave
    spline_zpt = zpt
    spline_zpt_err = zpt_err
    spline_order = order

    start_server()

    return spline


server = None


def start_server():
    global server

    # Setting num_procs here means we can't touch the IOLoop before now, we must
    # let Server handle that. If you need to explicitly handle IOLoops then you
    # will need to use the lower level BaseServer class.
    server = Server({'/': bkapp}, num_procs=1)
    server.start()

    # Setting num_procs here means we can't touch the IOLoop before now, we must
    # let Server handle that. If you need to explicitly handle IOLoops then you
    # will need to use the lower level BaseServer class.
    print('Opening Bokeh application on http://localhost:5006/')

    server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()
