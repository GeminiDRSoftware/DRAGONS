from bokeh.plotting import figure, show, output_file
import matplotlib.pyplot as plt

import numpy as np

def dgsplot_matplotlib(ad, aperture):
    plot_data = _setup_dgsplots(ad, aperture)

    plt.title(plot_data['title'])
    plt.xlabel(plot_data['xaxis'])
    plt.ylabel(plot_data['yaxis'])
    plt.plot(plot_data['wavelength'], plot_data['data'])
    plt.show()

    return

def dgsplot_bokeh(ad, aperture):
    plot_data = _setup_dgsplots(ad, aperture)

    # Default is to write this file where the dgsplot executable script is located.
    # This might not be writable by the user.  Setting it manually will write in
    # current directory.

    output_file('dgsplot.html')

    p = figure(title=plot_data['title'], x_axis_label=plot_data['xaxis'],
               y_axis_label=plot_data['yaxis'], width=1000, height=600)
    p.title.text_font_size = '15pt'
    p.title.align = 'center'
    p.xaxis.axis_label_text_font_size = '12pt'
    p.yaxis.axis_label_text_font_size = '12pt'
    p.line(plot_data['wavelength'], plot_data['data'])
    show(p)

    return

def _setup_dgsplots(ad, aperture):
    setup_plot = {}
    data = ad[aperture-1].data
    setup_plot['data'] = data
    setup_plot['wavelength'] = ad[aperture-1].wcs(np.arange(data.size)).astype(np.float32)
    setup_plot['wave_units'] = ad[aperture-1].wcs.output_frame.unit[0]
    setup_plot['signal_units'] = ad[aperture-1].hdr["BUNIT"]

    setup_plot['title'] = f'{ad.filename} - Aperture {aperture}'
    setup_plot['xaxis'] = f'Wavelength ({setup_plot["wave_units"]})'
    setup_plot['yaxis'] = f'Signal ({setup_plot["signal_units"]})'

    return setup_plot