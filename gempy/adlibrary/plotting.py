from bokeh.plotting import figure, show, output_file
import matplotlib.pyplot as plt
from geminidr.gemini.lookups import DQ_definitions as DQ

import numpy as np

COLORS = ['blue', 'orange', 'green', 'red', 'purple',
          'brown', 'pink', 'grey', 'olive', 'cyan']



def dgsplot_matplotlib(ad, aperture, ignore_mask):
    plot_data = _setup_dgsplots(ad, aperture, ignore_mask)

    plt.title(plot_data['title'])
    plt.xlabel(plot_data['xaxis'])
    plt.ylabel(plot_data['yaxis'])
    for i, (x, y) in enumerate(zip(plot_data['wavelength'], plot_data['data'])):
        plt.plot(x, y, color=COLORS[i % len(COLORS)])
    plt.show()

    return


def dgsplot_bokeh(ad, aperture, ignore_mask):
    plot_data = _setup_dgsplots(ad, aperture, ignore_mask)

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
    for i, (x, y) in enumerate(zip(plot_data['wavelength'], plot_data['data'])):
        p.line(x, y, color=COLORS[i % len(COLORS)])
    show(p)

    return


def _setup_dgsplots(ad, aperture, ignore_mask):
    if not (0 < aperture <= len(ad)):
        raise ValueError(f"Aperture {aperture} is invalid "
                         f"({ad.filename} has {len(ad)} extensions)")
    data = ad[aperture-1].data
    mask = ad[aperture-1].mask
    if mask is None or ignore_mask:  # avoid having to do this later
        mask = np.zeros_like(data, dtype=DQ.datatype)
    wcs = ad[aperture-1].wcs
    nworld_axes = wcs.output_frame.naxes
    if nworld_axes != 1:
        raise ValueError(f"{ad.filename} has {nworld_axes} world axes")
    pix = np.arange(data.shape[-1])

    setup_plot = {}
    if data.ndim == 1:
        setup_plot['data'] = [np.where(mask==0, data, np.nan)]
        setup_plot['wavelength'] = [wcs(pix).astype(np.float32)]
    else:
        setup_plot['data'] = np.where(mask==0, data, np.nan)
        grid = np.meshgrid(pix, np.arange(data.shape[0]),
                           sparse=True, indexing='xy')
        setup_plot['wavelength'] = wcs(*grid)

    setup_plot['wave_units'] = wcs.output_frame.unit[0]
    setup_plot['signal_units'] = ad[aperture-1].hdr["BUNIT"]
    setup_plot['title'] = f'{ad.filename} - Aperture {aperture}'
    setup_plot['xaxis'] = f'Wavelength ({setup_plot["wave_units"]})'
    setup_plot['yaxis'] = f'Signal ({setup_plot["signal_units"]})'

    return setup_plot