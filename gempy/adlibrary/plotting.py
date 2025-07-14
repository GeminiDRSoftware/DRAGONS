from bokeh.plotting import figure, show, output_file
import matplotlib.pyplot as plt
from geminidr.gemini.lookups import DQ_definitions as DQ

import numpy as np

COLORS = ['blue', 'orange', 'green', 'red', 'purple',
          'brown', 'pink', 'grey', 'olive', 'cyan']



def dgsplot_matplotlib(ad, aperture, ignore_mask=False, kwargs=None):
    plot_data = _setup_dgsplots(ad, aperture, ignore_mask)

    plt.title(plot_data['title'])
    plt.xlabel(plot_data['xaxis'])
    plt.ylabel(plot_data['yaxis'])
    for i, (x, y) in enumerate(zip(plot_data['wavelength'], plot_data['data'])):
        plt.plot(x, y, color=COLORS[i % len(COLORS)], **kwargs)
    plt.show()

    return


def dgsplot_bokeh(ad, aperture, ignore_mask=False):
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
    exts_to_plot = [ext for ext, apnum in zip(ad, ad.hdr.get("APERTURE")) if apnum == aperture]
    if not exts_to_plot:
        if not (0 < aperture <= len(ad)):
            raise ValueError(f"Aperture {aperture} is invalid "
                             f"({ad.filename} has {len(ad)} extensions)")
        exts_to_plot = [ad[aperture-1]]

    setup_plot = {'data': [], 'wavelength': []}
    wave_units = set()
    signal_units = set()
    for ext in exts_to_plot:
        nworld_axes = ext.wcs.output_frame.naxes
        if nworld_axes != 1:
            raise ValueError(f"{ad.filename} has {nworld_axes} world axes")

        pix = np.arange(ext.data.shape[-1])
        if ext.data.ndim == 1:
            setup_plot['data'].append(ext.data if ext.mask is None else
                                      np.where(ext.mask==0, ext.data, np.nan))
            setup_plot['wavelength'].append(ext.wcs(pix).astype(np.float32))
        else:
            setup_plot['data'].extend(ext.data if ext.mask is None else np.where(ext.mask==0, ext.data, np.nan))
            grid = np.meshgrid(pix, np.arange(ext.data.shape[0]),
                               sparse=True, indexing='xy')
            setup_plot['wavelength'].extend(ext.wcs(*grid).astype(np.float32))

        wave_units.add(ext.wcs.output_frame.unit[0])
        signal_units.add(ext.hdr["BUNIT"])

    if len(wave_units) > 1:
        raise ValueError(f"{ad.filename} has different wavelength units in the "
                         "extensions to be plotted")
    if len(signal_units) > 1:
        raise ValueError(f"{ad.filename} has different signal units in the "
                         f"extensions to be plotted")

    setup_plot['wave_units'] = wave_units.pop()
    setup_plot['signal_units'] = signal_units.pop()
    setup_plot['title'] = f'{ad.filename} - Aperture {aperture}'
    setup_plot['xaxis'] = f'Wavelength ({setup_plot["wave_units"]})'
    setup_plot['yaxis'] = f'Signal ({setup_plot["signal_units"]})'

    return setup_plot