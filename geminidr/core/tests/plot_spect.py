"""
Plots that are used in tests for the `primitives_spect.py` module.
"""

from copy import copy

import matplotlib as mpl

from astropy import visualization as vis
from matplotlib import gridspec
from matplotlib import pyplot as plt

def plot_():
    """

    Returns
    -------

    """
    palette = copy(plt.cm.cividis)
    palette.set_bad('r', 0.75)

    norm = vis.ImageNormalize(data[~data.mask],
                              stretch=vis.LinearStretch(),
                              interval=vis.PercentileInterval(97))

    fig = plt.figure(
        num=f"Slit Response from MEF - {_ad.filename}", figsize=(9, 9), dpi=110)

        gs = gridspec.GridSpec(nrows=2, ncols=2, figure=fig)

        # Display raw mosaicked data and its bins
        ax1 = fig.add_subplot(gs[0])
        ax1.imshow(data, cmap=palette, origin='lower', vmin=norm.vmin, vmax=norm.vmax)

        ax1.set_title("Mosaicked Data\n and Spectral Bins", fontsize=10)
        ax1.set_xlim(-1, data.shape[1])
        ax1.set_xticks([])
        ax1.set_ylim(-1, data.shape[0])
        ax1.set_yticks(bin_center)
        ax1.set_yticklabels(
            ["Bin {}".format(i) for i in range(len(bin_center))],
            fontsize=6)
        ax1.tick_params(axis=u'both', which=u'both', length=0)
        _ = [ax1.spines[s].set_visible(False) for s in ax1.spines]
        _ = [ax1.axhline(b, c='w', lw=0.5) for b in bin_limits]

        # Display non-smoothed bins
        ax2 = fig.add_subplot(gs[1])
        ax2.imshow(binned_data, cmap=palette, origin='lower')

        ax2.set_title("Binned, smoothed\n and normalized data ", fontsize=10)
        ax2.set_xlim(0, data.shape[1])
        ax2.set_xticks([])
        ax2.set_ylim(0, data.shape[0])
        ax2.set_yticks(bin_center)
        ax2.set_yticklabels(
            ["Bin {}".format(i) for i in range(len(bin_center))],
            fontsize=6)
        ax2.tick_params(axis=u'both', which=u'both', length=0)
        _ = [ax2.spines[s].set_visible(False) for s in ax2.spines]
        _ = [ax2.axhline(b, c='w', lw=0.5) for b in bin_limits]

        # Display reconstructed slit response
        vmin = slit_response_data.min()
        vmax = slit_response_data.max()

        ax3 = fig.add_subplot(gs[2])
        ax3.imshow(slit_response_data, cmap=palette, origin='lower',
                   vmin=vmin, vmax=vmax)

        ax3.set_title("Reconstructed\n Slit response", fontsize=10)
        ax3.set_xlim(0, data.shape[1])
        ax3.set_xticks([])
        ax3.set_ylim(0, data.shape[0])
        ax3.set_yticks([])
        ax3.tick_params(axis=u'both', which=u'both', length=0)
        _ = [ax3.spines[s].set_visible(False) for s in ax3.spines]

        # Display extensions
        sub_axs = []
        sub_gs = gridspec.GridSpecFromSubplotSpec(
            nrows=len(_ad), ncols=1, subplot_spec=gs[3], hspace=0.03)

        # The [::-1] is needed to put the fist extension in the bottom
        for i, ext in enumerate(slit_response_ad[::-1]):

            data, mask, variance = _transpose_if_needed(
                ext.data, ext.mask, ext.variance, transpose=dispaxis == 1)

            data = np.ma.masked_array(data, mask=mask)

            ax = fig.add_subplot(sub_gs[i])

            ax.imshow(data, origin="lower", vmin=vmin, vmax=vmax,
                      cmap=palette)
            ax.set_xlim(0, data.shape[1])
            ax.set_xticks([])
            ax.set_ylim(0, data.shape[0])
            ax.set_yticks([data.shape[0] // 2])

            ax.set_yticklabels(
                [f"Ext {len(slit_response_ad) - i - 1}"], fontsize=6)

            _ = [ax.spines[s].set_visible(False) for s in ax.spines]

            if i == 0:
                ax.set_title("Multi-extension\n Slit Response Function")

            sub_axs.append(ax)

        fig.tight_layout(rect=[0, 0, 0.95, 1], pad=0.5)
        plt.savefig("slit_illum_{}.png".format(_ad.data_label()))