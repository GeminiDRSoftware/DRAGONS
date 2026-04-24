# %%

# import astropy.io.fits as pyfits
import numpy as np
import astrodata, gemini_instruments
import matplotlib.pyplot as plt

# %%

# fn = "N20240429S0204_K.fits"
band = "H"
# band = "K"

fnroot = "N20240429S0204"

# fn = f"{fnroot}_{band}.fits"
# ad = astrodata.open(fn)
# adlist = [ad]

fnout = f"../../test_i2/{fnroot}_{band}_arc.fits"
adout = astrodata.open(fnout)

from geminidr.igrins2.primitives_igrins import get_ref_spectra, get_ref_data
from geminidr.igrins2.procedures.match_orders import match_specs_w_shift, match_specs

orders_ref, s_list_ref = get_ref_spectra(band)

# %%

ext = adout[0]
spec1d = ext.SPEC1D

s_list_ = spec1d["specs"]
s_list = [np.array(s, dtype=np.float64) for s in s_list_]


# to plot the order match result for visual inspection.
delta_indx, displacements = match_specs_w_shift(s_list_ref, s_list, frac_thresh=0.3)

norm = np.nanmax(s_list) / 1.e1
norm_ref = np.nanmax(s_list_ref) / 1.e1

x = np.arange(2048)
norders = len(displacements)

fig, axs = plt.subplots(norders, 1, num=1, clear=True)

for ax, (o, shift) in zip(axs, displacements):
    s_ref = s_list_ref[o + delta_indx]
    s = s_list[o]
    ax.plot(x, s_ref/norm_ref)
    ax.plot(x + shift, s/norm + 0.5)
    ax.set_ylim(-0.2, 1.5)
    ax.annotate(str(orders_ref[o+delta_indx]), (0, 1), xytext=(5, -5),
                xycoords="axes fraction", textcoords="offset points",
                ha="left", va="top", )

# %%

# This is to check if the shift_from_ref column in the spec1d is reasonable.
ext = adout[0]
spec1d = ext.SPEC1D

spec1d_orders = spec1d["orders"]
center_indx = len(spec1d_orders) // 2
selected_orders = spec1d_orders[max(center_indx - 5, 0):min(center_indx + 5, len(spec1d_orders))]

s_list_ = spec1d["specs"]
s_list = [np.array(s, dtype=np.float64) for s in s_list_]

s_dict = dict(zip(spec1d_orders, zip(s_list, spec1d["shift_from_ref"])))
s_dict_ref = dict(zip(orders_ref, s_list_ref))

norm = np.nanmax(s_list) / 1.e1
norm_ref = np.nanmax(s_list_ref) / 1.e1

x = np.arange(2048)

import matplotlib.pyplot as plt
fig, axs = plt.subplots(len(selected_orders), 1, num=1, clear=True)

for ax, o in zip(axs, selected_orders):
    s_ref = s_dict_ref[o]
    s, shift = s_dict[o]
    ax.plot(x + shift, s_ref/norm_ref)
    ax.plot(x, s/norm + 0.5)
    ax.set_ylim(-0.2, 1.5)
    ax.annotate(str(o), (0, 1), xytext=(5, -5),
                xycoords="axes fraction", textcoords="offset points",
                ha="left", va="top", )

# %%


if False:
    # simple test

    delta_indx = match_specs(s_list_ref, s_list, frac_thresh=0.3)
    assert delta_indx == dict(H=0, K=-1)[band]

    delta_indx = match_specs(s_list_ref, s_list[1:], frac_thresh=0.3)
    assert delta_indx == dict(H=0, K=-1)[band] + 1

