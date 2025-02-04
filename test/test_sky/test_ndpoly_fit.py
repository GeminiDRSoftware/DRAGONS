from igrinsdr.igrins.primitives_igrins import Igrins, NdPolyNamed

import astrodata
import igrins_instruments

# fn = "N20240429S0204_K.fits"
band = "H"
# band = "K"

fnroot = "N20240429S0204"

# fn = f"{fnroot}_{band}.fits"
# ad = astrodata.open(fn)
# adlist = [ad]

fnout = f"../../test_i2/{fnroot}_{band}_arc.fits"
ad = astrodata.open(fnout)
adinputs = [ad]

if True: # FIXME The code below needs to be reviewed.

    # we want draw qa plots for the volume fit. To access the intermediate
    # results, we will call _prepareVolumeFit.

    tbl_linefit = ad[0].LINEFIT

    dft = Igrins._prepareVolumFit(tbl_linefit)
    dd = dft[dft["badmask"] == 0].reset_index()
    # dd = dft.reset_index()

    names = ["pixel", "order", "slit"]
    orders = [3, 2, 1]

    # because the offset at slit center should be 0, we divide the
    # offset by slit_pos, and fit the data then multiply by slit_pos.

    cc0 = dd["slit_center"] - 0.5

    # points0, scalar0 is not used, but for the debugging purpose.

    # 3d points : x-pixel, order, location on the slit
    points0 = dict(zip(names, [dd["pixel0"],
                               dd["order"],
                               cc0]))
    # scalar is offset of the measured line from the location at slit center.
    scalar0 = dd["offset"]

    # load the volumefit coeffs

    params = ad[0].VOLUMEFIT_COEFFS.to_pandas()
    # ad[0].VOLUMEFIT_COEFFS.to_pandas()
    params = params.set_index(names)

    poly, coeffs = NdPolyNamed.from_pandas(params)

    #values = dict(zip(names, [pixels, orders, slit_pos]))
    offsets_fitted = poly.multiply(points0, coeffs)
    doffsets = scalar0 - offsets_fitted * cc0

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.scatter(dd["pixel0"], doffsets, c=cc0.values, cmap="gist_heat")

