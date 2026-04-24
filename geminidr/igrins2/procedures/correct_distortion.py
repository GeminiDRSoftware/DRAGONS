import numpy as np
import astropy.io.fits as pyfits
from scipy.interpolate import interp1d
from numpy.polynomial import Polynomial

# refactored version is avaiable at utisl/rectipy.py
# May need to use that version.

class ShiftX(object):
    AXIS = 1

    def __init__(self, shiftx_map):
        self.shiftx_map = shiftx_map

        iy0, ix0 = np.indices(shiftx_map.shape)

        self.ix = ix0 - shiftx_map
        self.ix0 = ix0[0]

    def __call__(self, d0):

        d0_acc = np.add.accumulate(d0, axis=self.AXIS)

        # Do not use assume_sorted, it results in incorrect interpolation.
        d0_acc_shft = np.array([interp1d(xx, dd,
                                         bounds_error=False,
                                         )(self.ix0) \
                                for xx, dd in zip(self.ix, d0_acc)])

        d0_shft = np.empty_like(d0_acc_shft)
        d0_shft[:,1:] = d0_acc_shft[:,1:]-d0_acc_shft[:,:-1]

        return d0_shft


def get_rectified_2dspec(data, order_map, ap, # bottom_up_solutions,
                         conserve_flux=False, height=0):

    # sl = slice(0, 2048), slice(0, 2048)
    data = data.copy()
    # resume from context does not work unless copying the data

    msk = (order_map > 0) & np.isfinite(data)
    data[~msk] = 0.

    data[~np.isfinite(data)] = 0.

    from scipy.interpolate import interp1d

    def get_shifted(data, normalize=False, height=0):

        acc_data = np.add.accumulate(data, axis=0)
        ny, nx = acc_data.shape
        yy = np.arange(ny)
        xx = np.arange(0, nx)

        # Do not use assume_sorted, it results in incorrect interpolation.
        d0_acc_interp = [interp1d(yy, dd,
                                  bounds_error=False)
                         for dd in acc_data.T]

        if height == 0:
            max_height = 0

            # for c in bottom_up_solutions:
            for o in ap.orders:
                bottom = ap.apcoeffs[o](xx, 0)
                up = ap.apcoeffs[o](xx, 1)
                # bottom = Polynomial(c[0][1])(xx)
                # up = Polynomial(c[1][1])(xx)

                _height = up - bottom
                max_height = max(int(np.ceil(max(_height))), max_height)

            height = max_height

        d_factor = 1./height

        bottom_up_list = []

        # for c in bottom_up_solutions:
        #     bottom = Polynomial(c[0][1])(xx)
        #     up = Polynomial(c[1][1])(xx)
        for o in ap.orders:
            bottom = ap.apcoeffs[o](xx, 0)
            up = ap.apcoeffs[o](xx, 1)
            dh = (up - bottom) * d_factor  # * 0.02

            bottom_up = zip(bottom - dh, up)
            bottom_up_list.append(bottom_up)

        d0_shft_list = []

        # for c in cent["bottom_up_solutions"]:
        for bottom_up in bottom_up_list:
            # p_bottom = Polynomial(c[0][1])
            # p_up = Polynomial(c[1][1])

            # height = p_up(xx) - p_bottom(xx)
            # bottom_up = zip(p_bottom(xx), p_up(xx))

            # max_height = int(np.ceil(max(height)))

            yy_list = [np.linspace(y1, y2, height+1)
                       for (y1, y2) in bottom_up]
            d0_acc_shft = np.array([intp(yy) for yy, intp
                                    in zip(yy_list, d0_acc_interp)]).T

            # d0_shft = np.empty_like(d0_acc_shft)
            d0_shft = d0_acc_shft[1:, :]-d0_acc_shft[:-1, :]
            if normalize:
                d0_shft = d0_shft/[yy[1]-yy[0] for yy in yy_list]
            d0_shft_list.append(d0_shft)

        return d0_shft_list

    d0_shft_list = get_shifted(data, height=height)
    msk_shft_list = get_shifted(msk, normalize=conserve_flux, height=height)

    return d0_shft_list, msk_shft_list, height


if __name__ == "__main__":
    fn = "SDCH_20240425_0067_arc.fits"
    import astrodata
    ad_sky = astrodata.open(fn)
    from igrinsdr.igrins.procedures.apertures import Apertures
    ap = Apertures(ad_sky[0].SLITEDGE)
    from igrinsdr.igrins.procedures.shifted_images import ShiftedImages

    data = ad_sky[0].data
    # data = ad_sky[0].WVLCOR
    order_map = ad_sky[0].ORDERMAP

    data = ShiftedImages.from_table(ad[0].WVLCOR).image
    d, m = get_rectified_2dspec(data, order_map, ap, # bottom_up_solutions,
                                conserve_flux=False, height=0)


    # d = pyfits.open("../outdata/20140525/SDCH_20140525_0016.combined_image.fits")[0].data

    # msk = np.isfinite(pyfits.open("../outdata/20140525/SDCH_20140525_0042.combined_image.fits")[0].data)

    # d[~msk] = np.nan

    # slitoffset = pyfits.open("../calib/primary/20140525/SKY_SDCH_20140525_0029.slitoffset_map.fits")[0].data

    # d[~np.isfinite(slitoffset)] = np.nan


    # # now shift
    # msk = np.isfinite(d)
    # d0 = d.copy()
    # d0[~msk] = 0.

    # shiftx = ShiftX(slitoffset)

    # d0_shft = shiftx(d0)
    # msk_shft = shiftx(msk)

    # variance = d0
    # variance_shft = shiftx(variance)

    # d0_flux = d0_shft / msk_shft
