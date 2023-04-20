import numpy as np

def iter_order(pp):
    xx = np.arange(2048)
    yy = np.tile(xx.reshape((-1, 1)), (1, 2048))

    for o, k in pp:
        y_top = k["top"](xx)
        y_bottom = k["bottom"](xx)

        ymax = int(np.ceil(y_top.max()))
        ymin = max(int(np.floor(y_bottom.min())), 0)

        sl = slice(ymin, ymax+1)
        ys = yy[sl]
        m = (ys < y_top) & (ys > y_bottom)

        yield o, sl, m


if __name__ == '__main__':
    import astropy.io.fits as pyfits
    from trace_flat import table_to_poly

    hdul = pyfits.open("/home/jjlee/git_personal/IGRINSDR/SDCH_20220301_0011_lampstack.fits")

    pp = table_to_poly(hdul["SLITEDGE"].data)

    import numpy as np

    mask = np.zeros((2048, 2048), dtype=int)
    for o, sl, m in iter_order(pp):
        print(o)
        mask[sl][m] = o + 1
