import numpy as np

igrins_orders = {}
igrins_orders["H"] = range(99, 122)
igrins_orders["K"] = range(72, 94)


# def get_ordered_line_data(identified_lines, orders=None):
#     """
#     identified_lines : dict of lines with key of orders_i.
#     lines[0] : list of x positions
#     lines[1] : list of wavelengths
#     """
#     x_list, y_list, z_list = [], [], []
#     # x:pixel, y:order, z:wavelength

#     if orders is None:
#         o_l = [(i, oh)  for i, oh in identified_lines.items()]
#     else:
#         o_l = zip(orders, identified_lines)

#     for o, oh in sorted(o_l):

#         x_list.extend(oh[0])
#         y_list.extend([o] * len(oh[0]))
#         z_list.extend(np.array(oh[1])*o)

#     return map(np.array, [x_list, y_list, z_list])


def check_dx1(ax, x, y, dx, gi, mystd):

    grid_z2 = gi(x, y, dx)
    im = ax.imshow(grid_z2, origin="lower", aspect="auto",
                   extent=(gi.xi[0], gi.xi[-1], gi.yi[0], gi.yi[-1]),
                   interpolation="none")
    im.set_clim(-mystd, mystd)

def check_dx2(ax, x, y, dx):
    m1 = dx >= 0
    ax.scatter(x[m1], y[m1], dx[m1]*10, color="r")
    m2 = dx < 0
    ax.scatter(x[m2], y[m2], -dx[m2]*10, color="b")

class GridInterpolator(object):
    def __init__(self, xi, yi, interpolator="mlab"):
        self.xi = xi
        self.yi = yi
        self.xx, self.yy = np.meshgrid(xi, yi)
        self._interpolator = interpolator


    def _grid_scipy(self, xl, yl, zl):
        from scipy.interpolate import griddata
        x_sample = 256
        z_gridded = griddata(np.array([yl*x_sample, xl]).T,
                             np.array(zl),
                             (self.yy*x_sample, self.xx),
                             method="linear")
        return z_gridded


    def __call__(self, xl, yl, zl):
        if self._interpolator == "scipy":
            z_gridded = self._grid_scipy(xl, yl, zl)
        elif self._interpolator == "mlab":
            from matplotlib.mlab import griddata
            try:
                z_gridded = griddata(xl, yl, zl, self.xi, self.yi)
            except Exception:
                z_gridded = self._grid_scipy(xl, yl, zl)

        return z_gridded

def show_grided_image(ax, gi, xl, yl, zl, orders):
    import matplotlib

    extent = [0, 2048, orders[0]-1, orders[-1]+1]

    z_max, z_min = zl.max(), zl.min()
    norm = matplotlib.colors.Normalize(vmin=z_min, vmax=z_max)

    z_gridded = gi(xl, yl, zl)

    ax.imshow(z_gridded, aspect="auto", origin="lower", interpolation="none",
              extent=extent, norm=norm)

    ax.scatter(xl, yl, 10, c=zl, norm=norm)
    ax.set_xlim(0, 2048)
    ax.set_ylim(orders[0]-1, orders[-1]+1)

def fit_2dspec(xl, yl, zl, x_degree=4, y_degree=3,
               x_domain=None, y_domain=None):
    from astropy.modeling import fitting
    # Fit the data using astropy.modeling
    if x_domain is None:
        x_domain = [min(xl), max(xl)]
    # more room for y_domain??
    if y_domain is None:
        #y_domain = [orders[0]-2, orders[-1]+2]
        y_domain = [min(yl), max(yl)]
    from astropy.modeling.polynomial import Chebyshev2D
    p_init = Chebyshev2D(x_degree=x_degree, y_degree=y_degree,
                         x_domain=x_domain, y_domain=y_domain)
    f = fitting.LinearLSQFitter()

    p = f(p_init, xl, yl, zl)

    for i in [0]:
        dd = p(xl, yl) - zl
        m = np.abs(dd) < 3.*dd.std()
        p = f(p, xl[m], yl[m], zl[m])

    return p, m


def get_dx(xl, yl, zl, orders, p):
    dlambda_order = {}
    for o in orders:
        wvl_minmax = p([0, 2047], [o]*2) / o
        dlambda = (wvl_minmax[1] - wvl_minmax[0]) / 2048.
        dlambda_order[o] = dlambda

    dlambda = [dlambda_order[y1] for y1 in yl]
    dx = (zl - p(xl, yl))/yl/dlambda

    return dx


def get_dx_from_identified_lines(p, identified_lines):
    dpix_list = {}
    for i, oh in sorted(identified_lines.items()):
        oh = identified_lines[i]
        o = i #orders[i]
        wvl = p(oh[0], [o]*len(oh[0])) / o

        wvl_minmax = p([0, 2047], [o]*2) / o
        dlambda = (wvl_minmax[1] - wvl_minmax[0]) / 2048.

        dpix_list[i] = (oh[1] - wvl)/dlambda

    return dpix_list

def check_fit(fig, xl, yl, zl, p, orders,
              identified_lines):

    xi = np.linspace(0, 2048, 256+1)
    yi = np.linspace(orders[0]-1, orders[-1]+1, len(orders)*10)
    #yi = orders
    gi = GridInterpolator(xi, yi)

    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 2, height_ratios=[4,1])
    ax = fig.add_subplot(gs[0,0])
    show_grided_image(ax, gi, xl, yl, zl, orders)

    dx = get_dx(xl, yl, zl, orders, p)


    ax2 = fig.add_subplot(gs[0,1], sharey=ax)
    check_dx1(ax2, xl, yl, dx, gi, mystd=0.5)
    check_dx2(ax2, xl, yl, dx)

    ax3 = fig.add_subplot(gs[1,1], sharex=ax2)
    for i, oh in sorted(identified_lines.items()):
        oh = identified_lines[i]
        o = i #orders[i]
        wvl = p(oh[0], [o]*len(oh[0])) / o

        wvl_minmax = p([0, 2047], [o]*2) / o
        dlambda = (wvl_minmax[1] - wvl_minmax[0]) / 2048.

        ax3.plot(oh[0], (oh[1] - wvl)/dlambda, "o-")

    ax.axhline(orders[0], linestyle=":", color="0.5")
    ax.axhline(orders[-1], linestyle=":", color="0.5")
    ax2.axhline(orders[0], linestyle=":", color="0.5")
    ax2.axhline(orders[-1], linestyle=":", color="0.5")

    ax.set_xlim(0, 2048)
    ax2.set_xlim(0, 2048)
    ax3.set_ylim(-1, 1)

    ax.set_xlabel("x-pixel")
    ax.set_ylabel("order")

    ax3.set_ylabel(r"$\Delta\lambda$ [pixel]")


def check_fit_simple(fig, xl, yl, zl, p, orders):
    #import matplotlib.pyplot as plt

    xi = np.linspace(0, 2048, 256+1)
    yi = np.linspace(orders[0]-1, orders[-1]+1, len(orders)*10)
    #yi = orders
    gi = GridInterpolator(xi, yi)

    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 2, height_ratios=[4,1])
    ax = fig.add_subplot(gs[0,0])
    show_grided_image(ax, gi, xl, yl, zl, orders)

    dx = (zl - p(xl, yl))

    mystd = dx.std()
    mystd = dx[np.abs(dx) < 3.*mystd].std()

    ax2 = fig.add_subplot(gs[0,1], sharey=ax)
    check_dx1(ax2, xl, yl, dx, gi, mystd=3*mystd)
    check_dx2(ax2, xl, yl, dx)


# if __name__ == "__main__":

#     utdate="20140316"
#     band = "K"

#     import json
#     ohlines = {}
#     for b_ in ["H","K"]:
#         ohlines_ = json.load(open("ohlines_%s_%s_r2.json" % (b_,utdate)))
#         ohlines[b_] = dict((int(k), (v["pixel"], v["wavelength"])) \
#                            for (k, v) in ohlines_.items())

#     for b_ in ["K"]:
#         hitran_ = json.load(open("../hitran_bootstrap_%s_%s.json" % (b_,utdate)))
#         hitran_ = dict((int(i_), s) for i_,s in hitran_.items())

#         for k, o in enumerate(igrins_orders["K"]):
#             if o not in hitran_: continue
#             kk = ohlines[b_][k]
#             v = hitran_[o]
#             kk[0].extend(v["pixel"])
#             kk[1].extend(v["wavelength"])

#         extra_ = json.load(open("../extra_%s_%s.json" % (b_,utdate)))
#         for k, v in extra_.items():
#             kk = ohlines[b_].setdefault(int(k), [[],[]])
#             kk[0].extend(v["pixel"])
#             kk[1].extend(v["wavelength"])


#     # identified_lines : dict of dict(pixel, wavelenth, weight) for each order.
#     orders = igrins_orders[band]
#     identified_lines = dict((orders[i], s) for i,s in ohlines[band].items())


#     xl, yl, zl = get_ordered_line_data(identified_lines)
#     # xl : pixel
#     # yl : order
#     # zl : wvl * order

#     x_domain = [0, 2047]
#     y_domain = [orders[0]-2, orders[-1]+2]
#     p, m = fit_2dspec(xl, yl, zl, x_degree=4, y_degree=3)

#     import matplotlib.pyplot as plt
#     fig = plt.figure(figsize=(12, 7))

#     #id_lines = dict((orders[o], s) for o, s in identified_lines.items())
#     check_fit(fig, xl, yl, zl, p, orders, identified_lines)
#     fig.tight_layout()


#     postfix = "%s_%s" % (utdate, band)
#     fig.savefig("ecfit_%s_fig1.png" % postfix)

#     if 0:
#         xx = np.arange(0, 2048)
#         wvl_list = []
#         figure()
#         for o in igrins_orders[band]:
#             oo = np.empty_like(xx)
#             oo.fill(o)
#             wvl = p(xx, oo)/o
#             plot(xx, wvl)
#             wvl_list.append(list(wvl))
#         import json
#         json.dump(wvl_list,
#                   open("wvl_sol_ohlines_%s_%s.json" % (band, utdate),"w"))
