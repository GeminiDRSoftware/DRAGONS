import json
import scipy.ndimage as ni

import numpy as np

from scipy.signal import savgol_filter as _savgol_filter


def savgol_filter(s, ws, n, **kwargs):
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",category=FutureWarning)

        return _savgol_filter(s, ws, n, **kwargs)


def sg_filter(s1, winsize1=15, winsize2=11):
    s1m = ni.median_filter(s1, 11)
    #s1m = s1

    #winsize1 = 15
    #winsize2 = 11

    f1 = savgol_filter(s1m, winsize1, 3)

    f1_std = np.nanstd(s1-f1)

    if 0: # calculate weight
        f1_mask = np.abs(s1-f1) > 2.*f1_std
        f1_mask2 = ni.binary_opening(f1_mask, iterations=int(winsize2*0.2))
        f1_mask3 = ni.binary_closing(f1_mask2, iterations=int(winsize2*0.2))
        f1_mask4 = ni.binary_dilation(f1_mask3, iterations=int(winsize2))

        weight = ni.gaussian_filter(f1_mask4.astype("d"), winsize2)
    else:
        fd2 = savgol_filter(s1m, winsize1, 3, deriv=2)
        fd2_std = np.std(fd2)
        f1_mask = np.abs(fd2) > 2.*fd2_std

        f1_mask = f1_mask | (s1m < s1m.max()*0.4)

        f1_mask4 = ni.binary_dilation(f1_mask, iterations=int(winsize2))
        #f1_mask4[:300] = True
        #f1_mask4[-300:] = True
        weight = ni.gaussian_filter(f1_mask4.astype("d"), winsize2*.5)

    # find a region where deviation is significant

    if np.any(weight):
        weight/=weight.max()
        f2 = savgol_filter(s1m, winsize2, 5)
        f12 = f1*(1.-weight) + f2*weight
    else:
        f12 = f1
        weight = np.zeros(f12.shape)


    if 0:
        ax1.cla()
        ax2.cla()
        ax1.plot(f12)
        ax2.plot(s1 - f1, color="0.5")
        ax2.plot(s1 - f12)
        ax2.plot(weight * f1_std*2)

        ax2.set_ylim(-0.02, 0.02)

    return f12, f1_std


def sv_iter(s1m, maxiter=30, pad=8, winsize1=15, winsize2=11,
            return_mask=False):
    xi = np.arange(len(s1m))

    from scipy.interpolate import interp1d
    mm_old = None

    s1m_orig = s1m
    s1m = s1m.copy()
    mm = ~np.isfinite(s1m)
    s1m[mm] = interp1d(xi[~mm], s1m[~mm])(xi[mm])

    for i in range(maxiter):
        if len(s1m) < winsize1:
            break
        f12, f1_std = sg_filter(s1m, winsize1=winsize1, winsize2=winsize2)
        mm = (s1m - f12) < -2*f1_std
        mm[:pad] = False
        mm[-pad:] = False

        if mm_old is not None and np.all(mm_old == mm):
            break
        else:
            mm_old = mm

        s1m[mm] = interp1d(xi[~mm], f12[~mm])(xi[mm])


    try:
        if return_mask:
            f12, f1_std = sg_filter(s1m, winsize1=winsize1, winsize2=winsize2)
            mm = (s1m_orig - f12)# < -2*f1_std
            r = f12, mm
        else:
            r = f12
    except UnboundLocalError:
        raise RuntimeError("no sv fit is made")

    return r


def get(s1_sl, f12):
    kk = (s1_sl - f12)/f12
    kk[~np.isfinite(kk)] = 0.
    kkr = np.fft.rfft(kk)

    #kkr = np.fft.rfft(kk[512:512+1024])
    #kkr0 = np.zeros(len(kk), dtype=complex)
    #kkr0[:len(kkr)] = kkr
    #fig = plt.figure(3)
    #ax3 = fig.add_subplot(111)

    power_s = np.abs(kkr)**2
    power_s_max = np.max(power_s[2:])
    kkr[ power_s < power_s_max*0.4] = 0
    kkrr = np.fft.irfft(kkr)

    ww = (kkrr * f12 + f12)/f12
    return ww


def _get_finite_boundary_indices(s1):
    # select finite number only. This may happen when orders go out of
    # chip boundary.
    s1 = np.array(s1)
    s1[:4] = np.nan
    s1[-4:] = np.nan
    # k1, k2 = np.nonzero(np.isfinite(s1))[0][[0, -1]]

    # k1, k2 = np.nonzero(s1>0.)[0][[0, -1]]
    with np.errstate(invalid="ignore"):
        nonzero_indices = np.nonzero(s1 > 0.)[0]  # [[0, -1]]

    # # return meaningless indices if non-zero spectra is too short
    #  if len(nonzero_indices) < 5:
    #      return 4, 4

    try:
        k1, k2 = nonzero_indices[[0, -1]]
    except IndexError:
        return 0, 0
    k1 = max(k1, 4)
    k2 = min(k2, 2047-4)
    return k1, k2


def get_smooth_continuum(s, wvl=None):
    """
    wvl : required for masking our some absorption features
    """

    # from .trace_flat import get_finite_boundary_indices
    k1, k2 = _get_finite_boundary_indices(s)
    if k1 == k2:
        r = np.empty(len(s), dtype="d")
        r.fill(np.nan)
        return r

    sl = slice(k1, k2+1)

    #s1m = ni.median_filter(np.array(s1[sl]), 150)

    s1m = np.array(s[sl])

    if wvl is not None:
        wvl11 = np.array(wvl[sl])
        for ww1, ww2 in [(1.9112, 1.9119),
                         (1.91946, 1.92139),
                         (1.92826, 1.92901),
                         (1.92372, 1.92457)]:
            msk = (ww1 < wvl11) & (wvl11 < ww2)
            s1m[msk] = np.nan

    if len(s1m) > 351:
        f12 = sv_iter(s1m, winsize1=351, winsize2=91)
    elif len(s1m) > 25:
        f12 = sv_iter(s1m, winsize1=25, winsize2=11)
    else:
        f12 = None

    r = np.empty(len(s), dtype="d")
    r.fill(np.nan)
    if f12 is not None:
        r[sl] = f12
    return r


if __name__ == "__main__":
    band = "K"
    wvl_sol = json.load(open("calib/primary/20140525/SKY_SDC%s_20140525_0029.wvlsol_v1.json" % (band,)))["wvl_sol"]


    fig1 = figure(1)
    clf()
    ax1 = fig1.add_subplot(211)
    ax2 = fig1.add_subplot(212, sharex=ax1)

    s = json.load(open("calib/primary/20140525/ORDERFLAT_SDC%s_20140525_0074.json" % (band,)))

    specs = s["mean_order_specs"]





    #for s1 in specs:
    #    plot(s1)

    import astropy.io.fits as pyfits
    dd = pyfits.open("outdata/20140525/SDC%s_20140525_0016.spec.fits" % band)[0].data

    #ii = 0

    fig2 = figure(2)
    fig2.clf()
    ax3 = fig2.add_subplot(111)

    s1, a0v1, wvl1 = zip(specs, dd, wvl_sol)[5]
    if 1:

        f12 = get_smooth_continuum(s1, wvl1)
        ax3.plot(wvl1, s1 / f12)


        f12[f12 < np.nanmax(f12) * 0.05] = np.nan
        ax2.plot(wvl1, s1 / f12, zorder=0.2, color="0.5")
        ax1.plot(wvl1, s1, color="0.8")
        ax1.plot(wvl1, f12)

        #ax4 = fig2.add_subplot(212, sharex=ax3)
        a0v_ss = a0v1/f12
        a0v_ss[f12<np.nanmax(f12)*0.1] = np.nan
        ax3.plot(wvl1, a0v_ss/np.nanmax(a0v_ss[100:-100]))
        #ax3.set_ylim(0.7, 1.1)
        #ax3.plot(wvl1[sl], a0v_ss/np.median(a0v_ss)*0.02)
        #ax3.set_ylim(-0.02, 0.05)



if 0:
    #for s1, a0v1, wvl1 in zip(specs, dd, wvl_sol):
        #s1 = specs[ii]
        #a0v1 = dd[ii]

        from .trace_flat import get_finite_boundary_indices
        k1, k2 = get_finite_boundary_indices(s1)
        sl = slice(k1, k2+1)

        #s1m = ni.median_filter(np.array(s1[sl]), 150)

        s1m = np.array(s1[sl])

        wvl11 = np.array(wvl1[sl])
        for ww1, ww2 in [(1.9112, 1.9119),
                         (1.91946, 1.92139),
                         (1.92826, 1.92901),
                         (1.92372, 1.92457)]:
            msk = (ww1 < wvl11) & (wvl11 < ww2)
            s1m[msk] = np.nan


        f12 = sv_iter(s1m, winsize1=351, winsize2=91)


        if 0:
            ww = get(s1[sl], f12)
            ax2.plot(ww)
            s1m2 = s1m/ww
            f12 = sv_iter(s1m2, winsize1=351, winsize2=71)

            f12[f12 < f12.max() * 0.05] = np.nan
            #ax2.plot(s1[sl] / f12, zorder=0.2, color="0.5")
            #ax1.plot(s1[sl], color="0.8")
            #ax1.plot(f12)
        else:
            s1m2 = s1m
