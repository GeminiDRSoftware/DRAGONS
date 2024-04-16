import sys
import astrodata, ghost_instruments
from ghostdr.ghost.primitives_ghost_spect import GHOSTSpect
from ghostdr.ghost.polyfit import GhostArm, SlitView
from gempy.library import tracing
from astropy.io import fits
import numpy as np
from scipy import optimize
from matplotlib import pyplot as plt

RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
END = '\033[0m'


def get_fibre_separation(p):
    # Returns the separation (in microns) between fibres in the slitviewer
    ad, ad_slitflat = p.streams['main']
    #p.getProcessedSlitFlat(refresh=False)
    #slitflat = p._get_cal(ad, 'processed_slitflat')
    #ad_slitflat = astrodata.open(slitflat)
    slitv_fn = p._get_slitv_polyfit_filename(ad)
    slitvpars = astrodata.open(slitv_fn)
    sv = SlitView(None, ad_slitflat[0].data, slitvpars.TABLE[0],
                  mode=ad.res_mode())
    slit_models = sv.model_profile(flat_image=ad_slitflat[0].data)
    sep_pixels = slit_models[ad.arm()].separation.value
    # sep_pixels is *unbinned* pixels
    sep_microns = sep_pixels * sv.microns_pix / sv.binning
    return sep_microns


def get_xpars(p):
    # Returns the locations (rows) of the orders on the echellogram
    ad = p.streams['main'][0]
    poly_xmod = p._get_polyfit_filename(ad, 'xmod')
    print(f"Using XMOD {poly_xmod}")
    xpars = astrodata.open(poly_xmod)
    return xpars[0].data


def get_initial_spatmod(p):
    # Returns the current SPATMOD parameters
    ad = p.streams['main'][0]
    poly_spat = p._get_polyfit_filename(ad, 'spatmod')
    print(f"Using SPATMOD {poly_spat}")
    spatpars = astrodata.open(poly_spat)
    return spatpars[0].data


def do_fit(arm, params, args, maxiter=5, sigma=3):
    # Fit with iterative sigma-clipping
    # params are (orders, y_values (i.e., x), values to fit, ydeg, xdeg, sigma)
    ngood = args[0].size
    for niter in range(maxiter):
        result, ierr = optimize.leastsq(arm.fit_resid, params, args)
        if ierr not in (1, 2, 3, 4):
            print(f"{RED}IERR {ierr}{END}")
        result = result.reshape(params.shape)
        fitted_values = arm.evaluate_poly(result, (args[1], args[0]))
        resid = fitted_values - args[2]
        resid_fitting = resid[args[-1] < 1e5]
        print("MAX", resid_fitting.max(), "RMS", resid_fitting.std())
        bad = np.where(np.logical_and(abs(resid) > sigma * resid_fitting.std(),
                                      args[-1] < 1e5))[0]
        if bad.size:
            ngood -= bad.size
            print(f"REJECTING {len(bad)} POINTS (now {ngood})")
            args[-1][bad] = 1e5
        else:
            break
    return result


def main(ad, ad_slitflat, flat_bin=8):
    ny, nx = ad[0].shape
    p = GHOSTSpect([ad, ad_slitflat])
    arm = GhostArm(arm=ad.arm(), mode=ad.res_mode())
    mid_order = (arm.m_min + arm.m_max) // 2
    read_noise = ad.read_noise()[0]
    sep_microns = get_fibre_separation(p)
    print(f"Using fibre separation of {sep_microns} microns")
    spatpars = get_initial_spatmod(p)
    xpars = get_xpars(p)
    xx, wave, blaze = arm.spectral_format(xparams=xpars)
    xmap = arm.adjust_x(xx, ad[0].data) + ny // 2

    # Bin the data horizontally
    averaged = np.median(ad[0].data.reshape(ny, nx // flat_bin, flat_bin),
                         axis=2)

    # Find all groups of 17 fibres
    found_groups = []
    for i, col in enumerate(averaged.T):
        if i % 2:  # speed up by factor of 2, still a good result
            continue
        x = (i + 0.5) * flat_bin - 0.5
        fwidth = 0.2 * sep_microns / arm.evaluate_poly(
            spatpars, (np.array([x]), np.array([mid_order])))
        print(".", end="")
        sys.stdout.flush()
        order_locations = xmap[:, int(x)]
        #print("X = ", x, "WIDTH = ", fwidth)
        #print(order_locations)

        # Find all peaks (fibres) and group them
        peaks = tracing.find_peaks(
            col, widths=np.arange(0.75, 1.26, 0.05) * fwidth,
            pinpoint_index=-1, variance=np.full_like(col, 10*read_noise**2))[0]
        order_sep = np.median(np.diff(peaks)) * 2
        grouped_peaks = []
        this_group = [peaks[0]]
        for p, s in zip(peaks[1:], np.diff(peaks)):
            if s < order_sep:
                this_group.append(p)
            else:
                grouped_peaks.append(this_group)
                this_group = [p]

        # Find groups of 17 fibres and determine the location of the central
        # fibre (which is close to the slit centre) and mean fibre separation
        for group in grouped_peaks:
            #print(len(group), group)
            if len(group) < 17:
                continue

            # Determine a more accurate centre for each fibre
            for i, g in enumerate(group):
                p = int(g + 0.5)
                data = col[p-1:p+2]
                p += 0.5 * (data[2] - data[0]) / (3 * data[1] - data.sum())
                group[i] = p

            pos_mean = np.mean(group)
            diffs = np.diff(group)
            sep_mean, sep_rms = diffs.mean(), diffs.std()
            #print(order, pos_mean, sep_mean, sep_rms)
            scale = sep_microns / sep_mean
            found_groups.append([pos_mean, x, scale,
                                 sep_rms / sep_mean * scale, False])
    print("\n")

    # Fit the XMOD. Again, we're fitting to the cental fibre, not the slit
    # centre, but that's close enough
    xmod_to_fit = []
    for i, (y, x, _, _, _) in enumerate(found_groups):
        expected_positions = xmap[:, int(x)]
        offsets = abs(expected_positions - y)
        order = np.argmin(offsets)
        if offsets[order] < 10:
            xmod_to_fit.append([order + arm.m_min, x, y - ny // 2])
            found_groups[i][-1] = True
    xmod_to_fit = np.asarray(xmod_to_fit).T
    ydeg = xpars.shape[0] - 1
    xdeg = xpars.shape[1] - 1
    xmod_sigma = np.ones((xmod_to_fit.shape[-1],))
    new_xpars = do_fit(arm, xpars, args=(xmod_to_fit[0], xmod_to_fit[1],
                                         xmod_to_fit[2], ydeg, xdeg,
                                         xmod_sigma), sigma=5)

    print("New XMOD created. No need to update unless warnings follow.")
    print(new_xpars)
    xx, wave, blaze = arm.spectral_format(xparams=new_xpars)
    new_xmap = xx + ny // 2  # no need to adjust

    xplot = np.arange(0, nx, 64)
    fig, ax = plt.subplots()
    ax.plot(xmod_to_fit[1], xmod_to_fit[2] + ny // 2, 'ko')
    for order in range(arm.m_min, arm.m_max+1):
        yplot = arm.evaluate_poly(xpars, (xplot, np.full_like(xplot, order)))
        ax.plot(xplot, yplot + ny // 2, 'k:')
        yplot = xmap[order-arm.m_min, xplot.astype(int)]
        ax.plot(xplot, yplot, 'k-')
        yplot = new_xmap[order-arm.m_min, xplot.astype(int)]
        ax.plot(xplot, yplot, 'r-')

    # See if any of our previously-found groups of 17 match to expected
    # positions *now* but didn't before... indicates the XMOD needs updating
    spatmod_to_fit = []
    for (y, x, scale, rms, found) in found_groups:
        expected_positions = xmap[:, int(x)]
        offsets = abs(expected_positions - y)
        order = np.argmin(offsets)
        if offsets[order] < 10:
            spatmod_to_fit.append([order + arm.m_min, y, x, scale, rms])
            if not found:
                print("{RED}Order {order+arm.m_min} "
                      "was not previously matched: Update XMOD{END}")
    plt.show()

    print("\nFitting SPATMOD")
    orders, y, x, scales, sigma = np.asarray(spatmod_to_fit).T
    ydeg = spatpars.shape[0] - 1
    xdeg = spatpars.shape[1] - 1
    new_spatpars = do_fit(arm, spatpars, args=(orders, x, scales,
                                               ydeg, xdeg, sigma))
    print(new_spatpars)

    # Plot the old and new SPATMODS and report the change in each order
    fig, ax = plt.subplots()
    ax.plot(x, scales + (orders - arm.m_min) * 50, 'ko')
    for order in range(arm.m_min, arm.m_max+1):
        dy = (order - arm.m_min) * 50
        s_orig = arm.evaluate_poly(spatpars,
                                   (xplot, np.full_like(xplot, order)))
        ax.plot(xplot, s_orig+dy, 'k-')
        s_new = arm.evaluate_poly(new_spatpars,
                                  (xplot, np.full_like(xplot, order)))
        ax.plot(xplot, s_new+dy, 'r-')
        ratio_min, ratio_max = np.min(s_new / s_orig), np.max(s_new / s_orig)
        deviation = max(abs(ratio_min - 1), abs(ratio_max - 1))
        if deviation > 0.05:
            color = RED
        elif deviation > 0.025:
            color = YELLOW
        else:
            color = GREEN
        print(f"{color}Order {order}: {ratio_min:6.3f} {ratio_max:6.3f}{END}")
    plt.show()

    fits.writeto("new_xmod.fits", new_xpars, overwrite=True)
    fits.writeto("new_spatmod.fits", new_spatpars, overwrite=True)


if __name__ == "__main__":
    try:
        flat_filename = sys.argv[1]
        slitflat_filename = sys.argv[2]
    except IndexError:
        print("Usage: mkspatmod.py <flat_filename> <slitflat_filename>")
    flat = astrodata.open(flat_filename)
    slitflat = astrodata.open(slitflat_filename)
    assert ({'FLAT', 'PROCESSED'}.issubset(flat.tags) and
            'SLIT' not in flat.tags), f"{flat.filename} is not a FLAT"
    assert flat.res_mode() == "std", "Must be run on SR data"
    assert {'FLAT', 'PROCESSED', 'SLIT'}.issubset(slitflat.tags), \
        f"{slitflat.filename} is not a SLITFLAT"
    
    main(flat, slitflat)
    
