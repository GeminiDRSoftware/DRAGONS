import numpy as np
import scipy.ndimage as ni

def find_feature_mask_simple(s_msk, sigma=1, ax=None, x_values=None):
    """
    sigma is estimated globally.
    """
    # find emission features from observed spec.

    filtered_spec = s_msk - ni.median_filter(s_msk, 15)
    #filtered_spec = ni.gaussian_filter1d(filtered_spec, 0.5)
    #smoothed_std = get_smoothed_std(filtered_spec,
    #                                rad=3, smooth_length=3)
    std = np.nanstd(filtered_spec)

    with np.errstate(invalid="ignore"):
        for i in [0, 1]:
            std = filtered_spec[np.abs(filtered_spec)<3*std].std()

        emission_feature_msk_ = filtered_spec > sigma*std

    #emission_feature_msk_ = ni.binary_closing(emission_feature_msk_)
    emission_feature_msk = ni.binary_opening(emission_feature_msk_,
                                             iterations=1)

    if ax is not None:
        if x_values is None:
            x_values = np.arange(len(s_msk))

        #ax.plot(x_values, s_msk)
        ax.plot(x_values, filtered_spec)
        #ax.plot(x_values, smoothed_std)
        ax.axhline(sigma*std)
        ax.plot(x_values[emission_feature_msk],
                emission_feature_msk[emission_feature_msk],
                "ys", mec="none")

    return emission_feature_msk

def find_peaks(s, sigma=3, ax=None):

        emission_feature_msk = find_feature_mask_simple(s, sigma=sigma, ax=ax)

        import scipy.ndimage as ni
        emission_feature_label, label_max = ni.label(emission_feature_msk)
        com_list = ni.center_of_mass(s, emission_feature_label,
                                     range(1, label_max+1))

        from .fit_gaussian import fit_gaussian_simple
        x = np.arange(len(s))
        sol_list = []
        for com in com_list:
            sol = fit_gaussian_simple(x, s, com, sigma_init=2., do_plot=False)
            sol_list.append(sol)

        #center_list = np.array([sol[0][0] for sol in sol_list])
        #width_list = np.array([sol[0][1] for sol in sol_list])
        #height_list = np.array([sol[0][2] for sol in sol_list])

        return [sol_[0] for sol_ in sol_list]






##### below is used by what???



dx = 400


def fitgaussian(s, lines, sigma_init=1.5, do_plot=False):
    """ Return (height, x, width)
    the gaussian parameters of a 1D distribution found by a fit"""

    lines = np.array(lines)
    xmin = max(int(np.floor(lines[0]))-10, 0)
    xmax = min(int(np.ceil(lines[-1]))+10, len(s))

    xx = np.arange(xmin, xmax)
    yy = s[xmin:xmax]
    ymax = yy.max()
    yy = yy / ymax
    d_centers0 = lines - lines[0]

    def _gauss0(params):
        """ Returns a gaussian function with the given parameters"""
        center, sigma, height, background = params

        y_models = []
        for d_center in d_centers0:
            y_models.append(np.exp(-(((xx - (center + d_center))/sigma)**2*0.5)))
        return height*np.array(y_models).sum(axis=0) + background
        #return (height*np.array(s).sum(axis=0) + background)

    def _gauss(params):
        return np.sum((yy - _gauss0(params))**2)

        #return (height*np.array(s).sum(axis=0) + background)

    params0 = np.array([lines[0], sigma_init, 1., 0.])
    params_min = np.array([xmin, 0., 0, 0.])
    params_max = np.array([xmax, 2*sigma_init, 2., 1.])
    from scipy.optimize import fmin_tnc
    sol_ = fmin_tnc(_gauss, params0,
                    bounds=zip(params_min, params_max),
                    approx_grad=True, disp=0,
                    epsilon=0.1)

    sol_[0][2] = ymax * sol_[0][2]
    if do_plot:
        fig = plt.figure(10)
        fig.clf()
        ax = fig.add_subplot(111)

        ax.plot(xx, yy)
        ax.plot(xx, _gauss0(sol_[0]))
        ax.vlines(sol_[0][0]+d_centers0, 0, 1)
        # print d_centers0
    return sol_



# standard deviation filter
from .stddev_filter import window_stdev

import scipy.ndimage as ni

def get_smoothed_std(smoothed, rad=3, smooth_length=25):
    windowed_stddev = window_stdev(smoothed, rad)
    smoothed_std = ni.median_filter(windowed_stddev,
                                    smooth_length, mode="wrap")
    #smoothed_std = np.empty_like(s, dtype="f")
    #smoothed_std.fill(np.nan)

    #smoothed_std[rad-1:-rad] = smoothed_std_

    return smoothed_std


def find_emission_features(s):
    from scipy import signal
    #kernel = signal.ricker(19, 2)
    #smoothed = np.convolve(s[msk1], kernel, mode="same")

    #smoothed = s[msk1] - ni.median_filter(s,15)[msk1]
    smoothed = s - ni.median_filter(s,15)

    smoothed_std = get_smoothed_std(smoothed, rad=3, smooth_length=25)
    emission_msk = smoothed > 1.*smoothed_std

    # emission_msk_ = np.zeros([len(s)], dtype=bool)
    # # the indices need to be checked.
    # emission_msk_[rad-1:-rad] = smoothed[rad-1:-rad] > 1.*smoothed_std
    # emission_msk = ni.binary_opening(emission_msk_)

    return dict(smoothed=smoothed,
                smoothed_std=smoothed_std,
                emission_msk=emission_msk)


def identify_line_candidates(emission_feature_msk,
                             ohlines_pixel,
                             nearby_lines_indices,
                             dpixel=2,
                             minpix=None,
                             maxpix=None,
                             ):
    """
    emission_feature_msk : 1d mask where detected features are True.
    line_group_list : list of line groups. A line group contain single line, double etc.
    """

    if minpix is None:
        minpix = 0
    if maxpix is None:
        maxpix = len(emission_feature_msk)

    #ohlines_pixel = um2pixel(ohline_um)

    line_group_idx_list = []
    line_group_list = []

    for line_group_indices in nearby_lines_indices:
        line_group = ohlines_pixel[line_group_indices]
        if (minpix < min(line_group)) and (max(line_group) < maxpix - 1):
            line_group_idx_list.append(line_group_indices)
            line_group_list.append(line_group)

    line_match_msk = identify_matching_mask(emission_feature_msk,
                                            line_group_list,
                                            dpixel=dpixel)

    matched_indices = itertools.compress(line_group_idx_list,
                                         line_match_msk)
    matched_pixelcoords = itertools.compress(line_group_list,
                                             line_match_msk)


    return list(matched_indices), list(matched_pixelcoords)


def identify_matching_mask(emission_feature_msk,
                           line_group_list,
                           dpixel=2):

    # for lines groups, check if there is matching emisison feature
    #nearby_lines_matched_indices = []
    if dpixel:
        mask2 = ni.binary_dilation(emission_feature_msk,
                                   iterations=dpixel)
    else:
        mask2 = emission_feature_msk

    line_match_msk = []
    for line_group in line_group_list:
        i_indices = np.floor(line_group).astype("i")
        #plot(i_indices, np.zeros_like(i_indices), "o")
        if np.any(mask2[i_indices]) or np.any(mask2[i_indices+1]):
            line_match_msk.append(True)
        else:
            line_match_msk.append(False)

    return line_match_msk


def find_feature_mask(s_msk, sigma=1, ax=None, x_values=None):
    # find emission features from observed spec.

    filtered_spec = s_msk - ni.median_filter(s_msk, 15)
    filtered_spec = ni.gaussian_filter1d(filtered_spec, 0.5)
    smoothed_std = get_smoothed_std(filtered_spec,
                                    rad=3, smooth_length=3)
    emission_feature_msk_ = filtered_spec > sigma*smoothed_std
    #emission_feature_msk_ = ni.binary_closing(emission_feature_msk_)
    emission_feature_msk = ni.binary_opening(emission_feature_msk_,
                                             iterations=1)

    if ax is not None:
        if x_values is None:
            x_values = np.arange(len(s_msk))

        #ax.plot(x_values, s_msk)
        ax.plot(x_values, filtered_spec)
        ax.plot(x_values, smoothed_std)

        ax.plot(x_values[emission_feature_msk],
                emission_feature_msk[emission_feature_msk],
                "ys", mec="none")

    return emission_feature_msk
