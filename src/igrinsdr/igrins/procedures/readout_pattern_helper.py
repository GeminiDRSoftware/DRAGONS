import numpy as np

from .image_combine import image_median
from .readout_pattern import pipes, apply as apply_pipe
from .bg_mask_helper import make_background_mask_from_combined

# from igrins.procedures.procedure_dark import (apply_rp_2nd_phase,
#                                               apply_rp_1st_phase)


def apply_rp_1st_phase(d, mask=None, subtract_col_wise_bias_c64=False):
    """
    Note that this only apply
     - amp_wise_bias_r2
     - p64_0th_order

    which is used for flat images
    """
    if mask is None:
        mask = np.zeros(d.shape, dtype=bool)
    else:
        mask = mask.copy()

    mask[:4] = True
    mask[-4:] = True

    keys = ['amp_wise_bias_r2', 'p64_0th_order']

    if subtract_col_wise_bias_c64:
        keys.append('col_wise_bias_c64')

    p = [pipes[k] for k in keys]

    return apply_pipe(d, p, mask=mask)


# def apply_rp_1st_phase(d, mask=None):
#     if mask is None:
#         mask = np.zeros(d.shape, dtype=bool)
#     else:
#         mask = mask.copy()

#     mask[:4] = True
#     mask[-4:] = True

#     p = [pipes[k] for k in ['amp_wise_bias_r2',
#                             'p64_0th_order',
#                             'col_wise_bias_c64']]

#     return apply_pipe(d, p, mask=mask)


def apply_rp_2nd_phase(d, mask=None):
    if mask is None:
        mask = np.zeros(d.shape, dtype=bool)
    else:
        mask = mask.copy()

    mask[:4] = True
    mask[-4:] = True

    p = [pipes[k] for k in ['p64_1st_order',
                            'col_wise_bias_c64',
                            'amp_wise_bias_r2',
                            'col_wise_bias']]

    return apply_pipe(d, p, mask=mask)


def apply_rp_3rd_phase(d):
    p = [pipes[k] for k in ['p64_per_column',
                            'row_wise_bias',
                            'amp_wise_bias_c64']]

    return apply_pipe(d, p)


def sub_bg_from_slice(d, bg_y_slice):
    s = np.nanmedian(d[bg_y_slice], axis=0)
    return d - s[np.newaxis, :]


def make_initial_flat_cube(data_list, mode, bg_y_slice):
    """
    data_list : list of images with guard removed
    mode : 0 - no pattern removal, 1 - 1st pahse
    """

    # mode = 1 for H, mode 0 for K. Using mode 1 for K is not okay
    # due to large background (at least for Gemini data)
    # We do not use bias_mask at this stage.

    cards = []

    cube = np.array(data_list)

    if mode == 1:
        # bg_mask, bg_dict = dh.make_background_mask(cube)
        c = image_median(cube)
        bg_mask, bg_dict = make_background_mask_from_combined(c)
        cards.append(("IGRFLAT0", bg_dict))
        cube2 = [apply_rp_1st_phase(d1, bg_mask) for d1 in cube]
    # elif mode == 2:
    #     cube2 = [apply_rp_2nd_phase(d1 - bbg) + bbg for d1 in cube]
    else:
        cube2 = cube

    cube3 = [sub_bg_from_slice(d1, bg_y_slice) for d1 in cube2]

    return cards, cube3

from .readout_pattern_guard import remove_pattern_from_guard
from .ro_pattern_fft import (get_amp_wise_rfft,
                             make_model_from_rfft)

def select_k_to_remove(c, n=2):
    ca = np.abs(c)
    # k = np.median(ca, axis=0)[1:]  # do no include the 1st column
    k = np.percentile(ca, 95, axis=0)[1:]  # do no include the 1st column
    # print(k[:10])
    x = np.arange(1, 1 + len(k))
    msk = (x < 5) | (15 < x)  # only select k from 5:15

    # polyfit with 5:15 data
    p = np.polyfit(np.log10(x[msk]), np.log10(k[msk]), 2,
                   w=1./x[msk])
    # p = np.polyfit(np.log10(x[msk][:30]), np.log10(k[msk][:30]), 2,
    #                w=1./x[msk][:30])
    # print(p)

    # sigma from last 256 values
    ss = np.std(np.log10(k[-256:]))

    # model from p with 3 * ss
    y = 10.**(np.polyval(p, np.log10(x)))

    di = 5
    dly = np.log10(k/y)[di:15]

    # select first two values above 3 * ss
    ii = np.argsort(dly)
    yi = [di + i1 + 1for i1 in ii[::-1][:n] if dly[i1] > 3 * ss]

    return yi


def remove_pattern(data_minus, mask=None, remove_level=1,
                   remove_amp_wise_var=True):

    d1 = remove_pattern_from_guard(data_minus)

    if remove_level == 2:
        d2 = apply_rp_2nd_phase(d1, mask=mask)
    elif remove_level == 3:
        d2 = apply_rp_2nd_phase(d1, mask=mask)
        d2 = apply_rp_3rd_phase(d2)
    else:
        d2 = d1

    if remove_amp_wise_var:
        c = get_amp_wise_rfft(d2)

        ii = select_k_to_remove(c)
        print(ii)
        # ii = [9, 6]

        new_shape = (32, 64, 2048)
        mm = np.zeros(new_shape)

        for i1 in ii:
            mm1 = make_model_from_rfft(c, slice(i1, i1+1))
            mm += mm1[:, np.newaxis, :]

        ddm = mm.reshape((-1, 2048))

        return d2 - ddm

    else:
        return d2
