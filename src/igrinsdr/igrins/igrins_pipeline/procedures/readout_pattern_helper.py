import numpy as np

from ..utils.image_combine import image_median
from .readout_pattern import pipes, apply as apply_pipe
from ..bg_mask_helper import make_background_mask_from_combined

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
