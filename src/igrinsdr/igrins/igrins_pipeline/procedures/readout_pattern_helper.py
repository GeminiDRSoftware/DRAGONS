import numpy as np

from .readout_pattern import pipes, apply as apply_pipe


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
