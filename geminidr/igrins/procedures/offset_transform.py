from __future__ import print_function

import numpy as np
from scipy.signal import correlate

__all__ = ["get_offset_transform_between_two_specs"]


def _get_offset_transform(thar_spec_src, thar_spec_dst):

    offsets = []
    cor_list = []
    center = 2048/2.

    for s_src, s_dst in zip(thar_spec_src, thar_spec_dst):
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",category=FutureWarning)

            cor = correlate(s_src, s_dst, mode="same")

        cor_list.append(cor)
        offset = center - np.argmax(cor)
        offsets.append(offset)

    # from skimage.measure import ransac, LineModel
    from .skimage_measure_fit import ransac, LineModel

    xi = np.arange(len(offsets))
    data = np.array([xi, offsets]).T
    model_robust, inliers = ransac(data,
                                   LineModel, min_samples=3,
                                   residual_threshold=2, max_trials=100)

    outliers_indices = xi[inliers == False]
    offsets2 = [o for o in offsets]
    for i in outliers_indices:
        # reduce the search range for correlation peak using the model
        # prediction.
        ym = int(model_robust.predict_y(i))
        x1 = int(max(0, (center - ym) - 20))
        x2 = int(min((center - ym) + 20 + 1, 2048))
        # print i, x1, x2
        ym2 = center - (np.argmax(cor_list[i][x1:x2]) + x1)
        # print ym2
        offsets2[i] = ym2

    def get_offsetter(o):
        def _f(x, o=o):
            return x+o
        return _f
    sol_list = [get_offsetter(offset_) for offset_ in offsets2]

    return dict(sol_type="offset",
                sol_list=sol_list,
                offsets_orig=offsets,
                offsets_revised=offsets2)


def _get_offset_transform_between_two_specs(ref_spec, tgt_spec):

    orders_ref = ref_spec["orders"]
    s_list_ref = ref_spec["specs"]

    orders_tgt = tgt_spec["orders"]
    s_list_tgt = tgt_spec["specs"]

    s_list_tgt = [np.array(s) for s in s_list_tgt]

    orders_intersection = set(orders_ref).intersection(orders_tgt)
    orders_intersection = sorted(orders_intersection)

    def filter_order(orders, s_list, orders_intersection):
        s_dict = dict(zip(orders, s_list))
        s_list_filtered = [s_dict[o] for o in orders_intersection]
        return s_list_filtered

    s_list_ref_filtered = filter_order(orders_ref, s_list_ref,
                                       orders_intersection)
    s_list_tgt_filtered = filter_order(orders_tgt, s_list_tgt,
                                       orders_intersection)

    offset_transform = _get_offset_transform(s_list_ref_filtered,
                                             s_list_tgt_filtered)

    return orders_intersection, offset_transform


def get_offset_transform_between_two_specs(ref_spec, tgt_spec):
    intersected_orders, d = _get_offset_transform_between_two_specs(ref_spec,
                                                                    tgt_spec)

    offsetfunc_map = dict(zip(intersected_orders, d["sol_list"]))

    return offsetfunc_map

