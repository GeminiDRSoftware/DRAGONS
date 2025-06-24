import numpy as np

from .readout_pattern import pipes, apply as apply_pipe
from .readout_pattern_helper import (remove_pattern_from_guard,
                                     apply_rp_1st_phase,
                                     apply_rp_2nd_phase,
                                     apply_rp_3rd_phase,
                                     sub_bg_from_slice)
from .bg_mask_helper import (make_background_mask_from_combined,
                             image_median,
                             )

def remove_readout_pattern_flat_off(data_list, band,
                                    flat_off_pattern_removal="global_median",
                                    rp_remove_mode="auto"
                                    ):
    # During the process we will stack the image and make a mask file, which
    # will be used to subtract the pattern. FIXME Maybe the Mask file could be
    # made in the separate step and we do image-by-image correction given the
    # mask is provided as a parameter..

    # IGRINS2 data before 202410 suffer from readout pattern noise which is
    # mostly suppressed during the mainternance in 202410. Removing these
    # patterns are tricky even for flat off due to thermal background. Usually,
    # information in the guard columns should be good enough (which is the case
    # for IGRINS1), however, IGRINS2's guard columns have not very usable
    # values. Therefore, we fall back to use "global_median" mode. For IGRINS,
    # you should use "guard" mode.

    # With IGRINS2, the mean value of background region is not 0 which is likely due to
    # incorrect reference pixel correction in the control software.
    # We use relatively background free region and subtract the constance background from the image.
    # bg_y_slice is used for the estimation of the background.


    # flat_off_pattern_removal="global_median" # guard' | 'none'
    # rp_remove_mode="auto"

    # band = adlist[0].band()

    if band == "K":
        _rp_remove_mode = 1 # apply_rp_1st_phase : note that due to the thermal
                            # background in K, this will have some effect.
                            # FIXME we should incorporate order mask from flat
                            # on image.s
        bg_y_slice = slice(-256, None)
    else:
        _rp_remove_mode = 2 # apply_rp_2n_phase
        bg_y_slice = slice(None, 512)

    if rp_remove_mode == "auto":
        rp_remove_mode = _rp_remove_mode

    assert rp_remove_mode in [0, 1, 2]

    # elif rp_remove_mode == "remove_bg":
    #     rp_remove_mode = 1
    # else:
    #     rp_remove_mode = 0

    # # we assume there is only one dataextension per astrodata which should be true for IGRINS1/2.
    # data_list = [ad[hdunum].data for ad in adlist]

    # we remove initial readout pattern.
    if flat_off_pattern_removal == "none":
        pass
    elif flat_off_pattern_removal == "guard":
        # subtract patterns using the guard column
        data_list = [remove_pattern_from_guard(d) for d in data_list]
    elif flat_off_pattern_removal == "global_median":
        p = [pipes["p64_global_median"]]
        data_list = [apply_pipe(d, p) for d in data_list]

    cube = np.array(data_list)
    cards = []

    # we further try to remove pattern with an adequate background mask for thermal background.
    if rp_remove_mode == 0:
        cube2 = cube
    else:
        # bg_mask, bg_dict = dh.make_background_mask(cube)
        c = image_median(cube)
        bg_mask, bg_dict = make_background_mask_from_combined(c)
        cards.append(("IGRFLAT0", bg_dict))
        if rp_remove_mode == 1:
            cube2 = [apply_rp_1st_phase(d1, bg_mask) for d1 in cube]

        elif rp_remove_mode == 2:
            cube2 = [apply_rp_2nd_phase(d1, bg_mask) for d1 in cube]
        else:
            cube2 = [apply_rp_3rd_phase(d1) for d1 in cube]

    cube3 = [sub_bg_from_slice(d1, bg_y_slice) for d1 in cube2]

    return cube3
