import numpy as np

from .shifted_images import ShiftedImages
from .lacosmics import get_cr_mask


def get_xshifted_all(ap, profile_map, variance_map,
                     data_minus_flattened, slitoffset_map,
                     debug=False):

    if isinstance(slitoffset_map, str) and (slitoffset_map == "none"):
        msk1 = np.isfinite(data_minus_flattened) & np.isfinite(variance_map)
        shifted = ShiftedImages(data=data_minus_flattened,
                                variance_map=variance_map,
                                profile_map=profile_map,
                                mask=msk1)
    else:
        _ = ap.get_shifted_images(profile_map,
                                  variance_map,
                                  data_minus_flattened,
                                  slitoffset_map=slitoffset_map,
                                  debug=debug)

        data_shft, variance_map_shft, profile_map_shft, msk1_shft = _

        shifted = ShiftedImages(image=data_shft,
                                variance=variance_map_shft,
                                profile_map=profile_map_shft,
                                mask=msk1_shft)

    return shifted


def get_updated_variance(variance_map, variance_map0, synth_map, gain):
    variance_map_r = variance_map0 + np.abs(synth_map) / gain
    variance_map = np.max([variance_map, variance_map_r], axis=0)

    return variance_map


def extract_spec_using_profile(ap, profile_map,
                               variance_map,
                               variance_map0,
                               data_minus_flattened,
                               orderflat,  #
                               ordermap, ordermap_bpixed,
                               slitpos_map,
                               slitoffset_map,
                               gain,
                               debug=False,
                               extraction_mode="optimal",
                               cr_rejection_thresh=30.,
                               lacosmic_thresh=0.):

    # This is used to test spec-extraction without slit-offset-map

    # slitoffset_map_extract = "none"
    slitoffset_map_extract = slitoffset_map

    shifted = get_xshifted_all(ap, profile_map, variance_map,
                               data_minus_flattened, slitoffset_map_extract,
                               debug=debug)

    # for khjeong
    weight_thresh = None
    remove_negative = False

    _ = ap.extract_stellar_from_shifted(ordermap_bpixed,
                                        shifted.profile_map,
                                        shifted.variance,
                                        shifted.image,
                                        shifted.mask,
                                        weight_thresh=weight_thresh,
                                        remove_negative=remove_negative)

    s_list, v_list = _

    # if self.debug_output:
    #     self.save_debug_output()

    # make synth_spec : profile * spectra
    synth_map = ap.make_synth_map(ordermap,
                                  slitpos_map,
                                  profile_map, s_list,
                                  slitoffset_map=slitoffset_map)

    # update variance map
    if variance_map0 is not None:
        variance_map = get_updated_variance(variance_map, variance_map0, synth_map,
                                            gain)

    # get cosmicray mask
    sig_map = np.abs(data_minus_flattened - synth_map) / (variance_map**.5)

    with np.errstate(invalid="ignore"):
        cr_mask = np.abs(sig_map) > cr_rejection_thresh

    if lacosmic_thresh > 0:

        # As our data is corrected for orderflat, it
        # actually amplifies order boundary so that they
        # can be more easily picked as CRs.  For now, we
        # use a workaround by multiplying by the
        # orderflat. But it would be better to improve it.

        cosmic_input = sig_map.copy() * orderflat
        cosmic_input[~np.isfinite(data_minus_flattened)] = np.nan
        cr_mask_cosmics = get_cr_mask(cosmic_input,
                                      readnoise=lacosmic_thresh)

        cr_mask = cr_mask | cr_mask_cosmics

    data_minus_flattened = np.ma.array(data_minus_flattened,
                                       mask=cr_mask).filled(np.nan)

    # extract spec

    # profile_map is not used for shifting.
    shifted = get_xshifted_all(ap, profile_map, variance_map,
                               data_minus_flattened, slitoffset_map,
                               debug=False)

    _ = ap.extract_stellar_from_shifted(ordermap_bpixed,
                                        shifted.profile_map,
                                        shifted.variance,
                                        shifted.image,
                                        shifted.mask,
                                        weight_thresh=weight_thresh,
                                        remove_negative=remove_negative)

    s_list, v_list = _

    if extraction_mode == "simple":

        # print("doing simple extraction")

        # if self.extraction_mode in ["optimal"]:
        #     msg = ("optimal extraction is not supported "
        #            "for extended source")
        #     print(msg)

        # synth_map = make_synth_map(ap, profile_map, s_list,
        #                            ordermap=ordermap,
        #                            slitpos_map=slitpos_map,
        #                            slitoffset_map=slitoffset_map)

        synth_map = ap.make_synth_map(ordermap,
                                      slitpos_map,
                                      profile_map, s_list,
                                      slitoffset_map=slitoffset_map)

        nan_msk = ~np.isfinite(data_minus_flattened)
        data_minus_flattened[nan_msk] = synth_map[nan_msk]
        variance_map[nan_msk] = 0.  # can we replace it with other values??

        # profile_map is not used for shifting.
        shifted = get_xshifted_all(ap, profile_map,
                                   variance_map,
                                   data_minus_flattened,
                                   slitoffset_map,
                                   debug=False)

        _ = ap.extract_simple_from_shifted(ordermap,
                                           shifted.profile_map,
                                           shifted.variance,
                                           shifted.image)
        s_list, v_list = _

        # regenerate synth_map using the new s_list, which will be saved.
        synth_map = ap.make_synth_map(ordermap,
                                      slitpos_map,
                                      profile_map, s_list,
                                      slitoffset_map=slitoffset_map)
        # synth_map = extractor.make_synth_map(
        #     ap, profile_map, s_list,
        #     ordermap=ordermap,
        #     slitpos_map=slitpos_map,
        #     slitoffset_map=slitoffset_map)

    elif extraction_mode in ["auto", "optimal"]:
        pass
    else:
        raise RuntimeError("")

    # assemble aux images that can be used by debug output
    aux_images = dict(sig_map=sig_map,
                      synth_map=synth_map,
                      shifted=shifted)

    return s_list, v_list, cr_mask, aux_images


def extract_spec_uniform(ap, profile_map,
                         variance_map,
                         variance_map0,
                         data_minus_flattened,
                         data_minus, orderflat,  #
                         ordermap, ordermap_bpixed,
                         slitpos_map,
                         slitoffset_map,
                         gain,
                         debug=False,
                         lacosmic_thresh=0.):

    # we need to update the variance map by rejecting
    # cosmic rays, but it is not clear how we do this
    # for extended source.
    # variance_map2 = variance_map

    # detect CRs

    if lacosmic_thresh > 0:

        # As our data is corrected for orderflat, it
        # actually amplifies order boundary so that they
        # can be more easily picked as CRs.  For now, we
        # use a workaround by multiplying by the
        # orderflat. But it would be better to improve it.

        cosmic_input = data_minus / (variance_map**.5) * orderflat

        cr_mask_p = get_cr_mask(cosmic_input,
                                readnoise=lacosmic_thresh)

        cr_mask_m = get_cr_mask(-cosmic_input,
                                readnoise=lacosmic_thresh)

        cr_mask = cr_mask_p | cr_mask_m

    else:
        cr_mask = np.zeros(data_minus_flattened.shape,
                           dtype=bool)

    # variance_map[cr_mask] = np.nan

    data_minus_flattened_orig = data_minus_flattened
    data_minus_flattened = data_minus_flattened_orig.copy()
    data_minus_flattened[cr_mask] = np.nan

    # extract spec

    shifted = get_xshifted_all(ap, profile_map,
                               variance_map,
                               data_minus_flattened,
                               slitoffset_map,
                               debug=debug)

    # # for khjeong
    # weight_thresh = None
    # remove_negative = False

    s_list, v_list = ap.extract_simple_from_shifted(ordermap,
                                                    shifted.profile_map,
                                                    shifted.variance,
                                                    shifted.image)

    # if self.debug_output:
    #     hdu_list = pyfits.HDUList()
    #     hdu_list.append(pyfits.PrimaryHDU(data=data_shft))
    #     hdu_list.append(pyfits.ImageHDU(data=variance_map_shft))
    #     hdu_list.append(pyfits.ImageHDU(data=profile_map_shft))
    #     hdu_list.append(pyfits.ImageHDU(data=ordermap_bpixed))
    #     #hdu_list.append(pyfits.ImageHDU(data=msk1_shft.astype("i")))
    #     #hdu_list.append(pyfits.ImageHDU(data=np.array(s_list)))
    #     hdu_list.writeto("test0.fits", clobber=True)

    aux_images = dict(shifted=shifted)

    return s_list, v_list, cr_mask, aux_images


####
# if 0:

#     # This is used to test spec-extraction without slit-offset-map

#     # slitoffset_map_extract = "none"
#     slitoffset_map_extract = slitoffset_map

#     shifted = get_shifted_all(ap, profile_map, variance_map,
#                               data_minus_flattened, slitoffset_map_extract,
#                               debug=debug)


#     # for khjeong
#     weight_thresh = None
#     remove_negative = False

#     _ = ap.extract_stellar_from_shifted(ordermap_bpixed,
#                                         shifted.profile_map,
#                                         shifted.variance,
#                                         shifted.image,
#                                         shifted.mask,
#                                         weight_thresh=weight_thresh,
#                                         remove_negative=remove_negative)

#     s_list, v_list = _

#     # if self.debug_output:
#     #     self.save_debug_output()

#     # make synth_spec : profile * spectra
#     synth_map = ap.make_synth_map(ordermap,
#                                   slitpos_map,
#                                   profile_map, s_list,
#                                   slitoffset_map=slitoffset_map)

#     # update variance map
#     variance_map = get_updated_variance(variance_map, variance_map0,
#                                         synth_map,
#                                         gain)
#     # get cosmicray mask
#     sig_map = np.abs(data_minus_flattened - synth_map) / (variance_map**.5)

#     with np.errstate(invalid="ignore"):
#         cr_mask = np.abs(sig_map) > cr_rejection_thresh

#     if lacosmic_thresh > 0:
#         from ..libs.lacosmics import get_cr_mask

#         # As our data is corrected for orderflat, it
#         # actually amplifies order boundary so that they
#         # can be more easily picked as CRs.  For now, we
#         # use a workaround by multiplying by the
#         # orderflat. But it would be better to improve it.

#         cosmic_input = sig_map.copy() * extractor.orderflat
#         cosmic_input[~np.isfinite(data_minus_flattened)] = np.nan
#         cr_mask_cosmics = get_cr_mask(cosmic_input,
#                                       readnoise=lacosmic_thresh)

#         cr_mask = cr_mask | cr_mask_cosmics


#     data_minus_flattened = np.ma.array(data_minus_flattened,
#                                        mask=cr_mask).filled(np.nan)

#     # extract spec

#     # profile_map is not used for shifting.
#     shifted = get_shifted_all(ap, profile_map, variance_map,
#                               data_minus_flattened, slitoffset_map,
#                               debug=False)

#     _ = ap.extract_stellar_from_shifted(ordermap_bpixed,
#                                         shifted.profile_map,
#                                         shifted.variance,
#                                         shifted.image,
#                                         shifted.mask,
#                                         weight_thresh=weight_thresh,
#                                         remove_negative=remove_negative)

#     s_list, v_list = _

#     if extraction_mode == "simple":

#         print("doing simple extraction")
#         # if self.extraction_mode in ["optimal"]:
#         #     msg = ("optimal extraction is not supported "
#         #            "for extended source")
#         #     print(msg)

#         synth_map = make_synth_map(ap, profile_map, s_list,
#                                    ordermap=ordermap,
#                                    slitpos_map=slitpos_map,
#                                    slitoffset_map=slitoffset_map)

#         nan_msk = ~np.isfinite(data_minus_flattened)
#         data_minus_flattened[nan_msk] = synth_map[nan_msk]
#         variance_map[nan_msk] = 0. # can we replace it with other values??

#         # profile_map is not used for shifting.
#         shifted = get_shifted_all(ap, profile_map,
#                                   variance_map,
#                                   data_minus_flattened,
#                                   slitoffset_map,
#                                   debug=False)

#         _ = ap.extract_simple_from_shifted(ordermap,
#                                            shifted.profile_map,
#                                            shifted.variance_map,
#                                            shifted.data)
#         s_list, v_list = _

#         # regenerate synth_map using the new s_list, which will be saved.
#         synth_map = extractor.make_synth_map(
#             ap, profile_map, s_list,
#             ordermap=ordermap,
#             slitpos_map=slitpos_map,
#             slitoffset_map=slitoffset_map)

#     elif extraction_mode in ["auto", "optimal"]:
#         pass
#     else:
#         raise RuntimeError("")

#     # assemble aux images that can be used by debug output
#     aux_images = dict(sig_map=sig_map,
#                       synth_map=synth_map,
#                       shifted=shifted)

#     return s_list, v_list, cr_mask, aux_images
