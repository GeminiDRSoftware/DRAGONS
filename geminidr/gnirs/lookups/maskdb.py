# gnirs/maskdb.py
#
# This file contains the bad pixel mask (BPMs), illumination mask,
# and mask definition file (MDF) lookup tables for GNIRS

bpm_dict = {
    "GNIRS_11": "gnirsn_2012dec05_bpm_alt.fits"
}

illumMask_dict = {
    # Table of GNIRS illumination masks by camera and filter type (filters with or
    # without wings, see http://www.gemini.edu/sciops/instruments/gnirs/imaging).
    # Updated 2016.02.03

    # (CAMERA, MASKSHAPE): Illumination mask
    ('ShortBlue_G5540', 'Wings'): 'illum_mask_ShortBlue_G5540_Wings.fits',
    ('ShortBlue_G5540', 'NoWings'): 'illum_mask_ShortBlue_G5540_NoWings.fits',
    ('ShortBlue_G5538', 'Wings'): 'illum_mask_ShortBlue_G5540_Wings.fits',
    ('ShortBlue_G5538', 'NoWings'): 'illum_mask_ShortBlue_G5540_NoWings.fits',
    ('LongBlue_G5542', 'Wings'): 'illum_mask_LongBlue_G5542_Wings.fits',
    ('LongBlue_G5542', 'NoWings'): 'illum_mask_LongBlue_G5542_NoWings.fits',
    ('LongRed_G5543', 'Wings'): 'illum_mask_LongRed_G5543_Wings.fits',
    ('LongRed_G5543', 'NoWings'): 'illum_mask_LongRed_G5543_NoWings.fits',

}
