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

mdf_key = ('telescope', '_grating', 'camera')

mdf_dict = {
    # Dictionary key is the instrument and the value of the MASKNAME keyword.
    # Dictionary value is the lookup path of the MDF file for that instrument
    # with that MASKNAME.
    'Gemini-South_32/mm_G5506_ShortBlue_G5513': 'gnirs-ls-short-32-mdf.fits',
    'Gemini-South_111/mm_G5505_ShortBlue_G5513': 'gnirs-ls-short-111-mdf.fits',
    'Gemini-South_32/mm_G5506_LongBlue_G5515': 'gnirs-ls-long-32-mdf.fits',
    'Gemini-South_111/mm_G5505_LongBlue_G5515': 'gnirs-ls-long-111-mdf.fits',
    'Gemini-South_32/mm_G5506_LongRed_G5516': 'gnirs-ls-long-32-mdf.fits',
    'Gemini-South_111/mm_G5505_LongRed_G5516': 'gnirs-ls-long-111-mdf.fits',
    'Gemini-South_32/mm_G5506_ShortRed_G5514': 'gnirs-ls-short-32-mdf.fits',
    'Gemini-South_111/mm_G5505_ShortRed_G5514': 'gnirs-ls-short-111-mdf.fits',
    'Gemini-South_32/mm_G5506_ShortBlue_G5521': 'gnirs-ls-short-32-mdf.fits',
    'Gemini-South_111/mm_G5505_ShortBlue_G5521': 'gnirs-ls-short-111-mdf.fits',
    'Gemini-South_32/mm_G5506_ShortRed_G5522': 'gnirs-ls-short-32-mdf.fits',
    'Gemini-South_111/mm_G5505_ShortRed_G5522': 'gnirs-ls-short-111-mdf.fits',
    'Gemini-North_32/mm_G5506_ShortBlue_G5521': 'gnirsn-ls-short-32-mdf.fits',
    'Gemini-North_111/mm_G5505_ShortBlue_G5521': 'gnirsn-ls-short-111-mdf.fits',
    'Gemini-North_32/mm_G5506_ShortRed_G5522': 'gnirsn-ls-short-32-mdf.fits',
    'Gemini-North_111/mm_G5505_ShortRed_G5522': 'gnirsn-ls-short-111-mdf.fits',
    'Gemini-North_10/mm_G5507_LongBlue_G5515': 'gnirsn-ls-long-10-mdf.fits',
    'Gemini-North_32/mm_G5506_LongBlue_G5515': 'gnirsn-ls-long-32-mdf.fits',
    'Gemini-North_111/mm_G5505_LongBlue_G5515': 'gnirsn-ls-long-111-mdf.fits',
    'Gemini-North_10/mm_G5507_LongRed_G5516': 'gnirsn-ls-long-10-mdf.fits',
    'Gemini-North_32/mm_G5506_LongRed_G5516': 'gnirsn-ls-long-32-mdf.fits',
    'Gemini-North_111/mm_G5505_LongRed_G5516': 'gnirsn-ls-long-111-mdf.fits',
    'Gemini-North_32/mm_G5533_ShortBlue_G5538': 'gnirsn-ls-short-32-mdf.fits',
    'Gemini-North_111/mm_G5534_ShortBlue_G5538': 'gnirsn-ls-short-111-mdf.fits',
    'Gemini-North_32/mm/_G5533_ShortRed_G5544': 'gnirsn-ls-short-32-mdf.fits',
    'Gemini-North_111/mm_G5534_ShortRed_G5544': 'gnirsn-ls-short-111-mdf.fits',
    'Gemini-North_10/mm_G5532_LongBlue_G5542': 'gnirsn-ls-long-10-mdf.fits',
    'Gemini-North_32/mm_G5533_LongBlue_G5542': 'gnirsn-ls-long-32-mdf.fits',
    'Gemini-North_111/mm_G5534_LongBlue_G5542': 'gnirsn-ls-long-111-mdf.fits',
    'Gemini-North_10/mm_G5532_LongRed_G5543': 'gnirsn-ls-long-10-mdf.fits',
    'Gemini-North_32/mm_G5533_LongRed_G5543': 'gnirsn-ls-long-32-mdf.fits',
    'Gemini-North_111/mm_G5534_LongRed_G5543': 'gnirsn-ls-long-111-mdf.fits',
    'Gemini-North_32/mm_G5533_ShortBlue_G5540': 'gnirsn-ls-short-32-mdf.fits',
    'Gemini-North_111/mm_G5534_ShortBlue_G5540': 'gnirsn-ls-short-111-mdf.fits'
}
