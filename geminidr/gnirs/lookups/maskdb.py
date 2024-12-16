# gnirs/maskdb.py
#
# This file contains the bad pixel mask (BPMs), illumination mask,
# and mask definition file (MDF) lookup tables for GNIRS

bpm_dict = {
    "GNIRS_11": "gnirsn_2012dec05_bpm_alt.fits"
}

# mdf_key = ('telescope', '_prism', 'decker', '_grating', 'camera')

# mdf_dict = {
    # Dictionary key is the telescope (for north/south), prism, decker, grating,
    # and camera.
    # Dictionary value is the lookup path of the MDF file for that combination.
    # Long-slit (LS) MDFs:
    # 'Gemini-South_MIR_G5511_SC_Long_32/mm_G5506_ShortBlue_G5513': 'gnirs-ls-short-32-mdf.fits',
    # 'Gemini-South_MIR_G5511_SC_Long_111/mm_G5505_ShortBlue_G5513': 'gnirs-ls-short-111-mdf.fits',
    # 'Gemini-South_MIR_G5511_LC_Long_32/mm_G5506_LongBlue_G5515': 'gnirs-ls-long-32-mdf.fits',
    # 'Gemini-South_MIR_G5511_LC_Long_111/mm_G5505_LongBlue_G5515': 'gnirs-ls-long-111-mdf.fits',
    # 'Gemini-South_MIR_G5511_LC_Long_32/mm_G5506_LongRed_G5516': 'gnirs-ls-long-32-mdf.fits',
    # 'Gemini-South_MIR_G5511_LC_Long_111/mm_G5505_LongRed_G5516': 'gnirs-ls-long-111-mdf.fits',
    # 'Gemini-South_MIR_G5511_SC_Long_32/mm_G5506_ShortRed_G5514': 'gnirs-ls-short-32-mdf.fits',
    # 'Gemini-South_MIR_G5511_SC_Long_111/mm_G5505_ShortRed_G5514': 'gnirs-ls-short-111-mdf.fits',
    # 'Gemini-South_MIR_G5511_SC_Long_32/mm_G5506_ShortBlue_G5521': 'gnirs-ls-short-32-mdf.fits',
    # 'Gemini-South_MIR_G5511_SC_Long_111/mm_G5505_ShortBlue_G5521': 'gnirs-ls-short-111-mdf.fits',
    # 'Gemini-South_MIR_G5511_SC_Long_32/mm_G5506_ShortRed_G5522': 'gnirs-ls-short-32-mdf.fits',
    # 'Gemini-South_MIR_G5511_SC_Long_111/mm_G5505_ShortRed_G5522': 'gnirs-ls-short-111-mdf.fits',
    # 'Gemini-North_MIR_G5511_SC_Long_32/mm_G5506_ShortBlue_G5521': 'gnirsn-ls-short-32-mdf.fits',
    # 'Gemini-North_MIR_G5511_SC_Long_111/mm_G5505_ShortBlue_G5521': 'gnirsn-ls-short-111-mdf.fits',
    # 'Gemini-North_MIR_G5511_SC_Long_32/mm_G5506_ShortRed_G5522': 'gnirsn-ls-short-32-mdf.fits',
    # 'Gemini-North_MIR_G5511_SC_Long_111/mm_G5505_ShortRed_G5522': 'gnirsn-ls-short-111-mdf.fits',
    # 'Gemini-North_MIR_G5511_LC_Long_10/mm_G5507_LongBlue_G5515': 'gnirsn-ls-long-10-mdf.fits',
    # 'Gemini-North_MIR_G5511_LC_Long_32/mm_G5506_LongBlue_G5515': 'gnirsn-ls-long-32-mdf.fits',
    # 'Gemini-North_MIR_G5511_LC_Long_111/mm_G5505_LongBlue_G5515': 'gnirsn-ls-long-111-mdf.fits',
    # 'Gemini-North_MIR_G5511_LC_Long_10/mm_G5507_LongRed_G5516': 'gnirsn-ls-long-10-mdf.fits',
    # 'Gemini-North_MIR_G5511_LC_Long_32/mm_G5506_LongRed_G5516': 'gnirsn-ls-long-32-mdf.fits',
    # 'Gemini-North_MIR_G5511_LC_Long_111/mm_G5505_LongRed_G5516': 'gnirsn-ls-long-111-mdf.fits',
    # 'Gemini-North_MIR_G5537_SCLong_G5531_32/mm_G5533_ShortBlue_G5538': 'gnirsn-ls-short-32-mdf.fits',
    # 'Gemini-North_MIR_G5537_SCLong_G5531_111/mm_G5534_ShortBlue_G5538': 'gnirsn-ls-short-111-mdf.fits',
    # 'Gemini-North_MIR_G5537_SCLong_G5531_32/mm/_G5533_ShortRed_G5544': 'gnirsn-ls-short-32-mdf.fits',
    # 'Gemini-North_MIR_G5537_SCLong_G5531_111/mm_G5534_ShortRed_G5544': 'gnirsn-ls-short-111-mdf.fits',
    # 'Gemini-North_MIR_G5537_LCLong_G5531_10/mm_G5532_LongBlue_G5542': 'gnirsn-ls-long-10-mdf.fits',
    # 'Gemini-North_MIR_G5537_LCLong_G5531_32/mm_G5533_LongBlue_G5542': 'gnirsn-ls-long-32-mdf.fits',
    # 'Gemini-North_MIR_G5537_LCLong_G5531_111/mm_G5534_LongBlue_G5542': 'gnirsn-ls-long-111-mdf.fits',
    # 'Gemini-North_MIR_G5537_LCLong_G5531_10/mm_G5532_LongRed_G5543': 'gnirsn-ls-long-10-mdf.fits',
    # 'Gemini-North_MIR_G5537_LCLong_G5531_32/mm_G5533_LongRed_G5543': 'gnirsn-ls-long-32-mdf.fits',
    # 'Gemini-North_MIR_G5537_LCLong_G5531_111/mm_G5534_LongRed_G5543': 'gnirsn-ls-long-111-mdf.fits',
    # 'Gemini-North_MIR_G5537_SCLong_G5531_32/mm_G5533_ShortBlue_G5540': 'gnirsn-ls-short-32-mdf.fits',
    # 'Gemini-North_MIR_G5537_SCLong_G5531_111/mm_G5534_ShortBlue_G5540': 'gnirsn-ls-short-111-mdf.fits',

    # Cross-dispersed (XD) MDFs
    # We don't actually use these MDFs, but we do use the keys to select the
    # appropriate MDF table from MDF/xd_MDF_table.py.
    # 'Gemini-South_XD_G5509_SC_XD/IFU_32/mm_G5506_ShortBlue_G5513': 'gnirs-xd-short-32-mdf.fits',
    # 'Gemini-South_XD_G5509_SC_XD_32/mm_G5506_ShortBlue_G5513': 'gnirs-xd-short-32-mdf.fits',
    # 'Gemini-South_XD_G5509_IFU_32/mm_G5506_ShortBlue_G5513': 'gnirs-xd-short-32-mdf.fits',
    # 'Gemini-South_XD_G5508_LC_XD_32/mm_G5506_LongBlue_G5515': 'gnirs-xd-long-32-mdf.fits',
    # 'Gemini-South_XD_G5509_SC_XD/IFU_111/mm_G5505_ShortBlue_G5513': 'gnirs-xd-short-111-mdf.fits',
    # 'Gemini-South_XD_G5509_SC_XD_111/mm_G5505_ShortBlue_G5513': 'gnirs-xd-short-111-mdf.fits',
    # 'Gemini-South_XD_G5509_IFU_111/mm_G5505_ShortBlue_G5513': 'gnirs-xd-short-111-mdf.fits',
    # 'Gemini-South_XD_G5509_SC_XD/IFU_32/mm_G5506_ShortBlue_G5521': 'gnirs-xd-short-32-mdf.fits',
    # 'Gemini-South_XD_G5509_SC_XD_32/mm_G5506_ShortBlue_G5521': 'gnirs-xd-short-32-mdf-newt.fits',
    # 'Gemini-South_XD_G5509_IFU_32/mm_G5506_ShortBlue_G5521': 'gnirs-xd-short-32-mdf.fits',
    # 'Gemini-South_XD_G5509_SC_XD/IFU_111/mm_G5505_ShortBlue_G5521': 'gnirs-xd-short-111-mdf.fits',
    # 'Gemini-South_XD_G5509_SC_XD_111/mm_G5505_ShortBlue_G5521': 'gnirs-xd-short-111-mdf.fits',
    # 'Gemini-South_XD_G5509_IFU_111/mm_G5505_ShortBlue_G5521': 'gnirs-xd-short-111-mdf.fits',
    # 'Gemini-North_XD_G5509_SC_XD_32/mm_G5506_ShortBlue_G5521': 'gnirsn-sxd-short-32-mdf.fits',
    # 'Gemini-North_XD_G5509_SC_XD_111/mm_G5505_ShortBlue_G5521': 'gnirsn-sxd-short-111-mdf.fits',
    # 'Gemini-North_XD_G5508_LC_XD_10/mm_G5507_LongBlue_G5515': 'gnirsn-lxd-long-10-mdf.fits',
    # 'Gemini-North_XD_G5508_LC_XD_32/mm_G5506_LongBlue_G5515': 'gnirsn-lxd-long-32-mdf.fits',
    # 'Gemini-North_XD_G5508_LC_XD_111/mm_G5505_LongBlue_G5515': 'gnirsn-lxd-long-111-mdf.fits',
    # 'Gemini-North_XD_G5509_SC_XD_10/mm_G5507_LongBlue_G5515': 'gnirsn-sxd-long-10-mdf.fits',
    # 'Gemini-North_SXD_G5536_SCXD_G5531_32/mm_G5533_ShortBlue_G5538': 'gnirsn-sxd-short-32-mdf.fits',
    # 'Gemini-North_SXD_G5536_SCXD_G5531_111/mm_G5534_ShortBlue_G5538': 'gnirsn-sxd-short-111-mdf.fits',
    # 'Gemini-North_LXD_G5535_LCXD_G5531_10/mm_G5532_LongBlue_G5542': 'gnirsn-lxd-long-10-mdf.fits',
    # 'Gemini-North_LXD_G5535_LCXD_G5531_32/mm_G5533_LongBlue_G5542': 'gnirsn-lxd-long-32-mdf.fits',
    # 'Gemini-North_LXD_G5535_LCXD_G5531_111/mm_G5534_LongBlue_G5542': 'gnirsn-lxd-long-111-mdf.fits',
    # 'Gemini-North_SXD_G5536_SCXD_G5531_10/mm_G5532_LongBlue_G5542': 'gnirsn-sxd-long-10-mdf.fits',
    # 'Gemini-North_SXD_G5536_SCXD_G5531_32/mm_G5533_LongBlue_G5542': 'gnirsn-sxd-long-32-mdf.fits',
    # 'Gemini-North_SXD_G5536_SCXD_G5531_111/mm_G5534_LongBlue_G5542': 'gnirsn-sxd-long-111-mdf.fits',
    # 'Gemini-North_SXD_G5536_SCXD_G5531_32/mm_G5533_ShortBlue_G5540': 'gnirsn-sxd-short-32-mdf.fits',
    # 'Gemini-North_SXD_G5536_SCXD_G5531_111/mm_G5534_ShortBlue_G5540': 'gnirsn-sxd-short-111-mdf.fits',

    # Integral field unit (IFU) MDFs
    # 'Gemini-South_MIR_G5511_SC_XD/IFU_32/mm_G5506_ShortBlue_G5513': 'gnirs-ifu-short-32-mdf2.fits',
    # 'Gemini-South_MIR_G5511_SC_XD_32/mm_G5506_ShortBlue_G5513': 'gnirs-ifu-short-32-mdf2.fits',
    # 'Gemini-South_MIR_G5511_IFU_32/mm_G5506_ShortBlue_G5513': 'gnirs-ifu-short-32-mdf2.fits',
    # 'Gemini-South_MIR_G5511_** ENG 49450 **_32/mm_G5506_ShortBlue_G5513': 'gnirs-ifu-short-32-mdf2.fits',
    # 'Gemini-South_MIR_G5511_IFU_111/mm_G5505_ShortBlue_G5513': 'gnirs-ifu-short-111-mdf2.fits',
    # 'Gemini-South_MIR_G5511_SC_XD/IFU_32/mm_G5506_ShortBlue_G5521': 'gnirs-ifu-short-32-mdf2.fits',
    # 'Gemini-South_MIR_G5511_SC_XD_32/mm_G5506_ShortBlue_G5521': 'gnirs-ifu-short-32-mdf2.fits',
    # 'Gemini-South_MIR_G5511_IFU_32/mm_G5506_ShortBlue_G5521': 'gnirs-ifu-short-32-mdf2.fits',
    # 'Gemini-South_MIR_G5511_** ENG 49450 **_32/mm_G5506_ShortBlue_G5521': 'gnirs-ifu-short-32-mdf2.fits',
    # 'Gemini-South_MIR_G5511_IFU_111/mm_G5505_ShortBlue_G5521': 'gnirs-ifu-short-111-mdf2.fits',
# }

# Cut-on and cut-off wavelengths (um) of GNIRS order-blocking filters, based on conservative transmissivity (1%),
# or inter-order minima in the flats (determined using the GN filter set)
#
# Instruction for finding filter cut-on and cut-off wvls (same as for F2):
# https://docs.google.com/document/d/1LVTUFWkXJygkRUvqjsFm_7VZnqy7I4fy/edit?usp=sharing&ouid=106387637049533476653&rtpof=true&sd=true
bl_filter_range_dict = {'X_G0518': (1.01, 1.19), # GN filters
                        'J_G0517': (1.15, 1.385),
                        'H_G0516': (1.46, 1.84),
                        'K_G0515': (1.89, 2.54),
                        'L_G0527': (2.77, 4.44),
                        'M_G0514': (4.2, 6.0),
                        'X_G0506': (1.01, 1.19), # GS filter (assumed to be the same as the GN filters)
                        'J_G0505': (1.15, 1.385),
                        'H_G0504': (1.46, 1.84),
                        'K_G0503': (1.89, 2.54),
                        'L_G0502': (2.77, 4.44),
                        'M_G0501': (4.2, 6.0)}
