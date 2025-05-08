# gmos/maskdb.py
#
# This file contains the illumination masks
# and mask definition file (MDF) lookup tables for GMOS


illumMask_dict = {
    # GMOS North EEV
    # - From old imaging BPMs.  Cleaned illumination window and removed
    #   bad pixels.
    "GMOS-N_EEV_11_3amp_v1":  "gmos-n_illum_EEV_11_3amp_v1.fits.bz2",
    "GMOS-N_EEV_22_3amp_v1":  "gmos-n_illum_EEV_22_3amp_v1.fits.bz2",

    # GMOS North e2V
    # - From old imaging BPMs.  Cleaned illumination window and removed
    #   bad pixels.
    "GMOS-N_e2v_11_6amp_v1": "gmos-n_illum_e2v_11_6amp_v1.fits.bz2",
    "GMOS-N_e2v_22_6amp_v1": "gmos-n_illum_e2v_22_6amp_v1.fits.bz2",

    # GMOS North Hamamatsu
    # - From old imaging BPMs.  Cleaned illumination window and removed
    #   bad pixels.
    "GMOS-N_Ham_11_12amp_v1": "gmos-n_illum_HAM_11_12amp_v1.fits.bz2",
    "GMOS-N_Ham_22_12amp_v1": "gmos-n_illum_HAM_22_12amp_v1.fits.bz2",
    "GMOS-N_Ham_44_12amp_v1": "gmos-n_illum_HAM_44_12amp_v1.fits.bz2",

    # ----

    # GMOS South EEV
    # - From old imaging BPMs.  Cleaned illumination window and removed
    #   bad pixels.
    "GMOS-S_EEV_11_3amp_v1": "gmos-s_illum_EEV_11_3amp_v1.fits.bz2",
    "GMOS-S_EEV_22_3amp_v1": "gmos-s_illum_EEV_22_3amp_v1.fits.bz2",

    # GMOS South Hamamatsu
    # - From old imaging BPMs.  Cleaned illumination window and removed
    #   bad pixels.
    "GMOS-S_Ham_11_12amp_v1": "gmos-s_illum_HAM_11_12amp_v1.fits.bz2",
    "GMOS-S_Ham_22_12amp_v1": "gmos-s_illum_HAM_22_12amp_v1.fits.bz2",
    "GMOS-S_Ham_44_12amp_v1": "gmos-s_illum_HAM_44_12amp_v2.fits.bz2",

}


mdf_key = ("instrument", "focal_plane_mask")

mdf_dict = {
    # Dictionary key is the instrument and the value of MASKNAME keyword
    # Dictionary value is lookup path of the MDF for that instrument with
    # that MASKNAME
    "GMOS-N_IFU-B": "gnifu_slitb_mdf.fits",
    "GMOS-N_IFU-R": "gnifu_slitr_mdf.fits",
    "GMOS-N_IFU-2": "gnifu_slits_mdf.fits",
    "GMOS-S_IFU-B": "gsifu_slitb_mdf.fits",
    "GMOS-S_IFU-R": "gsifu_slitr_mdf.fits",
    "GMOS-S_IFU-2": "gsifu_slits_mdf.fits",
    #"GMOS-S_IFU-B": "Gemini/GMOS/MDF/gsifu_slitb_mdf_2003nov.fits",
    #"GMOS-S_IFU-R": "Gemini/GMOS/MDF/gsifu_slitr_mdf_2003nov.fits",
    #"GMOS-S_IFU-2": "Gemini/GMOS/MDF/gsifu_slits_mdf_2003nov.fits",
    #"GMOS-S_IFU-B_HAM": "Gemini/GMOS/MDF/gsifu_slitb_mdf_HAM.fits",
    #"GMOS-S_IFU-R_HAM": "Gemini/GMOS/MDF/gsifu_slitr_mdf_HAM.fits",
    #"GMOS-S_IFU-2_HAM": "Gemini/GMOS/MDF/gsifu_slits_mdf_HAM.fits",
    "GMOS-S_IFU-NS-B": "gsifu_ns_slitb_mdf.fits",
    "GMOS-S_IFU-NS-R": "gsifu_ns_slitr_mdf.fits",
    "GMOS-S_IFU-NS-2": "gsifu_ns_slits_mdf.fits",
    "GMOS-N_0.25arcsec": "0.25arcsec.fits",
    "GMOS-S_0.25arcsec": "0.25arcsec.fits",
    "GMOS-N_0.5arcsec": "0.5arcsec.fits",
    "GMOS-S_0.5arcsec": "0.5arcsec.fits",
    "GMOS-N_0.75arcsec": "0.75arcsec.fits",
    "GMOS-S_0.75arcsec": "0.75arcsec.fits",
    "GMOS-N_1.0arcsec": "1.0arcsec.fits",
    "GMOS-S_1.0arcsec": "1.0arcsec.fits",
    "GMOS-N_1.5arcsec": "1.5arcsec.fits",
    "GMOS-S_1.5arcsec": "1.5arcsec.fits",
    "GMOS-N_2.0arcsec": "2.0arcsec.fits",
    "GMOS-S_2.0arcsec": "2.0arcsec.fits",
    "GMOS-N_5.0arcsec": "5.0arcsec.fits",
    "GMOS-S_5.0arcsec": "5.0arcsec.fits",
    "GMOS-N_NS0.5arcsec": "NS0.5arcsec.fits",
    "GMOS-S_NS0.5arcsec": "NS0.5arcsec.fits",
    "GMOS-N_NS0.75arcsec": "NS0.75arcsec.fits",
    "GMOS-S_NS0.75arcsec": "NS0.75arcsec.fits",
    "GMOS-N_NS1.0arcsec": "NS1.0arcsec.fits",
    "GMOS-S_NS1.0arcsec": "NS1.0arcsec.fits",
    "GMOS-N_NS1.5arcsec": "NS1.5arcsec.fits",
    "GMOS-S_NS1.5arcsec": "NS1.5arcsec.fits",
    "GMOS-N_NS2.0arcsec": "NS2.0arcsec.fits",
    "GMOS-S_NS2.0arcsec": "NS2.0arcsec.fits",
    }
