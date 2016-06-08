import os.path

mdf_dict = {
    # Dictionary key is the instrument and the value of the MASKNAME keyword
    # Dictionary value is the lookup path of the MDF for that instrument with
    # that MASKNAME
    "GMOS-N_IFU-B": "Gemini/GMOS/MDF/gnifu_slitb_mdf.fits",
    "GMOS-N_IFU-R": "Gemini/GMOS/MDF/gnifu_slitr_mdf.fits",
    "GMOS-N_IFU-2": "Gemini/GMOS/MDF/gnifu_slits_mdf.fits",
    "GMOS-S_IFU-B": "Gemini/GMOS/MDF/gsifu_slitb_mdf.fits",
    "GMOS-S_IFU-R": "Gemini/GMOS/MDF/gsifu_slitr_mdf.fits",
    "GMOS-S_IFU-2": "Gemini/GMOS/MDF/gsifu_slits_mdf.fits",
    #"GMOS-S_IFU-B": "Gemini/GMOS/MDF/gsifu_slitb_mdf_2003nov.fits",
    #"GMOS-S_IFU-R": "Gemini/GMOS/MDF/gsifu_slitr_mdf_2003nov.fits",
    #"GMOS-S_IFU-2": "Gemini/GMOS/MDF/gsifu_slits_mdf_2003nov.fits",
    #"GMOS-S_IFU-B_HAM": "Gemini/GMOS/MDF/gsifu_slitb_mdf_HAM.fits",
    #"GMOS-S_IFU-R_HAM": "Gemini/GMOS/MDF/gsifu_slitr_mdf_HAM.fits",
    #"GMOS-S_IFU-2_HAM": "Gemini/GMOS/MDF/gsifu_slits_mdf_HAM.fits",
    "GMOS-S_IFU-NS-B": "Gemini/GMOS/MDF/gsifu_ns_slitb_mdf.fits",
    "GMOS-S_IFU-NS-R": "Gemini/GMOS/MDF/gsifu_ns_slitr_mdf.fits",
    "GMOS-S_IFU-NS-2": "Gemini/GMOS/MDF/gsifu_ns_slits_mdf.fits",
    "GMOS-N_0.25arcsec": "Gemini/GMOS/MDF/0.25arcsec.fits",
    "GMOS-S_0.25arcsec": "Gemini/GMOS/MDF/0.25arcsec.fits",
    "GMOS-N_0.5arcsec": "Gemini/GMOS/MDF/0.5arcsec.fits",
    "GMOS-S_0.5arcsec": "Gemini/GMOS/MDF/0.5arcsec.fits",
    "GMOS-N_0.75arcsec": "Gemini/GMOS/MDF/0.75arcsec.fits",
    "GMOS-S_0.75arcsec": "Gemini/GMOS/MDF/0.75arcsec.fits",
    "GMOS-N_1.0arcsec": "Gemini/GMOS/MDF/1.0arcsec.fits",
    "GMOS-S_1.0arcsec": "Gemini/GMOS/MDF/1.0arcsec.fits",
    "GMOS-N_1.5arcsec": "Gemini/GMOS/MDF/1.5arcsec.fits",
    "GMOS-S_1.5arcsec": "Gemini/GMOS/MDF/1.5arcsec.fits",
    "GMOS-N_2.0arcsec": "Gemini/GMOS/MDF/2.0arcsec.fits",
    "GMOS-S_2.0arcsec": "Gemini/GMOS/MDF/2.0arcsec.fits",
    "GMOS-N_5.0arcsec": "Gemini/GMOS/MDF/5.0arcsec.fits",
    "GMOS-S_5.0arcsec": "Gemini/GMOS/MDF/5.0arcsec.fits",
    "GMOS-N_NS0.5arcsec": "Gemini/GMOS/MDF/NS0.5arcsec.fits",
    "GMOS-S_NS0.5arcsec": "Gemini/GMOS/MDF/NS0.5arcsec.fits",
    "GMOS-N_NS0.75arcsec": "Gemini/GMOS/MDF/NS0.75arcsec.fits",
    "GMOS-S_NS0.75arcsec": "Gemini/GMOS/MDF/NS0.75arcsec.fits",
    "GMOS-N_NS1.0arcsec": "Gemini/GMOS/MDF/NS1.0arcsec.fits",
    "GMOS-S_NS1.0arcsec": "Gemini/GMOS/MDF/NS1.0arcsec.fits",
    "GMOS-N_NS1.5arcsec": "Gemini/GMOS/MDF/NS1.5arcsec.fits",
    "GMOS-S_NS1.5arcsec": "Gemini/GMOS/MDF/NS1.5arcsec.fits",
    "GMOS-N_NS2.0arcsec": "Gemini/GMOS/MDF/NS2.0arcsec.fits",
    "GMOS-S_NS2.0arcsec": "Gemini/GMOS/MDF/NS2.0arcsec.fits",
    }

# Where to look for the MOS MDFs
# KL: Obviously having an internal path is the release software is
#     far from ideal.  Also, using the /net version of the path
#     is not great but it seems that /gemsoft does not have the
#     "masks" directory.   Eventually a better solution needs
#     to be implemented.  FYI, mdf_locations is used in addMDF.
#     Possible solution: have autoredux/redux figure out the path
#     and store it in the rc when reduce is called.
mdf_locations = ['.', 
                 os.path.join(os.path.sep, 
                              'net', 'mko-nfs', 'sci', 'dataflow', 'masks'),
                 os.path.join(os.path.sep,
                              'net', 'cpostonfs-nv1', 'dataflow', 'masks')]

