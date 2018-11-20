from __future__ import print_function
"""
Collection of functions to make pipeline-processed and 
IRAF-processed files compatible with the other system.

"""
# TODO: write unit tests for irafcompat

import re

def pipeline2iraf(ad, verbose=False):
        
    if "GMOS" in ad.tags:
        compat_with_iraf_GMOS(ad, verbose)
    else:
        if verbose:
            print("Data type not supported, {0}".format(ad.filename))
         
    return

def compat_with_iraf_GMOS(ad, verbose):
    
    # The mighty GMOS OBSMODE
    obsmode = _get_gmos_obsmode(ad)
    if verbose:
        print("Add OBSMODE {0} to PHU.".format(obsmode))
    ad.phu.set('OBSMODE', obsmode, "Observing mode (IMAGE|IFU|MOS|LONGSLIT)")
    
    # The other keywords required by the Gemini IRAF tasks.
    if "PREPARED" in ad.tags:
        if verbose:
            print("Add GPREPARE to PHU")
        ad.phu.set('GPREPARE', "Compatibility", "For IRAF compatibility")
    if 'STACKFRM' in ad.phu and  ad.phu['OBSTYPE'] == "BIAS":
        if verbose:
            print("Add GBIAS to PHU")
        ad.phu.set('GBIAS', "Compatibility", "For IRAF compatibility")
    if 'ADUTOELE' in ad.phu:
        if verbose:
            print("Add GGAIN to PHU")
            print("Add GAINMULT to PHU")
        ad.phu.set('GGAIN', "Compatibility", "For IRAF compatibility")
        ad.phu.set('GAINMULT', "Compatibility", "For IRAF compatibility")
    if 'NORMLIZE' in ad.phu:
        if verbose:
            print("Add GIFLAT to PHU")
        ad.phu.set('GIFLAT', "Compatibility", "For IRAF compatibility")
    if 'BIASIM' in ad.phu:
        if verbose:
            print("Add GIREDUCE to PHU")
        ad.phu.set('GIREDUCE', "Compatibility", "For IRAF compatibility")
    if 'MOSAIC' in ad.phu:
        if verbose:
            print("Add GMOSAIC to PHU")
        ad.phu.set('GMOSAIC', "Compatibility", "For IRAF compatibility")

    return

def _get_gmos_obsmode(ad):
    obstype = ad.phu['OBSTYPE']
    masktype = ad.phu['MASKTYP']
    maskname = ad.phu['MASKNAME']
    grating = ad.phu['GRATING']
        
    if obstype == "BIAS" or obstype == "DARK" or masktype == 0:
        obsmode = "IMAGE"
    elif masktype == -1:
        if maskname[0:3] == "IFU":
            obsmode = "IFU"
        else:
            msg = "MASKTYP and MASKNAME are inconsistent. Cannot "\
                   "assign OBSMODE."
            raise ValueError(msg)
    elif masktype == 1:
        if re.search('arcsec', maskname):
            obsmode = "LONGSLIT"
        elif maskname != "None" and maskname[0:3] != "IFU":
            obsmode = "MOS"
        else:
            if maskname == "None":
                looks_like = "IMAGE"
            elif maskname[0:3] == "IFU":
                looks_like = "IFU"
            print("ERROR: Should be LONGSLIT or MOS but looks like {0}".format(looks_like))
            msg = "MASKTYP and MASKNAME are inconsistent. Cannot assign OBSMODE."
            raise ValueError(msg)
    else:
        print("WARNING: Headers not standard, assuming OBSMODE=IMAGE.")
        obsmode = "IMAGE"
    
    if grating == "MIRROR" and obsmode != "IMAGE":
        print("WARNING: Mask or IFU used without grating, setting OBSMODE to IMAGE.")
        obsmode = "IMAGE"

    return obsmode

