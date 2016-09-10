"""
Collection of functions to make pipeline-processed and 
IRAF-processed files compatible with the other system.
"""
import re

def pipeline2iraf(ad, verbose=False):
        
    if "GMOS" in ad.types:
        compat_with_iraf_GMOS(ad, verbose)
    else:
        if verbose:
            print "Data type not supported, {0}".format(filename)
         
    return

def compat_with_iraf_GMOS(ad, verbose):
    
    # The mighty GMOS OBSMODE
    obsmode = _get_gmos_obsmode(ad)
    if verbose:
        print "Add OBSMODE {0} to PHU.".format(obsmode)
    ad.phu_set_key_value(keyword='OBSMODE', value=obsmode, 
                         comment="Observing mode (IMAGE|IFU|MOS|LONGSLIT)")  
    
    # The other keywords required by the Gemini IRAF tasks.
    if "PREPARED" in ad.types:
        if verbose:
            print "Add GPREPARE to PHU"
        ad.phu_set_key_value(keyword="GPREPARE", value="Compatibility",
                             comment="For IRAF compatibility")
    if 'STACKFRM' in ad.phu.header.keys() and  \
          ad.phu_get_key_value('OBSTYPE') == "BIAS":
        if verbose:
            print "Add GBIAS to PHU"
        ad.phu_set_key_value(keyword="GBIAS", value="Compatibility", 
                             comment="For IRAF compatibility")
    if 'ADUTOELE' in ad.phu.header.keys():
        if verbose:
            print "Add GGAIN to PHU"
            print "Add GAINMULT to PHU"
        ad.phu_set_key_value(keyword='GGAIN', value="Compatibility",
                             comment="For IRAF compatibility")
        ad.phu_set_key_value(keyword='GAINMULT', value="Compatibility",
                             comment="For IRAF compatibility")
    if 'NORMLIZE' in ad.phu.header.keys():
        if verbose:
            print "Add GIFLAT to PHU"
        ad.phu_set_key_value(keyword='GIFLAT', value="Compatibility",
                             comment="For IRAF compatibility")
    if 'BIASIM' in ad.phu.header.keys():
        if verbose:
            print "Add GIREDUCE to PHU"
        ad.phu_set_key_value(keyword='GIREDUCE', value="Compatibility",
                             comment="For IRAF compatibility")
    
    return

def _get_gmos_obsmode(ad):
    obstype = ad.phu_get_key_value('OBSTYPE')
    masktype = ad.phu_get_key_value('MASKTYP')
    maskname = ad.phu_get_key_value('MASKNAME')
    grating = ad.phu_get_key_value('GRATING')
        
    if obstype == "BIAS" or obstype == "DARK" or masktype == 0:
        obsmode = "IMAGE"
    elif masktype == -1:
        if maskname[0:3] == "IFU":
            obsmode = "IFU"
        else:
            msg = "MASKTYP and MASKNAME are inconsistent. Cannot "\
                   "assign OBSMODE."
            raise ValueError, msg
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
            print "ERROR: Should be LONGSLIT or MOS but "\
                   "looks like {0}".format(looks_like)
            msg = "MASKTYP and MASKNAME are inconsistent. Cannot "\
                   "assign OBSMODE."
            raise ValueError, msg
    else:
        print "WARNING: Headers not standard, assuming OBSMODE=IMAGE."
        obsmode = "IMAGE"
    
    if grating == "MIRROR" and obsmode != "IMAGE":
        print "WARNING: Mask or IFU used without grating, setting "\
                        "OBSMODE to IMAGE."
        obsmode = "IMAGE"

    return obsmode

