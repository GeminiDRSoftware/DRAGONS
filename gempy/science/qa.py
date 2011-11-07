# This module contains user level functions related to the quality assessment
# of the input dataset

import os
import sys
import math
import time
from copy import deepcopy
from datetime import datetime
import numpy as np
from astrodata import Errors
from astrodata import Lookups
from astrodata.adutils import gemLog
from gempy import geminiTools as gt
from gempy.science import display as ds
from gempy.science import resample as rs

# Load the timestamp keyword dictionary that will be used to define the keyword
# to be used for the time stamp for the user level function
timestamp_keys = Lookups.get_lookup_table("Gemini/timestamp_keywords",
                                          "timestamp_keys")

# Load the standard comments for header keywords that will be updated
# in these functions
keyword_comments = Lookups.get_lookup_table("Gemini/keyword_comments",
                                            "keyword_comments")

def iq_display_gmos(adinput=None, display=True, frame=1, threshold=None):

    # Instantiate the log. This needs to be done outside of the try block,
    # since the log object is used in the except block 
    log = gemLog.getGeminiLog()

    # The validate_input function ensures that adinput is not None and returns
    # a list containing one or more AstroData objects
    adinput = gt.validate_input(adinput=adinput)

    # Define the keyword to be used for the time stamp for this user level
    # function (use the one for measure_iq -- that's the only function
    # called that modifies the input data)
    timestamp_key = timestamp_keys["measure_iq"]

    # Initialize the list of output AstroData objects
    adoutput_list = []

    try:
        # Loop over each input AstroData object in the input list

        if frame is None:
            frame = 1
        display = True

        for ad in adinput:

            # Make a copy of the input, so that we can modify it
            # without affecting the original
            disp_ad = deepcopy(ad)

            # Tile the data into one science extension
            disp_ad = rs.tile_arrays(adinput=disp_ad,tile_all=True)

            # Measure IQ on the image
            disp_ad,stars = measure_iq(adinput=disp_ad,
                                       return_source_info=True)

            # Display the image with IQ stars marked
            if display:
                data_shape = disp_ad[0]["SCI"].data.shape

                if len(stars)==0:
                    iqmask = None
                else:
                    iqmask = _iq_overlay(stars,data_shape)

                    log.stdinfo('Sources used to measure IQ are marked ' +
                                'with blue circles.')

                try:
                    disp_ad = ds.display_gmos(adinput=disp_ad[0],
                                              frame=frame,
                                              threshold=threshold,
                                              overlay=iqmask)
                except:
                    log.warning("Could not display %s" % disp_ad[0].filename)
                    display = False

            frame+=1
            
            # Update headers in original file
            mean_fwhm = disp_ad[0].phu_get_key_value("MEANFWHM")
            mean_ellp = disp_ad[0].phu_get_key_value("MEANELLP")
            if mean_fwhm is not None:
                ad.phu_set_key_value("MEANFWHM",mean_fwhm,
                                     comment=keyword_comments["MEANFWHM"])
            if mean_ellp is not None:
                ad.phu_set_key_value("MEANELLP",mean_ellp,
                                     comment=keyword_comments["MEANELLP"])
            gt.mark_history(adinput=ad, keyword=timestamp_key)
            

            # Append the output AstroData object to the list of output
            # AstroData objects
            adoutput_list.append(ad)

        # Return the list of output AstroData objects
        return adoutput_list

    except:
        # Log the message from the exception
        log.critical(repr(sys.exc_info()[1]))
        raise


def measure_bg(adinput=None, separate_ext=False):
    """
    This function measures the background in an image and writes
    the average value to the SKYLEVEL keyword in the PHU.  If an
    OBJCAT source catalog is present and contains background values,
    these background values are averaged to give the final
    measurement.  If no OBJCAT is present, or there are no good
    background values, it will take a sigma-clipped median of all
    data not flagged in the DQ plane.
    """

    # Instantiate the log. This needs to be done outside of the try block,
    # since the log object is used in the except block 
    log = gemLog.getGeminiLog()

    # The validate_input function ensures that adinput is not None and returns
    # a list containing one or more AstroData objects
    adinput = gt.validate_input(adinput=adinput)

    # Define the keyword to be used for the time stamp for this user level
    # function
    timestamp_key = timestamp_keys["measure_bg"]

    # Initialize the list of output AstroData objects
    adoutput_list = []

    try:

        # Define a few useful numbers for formatting output
        llen = 23
        rlen = 24
        dlen = llen + rlen

        # Loop over each input AstroData object in the input list
        for ad in adinput: 
            
            # Loop over SCI extensions
            all_bg = None
            bunit = None
            for sciext in ad["SCI"]:
                extver = sciext.extver()
                objcat = ad["OBJCAT",extver]

                bunit = sciext.get_key_value("BUNIT")
                if bunit is None:
                    bunit = "adu"

                if objcat is None:
                    log.fullinfo("No OBJCAT found for %s[SCI,%d], taking "\
                                 "median of data instead." % 
                                 (ad.filename,extver))
                    bg = None
                else:
                    bg = objcat.data["BACKGROUND"]
                    if np.all(bg==-999):
                        log.fullinfo("No background values in %s[OBJCAT,%d], "\
                                     "taking median of data instead." %
                                     (ad.filename,extver))
                        bg = None

                if bg is not None:
                    flags = objcat.data["FLAGS"]
                    dqflag = objcat.data["IMAFLAGS_ISO"]
                    if not np.all(dqflag==-999):
                        flags |= dqflag
                    good_bg = bg[flags==0]

                    # sigma-clip
                    mean = np.mean(good_bg)
                    sigma = np.std(good_bg)
                    good_bg = good_bg[((good_bg < mean+sigma) & 
                                       (good_bg > mean-sigma))]

                    sci_bg = np.mean(good_bg)
                    sci_std = np.std(good_bg)

                else:
                    scidata = sciext.data

                    dqext = ad["DQ",extver]
                    if dqext is not None:
                        scidata = scidata[dqext.data==0]
                    
                    # Roughly mask sources
                    median = np.median(scidata)
                    sigma = np.std(scidata)
                    scidata = scidata[scidata<median+sigma]

                    sci_bg = np.median(scidata)
                    sci_std = np.std(scidata)

                if all_bg is None:
                    all_bg = sci_bg
                    all_std = sci_std
                else:
                    all_bg = np.mean([all_bg,sci_bg])
                    all_std = np.sqrt(all_std**2+sci_std**2)

                # Write sky background to science header and log the value
                # if not averaging all together
                sciext.set_key_value("SKYLEVEL", sci_bg,
                    comment="%s [%s]" % (keyword_comments["SKYLEVEL"],bunit))

                if separate_ext:
                    log.stdinfo("\n    Filename: %s[SCI,%d]" % 
                                (ad.filename,extver))
                    log.stdinfo("    "+"-"*dlen)
                    log.stdinfo("    "+"Sky level measurement:".ljust(llen) +
                                ("%.0f +/- %.0f %s" % 
                                 (sci_bg,sci_std,bunit)).rjust(rlen))
                    log.stdinfo("    "+"-"*dlen+"\n")

            # Write mean background to PHU if averaging all together
            # (or if there's only one science extension)
            if ad.count_exts("SCI")==1 or not separate_ext:
                ad.phu_set_key_value("SKYLEVEL", all_bg,
                    comment="%s [%s]" % (keyword_comments["SKYLEVEL"],bunit))

                if not separate_ext:
                    log.stdinfo("\n    Filename: %s" % ad.filename)
                    log.stdinfo("    "+"-"*dlen)
                    log.stdinfo("    "+"Sky level measurement:".ljust(llen) +
                                ("%.0f +/- %.0f %s" % 
                                 (all_bg,all_std,bunit)).rjust(rlen))
                    log.stdinfo("    "+"-"*dlen+"\n")

            # Add the appropriate time stamps to the PHU
            gt.mark_history(adinput=ad, keyword=timestamp_key)

            # Append the output AstroData object to the list of output
            # AstroData objects
            adoutput_list.append(ad)

        # Return the list of output AstroData objects
        return adoutput_list

    except:
        # Log the message from the exception
        log.critical(repr(sys.exc_info()[1]))
        raise


def measure_iq(adinput=None, return_source_info=False, separate_ext=False):
    """
    This function is for use with sextractor-style source-detection.
    FWHM are already in OBJCAT; this function does the clipping and
    reporting only.
    """

    # Instantiate the log. This needs to be done outside of the try block,
    # since the log object is used in the except block 
    log = gemLog.getGeminiLog()

    # The validate_input function ensures that adinput is not None and returns
    # a list containing one or more AstroData objects
    adinput = gt.validate_input(adinput=adinput)

    # Define the keyword to be used for the time stamp for this user level
    # function
    timestamp_key = timestamp_keys["measure_iq"]

    # Initialize the list of output AstroData objects
    adoutput_list = []
    
    try:
        
        # Loop over each input AstroData object
        for ad in adinput:

            # Clip sources from the OBJCAT
            good_source = _clip_sources(ad,separate_ext=separate_ext)

            if len(good_source.keys())==0:
                log.warning('No good sources found in %s' % ad.filename)
                gt.mark_history(adinput=ad, keyword=timestamp_key)
                adoutput_list.append(ad)
                good_source['all'] = []
                continue

            for key in good_source:
                src = good_source[key]

                if len(src)==0:
                    log.warning('No good sources found in %s, %s extensions' %
                                (ad.filename,key))
                    gt.mark_history(adinput=ad, keyword=timestamp_key)
                    continue

                # Clipped mean of FWHM and ellipticity
                if len(src)>1:
                    mean_fwhm,std_fwhm,mean_ellip,std_ellip = _clipped_mean(src)
                elif len(src)==1:
                    log.warning('Only one source found. IQ numbers may ' +
                                'not be accurate.')
                    mean_fwhm = src[0]['fwhm']
                    std_fwhm = np.nan
                    mean_ellip = src[0]['ellipticity']
                    std_ellip = np.nan
                    
                log.stdinfo("%d sources used to measure IQ." % len(src))

                airmass = float(ad.airmass())
                if airmass is None:
                    log.warning("Airmass not found, not correcting to zenith")
                    corr = mean_fwhm
                else:
                    corr = mean_fwhm * airmass**(-0.6)

                # Get IQ constraint band corresponding to
                # the corrected FWHM number
                iq_band = _iq_band(adinput=ad,fwhm=corr)[0]

                # Format output for printing or logging                
                llen = 32
                rlen = 24
                dlen = llen+rlen
                pm = '+/-'
                fnStr = 'Filename: %s' % ad.filename
                if separate_ext:
                    fnStr += "[%s,%s]" % key
                fmStr = ('FWHM Mean %s Sigma:' % pm).ljust(llen) + \
                        ('%.3f %s %.3f arcsec' % (mean_fwhm, pm,
                                                  std_fwhm)).rjust(rlen)
                emStr = ('Ellipticity Mean %s Sigma:' % pm).ljust(llen) + \
                        ('%.3f %s %.3f' % (mean_ellip, pm, 
                                           std_ellip)).rjust(rlen)
                csStr = ('Zenith-corrected FWHM (AM %.2f):'%airmass).ljust(llen) + \
                        ('%.3f arcsec' % corr).rjust(rlen)
                if iq_band!='':
                    filter = ad.filter_name(pretty=True)
                    iqStr = ('IQ band for %s filter:'%filter).ljust(llen)+\
                            iq_band.rjust(rlen)
                else:
                    iqStr = '(IQ band could not be determined)'

                # Warn if high ellipticity
                if mean_ellip>0.1:
                    ell_warn = "\n    "+\
                               "WARNING: high ellipticity".rjust(dlen)
                else:
                    ell_warn = ""                    

                # Create final formatted string
                finalStr = '\n    '+fnStr+'\n    '+'-'*dlen+\
                           '\n    '+fmStr+'\n    '+emStr+\
                           '\n    '+csStr+'\n    '+iqStr+ell_warn+\
                           '\n    '+'-'*dlen+'\n'
                # Log final string
                log.stdinfo(finalStr, category='IQ')
                
                # Store average FWHM and ellipticity to header
                if separate_ext:
                    sciext = ad[key]
                    sciext.set_key_value("MEANFWHM", mean_fwhm,
                                         comment=keyword_comments["MEANFWHM"])
                    sciext.set_key_value("MEANELLP", mean_ellip,
                                         comment=keyword_comments["MEANELLP"])
                else:
                    ad.phu_set_key_value("MEANFWHM", mean_fwhm,
                                         comment=keyword_comments["MEANFWHM"])
                    ad.phu_set_key_value("MEANELLP", mean_ellip,
                                         comment=keyword_comments["MEANELLP"])

                # Add the appropriate time stamps to the PHU
                gt.mark_history(adinput=ad, keyword=timestamp_key)

            # Append the output AstroData object to the list of output
            # AstroData objects
            adoutput_list.append(ad)

        if return_source_info:
            if separate_ext:
                return adoutput_list, good_source
            else:
                return adoutput_list, good_source['all']
        else:
            return adoutput_list
    except:
        # Log the message from the exception
        log.critical(repr(sys.exc_info()[1]))
        raise

def measure_zp(adinput=None):
    """
    This function is for use with sextractor-style source-detection.
    It relies on having already added a reference catalog and done the
    cross match to populate the refmag column of the objcat
    """

    # Instantiate the log. This needs to be done outside of the try block,
    # since the log object is used in the except block 
    log = gemLog.getGeminiLog()

    # The validate_input function ensures that adinput is not None and returns
    # a list containing one or more AstroData objects
    adinput = gt.validate_input(adinput=adinput)

    # Define the keyword to be used for the time stamp for this user level
    # function
    timestamp_key = timestamp_keys["measure_zp"]

    # Initialize the list of output AstroData objects
    adoutput_list = []

    try:

        # Loop over each input AstroData object
        for ad in adinput:

            # Loop over OBJCATs extensions
            objcats = ad['OBJCAT']
            if objcats is None:
                raise Errors.ScienceError("No OBJCAT found in %s" % ad.filename)
            for objcat in objcats:
                extver = objcat.extver()
                mags = objcat.data['MAG_AUTO']
                if np.all(mags==-999):
                    log.warning("No magnitudes found in %s[OBJCAT,%d]"%
                                (ad.filename,extver))
                    adoutput_list.append(ad)
                    continue

                # Need to correct the mags for the exposure time
                et = float(ad.exposure_time())
                magcor = 2.5*math.log10(et)
                mags = np.where(mags==-999,mags,mags+magcor)

                # FIXME: Need to determine if we're in electrons or ADUs and correct
                # the mags for the gain if we're in ADU here. ZPs are in electrons.

                # FIXME: Need to apply the appropreate nominal extinction correction here

                refmags = objcat.data['REF_MAG']

                zps = refmags - mags
 
                zps = np.where((zps > -500), zps, None)

                zps = zps[np.flatnonzero(zps)]

                if len(zps)==0:
                    log.warning('No reference sources found in %s[OBJCAT,%d]'%
                                (ad.filename,extver))
                    adoutput_list.append(ad)
                    continue

                mean = np.mean(zps)
                sigma = np.std(zps)

                log.fullinfo("Unclipped zeropoint measurement: %f +/- %f" % (mean, sigma))

                zps = np.where(((zps < mean+sigma) & (zps > mean-sigma)), zps, None)
                zps = zps[np.flatnonzero(zps)]
                mean = np.mean(zps)
                sigma = np.std(zps)
                log.stdinfo("Filename: %s" % ad.filename)
                log.stdinfo("--------------------------------------------------------")
                log.stdinfo("%d sources used to measure Zeropoint" % len(zps))
                log.stdinfo("Zeropoint measurement (%s band): %.3f +/- %.3f" % (ad.filter_name(pretty=True), mean, sigma))
                log.stdinfo("--------------------------------------------------------")

            # Append the output AstroData object to the list of output
            # AstroData objects
            adoutput_list.append(ad)

        return adoutput_list

    except:
        # Log the message from the exception
        log.critical(repr(sys.exc_info()[1]))
        raise

            
            


##############################################################################
# Below are the helper functions for the user level functions in this module #
##############################################################################


def _iq_band(adinput=None,fwhm=None):
    """
    Helper function to take WFS, filter, and airmass information from
    an AstroData instance and use it to convert a seeing FWHM into
    an IQ constraint band.

    :param adinput: Input images for which the FWHM has been measured
    :type adinput: Astrodata objects, either a single or a list of objects

    :param fwhm: Measured FWHM of stars in image, in arcsec
    :type fwhm: float, either a single or a list that matches the length
                of the adinput list
    """

    # Instantiate the log. This needs to be done outside of the try block,
    # since the log object is used in the except block 
    log = gemLog.getGeminiLog()

    # The validate_input function ensures that adinput is not None and returns
    # a list containing one or more AstroData objects
    adinput = gt.validate_input(adinput=adinput)
    fwhm = gt.validate_input(adinput=fwhm)

    # Initialize the list of output IQ band strings
    str_output = []

    try:

        # get the IQ band definitions from a lookup table
        iqConstraints = Lookups.get_lookup_table("Gemini/IQConstraints",
                                                 "iqConstraints")

        # if there is only one FWHM listed for all adinput, copy
        # that entry into a list that matches the length of the
        # adinput list
        if len(fwhm)==1:
            fwhm = [fwhm[0] for ad in adinput]
        else:
            # otherwise check that length of fwhm list matches 
            # the length of the adinput list
            if len(adinput)!=len(fwhm):
                raise Errors.InputError("fwhm list must match length of " +
                                        "adinput list")
        
        # Loop over each input AstroData object in the input list
        count=0
        for ad in adinput:

            try:
                wfs = str(ad.wavefront_sensor())
            except:
                wfs = None
            try:
                filter = str(ad.filter_name(pretty=True))
            except:
                filter = None

            # default value for iq band
            iq = ''

            # check that ad has valid WFS, filter
            if wfs is not None and filter is not None:
                if filter in iqConstraints.keys():
                    if wfs in iqConstraints[filter].keys():

                        # get limits for this observation
                        iq20 = iqConstraints[filter][wfs]['20']
                        iq70 = iqConstraints[filter][wfs]['70']
                        iq85 = iqConstraints[filter][wfs]['85']

                        # get iq band
                        if fwhm[count]<iq20:
                            iq='IQ20 (<%.2f arcsec)' % iq20
                        elif fwhm[count]<iq70:
                            iq='IQ70 (%.2f-%.2f arcsec)' % (iq20,iq70)
                        elif fwhm[count]<iq85:
                            iq='IQ85 (%.2f-%.2f arcsec)' % (iq70,iq85)
                        else:
                            iq='IQAny (>%.2f arcsec)' % iq85
            
            # Append the iq band string to the output
            str_output.append(iq)
            count+=1

        # Return the list of output AstroData objects
        return str_output

    except:
        # Log the message from the exception
        log.critical(repr(sys.exc_info()[1]))
        raise

def _iq_overlay(stars,data_shape):
    """
    Generates a tuple of numpy arrays that can be used to mask a display with
    circles centered on the stars' positions and radii that reflect the
    measured FWHM.
    Eg. data[iqmask] = some_value

    The circle definition is based on numdisplay.overlay.circle, but circles
    are two pixels wide to make them easier to see.
    """

    xind = []
    yind = []
    width = data_shape[1]
    height = data_shape[0]
    for star in stars:
        x0 = star['x']
        y0 = star['y']
        #radius = star['fwhm']
        radius = 16
        r2 = radius*radius
        quarter = int(math.ceil(radius * math.sqrt (0.5)))

        for dy in range(-quarter,quarter+1):
            if r2>dy**2:
                dx = math.sqrt(r2 - dy**2)
            else:
                dx = 0
            j = int(round(dy+y0))
            i = int(round(x0-dx))           # left arc
            if i>=0 and j>=0 and i<width and j<height:
                xind.extend([i-1,i-2])
                yind.extend([j-1,j-1])
            i = int(round(x0+dx))           # right arc
            if i>=0 and j>=0 and i<width and j<height:
                xind.extend([i-1,i])
                yind.extend([j-1,j-1])

        for dx in range(-quarter, quarter+1):
            if r2>dx**2:
                dy = math.sqrt(r2 - dx**2)
            else:
                dy = 0
            i = int(round(dx + x0))
            j = int(round(y0 - dy))           # bottom arc
            if i>=0 and j>=0 and i<width and j<height:
                xind.extend([i-1,i-1])
                yind.extend([j-1,j-2])
            j = int (round (y0 + dy))           # top arc
            if i>=0 and j>=0 and i<width and j<height:
                xind.extend([i-1,i-1])
                yind.extend([j-1,j])

    iqmask = (np.array(yind),np.array(xind))
    return iqmask
    

def _clip_sources(ad, separate_ext=False):
    """
    This function takes the source data from the OBJCAT and returns the best
    sources for IQ measurement.
    
    :param ad: input image
    :type ad: AstroData instance with OBJCAT attached

    :param separate_ext: Flag to treat each extension separately
    :type separate_ext: boolean
    """

    good_source = {}
    first = True
    for sciext in ad["SCI"]:
        extver = sciext.extver()

        objcat = ad["OBJCAT",extver]
        if objcat is None:
            continue
        if objcat.data is None:
            continue

        x = objcat.data.field("X_IMAGE")
        y = objcat.data.field("Y_IMAGE")
        fwhm_pix = objcat.data.field("FWHM_IMAGE")
        fwhm_arcsec = objcat.data.field("FWHM_WORLD")
        ellip = objcat.data.field("ELLIPTICITY")
        sxflag = objcat.data.field("FLAGS")
        dqflag = objcat.data.field("IMAFLAGS_ISO")
        class_star = objcat.data.field("CLASS_STAR")
        area = objcat.data.field("ISOAREA_IMAGE")

        # Source is good if ellipticity defined and <0.5
        eflag = np.where((ellip>0.5)|(ellip==-999),1,0)

        # Source is good if probability of being a star >0.6
        sflag = np.where(class_star<0.6,1,0)

        flags = sxflag | eflag | sflag

        # Source is good if greater than 10 connected pixels
        # Ignore criterion if all undefined (-999)
        if not np.all(area==-999):
            aflag = np.where(area<100,1,0)
            flags |= aflag

        # Source is good if not flagged in DQ plane
        # Ignore criterion if all undefined (-999)
        if not np.all(dqflag==-999):
            flags |= dqflag

        # Use flag=0 to find good data
        good = (flags==0)
        rec = np.rec.fromarrays(
            [x[good],y[good],fwhm_pix[good],fwhm_arcsec[good],ellip[good]],
            names=["x","y","fwhm","fwhm_arcsec","ellipticity"])

        # Store data for extensions separately if desired
        if separate_ext:
            good_source[("SCI",extver)] = rec
        else:
            if first:
                good_source["all"] = rec
                first = False
            else:
                good_source["all"] = np.hstack([good_source["all"],rec])

    return good_source

def _clipped_mean(src):
    """
    Clipping is done on FWHM
    """

    fwhm = src['fwhm_arcsec']
    ellip = src['ellipticity']

    mean=0.7
    sigma=1
    num_total = len(fwhm)
    if num_total < 3:
        return np.mean(fwhm),np.std(fwhm),np.mean(ellip),np.std(ellip)

    num = num_total
    clip=0
    while (num > 0.5*num_total):

        num=0

        # for fwhm
        sum = 0
        sumsq = 0

        # for ellipticity
        esum = 0
        esumsq = 0

        upper = mean+sigma
        lower = mean-(3*sigma)

        i = 0
        for f in fwhm:
            e = ellip[i]
            if(f<upper and f>lower):
                sum += f
                sumsq += f*f
                esum += e
                esumsq += e*e
                num+=1
            i+=1

        if num>0:
            # for fwhm
            mean = sum / num
            var = (sumsq / num) - mean*mean
            if var>=0:
                sigma = math.sqrt(var)
            else:
                var = np.nan

            # for ellipticity
            emean = esum / num
            evar = (esumsq / num) - emean*emean
            if evar>=0:
                esigma = math.sqrt(evar)
            else:
                esigma = np.nan

        elif clip==0:
            return np.mean(fwhm),np.std(fwhm),np.mean(ellip),np.std(ellip)
        else:
            break

        clip+=1
        if clip>10:
            break

    return mean,sigma,emean,esigma
