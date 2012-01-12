import sys
import math
from copy import deepcopy
import numpy as np
from astrodata import Errors
from astrodata import Lookups
from astrodata.adutils import gemLog
from gempy import geminiTools as gt
from primitives_GENERAL import GENERALPrimitives

class QAPrimitives(GENERALPrimitives):
    """
    This is the class containing all of the primitives for the GEMINI level of
    the type hierarchy tree. It inherits all the primitives from the level
    above, 'GENERALPrimitives'.
    """
    astrotype = "GEMINI"
    
    def init(self, rc):
        GENERALPrimitives.init(self, rc)
        return rc
    init.pt_hide = True
    
    def measureBG(self, rc):

        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "measureBG", "starting"))
        
        # Define the keyword to be used for the time stamp for this primitive
        timestamp_key = self.timestamp_keys["measureBG"]

        # Initialize the list of output AstroData objects
        adoutput_list = []

        # Define a few useful numbers for formatting output
        llen = 23
        rlen = 24
        dlen = llen + rlen

        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():

            # Get the necessary parameters from the RC
            separate_ext = rc["separate_ext"]

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
                    if len(bg)==0 or np.all(bg==-999):
                        log.fullinfo("No background values in %s[OBJCAT,%d], "\
                                     "taking median of data instead." %
                                     (ad.filename,extver))
                        bg = None
                    else:
                        flags = objcat.data["FLAGS"]
                        dqflag = objcat.data["IMAFLAGS_ISO"]
                        if not np.all(dqflag==-999):
                            flags |= dqflag
                        good_bg = bg[flags==0]

                        if len(good_bg)<3:
                            log.fullinfo("No good background values in "\
                                         "%s[OBJCAT,%d], "\
                                         "taking median of data instead." %
                                         (ad.filename,extver))
                            bg = None
                        else:

                            # sigma-clip
                            mean = np.mean(good_bg)
                            sigma = np.std(good_bg)
                            good_bg = good_bg[((good_bg < mean+sigma) & 
                                               (good_bg > mean-sigma))]
                            

                            if len(good_bg)<3:
                                log.fullinfo("No good background values in "\
                                             "%s[OBJCAT,%d], "\
                                             "taking median of data instead." %
                                             (ad.filename,extver))
                                bg = None
                            else:
                                sci_bg = np.mean(good_bg)
                                sci_std = np.std(good_bg)

                if bg is None:
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

                # OK, make sure we have a number in electrons
                if(bunit=='adu'):
                    bg_e = all_bg * float(sciext.gain())
                else:
                    bg_e = all_bg
                log.fullinfo("BG electrons = %f" % bg_e)
                # Now divide it by the exposure time
                bg_e /= float(sciext.exposure_time())
                log.fullinfo("BG electrons/s = %f" % bg_e)

                # Now, it's in pixels, divide it by the area of a pixel to get arcsec^2
                pixscale = float(sciext.pixel_scale())
                bg_e /= (pixscale*pixscale)
                log.fullinfo("BG electrons/s/as^2 = %f" % bg_e)
                # Now get that in (instrumental) magnitudes...
                if bg_e<=0:
                    raise Errors.ScienceError("Background in electrons is "\
                                              "less than or equal to 0 for "\
                                              "%s[SCI,%d]" % 
                                              (ad.filename,extver))
                bg_im = -2.5 * math.log10(bg_e)
                log.fullinfo("BG inst mag = %f" % bg_im)
                # And convert to apparent magnitude using the nominal zeropoint
                bg_am = bg_im + float(sciext.nominal_photometric_zeropoint())
                log.fullinfo("BG mag = %f" % bg_am)


                # Write sky background to science header and log the value
                # if not averaging all together
                sciext.set_key_value(
                    "SKYLEVEL", sci_bg, comment="%s [%s]" % 
                    (self.keyword_comments["SKYLEVEL"],bunit))

                if separate_ext:
                    log.stdinfo("\n    Filename: %s[SCI,%d]" % 
                                (ad.filename,extver))
                    log.stdinfo("    "+"-"*dlen)
                    log.stdinfo("    "+"Sky level measurement:".ljust(llen) +
                                ("%.0f +/- %.0f %s" % 
                                 (sci_bg,sci_std,bunit)).rjust(rlen))
                    log.stdinfo("    mag / sq arcsec in %s: %.1f" % (sciext.filter_name(pretty=True), bg_am))
                    log.stdinfo("    "+"-"*dlen+"\n")

            # Write mean background to PHU if averaging all together
            # (or if there's only one science extension)
            if ad.count_exts("SCI")==1 or not separate_ext:
                ad.phu_set_key_value(
                    "SKYLEVEL", all_bg, comment="%s [%s]" % 
                    (self.keyword_comments["SKYLEVEL"],bunit))

                # OK, make sure we have a number in electrons
                if(bunit=='adu'):
                    bg_e = all_bg * float(sciext.gain())
                else:
                    bg_e = all_bg
                log.fullinfo("BG electrons = %f" % bg_e)
                # Now divide it by the exposure time
                bg_e /= float(sciext.exposure_time())
                log.fullinfo("BG electrons/s = %f" % bg_e)

                # Now, it's in pixels, divide it by the area of a pixel to get arcsec^2
                pixscale = float(sciext.pixel_scale())
                bg_e /= (pixscale*pixscale)
                log.fullinfo("BG electrons/s/as^2 = %f" % bg_e)
                # Now get that in (instrumental) magnitudes...
                bg_im = -2.5 * math.log10(bg_e)
                log.fullinfo("BG inst mag = %f" % bg_im)
                # And convert to apparent magnitude using the nominal zeropoint
                bg_am = bg_im + float(sciext.nominal_photometric_zeropoint())
                log.fullinfo("BG mag = %f" % bg_am)

                if not separate_ext:
                    log.stdinfo("\n    Filename: %s" % ad.filename)
                    log.stdinfo("    "+"-"*dlen)
                    log.stdinfo("    "+"Sky level measurement:".ljust(llen) +
                                ("%.0f +/- %.0f %s" % 
                                 (all_bg,all_std,bunit)).rjust(rlen))
                    log.stdinfo("    Mag / sq arcsec in %s: %.1f" % (sciext.filter_name(pretty=True), bg_am))
                    log.stdinfo("    "+"-"*dlen+"\n")

            # Add the appropriate time stamps to the PHU
            gt.mark_history(adinput=ad, keyword=timestamp_key)

            # Change the filename
            ad.filename = gt.fileNameUpdater(adIn=ad, suffix=rc["suffix"], 
                                             strip=True)

            # Append the output AstroData object to the list 
            # of output AstroData objects
            adoutput_list.append(ad)

        # Report the list of output AstroData objects to the reduction
        # context
        rc.report_output(adoutput_list)

        yield rc

    def measureIQ(self, rc):
        """
        This primitive is for use with sextractor-style source-detection.
        FWHM are already in OBJCAT; this function does the clipping and
        reporting only.
        """
        
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "measureIQ", "starting"))
        
        # Define the keyword to be used for the time stamp for this primitive
        timestamp_key = self.timestamp_keys["measureIQ"]

        # Initialize the list of output AstroData objects
        adoutput_list = []
        
        # Check whether display is desired
        display = rc["display"]

        # Check whether inputs need to be tiled, ie. separate_ext=False
        # and at least one input has more than one science extension
        separate_ext = rc["separate_ext"]
        remove_bias = rc["remove_bias"]
        adinput = rc.get_inputs_as_astrodata()
        orig_input = adinput
        if not separate_ext:
            next = np.array([ad.count_exts("SCI") for ad in adinput])
            if np.any(next>1):
                # Keep a deep copy of the original, untiled input to
                # report back to RC
                orig_input = [deepcopy(ad) for ad in adinput]

                # If necessary, remove an approximate bias before tiling
                if remove_bias and display:

                    # Set the remove_bias parameter to False
                    # so it doesn't get removed again when
                    # display is run; leave it at default if no
                    # tiling is being done at this point, so the
                    # display will handle it later
                    remove_bias = False

                    new_adinput = []
                    for ad in adinput:
                        # Check whether data has been bias- or dark-subtracted
                        biasim = ad.phu_get_key_value("BIASIM")
                        darkim = ad.phu_get_key_value("DARKIM")

                        # Check whether data has been overscan-subtracted
                        overscan = np.array([ext.get_key_value("OVERSCAN") 
                                             for ext in ad["SCI"]])
                        if np.any(overscan) or biasim or darkim:
                            log.fullinfo("Bias level has already been removed "\
                                         "from data; no approximate "\
                                         "correction will be performed")
                        else:
                            # Try to get the bias level from the descriptor
                            try:
                                bias_level = ad.bias_level()
                            except:
                                log.warning("Bias level not found for %s; " \
                                            "approximate bias will not be "\
                                            "removed from displayed image" % 
                                            ad.filename)
                            else:
                                # Subtract the bias level from each
                                # science extension
                                log.stdinfo("\nSubtracting approximate bias "\
                                            "level from %s for display\n" \
                                            % ad.filename)
                                log.fullinfo("Bias levels used: %s" %
                                             str(bias_level))
                                ad = ad.sub(bias_level.dict_val)

                        new_adinput.append(ad)
                    adinput = new_adinput

                log.fullinfo("Tiling extensions together in order to compile "\
                             "IQ data from all extensions")
                rc.run("tileArrays(tile_all=True)")
            
        # Loop over each input AstroData object in the input list
        iq_overlays = []
        mean_fwhms = []
        mean_ellips = []
        for ad in rc.get_inputs_as_astrodata():
                
            # Clip sources from the OBJCAT
            good_source = gt.clip_sources(ad)
            keys = good_source.keys()

            # Check for no sources found
            if len(keys)==0:
                log.warning('No good sources found in %s' % ad.filename)
                if display:
                    iq_overlays.append(None)
                mean_fwhms.append(None)
                mean_ellips.append(None)
                continue

            # Go through all extensions
            keys.sort()
            for key in keys:
                src = good_source[key]

                if len(src)==0:
                    log.warning('No good sources found in %s, %s extensions' %
                                (ad.filename,key))
                    if display:
                        iq_overlays.append(None)
                    mean_fwhms.append(None)
                    mean_ellips.append(None)
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
                srcStr = "%d sources used to measure IQ." % len(src)
                fmStr = ('FWHM Mean %s Sigma:' % pm).ljust(llen) + \
                        ('%.3f %s %.3f arcsec' % (mean_fwhm, pm,
                                                  std_fwhm)).rjust(rlen)
                emStr = ('Ellipticity Mean %s Sigma:' % pm).ljust(llen) + \
                        ('%.3f %s %.3f' % (mean_ellip, pm, 
                                           std_ellip)).rjust(rlen)
                csStr = (
                    'Zenith-corrected FWHM (AM %.2f):'%airmass).ljust(llen) + \
                    ('%.3f arcsec' % corr).rjust(rlen)

                iq_warn = ""
                if iq_band is not None:

                    # iq_band is (percentile, lower bound, upper bound)
                    if iq_band[0]==20:
                        iq = 'IQ20 (<%.2f arcsec)' % iq_band[2]
                    elif iq_band[0]==100:
                        iq = 'IQAny (>%.2f arcsec)' % iq_band[1]
                    else:
                        iq = 'IQ%d (%.2f-%.2f arcsec)' % iq_band
 
                    filter = ad.filter_name(pretty=True)
                    iqStr = ('IQ band for %s filter:'%filter).ljust(llen)+\
                            iq.rjust(rlen)

                    # Get requested IQ band
                    try:
                        req_iq = int(ad.requested_iq())
                    except:
                        req_iq = None

                    if req_iq is not None:
                        reqStr = 'Requested IQ:'.ljust(llen) + \
                                 ('IQ%d' % req_iq).rjust(rlen)
                        if req_iq<iq_band[0]:
                            iq_warn = "\n    "+\
                                "WARNING: IQ requirement not met".rjust(dlen)
                    else:
                        reqStr = '(Requested IQ could not be determined)'
                else:
                    iqStr = '(IQ band could not be determined)'

                # Warn if high ellipticity
                if mean_ellip>0.1:
                    ell_warn = "\n    "+\
                        "WARNING: high ellipticity".rjust(dlen)
                else:
                    ell_warn = ""                    

                # Create final formatted string
                finalStr = '\n    '+fnStr+'\n    '+srcStr+\
                           '\n    '+'-'*dlen+\
                           '\n    '+fmStr+'\n    '+emStr+\
                           '\n    '+csStr+'\n    '+iqStr+\
                           '\n    '+reqStr+ell_warn+iq_warn+\
                           '\n    '+'-'*dlen+'\n'
                # Log final string
                log.stdinfo(finalStr, category='IQ')
                
                # Store average FWHM and ellipticity, for writing
                # to output header
                mean_fwhms.append(mean_fwhm)
                mean_ellips.append(mean_ellip)

                # If displaying, make a mask to display along with image
                # that marks which stars were used
                if display:
                    data_shape=ad[key].data.shape
                    iqmask = _iq_overlay(src,data_shape)
                    iq_overlays.append(iqmask)

        # Display image with stars used circled
        if display:
            # If separate_ext is True, we want the tile parameter
            # for the display primitive to be False
            tile = str(not separate_ext)

            # Stuff overlays into RC; display primitive will look for
            # them there
            rc["overlay"] = iq_overlays
            log.stdinfo("Sources used to measure IQ are marked " +
                        "with blue circles.\n")
            rc.run("display(tile=%s,remove_bias=%s)" % (tile,str(remove_bias)))

        # Update headers and filename for original input to report
        # back to RC
        for ad in orig_input:

            # Write FWHM and ellipticity to header
            if separate_ext:
                count = 0
                for sciext in ad["SCI"]:
                    mean_fwhm = mean_fwhms[count]
                    mean_ellip = mean_ellips[count]
                    if mean_fwhm is not None:
                        sciext.set_key_value(
                            "MEANFWHM", mean_fwhm,
                            comment=self.keyword_comments["MEANFWHM"])
                    if mean_ellip is not None:
                        sciext.set_key_value(
                            "MEANELLP", mean_ellip,
                            comment=self.keyword_comments["MEANELLP"])
                    count+=1
            if ad.count_exts("SCI")==1 or not separate_ext:
                mean_fwhm = mean_fwhms[0]
                mean_ellip = mean_ellips[0]
                if mean_fwhm is not None:
                    ad.phu_set_key_value(
                        "MEANFWHM", mean_fwhm,
                        comment=self.keyword_comments["MEANFWHM"])
                if mean_ellip is not None:
                    ad.phu_set_key_value(
                        "MEANELLP", mean_ellip,
                        comment=self.keyword_comments["MEANELLP"])

            # Add the appropriate time stamps to the PHU
            gt.mark_history(adinput=ad, keyword=timestamp_key)

            # Change the filename
            ad.filename = gt.fileNameUpdater(adIn=ad, suffix=rc["suffix"], 
                                             strip=True)
                
            # Append the output AstroData object to the list 
            # of output AstroData objects
            adoutput_list.append(ad)

        # Report the list of output AstroData objects to the reduction
        # context
        rc.report_output(adoutput_list)
        
        yield rc
    
    def measureZP(self, rc):
        """
        This primitive will determine the zeropoint by looking at
        sources in the OBJCAT for which a reference catalog magnitude
        has been determined.

        It will also compare the measured zeropoint against the nominal
        zeropoint for the instrument and the nominal atmospheric extinction
        as a function of airmass, to compute the estimated cloud attenuation.

        This function is for use with sextractor-style source-detection.
        It relies on having already added a reference catalog and done the
        cross match to populate the refmag column of the objcat

        The reference magnitudes (refmag) are straight from the reference
        catalog. The measured magnitudes (mags) are straight from the object
        detection catalog.

        We correct for astromepheric extinction at the point where we
        calculate the zeropoint, ie we define:
        actual_mag = zeropoint + instrumental_mag + extinction_correction

        where in this case, actual_mag is the refmag, instrumental_mag is
        the mag from the objcat, and we use the nominal extinction value as
        we don't have a measured one at this point. ie  we're actually
        computing zeropoint as:
        zeropoint = refmag - mag - nominal_extinction_correction

        Then we can treat zeropoint as: 
        zeropoint = nominal_photometric_zeropoint - cloud_extinction
        to estimate the cloud extinction.
        """

        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "measureZP", "starting"))
        
        # Define the keyword to be used for the time stamp for this primitive
        timestamp_key = self.timestamp_keys["measureZP"]

        # Initialize the list of output AstroData objects
        adoutput_list = []

        # get the CC band definitions from a lookup table
        ccConstraints = Lookups.get_lookup_table("Gemini/CCConstraints",
                                                 "ccConstraints")

        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():
            
            found_mag = True

            detzp_means=[]
            detzp_clouds=[]
            detzp_sigmas=[]
            total_sources=0
            # Loop over OBJCATs extensions
            objcats = ad['OBJCAT']
            if objcats is None:
                log.warning("No OBJCAT found in %s" % ad.filename)
                adoutput_list.append(ad)
                continue
                #raise Errors.ScienceError("No OBJCAT found in %s" % ad.filename)
            # We really want to check for the presence of reference mags in the objcats
            # at this point, but we can more easily do a quick check for the presence of
            # reference catalogs, which are a pre-requisite for this and not bother with
            # any of this if there are no reference catalogs
            if ad['REFCAT'] is None:
                log.warning("No Reference Catalogs Present - not attempting to measure photometric Zeropoints")
                # This is a bit of a hack, but it circumvents the big for loop
                objcats = []

            for objcat in objcats:
                extver = objcat.extver()
                mags = objcat.data['MAG_AUTO']
                mag_errs = objcat.data['MAGERR_AUTO']
                flags = objcat.data['FLAGS']
                iflags = objcat.data['IMAFLAGS_ISO']
                ids = objcat.data['NUMBER']
                if np.all(mags==-999):
                    log.warning("No magnitudes found in %s[OBJCAT,%d]"%
                                (ad.filename,extver))
                    continue

                # Need to correct the mags for the exposure time
                et = float(ad.exposure_time())
                magcor = 2.5*math.log10(et)
                mags = np.where(mags==-999,mags,mags+magcor)

                # Need to get the nominal atmospheric extinction
                nom_at_ext = float(ad.nominal_atmospheric_extinction())

                refmags = objcat.data['REF_MAG']
                refmag_errs = objcat.data['REF_MAG_ERR']
                if np.all(refmags==-999):
                    log.warning("No reference magnitudes found in %s[OBJCAT,%d]"%
                                (ad.filename,extver))
                    continue

                zps = refmags - mags - nom_at_ext
       
                # Is this mathematically correct? These are logarithmic values... (PH)
                # It'll do for now as an estimate at least
                zperrs = np.sqrt((refmag_errs * refmag_errs) + (mag_errs * mag_errs))
 
                # OK, trim out bad values
                zps = np.where((zps > -500), zps, None)
                zps = np.where((flags == 0), zps, None)
                zps = np.where((iflags == 0), zps, None)
                zperrs = np.where((zps > -500), zperrs, None)
                zperrs = np.where((flags == 0), zperrs, None)
                zperrs = np.where((iflags == 0), zperrs, None)
                ids = np.where((zps > -500), ids, None)
                ids = np.where((flags == 0), ids, None)
                ids = np.where((iflags == 0), ids, None)

                # Trim out where zeropoint error > 0.1
                zps = np.where((zperrs < 0.1), zps, None)
                zperrs = np.where((zperrs < 0.1), zperrs, None)
                ids = np.where((zperrs < 0.1), ids, None)
                
                # Discard the None values we just patched in
                zps = zps[np.flatnonzero(zps)]
                zperrs = zperrs[np.flatnonzero(zperrs)]
                ids = ids[np.flatnonzero(ids)]

                if len(zps)==0:
                    log.warning('No good reference sources found in %s[OBJCAT,%d]'%
                                (ad.filename,extver))
                    continue

                # Because these are magnitude (log) values, we weight directly from the
                # 1/variance, not signal / variance
                weights = 1.0 / (zperrs * zperrs)

                wzps = zps * weights
                zp = wzps.sum() / weights.sum()


                d = zps - zp
                d = d*d * weights
                zpv = d.sum() / weights.sum()
                zpe = math.sqrt(zpv)

                nominal_zeropoint = float(ad['SCI', extver].nominal_photometric_zeropoint())
                cloud = nominal_zeropoint - zp
                detzp_means.append(zp)
                detzp_clouds.append(cloud)
                detzp_sigmas.append(zpe)
                total_sources += len(zps)
                
                # Calculate which CC band we're in. 
                # Initially, I'm going to base this on a 2-sigma result...
                adj_cloud = cloud - 2.0*zpe
                # get cc constraints
                cc50 = ccConstraints['50']
                cc70 = ccConstraints['70']
                cc80 = ccConstraints['80']
                ccband = 'CCAny'
                ccnum = 100
                if(adj_cloud < cc80):
                    ccband = 'CC80'
                    ccnum = 80
                if(adj_cloud < cc70):
                    ccband = 'CC70'
                    ccnum = 70
                if(adj_cloud < cc50):
                    ccband = 'CC50'
                    ccnum = 50

                # Get requested CC band
                cc_warn = None
                try:
                    req_cc = int(ad.requested_cc())
                except:
                    req_cc = None
                if req_cc is not None:
                    if req_cc<ccnum:
                        cc_warn = '    WARNING: CC requirement not met'
                    if req_cc==100:
                        req_cc = 'CCAny'
                    else:
                        req_cc = 'CC%d' % req_cc
                

                log.fullinfo("    Filename: %s ['OBJCAT', %d]" % (ad.filename, extver))
                log.fullinfo("    --------------------------------------------------------")
                log.fullinfo("    %d sources used to measure Zeropoint" % len(zps))
                log.fullinfo("    Zeropoint measurement (%s band): %.3f +/- %.3f" % (ad.filter_name(pretty=True), zp, zpe))
                log.fullinfo("    Nominal Zeropoint in this configuration: %.3f" % nominal_zeropoint)
                log.fullinfo("    Estimated Cloud Extinction: %.3f +/- %.3f magnitudes" % (cloud, zpe))
                log.fullinfo("\n    This corresponds to %s" % ccband)
                if req_cc is not None:
                    log.fullinfo("    Requested CC band: %s" % req_cc)
                else:
                    log.fullinfo("    (Requested CC could not be determined)")
                if cc_warn is not None:
                    log.fullinfo(cc_warn)
                log.fullinfo("    --------------------------------------------------------")

            
            zp_str = []
            cloud_sum = 0
            cloud_esum = 0
            if(len(detzp_means)):
                for i in range(len(detzp_means)):
                    zp_str.append("%.3f +/- %.3f" % (detzp_means[i], detzp_sigmas[i]))
                    cloud_sum += detzp_clouds[i]
                    cloud_esum += (detzp_sigmas[i] * detzp_sigmas[i])
                cloud = cloud_sum / len(detzp_means)
                clouderr = math.sqrt(cloud_esum) / len(detzp_means)

                # Calculate which CC band we're in. 
                # Initially, I'm going to base this on a 2-sigma result...
                adj_cloud = cloud - 2.0*clouderr
                # get cc constraints
                cc50 = ccConstraints['50']
                cc70 = ccConstraints['70']
                cc80 = ccConstraints['80']
                ccband = 'CCany'
                ccnum = 100
                if(adj_cloud < cc80):
                    ccband = 'CC80'
                    ccnum = 80
                if(adj_cloud < cc70):
                    ccband = 'CC70'
                    ccnum = 70
                if(adj_cloud < cc50):
                    ccband = 'CC50'
                    ccnum = 50

                # Get requested CC band
                cc_warn = None
                try:
                    req_cc = int(ad.requested_cc())
                except:
                    req_cc = None
                if req_cc is not None:
                    if req_cc<ccnum:
                        cc_warn = '    WARNING: CC requirement not met'
                    if req_cc==100:
                        req_cc = 'CCAny'
                    else:
                        req_cc = 'CC%d' % req_cc
            
                log.stdinfo("    Filename: %s" % ad.filename)
                log.stdinfo("    --------------------------------------------------------")
                log.stdinfo("    %d sources used to measure Zeropoint" % total_sources)
                log.stdinfo("    Zeropoint measurements per detector (%s band): %s" % (ad.filter_name(pretty=True), ', '.join(zp_str)))
                log.stdinfo("    Estimated Cloud Extinction: %.3f +/- %.3f magnitudes" % (cloud, clouderr))
                log.stdinfo("\n    This corresponds to %s" % ccband)
                if req_cc is not None:
                    log.stdinfo("    Requested CC band: %s" % req_cc)
                else:
                    log.stdinfo("    (Requested CC could not be determined)")
                if cc_warn is not None:
                    log.stdinfo(cc_warn)
                log.stdinfo("    --------------------------------------------------------")
            else:
                log.stdinfo("    Filename: %s" % ad.filename)
                log.stdinfo("    Could not measure Zeropoint - no catalog sources associated")

            # Add the appropriate time stamps to the PHU
            gt.mark_history(adinput=ad, keyword=timestamp_key)

            # Change the filename
            ad.filename = gt.fileNameUpdater(adIn=ad, suffix=rc["suffix"], 
                                             strip=True)

            # Append the output AstroData object to the list 
            # of output AstroData objects
            adoutput_list.append(ad)

        # Report the list of output AstroData objects to the reduction
        # context
        rc.report_output(adoutput_list)

        yield rc


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

    # Initialize the list of output IQ band tuples
    list_output = []

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
            iq = None

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
                            iq = (20,None,iq20)
                            #iq='IQ20 (<%.2f arcsec)' % iq20
                        elif fwhm[count]<iq70:
                            iq = (70,iq20,iq70)
                            #iq='IQ70 (%.2f-%.2f arcsec)' % (iq20,iq70)
                        elif fwhm[count]<iq85:
                            iq = (85,iq70,iq85)
                            #iq='IQ85 (%.2f-%.2f arcsec)' % (iq70,iq85)
                        else:
                            iq = (100,iq85,None)
                            #iq='IQAny (>%.2f arcsec)' % iq85
            
            # Append the iq band tuple to the output
            # Return value is (percentile, lower bound, upper bound)
            list_output.append(iq)
            count+=1

        # Return the list of output AstroData objects
        return list_output

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
