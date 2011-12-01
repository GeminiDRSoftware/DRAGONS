import math
import numpy as np
from astrodata import Errors
from astrodata.adutils import gemLog
from gempy import geminiTools as gt
from gempy.science import qa
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
        
        # Initialize the list of output AstroData objects
        adoutput_list = []

        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():

            # Call the measure_bg user level function
            ad = qa.measure_bg(adinput=ad,separate_ext=rc["separate_ext"])[0]

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
        This primitive will detect the sources in the input images and fit
        both Gaussian and Moffat models to their profiles and calculate the 
        Image Quality and seeing from this.
        
        :param function: Function for centroid fitting
        :type function: string, can be: 'moffat','gauss' or 'both'; 
                        Default 'moffat'
                        
        :param display: Flag to turn on displaying the fitting to ds9
        :type display: Python boolean (True/False)
                       Default: True
        
        :param qa: flag to limit the number of sources used
        :type qa: Python boolean (True/False)
                  default: True
        """
        
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "measureIQ", "starting"))
        
        # Initialize the list of output AstroData objects
        adoutput_list = []
        
        # Loop over each input AstroData object in the input list
        if rc["display"]==False:
            for ad in rc.get_inputs_as_astrodata():
                
                # Call the measure_iq user level function,
                # which returns a list; take the first entry
                ad = qa.measure_iq(adinput=ad)[0]
                
                # Change the filename
                ad.filename = gt.fileNameUpdater(adIn=ad, suffix=rc["suffix"], 
                                                 strip=True)
                
                # Append the output AstroData object to the list 
                # of output AstroData objects
                adoutput_list.append(ad)
            
            # Report the list of output AstroData objects to the reduction
            # context
            rc.report_output(adoutput_list)
        
        else:
            
            # If display is needed, there may be instrument dependencies.
            # Call the iqDisplay primitive.
            rc.run("iqDisplay")
        
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
                raise Errors.ScienceError("No OBJCAT found in %s" % ad.filename)
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
                    log.warning('No reference sources found in %s[OBJCAT,%d]'%
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
                log.fullinfo("    Filename: %s ['OBJCAT', %d]" % (ad.filename, extver))
                log.fullinfo("    --------------------------------------------------------")
                log.fullinfo("    %d sources used to measure Zeropoint" % len(zps))
                log.fullinfo("    Zeropoint measurement (%s band): %.3f +/- %.3f" % (ad.filter_name(pretty=True), zp, zpe))
                log.fullinfo("    Nominal Zeropoint in this configuration: %.3f" % nominal_zeropoint)
                log.fullinfo("    Estimated Cloud Extinction: %.3f +/- %.3f magnitudes" % (cloud, zpe))
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
            
                log.stdinfo("    Filename: %s" % ad.filename)
                log.stdinfo("    --------------------------------------------------------")
                log.stdinfo("    %d sources used to measure Zeropoint" % total_sources)
                log.stdinfo("    Zeropoint measurements per detector: (%s band): %s" % (ad.filter_name(pretty=True), ', '.join(zp_str)))
                log.stdinfo("    Estimated Cloud Extinction: %.3f +/- %.3f magnitudes" % (cloud, clouderr))
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

