import os
from copy import deepcopy
import numpy as np
from astrodata import AstroData
from astrodata import Errors
from astrodata import Lookups
from astrodata.adutils import gemLog
from astrodata.adutils.gemutil import pyrafLoader
from gempy import geminiTools as gt
from gempy import managers as mgr
from gempy.geminiCLParDicts import CLDefaultParamsDict
from primitives_GMOS import GMOSPrimitives

class GMOS_IMAGEPrimitives(GMOSPrimitives):
    """
    This is the class containing all of the primitives for the GMOS_IMAGE
    level of the type hierarchy tree. It inherits all the primitives from the
    level above, 'GMOSPrimitives'.
    """
    astrotype = "GMOS_IMAGE"
    
    def init(self, rc):
        GMOSPrimitives.init(self, rc)
        return rc
    
    def fringeCorrect(self,rc):
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "fringeCorrect", "starting"))
        
        # Loop over each input AstroData object in the input list to
        # test whether it's appropriate to try to remove the fringes
        rm_fringe = False
        for ad in rc.get_inputs_as_astrodata():
            
            # Test the filter to see if we need to fringeCorrect at all
            filter = ad.filter_name(pretty=True)
            exposure = ad.exposure_time()
            if filter not in ['i','z']:
                log.stdinfo("No fringe correction necessary for filter " +
                            filter)
                break
            elif exposure<60.0 and filter=="i":
                log.stdinfo("No fringe necessary for filter " +
                            filter + " with exposure time %.1fs" % exposure)
                break
            elif exposure<6.0 and filter=="z":
                log.stdinfo("No fringe necessary for filter " +
                            filter + " with exposure time %.1fs" % exposure)
                break
            else:
                rm_fringe = True

        if rm_fringe:
            # Retrieve processed fringes for the input
            
            # Check for a fringe in the "fringe" stream first; the makeFringe
            # primitive, if it was called, would have added it there;
            # this avoids the latency involved in storing and retrieving
            # a calibration in the central system
            fringes = rc.get_stream("fringe",empty=True)
            if fringes is None or len(fringes)!=1:
                rc.run("getProcessedFringe")

                # If using generic fringe, scale by calculated statistics
                stats_scale=True

            else:
                log.stdinfo("Using fringe: %s" % fringes[0].filename)
                for ad in rc.get_inputs_as_astrodata():
                    rc.add_cal(ad,"processed_fringe",
                               os.path.abspath(fringes[0].filename))

                # If fringe was created from science, scale by exposure time
                stats_scale=False
            
            rc.run("removeFringe")
        
        yield rc
    
    def getProcessedFringe(self, rc):
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        
        caltype = "processed_fringe"
        source = rc["source"]
        if source == None:
            rc.run("getCalibration(caltype=%s)" % caltype)
        else:
            rc.run("getCalibration(caltype=%s, source=%s)" % (caltype,source))
            
        # List calibrations found
        # Fringe correction is always optional, so don't raise errors if fringe
        # not found
        first = True
        for ad in rc.get_inputs_as_astrodata():
            calurl = rc.get_cal(ad, caltype) #get from cache
            if calurl:
                cal = AstroData(calurl)
                if cal.filename is not None:
                    if first:
                        log.stdinfo("getCalibration: Results")
                        first = False
                    log.stdinfo("   %s\n      for %s" % (cal.filename,
                                                     ad.filename))
        
        yield rc
    
    def makeFringe(self, rc):

        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])

        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "fringeCorrectFromScience", 
                                 "starting"))

        # Get input, initialize output
        orig_input = rc.get_inputs_as_astrodata()
        adoutput_list = []

        # Check that filter is either i or z; this step doesn't
        # help data taken in other filters
        # Also check that exposure time is not too short;
        # there isn't much fringing for the shortest exposure times
        red = True
        long_exposure = True
        all_filter = None
        for ad in orig_input:
            filter = ad.filter_name(pretty=True)
            exposure = ad.exposure_time()
            if all_filter is None:
                # Keep the first filter found
                all_filter = filter

            # Check for matching filters in input files
            elif filter!=all_filter:
                red = False
                log.warning("Mismatched filters in input; not making " +
                            "fringe frame")
                adoutput_list = orig_input
                break

            # Check for red filters
            if filter not in ["i","z"]:
                red = False
                log.stdinfo("No fringe necessary for filter " +
                            filter)
                adoutput_list = orig_input
                break

            # Check for long exposure times
            if exposure<60.0 and filter=="i":
                long_exposure=False
                log.stdinfo("No fringe necessary for filter " +
                            filter + " with exposure time %.1fs" % exposure)
                adoutput_list = orig_input
                break
            elif exposure<6.0 and filter=="z":
                long_exposure=False
                log.stdinfo("No fringe necessary for filter " +
                            filter + " with exposure time %.1fs" % exposure)
                adoutput_list = orig_input
                break

        enough = False
        if red and long_exposure:

            # Add the current frame to a list
            rc.run("addToList(purpose=forFringe)")

            # Get other frames from the list
            rc.run("getList(purpose=forFringe)")

            # Check that there are enough input files
            adinput = rc.get_inputs_as_astrodata()
            
            enough = True
            if len(adinput)<3:
                # Can't make a useful fringe frame without at least
                # three input frames
                enough = False
                log.stdinfo("Fewer than 3 frames provided as input. " +
                            "Not making fringe frame.")
                adoutput_list = orig_input

            elif filter=="i" and len(adinput)<5:
                if "qa" in rc.context:
                    # If fewer than 5 frames and in QA context, don't
                    # bother making a fringe -- it'll just make the data
                    # look worse.
                    enough = False
                    log.stdinfo("Fewer than 5 frames provided as input " +
                                "with filter i. Not making fringe frame.")
                    adoutput_list = orig_input
                else:
                    # Allow it in the science case, but warn that it
                    # may not be helpful.
                    log.warning("Fewer than 5 frames " +
                                "provided as input with filter i. Fringe " +
                                "frame generation is not recommended.")

        if enough:

            # Call the makeFringeFrame primitive
            rc.run("makeFringeFrame")

            # Store the generated fringe
            rc.run("storeProcessedFringe")

            # Report the fringe to the "fringe" stream
            # This is because the calibration system has a potential latency
            # between storage and retrieval; sending the file to a stream
            # ensures that will be available to the fringeCorrect
            # primitive if it is called immediately after makeFringe
            fringe_frame = rc.get_inputs_as_astrodata()
            rc.report_output(fringe_frame,stream="fringe")

            # Get the list of science frames back into the main stream
            adoutput_list = adinput

        # Report files back to the reduction context: if all went well,
        # these are the fringe-corrected frames.  If fringe generation/
        # correction did not happen, these are the original inputs
        rc.report_output(adoutput_list)
        yield rc

    def makeFringeFrame(self, rc):
        """
        This primitive makes a fringe frame by masking out sources
        in the science frames and stacking them together. It calls 
        gifringe to do so, so works only for GMOS imaging currently.
        """
        
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "makeFringeFrame", "starting"))
        
        # Initialize the list of output AstroData objects
        adoutput_list = []

        # Define the keyword to be used for the time stamp for this primitive
        timestamp_key = self.timestamp_keys["makeFringeFrame"]

        # Check for at least 3 input frames
        adinput = rc.get_inputs_as_astrodata()
        if len(adinput)<3:
            log.stdinfo('Fewer than 3 frames provided as input. ' +
                        'Not making fringe frame.')
            adoutput_list = adinput
        else:
        
            # Get the parameters from the RC
            suffix = rc["suffix"]
            operation = rc["operation"]
            reject_method = rc["reject_method"]

            # load and bring the pyraf related modules into the name-space
            pyraf, gemini, yes, no = pyrafLoader()
        
            # Determine whether VAR/DQ needs to be propagated 
            for ad in adinput:
                if (ad.count_exts("VAR") == 
                    ad.count_exts("DQ") == 
                    ad.count_exts("SCI")):
                    fl_vardq=yes
                else:
                    fl_vardq=no
                    break
                
            # Prepare input files, lists, parameters... for input to 
            # the CL script
            clm = mgr.CLManager(imageIns=adinput, suffix=suffix, 
                                funcName="makeFringeFrame", 
                                combinedImages=True, log=log)
        
            # Check the status of the CLManager object, 
            # True=continue, False= issue warning
            if not clm.status:
                raise Errors.InputError("Inputs must be prepared")
        
            # Parameters set by the mgr.CLManager or the definition 
            # of the primitive 
            clPrimParams = {
                # Retrieve the inputs as a list from the CLManager
                "inimages"    :clm.imageInsFiles(type="listFile"),
                # Maybe allow the user to override this in the future. 
                "outimage"    :clm.imageOutsFiles(type="string"), 
                # This returns a unique/temp log file for IRAF
                "logfile"     :clm.templog.name,
                "fl_vardq"    :fl_vardq,
                }
        
            # Create a dictionary of the parameters from the Parameter 
            # file adjustable by the user
            clSoftcodedParams = {
                "combine"       :operation,
                "reject"        :reject_method,
                }
        
            # Grab the default parameters dictionary and update 
            # it with the two above dictionaries
            clParamsDict = CLDefaultParamsDict("gifringe")
            clParamsDict.update(clPrimParams)
            clParamsDict.update(clSoftcodedParams)
        
            # Log the parameters
            mgr.logDictParams(clParamsDict)
        
            log.debug("Calling the gifringe CL script for input list "+
                      clm.imageInsFiles(type="listFile"))
        
            gemini.gifringe(**clParamsDict)
        
            if gemini.gifringe.status:
                raise Errors.ScienceError("gifringe failed for inputs "+
                                          clm.imageInsFiles(type="string"))
            else:
                log.fullinfo("Exited the gifringe CL script successfully")
        
            # Rename CL outputs and load them back into memory 
            # and clean up the intermediate temp files written to disk
            # refOuts and arrayOuts are None here
            imageOuts, refOuts, arrayOuts = clm.finishCL() 
            ad_out = imageOuts[0]

            # Change type of DQ plane back to int16 
            # (gemcombine sets it to int32)
            if ad_out["DQ"] is not None:
                for dqext in ad_out["DQ"]:
                    dqext.data = dqext.data.astype(np.int16)

                    # Also delete the BUNIT keyword (gemcombine
                    # sets it to same value as SCI)
                    if dqext.get_key_value("BUNIT") is not None:
                        del dqext.header['BUNIT']

            # Fix BUNIT in VAR plane as well
            # (gemcombine sets it to same value as SCI)
            bunit = ad_out["SCI",1].get_key_value("BUNIT")
            if ad_out["VAR"] is not None:
                for varext in ad_out["VAR"]:
                    varext.set_key_value("BUNIT","%s*%s" % (bunit,bunit),
                                         comment=self.keyword_comments["BUNIT"])

            # Add the appropriate time stamps to the PHU
            gt.mark_history(adinput=ad_out, keyword=timestamp_key)

            # Append the output AstroData object to the list
            # of output AstroData objects
            adoutput_list.append(ad_out)

        # Report the list of output AstroData objects to the reduction
        # context
        rc.report_output(adoutput_list)
        
        yield rc
    
    def normalize(self, rc):
        """
        This primitive will calculate a normalization factor from statistics
        on CCD2, then divide by this factor and propagate variance accordingly.
        CCD2 is used because of the dome-like shape of the GMOS detector
        response: CCDs 1 and 3 have lower average illumination than CCD2, 
        and that needs to be corrected for by the flat.
        """
        
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "normalize", "starting"))
        
        # Define the keyword to be used for the time stamp for this primitive
        timestamp_key = self.timestamp_keys["normalize"]

        # Initialize the list of output AstroData objects
        adoutput_list = []
        
        # Check whether inputs need to be tiled to get CCD2 data
        adinput = rc.get_inputs_as_astrodata()
        orig_input = adinput
        next = np.array([ad.count_exts("SCI") for ad in adinput])
        if np.any(next>1):
            # Keep a deep copy of the original, untiled input to
            # report back to RC
            orig_input = [deepcopy(ad) for ad in adinput]
            log.fullinfo("Tiling extensions together to get statistics "\
                         "from CCD2")
            rc.run("tileArrays")

        # Loop over each tiled input AstroData object
        count = 0
        for ad in rc.get_inputs_as_astrodata():
            
            if ad.count_exts("SCI")==1:
                # Only one CCD present; use it
                sciext = ad["SCI"]
            else:
                # Otherwise, take the second science extension
                # from the tiled data
                sciext = ad["SCI",2]

            sci_data = sciext.data

            # Take off 5% of the width as a border
            xborder = int(0.05 * sci_data.shape[1])
            yborder = int(0.05 * sci_data.shape[0])
            if xborder<20:
                xborder = 20
            if yborder<20:
                yborder = 20
            log.fullinfo("Using data section [%i:%i,%i:%i] from "\
                         "CCD2 for statistics" %
                         (xborder,sci_data.shape[1]-xborder,
                          yborder,sci_data.shape[0]-yborder))
            stat_region = sci_data[yborder:-yborder,
                                   xborder:-xborder]
                        
            # Remove DQ-flagged values (including saturated values)
            dqext = ad["DQ",sciext.extver()]
            if dqext is not None:
                dqdata = dqext.data[yborder:-yborder,
                                    xborder:-xborder]
                stat_region = stat_region[dqdata==0]

            # Remove negative values
            stat_region = stat_region[stat_region>0]

            # Find the mode and standard deviation
            hist,edges = np.histogram(stat_region,
                                      bins=np.max(sci_data)/0.1)
            mode = edges[np.argmax(hist)]
            std = np.std(stat_region)
            
            # Find the values within 3 sigma of the mode; the normalization
            # factor is the median of these values
            central_values = stat_region[
                np.logical_and(stat_region > mode - 3 * std,
                               stat_region < mode + 3 * std)]
            norm_factor = np.median(central_values)
            log.fullinfo("Normalization factor: %.2f" % norm_factor)

            # Now apply the factor to the original input
            ad = orig_input[count]
            count +=1

            # Divide by the normalization factor and propagate the
            # variance appropriately
            ad = ad.div(norm_factor)
            
            # Set any values flagged in the DQ plane to 1
            # (to avoid dividing by zero)
            for sciext in ad["SCI"]:
                extver = sciext.extver()
                dqext = ad["DQ",extver]
                if dqext is not None:
                    mask = np.where(dqext.data>0)
                    sciext.data[mask] = 1.0

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
    
    def removeFringe(self, rc):
        """
        This primitive will scale the fringes to their matching science data
        in the inputs, then subtract them.
        The primitive getProcessedFringe must have been run prior to this in 
        order to find and load the matching fringes into memory.
        
        There are two ways to find the value to scale fringes by:
        1. If stats_scale is set to True, the equation:
        (letting science data = b (or B), and fringe = a (or A))
    
        arrayB = where({where[SCIb < (SCIb.median+2.5*SCIb.std)]} 
                          > [SCIb.median-3*SCIb.std])
        scale = arrayB.std / SCIa.std
    
        The section of the SCI arrays to use for calculating these statistics
        is the CCD2 SCI data excluding the outer 5% pixels on all 4 sides.
        Future enhancement: allow user to choose section
    
        2. If stats_scale=False, then scale will be calculated using:
        exposure time of science / exposure time of fringe

        :param stats_scale: Use statistics to calculate the scale values,
                            rather than exposure time
        :type stats_scale: Python boolean (True/False)
        """
        
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "removeFringe",
                                 "starting"))
        
        # Define the keyword to be used for the time stamp for this primitive
        timestamp_key = self.timestamp_keys["removeFringe"]

        # Initialize the list of output AstroData objects
        adoutput_list = []
        
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():
            
            # Check whether the removeFringe primitive has been run
            # previously
            if ad.phu_get_key_value(timestamp_key):
                log.warning("No changes will be made to %s, since it has " \
                            "already been processed by removeFringe" \
                            % (ad.filename))
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue
            
            # Get the appropriate fringe frame
            fringe = AstroData(rc.get_cal(ad, "processed_fringe"))
            
            # Take care of the case where there was no fringe 
            if fringe.filename is None:
                log.warning("Could not find an appropriate fringe for %s" \
                            % (ad.filename))
                # Append the input to the output without further processing
                adoutput_list.append(ad)
                continue
            
            # Check the inputs have matching filters, binning and SCI shapes.
            try:
                gt.checkInputsMatch(adInsA=ad, adInsB=fringe)
            except Errors.ToolboxError:
                # If not, try to clip the fringe frame to the size of the
                # science data
                # For a GMOS example, this allows a full frame fringe to
                # be used for a CCD2-only science frame. 
                fringe = gt.clip_auxiliary_data(
                    adinput=ad, aux=fringe, aux_type="cal")[0]

                # Check again, but allow it to fail if they still don't match
                gt.checkInputsMatch(adInsA=ad, adInsB=fringe)

            # Check whether statistics should be used
            stats_scale = rc["stats_scale"]

            # Calculate the scale value
            scale = 1.0
            if not stats_scale:
                # Use the exposure times to calculate the scale
                log.fullinfo("Using exposure times to calculate the scaling"+
                             " factor")
                try:
                    scale = ad.exposure_time() / fringe.exposure_time()
                except:
                    raise Errors.InputError("Could not get exposure times " +
                                            "for %s, %s. Try stats_scale=True" %
                                            (ad.filename,fringe.filename))
            else:

                # Use statistics to calculate the scaling factor, following
                # masked_sci = where({where[sciExt < 
                #                    (sciExt.median+2.5*sciExt.std)]} 
                #                 > [sciExt.median-3*sciExt.std])
                # scale = masked_sci.std / fringeExt.std
                log.fullinfo("Using statistics to calculate the " +
                             "scaling factor")

                # Deepcopy the input so it can be manipulated without
                # affecting the original
                statsad = deepcopy(ad)
                statsfringe = deepcopy(fringe)

                # Trim off any overscan region still present
                statsad,statsfringe = gt.trim_to_data_section([statsad,
                                                               statsfringe])

                # Check the number of science extensions; if more than
                # one, use CCD2 data only
                nsciext = statsad.count_exts("SCI")
                if nsciext>1:

                    # Get the CCD numbers and ordering information
                    # corresponding to each extension
                    log.fullinfo("Trimming data to data section to remove "\
                                 "overscan region")
                    sci_info,frng_info = gt.array_information([statsad,
                                                               statsfringe])

                    # Pull out CCD2 data
                    scidata = []
                    frngdata = []
                    for i in range(nsciext):

                        # Get the next extension in physical order
                        sciext = statsad["SCI",sci_info["amps_order"][i]]
                        frngext = statsfringe["SCI",frng_info["amps_order"][i]]

                        # Check to see if it is on CCD2; if so, keep it
                        if sci_info[
                            "array_number"][("SCI",sciext.extver())]==2:
                            scidata.append(sciext.data)
                        if frng_info[
                            "array_number"][("SCI",frngext.extver())]==2:
                            frngdata.append(frngext.data)
                        
                    # Stack data if necessary
                    if len(scidata)>1:
                        scidata = np.hstack(scidata)
                        frngdata = np.hstack(frngdata)
                    else:
                        scidata = scidata[0]
                        frngdata = frngdata[0]
                else:
                    scidata = statsad["SCI"].data
                    frngdata = statsfringe["SCI"].data


                # Take off 5% of the width as a border
                xborder = int(0.05 * scidata.shape[1])
                yborder = int(0.05 * scidata.shape[0])
                if xborder<20:
                    xborder = 20
                if yborder<20:
                    yborder = 20
                log.fullinfo("Using CCD2 data section "\
                             "[%i:%i,%i:%i] for statistics" %
                             (xborder,scidata.shape[1]-xborder,
                              yborder,scidata.shape[0]-yborder))

                s = scidata[yborder:-yborder,xborder:-xborder]
                f = frngdata[yborder:-yborder,xborder:-xborder]

                # Get median and standard deviation
                # (Must flatten for compatibility with 
                # older versions of numpy)
                smed = np.median(s.flatten()) 
                sstd = s.std()
                      
                # Remove sources from the science data:
                # Make an array of all the points where the pixel value is 
                # less than the median value + 2.5 x the standard deviation.
                # and greater than the median -3 x the standard deviation.
                smiddle = s[np.logical_and(s<(smed+(2.5*sstd)),
                                           s>(smed-(3.0*sstd)))]
                        
                # Scale factor
                # This is the same logic as used in the IRAF girmfringe,
                # but it doesn't seem to work well in either case.
                scale = smiddle.std() / f.std() 
        
            log.fullinfo("Scale factor found = "+str(scale))
                
            # Use mult from the arith toolbox to perform the scaling of 
            # the fringe frame
            scaled_fringe = fringe.mult(scale)
            
            # Subtract the scaled fringe from the science
            ad = ad.sub(scaled_fringe)
            
            # Record the fringe file used
            ad.phu_set_key_value("FRINGEIM", 
                                 os.path.basename(fringe.filename),
                                 comment=self.keyword_comments["FRINGEIM"])

            # Add the appropriate time stamps to the PHU
            gt.mark_history(adinput=ad, keyword=timestamp_key)

            # Change the filename
            ad.filename = gt.fileNameUpdater(adIn=ad, suffix=rc["suffix"], 
                                             strip=True)
            
            # Append the output AstroData object to the list 
            # of output AstroData objects
            adoutput_list.append(ad)
            
        # Report the list of output AstroData objects to the reduction context
        rc.report_output(adoutput_list)
        yield rc
    
    def scaleByIntensity(self, rc):
        """
        This primitive scales input images to the mean value of the first
        image.  It is intended to be used to scale flats to the same
        level before stacking.
        """
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])

        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "scaleByIntensity", "starting"))

        # Define the keyword to be used for the time stamp for this primitive
        timestamp_key = self.timestamp_keys["scaleByIntensity"]

        # Initialize the list of output AstroData objects
        adoutput_list = []

        # Check whether inputs need to be tiled to get CCD2 data
        adinput = rc.get_inputs_as_astrodata()
        orig_input = adinput
        next = np.array([ad.count_exts("SCI") for ad in adinput])
        if np.any(next>1):
            # Keep a deep copy of the original, untiled input to
            # report back to RC
            orig_input = [deepcopy(ad) for ad in adinput]
            log.fullinfo("Tiling extensions together to get statistics "\
                         "from CCD2")
            rc.run("tileArrays")

        # Loop over each tiled AstroData object
        first = True
        reference_mean = 1.0
        count = 0
        for ad in rc.get_inputs_as_astrodata():

            # Check the number of science extensions; if more than
            # one, use CCD2 data only
            if ad.count_exts("SCI")==1:
                # Only one CCD present; use it
                central_data = ad["SCI"].data
            else:
                # Otherwise, take the second science extension
                # from the tiled data
               central_data = ad["SCI",2].data

            # Take off 5% of the width as a border
            xborder = int(0.05 * central_data.shape[1])
            yborder = int(0.05 * central_data.shape[0])
            if xborder<20:
                xborder = 20
            if yborder<20:
                yborder = 20
            log.fullinfo("Using data section [%i:%i,%i:%i] from CCD2 "\
                             "for statistics" %
                         (xborder,central_data.shape[1]-xborder,
                          yborder,central_data.shape[0]-yborder))
            stat_region = central_data[yborder:-yborder,xborder:-xborder]
            
            # Get mean value
            this_mean = np.mean(stat_region)

            # Get relative intensity
            if first:
                reference_mean = this_mean
                scale = 1.0
                first = False
            else:
                scale = reference_mean / this_mean

            # Get the original, untiled input to apply the scale to
            ad = orig_input[count]
            count +=1

            # Log and save the scale factor
            log.fullinfo("Relative intensity for %s: %.3f" % (ad.filename,
                                                              scale))
            ad.phu_set_key_value("RELINT", scale,
                                 comment=self.keyword_comments["RELINT"])

            # Multiply by the scaling factor
            ad.mult(scale)

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

    def stackFlats(self, rc):
        """
        This primitive will combine the input flats with rejection
        parameters set appropriately for GMOS imaging twilight flats.
        """
        
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "stackFlats", "starting"))
        
        # Check for at least 2 input frames
        adinput = rc.get_inputs_as_astrodata()
        nframes = len(adinput)
        if nframes<2:
            log.stdinfo("At least two frames must be provided to " +
                        "stackFlats")
            # Report input to RC without change
            adoutput_list = adinput
            rc.report_output(adoutput_list)
        else:
            # Define rejection parameters based on number of input frames,
            # to be used with minmax rejection. Note: if reject_method
            # parameter is overridden, these parameters will just be
            # ignored
            reject_method = rc["reject_method"]
            nlow = 0
            nhigh = 0
            if nframes <= 2:
                reject_method = "none"
            elif nframes <= 5:
                nlow = 1
                nhigh = 1
            elif nframes <= 10:
                nlow = 2
                nhigh = 2
            else:
                nlow = 2
                nhigh = 3
            log.fullinfo("For %d input frames, using reject_method=%s, "\
                         "nlow=%d, nhigh=%d" % 
                         (nframes,reject_method, nlow, nhigh))

            # Run the scaleByIntensity primitive to scale flats to the
            # same level
            rc.run("scaleByIntensity")

            # Run the stackFrames primitive with the defined parameters
            prim_str = "stackFrames(suffix=%s,operation=%s,mask_type=%s," \
                       "reject_method=%s,nlow=%s,nhigh=%s)" % \
                       (rc["suffix"],rc["operation"],rc["mask_type"],
                        reject_method,nlow,nhigh)
            rc.run(prim_str)
        
        yield rc

    def storeProcessedFringe(self, rc):
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "storeProcessedFringe",
                                 "starting"))
        
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():
            
            # Updating the file name with the suffix for this primitive and
            # then report the new file to the reduction context
            ad.filename = gt.fileNameUpdater(adIn=ad, suffix=rc["suffix"],
                                             strip=True)
            
            # Sanitize the headers of the file so that it looks like
            # a public calibration file rather than a science file
            ad = gt.convert_to_cal_header(adinput=ad, caltype="fringe")[0]

            # Adding a PROCFRNG time stamp to the PHU
            gt.mark_history(adinput=ad, keyword="PROCFRNG")
            
            # Refresh the AD types to reflect new processed status
            ad.refresh_types()
        
        # Upload to cal system
        rc.run("storeCalibration")
        
        yield rc
    
