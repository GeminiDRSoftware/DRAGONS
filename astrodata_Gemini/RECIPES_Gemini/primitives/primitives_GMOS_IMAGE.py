import os
from astrodata import AstroData
from astrodata import Errors
from astrodata.adutils import gemLog
from gempy import geminiTools as gt
from gempy.science import preprocessing as pp
from gempy.science import qa
from gempy.science import stack as sk
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
    
    def iqDisplay(self, rc):
        
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "iqDisplay", "starting"))
        
        # Initialize the list of output AstroData objects
        adoutput_list = []

        frame = rc["frame"]
        if frame is None:
            frame = 1
        
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():
            
            # Call the iq_display_gmos user level function,
            # which returns a list; take the first entry
            ad = qa.iq_display_gmos(adinput=ad, frame=frame,
                                    saturation=rc["saturation"])[0]
            
            # Change the filename
            ad.filename = gt.fileNameUpdater(adIn=ad, suffix=rc["suffix"], 
                                             strip=True)
            
            # Append the output AstroData object to the list
            # of output AstroData objects
            adoutput_list.append(ad)
            
            # Increment frame number
            frame += 1
        
        # Report the list of output AstroData objects to the reduction
        # context
        rc.report_output(adoutput_list)
        
        yield rc
    
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
            else:
                log.stdinfo("Using fringe: %s" % fringes[0].filename)
                for ad in rc.get_inputs_as_astrodata():
                    rc.add_cal(ad,"processed_fringe",
                               os.path.abspath(fringes[0].filename))
            
            rc.run("removeFringe")
        
        yield rc
    
    def fringeCorrectFromScience(self, rc):

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
                if "QA" in rc.context:
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
            rc.report_output(adinput)

            # Call the fringeCorrect primitive
            rc.run("fringeCorrect")

            rc.run("showInputs")

            adoutput_list = rc.get_inputs_as_astrodata()

        # Report files back to the reduction context: if all went well,
        # these are the fringe-corrected frames.  If fringe generation/
        # correction did not happen, these are the original inputs
        rc.report_output(adoutput_list)
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

        # Check for at least 3 input frames
        adinput = rc.get_inputs_as_astrodata()
        if len(adinput)<3:
            log.stdinfo('Fewer than 3 frames provided as input. ' +
                        'Not making fringe frame.')
            adoutput_list = adinput
        else:
            # Call the make_fringe_image_gmos user level function,
            # which returns a list with filenames already updated
            adoutput_list = pp.make_fringe_image_gmos(
                adinput=adinput, suffix=rc["suffix"],
                operation=rc["operation"], reject_method=rc["reject_method"])
        
        # Report the list of output AstroData objects to the reduction
        # context
        rc.report_output(adoutput_list)
        
        yield rc
    
    def normalize(self, rc):
        """
        This primitive will normalize a stacked flat frame
        
        :param saturation: Defines saturation level for the raw frame, in ADU
        :type saturation: string, can be 'default', or a number (default
                          value for this primitive is '45000')
        """
        
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "normalize", "starting"))
        
        # Initialize the list of output AstroData objects
        adoutput_list = []
        
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():
            
            # Check whether the normalize primitive has been run previously
            timestamp_key = self.timestamp_keys["normalize_image_gmos"]
            if ad.phu_get_key_value(timestamp_key):
                log.warning("No changes will be made to %s, since it has " \
                            "already been processed by normalize" \
                            % (ad.filename))
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue
            
            # Call the normalize_image_gmos user level function,
            # which returns a list; take the first entry
            ad = pp.normalize_image_gmos(adinput=ad,
                                         saturation=rc["saturation"])[0]
            
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
        
        :param stats_scale: Use statistics to calculate the scale values?
        :type stats_scale: Python boolean (True/False)
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to 
                        screen. OR the message level as a string (i.e.,
                        'critical', 'status', 'fullinfo'...)
        """
        
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "removeFringe",
                                 "starting"))
        
        # Initialize the list of output AstroData objects
        adoutput_list = []
        
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():
            
            # Check whether the removeFringe primitive has been run
            # previously
            timestamp_key = self.timestamp_keys["remove_fringe_image_gmos"]
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
            
            # Call the remove_fringe_image_gmos user level function,
            # which returns a list; take the first entry
            ad = pp.remove_fringe_image_gmos(adinput=ad, fringe=fringe,
                                             stats_scale=rc["stats_scale"])[0]
            
            # Change the filename
            ad.filename = gt.fileNameUpdater(adIn=ad, suffix=rc["suffix"], 
                                             strip=True)
            
            # Append the output AstroData object to the list 
            # of output AstroData objects
            adoutput_list.append(ad)
        
        # Report the list of output AstroData objects to the reduction context
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

            # Scale images by relative intensity before stacking
            adinput = pp.scale_by_intensity_gmos(adinput=adinput)

            # Call the stack_frames user level function,
            # which returns a list with filenames already updated
            adoutput_list = sk.stack_frames(
                adinput=adinput, suffix=rc["suffix"],
                operation=rc["operation"], mask_type=rc["mask_type"],
                reject_method=reject_method, grow=rc["grow"], nlow=nlow,
                nhigh=nhigh)
        
        # Report the list of output AstroData objects to the reduction
        # context
        rc.report_output(adoutput_list)
        
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
    
