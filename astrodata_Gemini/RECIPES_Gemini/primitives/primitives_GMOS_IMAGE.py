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

        # Loop over each input AstroData object in the input list
        frame = rc["frame"]
        if frame is None:
            frame = 1
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

    def makeFringe(self, rc):
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])

        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "makeFringe", "starting"))

        adinput = rc.get_inputs_as_astrodata()

        # Check that filter is either i or z; this step doesn't
        # help data taken in other filters
        red = True
        for ad in adinput:
            filter = ad.filter_name(pretty=True)
            if filter not in ["i","z"]:
                if "QA" in rc.context:
                    # in QA context, don't bother trying
                    red = False
                    log.warning("No fringe necessary for filter " +
                                filter)
                    break
                else:
                    # in science context, let the user do it, but warn
                    # that it's pointless
                    log.warning("No fringe necessary for filter " + filter)
            elif filter=="i" and len(adinput)<5:
                if "QA" in rc.context:
                    # If fewer than 5 frames and in QA context, don't
                    # bother making a fringe -- it'll just make the data
                    # look worse.
                    red = False
                    log.warning("Fewer than 5 frames provided as input " +
                                "with filter i. Not making fringe frame.")
                    break
                else:
                    # Allow it in the science case, but warn that it
                    # may not be helpful.
                    log.warning("Fewer than 5 frames " +
                                "provided as input with filter i. Fringe " +
                                "correction is not recommended.")
            elif len(adinput)<2:
                # Can't make a fringe frame without at least 2 input frames
                red = False
                log.warning("Fewer than 2 frames provided as input. " +
                            "Not making fringe frame.")
                break

        if red:
            recipe_list = []

            # Call the makeFringeFrame primitive
            recipe_list.append("makeFringeFrame")

            # Store the generated fringe
            recipe_list.append("storeProcessedFringe")
                
            # Run the specified primitives
            rc.run("\n".join(recipe_list))

        # Report all the unchanged input files back to the reduction context
        rc.report_output(adinput)
        yield rc

    def makeFringeFrame(self, rc):
        """
        This primitive makes a fringe frame by masking out sources
        in the science frames and stacking them together.  It calls 
        gifringe to do so, so works only for GMOS imaging currently.
        """
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])

        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "makeFringeFrame", "starting"))

        # Check for at least 2 input frames
        adinput = rc.get_inputs_as_astrodata()
        if len(adinput)<2:
            log.warning('Fewer than 2 frames provided as input. ' +
                        'Not making fringe frame.')
            adoutput_list = adinput
        else:
            # Call the make_fringe_image_gmos user level function,
            # which returns a list with filenames already updated
            adoutput_list = pp.make_fringe_image_gmos(adinput=adinput,
                                                      suffix=rc["suffix"],
                                                      operation=rc["operation"])

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

        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to 
                        screen. OR the message level as a string (i.e.,
                        'critical', 'status', 'fullinfo'...)
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

            # Check whether normalize has been run previously
            if ad.phu_get_key_value("NORMFLAT"):
                log.warning("%s has already been processed by normalize" %
                            (ad.filename))
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

        # Report the list of output AstroData objects to
        # the reduction context
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
            if ad.phu_get_key_value("RMFRINGE"):
                log.warning("%s has already been processed by " \
                            "removeFringe" % (ad.filename))
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

        # Report the list of output AstroData objects and the scaled fringe
        # frames to the reduction context
        rc.report_output(adoutput_list)
        
        yield rc

    def stackFlats(self, rc):
        """
        This primitive will combine the input flats with rejection
        parameters set appropriately for GMOS imaging twilight flats.
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to 
                        screen. OR the message level as a string (i.e.,
                        'critical', 'status', 'fullinfo'...)
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
            log.warning("At least two frames must be provided to " +
                        "stackFlats")
            # Report input to RC without change
            adoutput_list = adinput

        else:            
            # Define rejection parameters based on number of input frames,
            # to be used with minmax rejection.  Note: if reject_method
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
                
            # Call the stack_frames user level function,
            # which returns a list with filenames already updated
            adoutput_list = sk.stack_frames(adinput=adinput,
                                   suffix=rc["suffix"],
                                   operation=rc["operation"],
                                   mask_type=rc["mask_type"],
                                   reject_method=reject_method,
                                   grow=rc["grow"],
                                   nlow=nlow,
                                   nhigh=nhigh)

        rc.report_output(adoutput_list)
        yield rc
    

