import os, shutil, time
from datetime import datetime
from astrodata import AstroData
from astrodata import Errors
from astrodata import IDFactory
from astrodata.adutils import gemLog
from astrodata.adutils.reduceutils.prsproxyutil import upload_calibration
from gempy import geminiTools as gt
from gempy.science import photometry as ph
from gempy.science import preprocessing as pp
from gempy.science import registration as rg
from gempy.science import resample as rs
from gempy.science import qa
from gempy.science import stack as sk
from gempy.science import standardization as sdz
from primitives_GENERAL import GENERALPrimitives

class GEMINIPrimitives(GENERALPrimitives):
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
    
    def addDQ(self, rc):
        """
        This primitive creates a data quality extension for each science
        extension in the input AstroData object. The value of a pixel will be
        the sum of the following: 0=good, 1=bad pixel (found in bad pixel
        mask), 2=value is non linear, 4=pixel is saturated
        """
        
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "addDQ", "starting"))
        
        # Initialize the list of output AstroData objects
        adoutput_list = []
        
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():
            
            # Check whether the addDQ primitive has been run previously
            timestamp_key = self.timestamp_keys["add_dq"]
            if ad.phu_get_key_value(timestamp_key):
                log.warning("No changes will be made to %s, since it has " \
                            "already been processed by addDQ" \
                            % (ad.filename))
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue
            
            # Call the add_dq user level function,
            # which returns a list; take the first entry
            try:
                ad = sdz.add_dq(adinput=ad)[0]
            except:
                if "QA" in rc.context:
                    # If DQ fails in QA context, continue on
                    log.warning("Unable to add DQ plane")
                    adoutput_list.append(ad)
                    continue
                else:
                    # Otherwise re-raise the error
                    raise
            
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
    
    def addToList(self, rc):
        """
        This primitive will update the lists of files to be stacked
        that have the same observationID with the current inputs.
        This file is cached between calls to reduce, thus allowing
        for one-file-at-a-time processing.
        
        :param purpose: 
        :type purpose: string
        """
        
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        
        # Perform an update to the stack cache file (or create it) using the
        # current inputs in the reduction context
        purpose = rc["purpose"]
        if purpose is None:
            purpose = ""
        if purpose=="":
            suffix = "_list"
        else:
            suffix = "_"+purpose
 
        # Update file names and write the files to disk to ensure the right
        # version is stored before adding it to the list.
        adoutput = []
        for ad in rc.get_inputs_as_astrodata():
            ad.filename = gt.fileNameUpdater(adIn=ad, suffix=suffix,
                                             strip=True)
            log.stdinfo("Writing %s to disk" % ad.filename,
                         category="list")
            ad.write(clobber=rc["clobber"])
            adoutput.append(ad)
        
        rc.report_output(adoutput)
        
        # Call the rq_stack_update method
        rc.rq_stack_update(purpose=purpose)
        
        yield rc
    
    def addVAR(self, rc):
        """
        This primitive calculates the variance of each SCI frame in the input
        files and appends it as a VAR frame using AstroData.
        """
        
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "addVAR", "starting"))
        
        # Initialize the list of output AstroData objects
        adoutput_list = []

        # Log a message about which type of variance is being added
        if rc["read_noise"] and not rc["poisson_noise"]:
            log.stdinfo("Adding the read noise component of the variance")
        if not rc["read_noise"] and rc["poisson_noise"]:
            log.stdinfo("Adding the poisson noise component of the variance")
        if rc["read_noise"] and rc["poisson_noise"]:
            log.stdinfo("Adding the read noise component and the poisson " +
                        "noise component of the variance")

        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():
            
            # Call the add_var user level function,
            # which returns a list; take the first entry
            ad = sdz.add_var(adinput=ad, read_noise=rc["read_noise"],
                             poisson_noise=rc["poisson_noise"])[0]
            
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
    
    def aduToElectrons(self, rc):
        """
        This primitive will convert the inputs from having pixel units of ADU
        to electrons.
        """
        
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "aduToElectrons", "starting"))
        
        # Initialize the list of output AstroData objects
        adoutput_list = []
        
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():
            
            # Check whether the aduToElectrons primitive has been run
            # previously
            timestamp_key = self.timestamp_keys["adu_to_electrons"]
            if ad.phu_get_key_value(timestamp_key):
                log.warning("No changes will be made to %s, since it has " \
                            "already been processed by aduToElectrons" \
                            % (ad.filename))
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue
            
            # Call the adu_to_electrons user level function,
            # which returns a list; take the first entry
            ad = pp.adu_to_electrons(adinput=ad)[0]
            
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
    
    def alignAndStack(self, rc):
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "alignAndStack", "starting"))
         
        # Add the input frame to the forStack list and 
        # get other available frames from the same list
        rc.run("addToList(purpose=forStack)")
        rc.run("getList(purpose=forStack)")

        # Check whether two or more input AstroData objects were provided
        adinput = rc.get_inputs_as_astrodata()
        if len(adinput) <= 1:
            log.stdinfo("No alignment or correction will be performed, " \
                        "since at least two input AstroData objects are " \
                        "required for alignAndStack")
            rc.report_output(adinput)
        else:
            recipe_list = []

            # Check to see if detectSources needs to be run
            run_ds = False
            for ad in adinput:
                objcat = ad["OBJCAT"]
                if objcat is None:
                    run_ds = True
                    break
            if run_ds:
                recipe_list.append("detectSources")
            
            # Register all images to the first one
            recipe_list.append("correctWCSToReferenceImage")
            
            # Align all images to the first one
            recipe_list.append("alignToReferenceImage")
            
            # Correct background level in all images to the first one
            recipe_list.append("correctBackgroundToReferenceImage")

            # Stack all frames
            recipe_list.append("stackFrames")
            
            # Run all the needed primitives
            rc.run("\n".join(recipe_list))
        
        yield rc
    
    def alignToReferenceImage(self, rc):
        """
        This primitive applies the transformation encoded in the input images
        WCSs to align them with a reference image, in reference image pixel
        coordinates. The reference image is taken to be the first image in
        the input list.
        
        By default, the transformation into the reference frame is done via
        interpolation. The interpolator parameter specifies the interpolation 
        method. The options are nearest-neighbor, bilinear, or nth-order 
        spline, with n = 2, 3, 4, or 5. If interpolator is None, 
        no interpolation is done: the input image is shifted by an integer
        number of pixels, such that the center of the frame matches up as
        well as possible. The variance plane, if present, is transformed in
        the same way as the science data.
        
        The data quality plane, if present, must be handled a little
        differently. DQ flags are set bit-wise, such that each pixel is the 
        sum of any of the following values: 0=good pixel,
        1=bad pixel (from bad pixel mask), 2=nonlinear, 4=saturated, etc.
        To transform the DQ plane without losing flag information, it is
        unpacked into separate masks, each of which is transformed in the same
        way as the science data. A pixel is flagged if it had greater than
        1% influence from a bad pixel. The transformed masks are then added
        back together to generate the transformed DQ plane.
        
        In order not to lose any data, the output image arrays (including the
        reference image's) are expanded with respect to the input image arrays.
        The science and variance data arrays are padded with zeros; the DQ
        plane is padded with ones.
        
        The WCS keywords in the headers of the output images are updated
        to reflect the transformation.
        
        :param interpolator: type of interpolation desired
        :type interpolator: string, possible values are None, 'nearest', 
                            'linear', 'spline2', 'spline3', 'spline4', 
                            or 'spline5'
        
        :param suffix: string to add on the end of the input filenames to 
                       generate output filenames
        :type suffix: string
        """
        
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "alignToReferenceImage",
                                 "starting"))
        
        # Initialize the list of output AstroData objects
        adoutput_list = []
        
        # Check whether two or more input AstroData objects were provided
        adinput = rc.get_inputs_as_astrodata()
        if len(adinput) <= 1:
            log.warning("No alignment will be performed, since at least two " \
                        "input AstroData objects are required for " \
                        "alignToReferenceImage")
            # Set the input AstroData object list equal to the output AstroData
            # objects list without further processing
            adoutput_list = adinput
        else:
            
            # Call the align_to_reference_image user level function,
            # which returns a list
            adoutput = rs.align_to_reference_image(
                adinput=adinput, interpolator=rc["interpolator"])
            
            # Change the filenames and append to output list
            for ad in adoutput:
                ad.filename = gt.fileNameUpdater(adIn=ad, suffix=rc["suffix"], 
                                                 strip=True)
                adoutput_list.append(ad)
        
        # Report the list of output AstroData objects to the reduction
        # context
        rc.report_output(adoutput_list)
        
        yield rc
    
    def clearCalCache(self, rc):
        # print "pG61:", rc.calindfile
        rc.persist_cal_index(rc.calindfile, newindex={})
        scals = rc["storedcals"]
        if scals:
            if os.path.exists(scals):
                shutil.rmtree(scals)
            cachedict = rc["cachedict"]
            for cachename in cachedict:
                cachedir = cachedict[cachename]
                if not os.path.exists(cachedir):
                    os.mkdir(cachedir)
        
        yield rc
    
    def contextReport(self, rc):
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                 logLevel=rc["logLevel"])
        
        log.fullinfo(rc.report(report_history=rc["report_history"],
                               internal_dict=rc["internal_dict"],
                               context_vars=rc["context_vars"],
                               report_inputs=rc["report_inputs"],
                               report_parameters=rc["report_parameters"],
                               showall=rc["showall"]))
        
        yield rc
    
    def correctBackgroundToReferenceImage(self, rc):
        """
        This primitive does an additive correction to a set
        of images to put their sky background at the same level
        as the reference image before stacking.
        """

        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])

        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", 
                                 "correctBackgroundToReferenceImage",
                                 "starting"))

        # Initialize the list of output AstroData objects
        adoutput_list = []

        # Check whether two or more input AstroData objects were provided
        adinput = rc.get_inputs_as_astrodata()
        if len(adinput) <= 1:
            log.warning("No correction will be performed, since at least " \
                        "two input AstroData objects are required for " \
                        "correctBackgroundToReferenceImage")

            # Set the input AstroData object list equal to the output AstroData
            # objects list without further processing
            adoutput_list = adinput

        else:
            # Call the user level function
            adoutput = pp.correct_background_to_reference_image(adinput=adinput)
            
            # Change the filenames and append to output list
            for ad in adoutput:
                ad.filename = gt.fileNameUpdater(adIn=ad, suffix=rc["suffix"], 
                                                 strip=True)
                adoutput_list.append(ad)
 
        # Report the list of output AstroData objects to the reduction
        # context
        rc.report_output(adoutput_list)

        yield rc

    def correctWCSToReferenceCatalog(self, rc):
        """
        This primitive calculates the average astrometric offset between
        the positions of sources in the reference catalog, and their
        corresponding object in the object catalog.
        It then reports the astrometric correction vector.
        If the 'correctWCS' parameter == True, it then applies that
        correction to the WCS of the image and also applies the same
        correction to the RA, DEC columns of the object catalog.
        """

        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])

        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "correctWCSToReferenceCatalog",
                                 "starting"))

        # Initialize the list of output AstroData objects
        adoutput_list = []

        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():

            # Call the correct_wcs_to_reference_catalog user level function
            ad = rg.correct_wcs_to_reference_catalog(adinput=ad, correctWCS=rc["correct_WCS"])[0]

            # Change the filename
            ad.filename = gt.fileNameUpdater(adIn=ad, suffix=rc["suffix"],
                                             strip=True)

            adoutput_list.append(ad)

        # Report the list of output AstroData objects to the reduction
        # context
        rc.report_output(adoutput_list)

        yield rc

    def correctWCSToReferenceImage(self, rc):
        """ 
        This primitive registers images to a reference image by correcting
        the relative error in their world coordinate systems. The function
        uses points of reference common to the reference image and the
        input images to fit the input WCS to the reference one. The fit
        is done by a least-squares minimization of the difference between
        the reference points in the input image pixel coordinate system.
        This function is intended to be followed by the
        align_to_reference_image function, which applies the relative
        transformation encoded in the WCS to transform input images into the
        reference image pixel coordinate system.
        
        The primary registration method is intended to be by direct mapping
        of sources in the image frame to correlated sources in the reference
        frame. This method fails when there are no correlated sources in the
        field, or when the WCSs are very far off to begin with. As a back-up
        method, the user can try correcting the WCS by the shifts indicated 
        in the POFFSET and QOFFSET header keywords (option fallback='header'), 
        or by hand-selecting common points of reference in an IRAF display
        (option fallback='user'). By default, only the direct method is
        attempted, as it is expected that the relative WCS will generally be
        more correct than either indirect method. If the user prefers not to
        attempt direct mapping at all, they may set method to either 'user'
        or 'header'.
        
        In order to use the direct mapping method, sources must have been
        detected in the frame and attached to the AstroData instance in an 
        OBJCAT extension. This can be accomplished via the detectSources
        primitive. Running time is optimal, and sometimes the solution is 
        more robust, when there are not too many sources in the OBJCAT. Try
        running detectSources with threshold=20. The solution may also be
        more robust if sub-optimal sources are rejected from the set of 
        correlated sources (use option cull_sources=True). This option may
        substantially increase the running time if there are many sources in
        the OBJCAT.
        
        It is expected that the relative difference between the WCSs of 
        images to be combined should be quite small, so it may not be necessary
        to allow rotation and scaling degrees of freedom when fitting the image
        WCS to the reference WCS. However, if it is desired, the options 
        rotate and scale can be used to allow these degrees of freedom. Note
        that these options refer to rotation/scaling of the WCS itself, not the
        images. Significant rotation and scaling of the images themselves 
        will generally already be encoded in the WCS, and will be corrected for
        when the images are aligned.
        
        The WCS keywords in the headers of the output images are updated
        to contain the optimal registration solution.
        
        :param method: method to use to generate reference points. Options
                       are 'sources' to directly map sources from the input
                       image to the reference image, 'user' to select 
                       reference points by cursor from an IRAF display, 
                       or 'header' to generate reference points from the 
                       POFFSET and QOFFSET keywords in the image headers.
        :type method: string, either 'sources', 'user', or 'header'
        
        :param fallback: back-up method for generating reference points.
                         if the primary method fails. The 'sources' option
                         cannot be used as the fallback.
        :type fallback: string, either 'user' or 'header'.
        
        :param cull_sources: flag to indicate whether sub-optimal sources 
                             should be rejected before attempting a direct
                             mapping. If True, sources that are saturated, 
                             not well-fit by a Gaussian, too broad, or too
                             elliptical will be eliminated from the
                             list of reference points.
        :type cull_sources: bool
        
        :param rotate: flag to indicate whether the input image WCSs should
                       be allowed to rotate with respect to the reference image
                       WCS
        :type rotate: bool
        
        :param scale: flag to indicate whether the input image WCSs should
                      be allowed to scale with respect to the reference image
                      WCS. The same scale factor is applied to all dimensions.
        :type scale: bool
        """
        
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "correctWCSToReferenceImage",
                                 "starting"))
        
        # Initialize the list of output AstroData objects
        adoutput_list = []
        
        # Check whether two or more input AstroData objects were provided
        adinput = rc.get_inputs_as_astrodata()
        if len(adinput) <= 1:
            log.warning("No correction will be performed, since at least " \
                        "two input AstroData objects are required for " \
                        "correctWCSToReferenceImage")
            # Set the input AstroData object list equal to the output AstroData
            # objects list without further processing
            adoutput_list = adinput
        else:
            # Call the correct_wcs_to_reference_image user level function
            adoutput = rg.correct_wcs_to_reference_image(
                adinput=adinput, method=rc["method"], fallback=rc["fallback"],
                cull_sources=rc["cull_sources"], rotate=rc["rotate"],
                scale=rc["scale"])
            
            # Change the filenames and append to output list
            for ad in adoutput:
                ad.filename = gt.fileNameUpdater(adIn=ad, suffix=rc["suffix"], 
                                                 strip=True)
                adoutput_list.append(ad)
        
        # Report the list of output AstroData objects to the reduction
        # context
        rc.report_output(adoutput_list)
        
        yield rc
    
    def crashReduce(self, rc):
        raise "Crashing"
        yield rc
    
    def detectSources(self, rc):
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "detectSources", "starting"))
        
        # Initialize the list of output AstroData objects
        adoutput_list = []
        
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():
            
            # Call the detect_sources user level function,
            # which returns a list; take the first entry
            ad = ph.detect_sources(adinput=ad, sigma=rc["sigma"],
                                   threshold=rc["threshold"], fwhm=rc["fwhm"],
                                   max_sources=rc["max_sources"],
                                   centroid_function=rc["centroid_function"],
                                   method=rc["method"])[0]
            
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
    
    def addReferenceCatalog(self, rc):
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])

        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "addReferenceCatalog", "starting"))

        # Initialize the list of output AstroData objects
        adoutput_list = []

        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():

            # Call the add_reference_catalog user level function
            # which returns a list, we take the first entry 
            ad = ph.add_reference_catalog(adinput=ad, source='sdss7')[0]

            # Match the object catalog against the reference catalog
            # Update the refid and refmag columns in the object catalog
            ad = ph.match_objcat_refcat(adinput=ad)[0]

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


    def display(self, rc):
        rc.rq_display(display_id=rc["display_id"])
        
        yield rc
    
    def divideByFlat(self, rc):
        """
        This primitive will divide each SCI extension of the inputs by those
        of the corresponding flat. If the inputs contain VAR or DQ frames,
        those will also be updated accordingly due to the division on the data.
        """
        
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "divideByFlat", "starting"))
        
        # Initialize the list of output AstroData objects
        adoutput_list = []
        
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():
            
            # Check whether the divideByFlat primitive has been run previously
            timestamp_key = self.timestamp_keys["divide_by_flat"]
            if ad.phu_get_key_value(timestamp_key):
                log.warning("No changes will be made to %s, since it has " \
                            "already been processed by divideByFlat" \
                            % (ad.filename))
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue
            
            # Retrieve the appropriate flat from the reduction context
            flat = AstroData(rc.get_cal(ad, "processed_flat"))
            
            # If there is no appropriate flat, there is no need to divide by
            # the flat in QA context; in SQ context, raise an error
            if flat.filename is None:
                if "QA" in rc.context:
                    log.warning("No changes will be made to %s, since no " \
                                "appropriate flat could be retrieved" \
                                % (ad.filename))
                    # Append the input AstroData object to the list of output
                    # AstroData objects without further processing
                    adoutput_list.append(ad)
                    continue
                else:
                    raise Errors.PrimitiveError("No processed flat found for %s" % 
                                                ad.filename)
            
            # Call the divide_by_flat user level function,
            # which returns a list; take the first entry
            ad = pp.divide_by_flat(adinput=ad, flat=flat)[0]
            
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
     
    def failCalibration(self,rc):
        # Mark a given calibration "fail" and upload it 
        # to the system. This is intended to be used to mark a 
        # calibration file that has already been uploaded, so that
        # it will not be returned as a valid match for future data.
        
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        
        # Initialize the list of output AstroData objects 
        adoutput_list = []
        
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():
            
            # Change the two keywords -- BAD and NO = Fail
            ad.phu_set_key_value("RAWGEMQA","BAD")
            ad.phu_set_key_value("RAWPIREQ","NO")
            log.fullinfo("%s has been marked %s" % (ad.filename,ad.qa_state()))
            
            # Append the output AstroData object to the list
            # of output AstroData objects
            adoutput_list.append(ad)
        
        # Report the list of output AstroData objects to the 
        # reduction context
        rc.report_output(adoutput_list)
        
        # Run the storeCalibration primitive, so that the 
        # failed file gets re-uploaded
        rc.run("storeCalibration")
        
        yield rc
            
    def getCalibration(self, rc):
        
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        
        # Retrieve type of calibration requested
        caltype = rc["caltype"]
        if caltype == None:
            log.error("getCalibration: caltype not set")
            raise Errors.PrimitiveError("getCalibration: caltype not set")
        
        # Retrieve source of calibration
        source = rc["source"]
        if source == None:
            source = "all"
            
        # Check whether calibrations are already available
        calibrationless_adlist = []
        adinput = rc.get_inputs_as_astrodata()
        #for ad in adinput:
        #    ad.mode = "update"
        #    calurl = rc.get_cal(ad,caltype)
        #    if not calurl:
        #        calibrationless_adlist.append(ad)
        calibrationless_adlist = adinput
        # Request any needed calibrations
        if len(calibrationless_adlist) ==0:
            # print "pG603: calibrations for all files already present"
            pass
        else:
            rc.rq_cal(caltype, calibrationless_adlist, source=source)
        
        yield rc
        
    def getList(self, rc):
        """
        This primitive will check the files in the stack lists are on disk,
        and then update the inputs list to include all members of the stack 
        for stacking.
        
        :param purpose: 
        :type purpose: string
        """
        
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        
        # Get purpose of list
        sidset = set()
        purpose=rc["purpose"]
        if purpose is None:
            purpose = ""
        
        # Get ID for all inputs
        for inp in rc.inputs:
            sidset.add(purpose+IDFactory.generate_stackable_id(inp.ad))
        
        # Import inputs from all lists
        for sid in sidset:
            stacklist = rc.get_list(sid) #.filelist
            log.stdinfo("List for stack id %s(...):" % sid[0:35])
            for f in stacklist:
                rc.report_output(f, stream=rc["to_stream"])
                log.stdinfo("   %s" % os.path.basename(f),
                             category="list")
        
        yield rc
    
    def getProcessedBias(self, rc):
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        
        caltype = "processed_bias"
        source = rc["source"]
        if source == None:
            rc.run("getCalibration(caltype=%s)" % caltype)
        else:
            rc.run("getCalibration(caltype=%s, source=%s)" % (caltype,source))
        
        # List calibrations found
        first = True
        for ad in rc.get_inputs_as_astrodata():
            calurl = rc.get_cal(ad, caltype) #get from cache
            if calurl:
                cal = AstroData(calurl)
                if cal.filename is None:
                    if "QA" not in rc.context:
                        raise Errors.InputError("Calibration not found for " \
                                                "%s" % ad.filename)
                else:
                    if first:
                        log.stdinfo("getCalibration: Results")
                        first = False
                    log.stdinfo("   %s\n      for %s" % (cal.filename,
                                                         ad.filename))
            else: 
                if "QA" not in rc.context:
                    raise Errors.InputError("Calibration not found for %s" % 
                                            ad.filename)
        
        yield rc
    
    def getProcessedDark(self, rc):
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        
        caltype = "processed_dark"
        source = rc["source"]
        if source == None:
            rc.run("getCalibration(caltype=%s)" % caltype)
        else:
            rc.run("getCalibration(caltype=%s, source=%s)" % (caltype,source))
            
        # List calibrations found
        first = True
        for ad in rc.get_inputs_as_astrodata():
            calurl = rc.get_cal(ad, caltype) #get from cache
            if calurl:
                cal = AstroData(calurl)
                if cal.filename is None:
                    if "QA" not in rc.context:
                        raise Errors.InputError("Calibration not found for " \
                                                "%s" % ad.filename)
                else:
                    if first:
                        log.stdinfo("getCalibration: Results")
                        first = False
                    log.stdinfo("   %s\n      for %s" % (cal.filename,
                                                         ad.filename))
            else: 
                if "QA" not in rc.context:
                    raise Errors.InputError("Calibration not found for %s" % 
                                            ad.filename)
        
        yield rc
    
    def getProcessedFlat(self, rc):
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        
        caltype = "processed_flat"
        source = rc["source"]
        if source == None:
            rc.run("getCalibration(caltype=%s)" % caltype)
        else:
            rc.run("getCalibration(caltype=%s, source=%s)" % (caltype,source))
        
        # List calibrations found
        first = True
        for ad in rc.get_inputs_as_astrodata():
            calurl = rc.get_cal(ad, caltype) #get from cache
            if calurl:
                cal = AstroData(calurl)
                if cal.filename is None:
                    if "QA" not in rc.context:
                        raise Errors.InputError("Calibration not found for " \
                                                "%s" % ad.filename)
                else:
                    if first:
                        log.stdinfo("getCalibration: Results")
                        first = False
                    log.stdinfo("   %s\n      for %s" % (cal.filename,
                                                         ad.filename))
            else: 
                if "QA" not in rc.context:
                    raise Errors.InputError("Calibration not found for %s" % 
                                            ad.filename)
        
        yield rc
    
    def measureZP(self, rc):
        """
        This primitive will determine the zeropoint by looking at
        sources in the OBJCAT for whic a reference catalog magnitude
        has been determined.
        """

        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "measureZP", "starting"))
        
        # Initialize the list of output AstroData objects
        adoutput_list = []

        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():

            # Call the measure_zp user level function
            ad = qa.measure_zp(adinput=ad)[0]

            # Append the output AstroData object to the list 
            # of output AstroData objects
            adoutput_list.append(ad)

        # Report the list of output AstroData objects to the reduction
        # context
        rc.report_output(adoutput_list)

        yield rc

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
    
    def nonlinearityCorrect(self, rc):
        """
        This primitive corrects the input for non-linearity
        """
        
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "nonlinearityCorrect",
                                "starting"))
        
        # Initialize the list of output AstroData objects
        adoutput_list = []
        
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():
            
            # Check whether the nonlinearityCorrect primitive has been run
            # previously
            timestamp_key = self.timestamp_keys["nonlinearity_correct"]
            if ad.phu_get_key_value(timestamp_key):
                log.warning("No changes will be made to %s, since it has " \
                            "already been processed by nonlinearityCorrect" \
                            % (ad.filename))
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue
            
            # Call the nonlinearity_correct user level function,
            # which returns a list; take the first entry
            ad = pp.nonlinearity_correct(adinput=ad)[0]
            
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
    
    def pause(self, rc):
        rc.request_pause()
        yield rc
    
    def setContext(self, rc):
        rc.update(rc.localparms)
        yield rc
    
    def showCals(self, rc):
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        
        if str(rc["showcals"]).lower() == "all":
            num = 0
            # print "pG256: showcals=all", repr (rc.calibrations)
            for calkey in rc.calibrations:
                num += 1
                log.stdinfo(rc.calibrations[calkey], category="calibrations")
            if (num == 0):
                log.stdinfo("There are no calibrations in the cache.")
        else:
            for adr in rc.inputs:
                sid = IDFactory.generate_astro_data_id(adr.ad)
                num = 0
                for calkey in rc.calibrations:
                    if sid in calkey :
                        num += 1
                        log.stdinfo(rc.calibrations[calkey], 
                                     category="calibrations")
            if (num == 0):
                log.stdinfo("There are no calibrations in the cache.")
        
        yield rc
    ptusage_showCals="Used to show calibrations currently in cache for inputs."
    
    def showInputs(self, rc):
        """
        A simple primitive to show the filenames for the current inputs to 
        this primitive.
        """
        
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        log.stdinfo("Inputs:", category="inputs")
        print "pG977:", id(rc), repr(rc.inputs)
        #if "stream" in rc:
        #    stream = rc["stream"]
        #else:
        #    stream = "main"
        
        log.stdinfo("stream: %s" % (rc._current_stream))
        for inf in rc.inputs:
            log.stdinfo("  %s" % inf.filename, category="inputs")
        
        yield rc
    showFiles = showInputs
    
    def showList(self, rc):
        """
        This primitive will log the list of files in the stacking list matching
        the current inputs and 'purpose' value.
        
        :param purpose: 
        :type purpose: string
        """
        
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        
        sidset = set()
        purpose = rc["purpose"]
        if purpose is None:
            purpose = ""
        # print "pG710"
        if purpose == "all":
            allsids = rc.get_stack_ids()
            # print "pG713:", repr(allsids)
            for sid in allsids:
                sidset.add(sid)
        else:
            for inp in rc.inputs:
                sidset.add(purpose+IDFactory.generate_stackable_id(inp.ad))
        for sid in sidset:
            stacklist = rc.get_list(sid) #.filelist
            log.status("List for stack id=%s" % sid, category="list")
            if len(stacklist) > 0:
                for f in stacklist:
                    log.status("   %s" % os.path.basename(f), category="list")
            else:
                log.status("No datasets in list", category="list")
        
        yield rc
    
    def showParameters(self, rc):
        """
        A simple primitive to log the currently set parameters in the 
        reduction context dictionary.
        """
        
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        
        rcparams = rc.param_names()
        if (rc["show"]):
            toshows = rc["show"].split(":")
            for toshow in toshows:
                if toshow in rcparams:
                    log.stdinfo("%s = %s" % (toshow, repr(rc[toshow])),
                                 category="parameters")
                else:
                    log.stdinfo("%s is not set" % (toshow),
                                 category="parameters")
        else:
            for param in rcparams:
                log.stdinfo("%s = %s" % (param, repr(rc[param])),
                             category="parameters")
        # print "all",repr(rc.parm_dict_by_tag("showParams", "all"))
        # print "iraf",repr(rc.parm_dict_by_tag("showParams", "iraf"))
        # print "test",repr(rc.parm_dict_by_tag("showParams", "test"))
        # print "sdf",repr(rc.parm_dict_by_tag("showParams", "sdf"))
        # print repr(dir(rc.ro.primDict[rc.ro.curPrimType][0]))
        
        yield rc
    
    def sleep(self, rc):
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        if rc["duration"]:
            dur = float(rc["duration"])
        else:
            dur = 5.
        log.status("Sleeping for %f seconds" % dur)
        time.sleep(dur)
        
        yield rc
    
    def stackFrames(self, rc):
        """
        This primitive will stack each science extension in the input dataset.
        New variance extensions are created from the stacked science extensions
        and the data quality extensions are propagated through to the final
        file.
        
        :param method: type of combining method to use. The options are
                       'average' or 'median'.
        :type method: string
        """
        
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "stackFrames", "starting"))
        
        # Initialize the list of output AstroData objects
        adoutput_list = []
        
        # Check whether two or more input AstroData objects were provided
        adinput = rc.get_inputs_as_astrodata()
        if len(adinput) <= 1:
            log.stdinfo("No stacking will be performed, since at least " \
                        "two input AstroData objects are required for " \
                        "stackFrames")
            # Set the input AstroData object list equal to the output AstroData
            # objects list without further processing
            adoutput_list = adinput
        else:
            # Call the stack_frames user level function
            adoutput_list = sk.stack_frames(
                adinput=adinput, suffix=rc["suffix"],
                operation=rc["operation"], reject_method=rc["reject_method"],
                mask_type=rc["mask_type"])
        
        # Report the list containing a single AstroData object to the reduction
        # context
        rc.report_output(adoutput_list)
        
        yield rc
    
    def storeCalibration(self, rc):
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        
        # print repr(rc)
        storedcals = rc["cachedict"]["storedcals"]
        
        for ad in rc.get_inputs_as_astrodata():
            fname = os.path.join(storedcals, os.path.basename(ad.filename))
            ad.filename = fname
            # always clobber, perhaps save clobbered file somewhere
            ad.write(filename = fname, rename = True, clobber=True)
            log.stdinfo("File saved to %s" % fname)
            if "upload" in rc.context:
                try:
                    upload_calibration(ad.filename)
                except:
                    log.warning("Unable to upload file to calibration system")
                else:
                    log.stdinfo("File stored in calibration system")
            yield rc
        
        yield rc
    
    def storeProcessedBias(self, rc):
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "storeProcessedBias",
                                 "starting"))
        
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():
            
            # Updating the file name with the suffix for this primitive and
            # then report the new file to the reduction context
            ad.filename = gt.fileNameUpdater(adIn=ad, suffix=rc["suffix"],
                                             strip=True)
            
            # Adding a PROCBIAS time stamp to the PHU
            gt.mark_history(adinput=ad, keyword="PROCBIAS")
            
            # Refresh the AD types to reflect new processed status
            ad.refresh_types()
        
        # Upload bias(es) to cal system
        rc.run("storeCalibration")
        
        yield rc
    
    def storeProcessedDark(self, rc):
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "storeProcessedDark",
                                 "starting"))
        
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():
            
            # Updating the file name with the suffix for this primitive and
            # then report the new file to the reduction context
            ad.filename = gt.fileNameUpdater(adIn=ad, suffix=rc["suffix"],
                                             strip=True)
            
            # Adding a PROCDARK time stamp to the PHU
            gt.mark_history(adinput=ad, keyword="PROCDARK")
            
            # Refresh the AD types to reflect new processed status
            ad.refresh_types()
        
        # Upload to cal system
        rc.run("storeCalibration")
        
        yield rc
    
    def storeProcessedFlat(self, rc):
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "storeProcessedFlat",
                                 "starting"))
        
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():
            
            # Updating the file name with the suffix for this primitive and
            # then report the new file to the reduction context
            ad.filename = gt.fileNameUpdater(adIn=ad, suffix=rc["suffix"],
                                             strip=True)
            
            # Adding a PROCFLAT time stamp to the PHU
            gt.mark_history(adinput=ad, keyword="PROCFLAT")
            
            # Refresh the AD types to reflect new processed status
            ad.refresh_types()
        
        # Upload to cal system
        rc.run("storeCalibration")
        
        yield rc
    
    def subtractDark(self, rc):
        """
        This primitive will subtract each SCI extension of the inputs by those
        of the corresponding dark. If the inputs contain VAR or DQ frames,
        those will also be updated accordingly due to the subtraction on the 
        data.
        """
        
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "subtractDark", "starting"))
        
        # Initialize the list of output AstroData objects
        adoutput_list = []
        
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():
            
            # Check whether the subtractDark primitive has been run previously
            timestamp_key = self.timestamp_keys["subtract_dark"]
            if ad.phu_get_key_value(timestamp_key):
                log.warning("No changes will be made to %s, since it has " \
                            "already been processed by subtractDark" \
                            % (ad.filename))
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue
            
            # Retrieve the appropriate dark from the reduction context
            dark = AstroData(rc.get_cal(ad, "processed_dark"))

            # If there is no appropriate dark, there is no need to subtract the
            # dark
            if dark.filename is None:
                log.warning("No changes will be made to %s, since no " \
                            "appropriate dark could be retrieved" \
                            % (ad.filename))
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue
            
            # Call the subtract_dark user level function,
            # which returns a list; take the first entry
            ad = pp.subtract_dark(adinput=ad, dark=dark)[0]
            
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
    
    def time(self, rc):
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        
        cur = datetime.now()
        elap = ""
        if rc["lastTime"] and not rc["start"]:
            td = cur - rc["lastTime"]
            elap = " (%s)" % str(td)
        log.stdinfo("Time: %s %s" % (str(datetime.now()), elap))
        rc.update({"lastTime":cur})
        
        yield rc
    
    def writeOutputs(self, rc):
        """
        A primitive that may be called by a recipe at any stage to
        write the outputs to disk.
        If suffix is set during the call to writeOutputs, any previous 
        suffixs will be striped and replaced by the one provided.
        examples: 
        writeOutputs(suffix= '_string'), writeOutputs(prefix= '_string') 
        or if you have a full file name in mind for a SINGLE file being 
        ran through Reduce you may use writeOutputs(outfilename='name.fits').
        
        :param strip: Strip the previously suffixed strings off file name?
        :type strip: Python boolean (True/False)
                     default: False
        
        :param clobber: Write over any previous file with the same name that
                        all ready exists?
        :type clobber: Python boolean (True/False)
                       default: False
        
        :param suffix: Value to be post pended onto each input name(s) to 
                       create the output name(s).
        :type suffix: string
        
        :param prefix: Value to be post pended onto each input name(s) to 
                         create the output name(s).
        :type prefix: string
        
        :param outfilename: The full filename you wish the file to be written
                            to. Note: this only works if there is ONLY ONE file
                            in the inputs.
        :type outfilename: string
        """
        
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        
        # Logging current values of suffix and prefix
        log.fullinfo("suffix = %s" % str(rc["suffix"]))
        log.fullinfo("prefix = %s" % str(rc["prefix"]))
        log.fullinfo("strip = %s" % str(rc["strip"]))
        
        # Initialize the list of output AstroData objects
        adoutput_list = []
        
        for ad in rc.get_inputs_as_astrodata():
            if rc["suffix"] and rc["prefix"]:
                ad.filename = gt.fileNameUpdater(adIn=ad,
                                                 prefix=rc["prefix"],
                                                 suffix=rc["suffix"],
                                                 strip=rc["strip"])
                log.fullinfo("File name updated to %s" % ad.filename)
                outfilename = os.path.basename(ad.filename)
            
            elif rc["suffix"]:
                # If the value of "suffix" was set, then set the file name 
                # to be written to disk to be postpended by it
                ad.filename = gt.fileNameUpdater(adIn=ad,
                                                 suffix=rc["suffix"],
                                                 strip=rc["strip"])
                log.fullinfo("File name updated to %s" % ad.filename)
                outfilename = os.path.basename(ad.filename)
            
            elif rc["prefix"]:
                # If the value of "prefix" was set, then set the file name 
                # to be written to disk to be pre pended by it
                ad.filename = gt.fileNameUpdater(adIn=ad,
                                                 prefix=rc["prefix"],
                                                 strip=rc["strip"])
                log.fullinfo("File name updated to %s" % ad.filename)
                outfilename = os.path.basename(ad.filename)
            
            elif rc["outfilename"]:
                # If the "outfilename" was set, set the file name of the file 
                # file to be written to this
                
                # Check that there is not more than one file to be written
                # to this file name, if so throw exception
                if len(rc.get_inputs_as_astrodata()) > 1:
                    message = """
                        More than one file was requested to be written to
                        the same name %s""" % (rc["outfilename"])
                    log.critical(message)
                    raise Errors.PrimitiveError(message)
                else:
                    outfilename = rc["outfilename"]
            else:
                # If no changes to file names are requested then write inputs
                # to their current file names
                outfilename = os.path.basename(ad.filename) 
                log.fullinfo("not changing the file name to be written " \
                             "from its current name")
            
            # Finally, write the file to the name that was decided 
            # upon above
            log.stdinfo("Writing to file %s" % outfilename)
            
            # AstroData checks if the output exists and raises an exception
            ad.write(filename=outfilename, clobber=rc["clobber"])
            
            # Append the output AstroData object to the list of output
            # AstroData objects
            adoutput_list.append(ad)
        
        # Report the list of output AstroData objects to the
        # reduction context
        rc.report_output(adoutput_list)
        
        yield rc
