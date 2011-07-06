import os
from datetime import datetime
import shutil
import time

from astrodata import AstroData
from astrodata import Errors
from astrodata import IDFactory
from astrodata.adutils import gemLog
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
        This primitive will create a numpy array for the data quality 
        of each SCI frame of the input data. This will then have a 
        header created and append to the input using AstroData as a DQ 
        frame. The value of a pixel will be the sum of the following: 
        (0=good, 1=bad pixel (found in bad pixel mask), 
        2=value is non linear, 4=pixel is saturated)
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to 
                        screen. OR the message level as a string (i.e.,
                        'critical', 'status', 'fullinfo'...)
        """
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "addDQ", "starting"))
        # Initialize the list of output AstroData objects
        adoutput_list = []
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs(style="AD"):
            # Check whether the addDQ primitive has been run previously
            if ad.phu_get_key_value("ADDDQ"):
                log.warning("%s has already been processed by addDQ" \
                            % (ad.filename))
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue
            # Call the add_dq user level function
            ad = sdz.add_dq(adinput=ad)
            # Append the output AstroData object (which is currently in the
            # form of a list) to the list of output AstroData objects
            adoutput_list.append(ad[0])
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
        :type purpose: string, either: "" for regular image stacking, 
                       or 'fringe' for fringe stacking.
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to 
                        screen. OR the message level as a string (i.e.,
                        'critical', 'status', 'fullinfo'...)
        """
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        # Perform an update to the stack cache file (or create it) using the
        # current inputs in the reduction context
        purpose = rc["purpose"]
        if purpose is None:
            purpose = ""
        # Call the rq_stack_update method
        rc.rq_stack_update(purpose=purpose)
        # Write the files in the stack to disk if they do not already exist
        for ad in rc.get_inputs(style="AD"):
            if not os.path.exists(ad.filename):
                log.fullinfo("writing %s to disk" % ad.filename,
                             category="list")
                ad.write(ad.filename)
        
        yield rc
    
    def addVAR(self, rc):
        """
        This primitive uses numpy to calculate the variance of each SCI frame
        in the input files and appends it as a VAR frame using AstroData.
        
        The calculation will follow the formula:
        variance = (read noise/gain)2 + max(data,0.0)/gain
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to 
                        screen. OR the message level as a string (i.e.,
                        'critical', 'status', 'fullinfo'...)
        """
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "addVAR", "starting"))
        # Initialize the list of output AstroData objects
        adoutput_list = []
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs(style="AD"):
            # Call the add_var user level function
            ad = sdz.add_var(adinput=ad, read_noise=rc["read_noise"],
                             poisson_noise=rc["poisson_noise"])
            # Append the output AstroData object (which is currently in the
            # form of a list) to the list of output AstroData objects
            adoutput_list.append(ad[0])
        # Report the list of output AstroData objects to the reduction
        # context
        rc.report_output(adoutput_list)
        
        yield rc 
    
    def aduToElectrons(self, rc):
        """
        This primitive will convert the inputs from having pixel 
        units of ADU to electrons.
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to 
                        screen. OR the message level as a string (i.e.,
                        'critical', 'status', 'fullinfo'...)
        """
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "aduToElectrons", "starting"))
        # Initialize the list of output AstroData objects
        adoutput_list = []
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs(style="AD"):
            # Check whether the aduToElectrons primitive has been run
            # previously
            if ad.phu_get_key_value("ADU2ELEC"):
                log.warning("%s has already been processed by aduToElectrons" \
                            % (ad.filename))
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue
            # Call the adu_to_electrons user level function
            ad = pp.adu_to_electrons(adinput=ad)
            # Append the output AstroData object (which is currently in the
            # form of a list) to the list of output AstroData objects
            adoutput_list.append(ad[0])
        # Report the list of output AstroData objects to the reduction
        # context
        rc.report_output(adoutput_list)
        
        yield rc
    
    def alignToReferenceImage(self, rc):
        """
        This primitive applies the transformation encoded in the input images
        WCSs to align them with a reference image, in reference image pixel
        coordinates.  The reference image is taken to be the first image in
        the input list.

        By default, the transformation into the reference frame is done via
        interpolation.  The interpolator parameter specifies the interpolation 
        method.  The options are nearest-neighbor, bilinear, or nth-order 
        spline, with n = 2, 3, 4, or 5.  If interpolator is None, 
        no interpolation is done: the input image is shifted by an integer
        number of pixels, such that the center of the frame matches up as
        well as possible.  The variance plane, if present, is transformed in
        the same way as the science data.  

        The data quality plane, if present, must be handled a little
        differently.  DQ flags are set bit-wise, such that each pixel is the 
        sum of any of the following values: 0=good pixel,
        1=bad pixel (from bad pixel mask), 2=nonlinear, 4=saturated, etc.
        To transform the DQ plane without losing flag information, it is
        unpacked into separate masks, each of which is transformed in the same
        way as the science data.  A pixel is flagged if it had greater than
        1% influence from a bad pixel.  The transformed masks are then added
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
        log = gemLog.getGeminiLog(logType=rc['logType'],logLevel=rc['logLevel'])
        log.debug(gt.log_message("primitive", "alignToReferenceImage", 
                                 "starting"))
        adinput = rc.get_inputs(style='AD')
        if len(adinput)<2:
            log.warning("At least two images must be provided to " +
                        "alignToReferenceImage")
            # Report input to RC without change
            adoutput_list = adinput
        else:
            adoutput_list = rs.align_to_reference_image(
                                         adinput=adinput,
                                         interpolator=rc['interpolator'])
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
     
    def correctWCSToReferenceImage(self, rc):
        """ 
        This primitive registers images to a reference image by correcting
        the relative error in their world coordinate systems.  The function
        uses points of reference common to the reference image and the
        input images to fit the input WCS to the reference one.  The fit
        is done by a least-squares minimization of the difference between
        the reference points in the input image pixel coordinate system.
        This function is intended to be followed by the align_to_reference_image
        function, which applies the relative transformation encoded in the
        WCS to transform input images into the reference image pixel
        coordinate system.
        
        The primary registration method is intended to be by direct mapping
        of sources in the image frame to correlated sources in the reference
        frame. This method fails when there are no correlated sources in the
        field, or when the WCSs are very far off to begin with.  As a back-up
        method, the user can try correcting the WCS by the shifts indicated 
        in the POFFSET and QOFFSET header keywords (option fallback='header'), 
        or by hand-selecting common points of reference in an IRAF display
        (option fallback='user').  By default, only the direct method is
        attempted, as it is expected that the relative WCS will generally be
        more correct than either indirect method.  If the user prefers not to
        attempt direct mapping at all, they may set method to either 'user'
        or 'header'.

        In order to use the direct mapping method, sources must have been
        detected in the frame and attached to the AstroData instance in an 
        OBJCAT extension.  This can be accomplished via the detectSources
        primitive.  Running time is optimal, and sometimes the solution is 
        more robust, when there are not too many sources in the OBJCAT.  Try
        running detectSources with threshold=20.  The solution may also be
        more robust if sub-optimal sources are rejected from the set of 
        correlated sources (use option cull_sources=True).  This option may
        substantially increase the running time if there are many sources in
        the OBJCAT.

        It is expected that the relative difference between the WCSs of 
        images to be combined should be quite small, so it may not be necessary
        to allow rotation and scaling degrees of freedom when fitting the image
        WCS to the reference WCS.  However, if it is desired, the options 
        rotate and scale can be used to allow these degrees of freedom.  Note
        that these options refer to rotation/scaling of the WCS itself, not the
        images.  Significant rotation and scaling of the images themselves 
        will generally already be encoded in the WCS, and will be corrected for
        when the images are aligned.

        The WCS keywords in the headers of the output images are updated
        to contain the optimal registration solution.

        Log messages will go to a 'main' type logger object, if it exists.
        or a null logger (ie. no log file, no messages to screen) if it does 
        not.

        :param method: method to use to generate reference points. Options
                       are 'sources' to directly map sources from the input
                       image to the reference image, 'user' to select 
                       reference points by cursor from an IRAF display, 
                       or 'header' to generate reference points from the 
                       POFFSET and QOFFSET keywords in the image headers.
        :type method: string, either 'sources', 'user', or 'header'
        
        :param fallback: back-up method for generating reference points.
                         if the primary method fails.  The 'sources' option
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
                      WCS.  The same scale factor is applied to all dimensions.
        :type scale: bool

        """
        log = gemLog.getGeminiLog(logType=rc['logType'],logLevel=rc['logLevel'])
        log.debug(gt.log_message("primitive", "correctWCSToReferenceImage", 
                                 "starting"))

        # Check that there are at least two images to register
        adinput = rc.get_inputs(style='AD')
        if len(adinput)<2:
            log.warning("At least two images must be provided to " +
                        "correctWCSToReferenceImage")
            # Report input to RC without change
            adoutput_list = adinput
        else:
            adoutput_list = rg.correct_wcs_to_reference_image(
                                               adinput=adinput,
                                               method=rc['method'], 
                                               fallback=rc['fallback'],
                                               cull_sources=rc['cull_sources'],
                                               rotate=rc['rotate'], 
                                               scale=rc['scale'])
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
        for ad in rc.get_inputs(style="AD"):
            # Call the detect_sources user level function
            ad = ph.detect_sources(adinput=ad,
                                   sigma=rc["sigma"],
                                   threshold=rc["threshold"],
                                   fwhm=rc["fwhm"],
                                   method=rc["method"])
            # Append the output AstroData object (which is currently in the
            # form of a list) to the list of output AstroData objects
            adoutput_list.append(ad[0])
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
        
        This is all conducted in pure Python through the arith 'toolbox' of 
        astrodata. 
        
        It is currently assumed that the same flat file will be applied to all
        input images.
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to 
                        screen. OR the message level as a string (i.e.,
                        'critical', 'status', 'fullinfo'...)
        """
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "divideByFlat", "starting"))

        # Initialize the list of output AstroData objects
        adoutput_list = []
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs(style="AD"):
            # Check whether the divideByFlat primitive has been run previously
            if ad.phu_get_key_value("DIVFLAT"):
                log.warning("%s has already been processed by divideByFlat" \
                            % (ad.filename))
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue

            # Retrieve the appropriate flat
            flat = AstroData(rc.get_cal(ad,"flat"))

            # Take care of the case where there was no, or an invalid flat 
            if flat is None or flat.count_exts("SCI") == 0:
                log.warning("Could not find an appropriate flat for %s" \
                            % (ad.filename))
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue

            # Call the divide_by_flat user level function
            ad = pp.divide_by_flat(adinput=rc.get_inputs(style="AD"),
                                       flat=flat)

            # Append the output AstroData object (which is currently in the
            # form of a list) to the list of output AstroData objects
            adoutput_list.append(ad[0])        
        
        # Report the list of output AstroData objects to the reduction
        # context
        rc.report_output(adoutput_list)
        
        yield rc
     
    def getCal(self, rc):
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        caltype = rc["caltype"]
        if caltype is None:
            log.critical("Requested a calibration no particular " +
                         "calibration type")
            raise Errors.PrimitiveError("get_cal: %s was None" % caltype)
        source = rc["source"]
        if source is None:
            source = "all"
        
        centralSource = False
        localSource = False
        if source == "all":
            centralSource = True
            localSource = True
        if source == "central":
            centralSource = True
        if source == "local":
            localSource = True
        
        inps = rc.get_inputs_as_astro_data()
        
        if localSource:
            rc.rq_cal(caltype, inps, source="local")
            for ad in inps:
                cal = rc.get_cal(ad, caltype)
                if cal is None:
                    print "get central"
                else:
                    print "got local", cal
            
            yield rc
    
    def getList(self, rc):
        """
        This primitive will check the files in the stack lists are on disk,
        and then update the inputs list to include all members of the stack 
        for stacking.
        
        :param purpose: 
        :type purpose: string, either: '' for regular image stacking, 
                       or 'fringe' for fringe stacking.
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to 
                        screen. OR the message level as a string (i.e.,
                        'critical', 'status', 'fullinfo'...)
        """
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        sidset = set()
        purpose=rc["purpose"]
        if purpose is None:
            purpose = ""
        for inp in rc.inputs:
            sidset.add(purpose+IDFactory.generate_stackable_id(inp.ad))
        for sid in sidset:
            stacklist = rc.get_list(sid) #.filelist
            log.stdinfo("List for stack id=%s" % sid, category="list")
            for f in stacklist:
                rc.report_output(f)
                log.stdinfo("   %s" % os.path.basename(f),
                             category="list")
        
        yield rc
    
    def getProcessedBias(self, rc):
        """
        This primitive will check the files in the lists that are on disk,
        and then update the inputs list to include all members of the list.
        """
        rc.rq_cal("bias", rc.get_inputs(style="AD"))
        yield rc
    
    def getProcessedDark(self, rc):
        """
        A primitive to search and return the appropriate calibration dark from
        a server for the given inputs.
        """
        rc.rq_cal("dark", rc.get_inputs(style="AD"))
        yield rc
    
    def getProcessedFlat(self, rc):
        """
        A primitive to search and return the appropriate calibration flat from
        a server for the given inputs.
        
        """
        rc.rq_cal("flat", rc.get_inputs(style="AD"))
        yield rc
    
    def getProcessedFringe(self, rc):
        """
        A primitive to search and return the appropriate calibration flat from
        a server for the given inputs.
        
        """
        rc.rq_cal("fringe", rc.get_inputs(style="AD"))
        yield rc
    
    def measureIQ(self, rc):
        """
        This primitive will detect the sources in the input images and fit
        both Gaussian and Moffat models to their profiles and calculate the 
        Image Quality and seeing from this.
        
        :param function: Function for centroid fitting
        :type function: string, can be: 'moffat','gauss' or 'both'; 
                        Default 'both'
                        
        :param display: Flag to turn on displaying the fitting to ds9
        :type display: Python boolean (True/False)
                       Default: True
        
        :param qa: flag to use a grid of sub-windows for detecting the sources
                   in the image frames, rather than the entire frame all at
                   once.
        :type qa: Python boolean (True/False)
                  default: True
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to 
                        screen. OR the message level as a string (i.e.,
                        'critical', 'status', 'fullinfo'...)
        """
        #@@FIXME: Detecting sources is done here as well. This should
        # eventually be split up into separate primitives, i.e. detectSources
        # and measureIQ.
        
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "measureIQ", "starting"))
        # Initialize the list of output AstroData objects
        adoutput_list = []
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs(style="AD"):
            # Call the measure_iq user level function
            ad = qa.measure_iq(adinput=ad, 
                               centroid_function=rc["centroid_function"],
                               display=rc["display"], qa=rc["qa"])
            # Append the output AstroData object (which is currently in the
            # form of a list) to the list of output AstroData objects
            adoutput_list.append(ad[0])
        # Report the list of output AstroData objects to the reduction
        # context
        rc.report_output(adoutput_list)
        
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
        for ad in rc.get_inputs(style="AD"):
            # Check whether the nonlinearityCorrect primitive has been run
            # previously
            if ad.phu_get_key_value("LINCOR"):
                log.warning("%s has already been processed by " \
                            "nonlinearityCorrect" % (ad.filename))
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue
            # Call the nonlinearity_correct user level function
            ad = pp.nonlinearity_correct(adinput=ad)
            # Append the output AstroData object (which is currently in the
            # form of a list) to the list of output AstroData objects
            adoutput_list.append(ad[0])
        # Report the list of output AstroData objects to the reduction
        # context
        rc.report_output(adoutput_list)
        
        yield rc
    
    def pause(self, rc):
        rc.request_pause()
        yield rc
    
    def scaleFringeToScience(self, rc):
        """
        This primitive will scale the fringes to their matching science data
        in the inputs.
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
        log.debug(gt.log_message("primitive", "scaleFringeToScience",
                                 "starting"))
        # Initialize the list of output AstroData objects
        adoutput_list = []
        # Loop over each input AstroData object in the input list
        count = 0
        for ad in rc.get_inputs(style="AD"):
            # Check whether the scaleFringeToScience primitive has been run
            # previously
            if ad.phu_get_key_value("SCALEFRG"):
                log.warning("%s has already been processed by " \
                            "scaleFringeToScience" % (ad.filename))
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue

            # Get the appropriate fringe frame
            fringe = AstroData(rc.get_cal(ad, "fringe"))

            # Take care of the case where there was no, or an invalid fringe 
            if fringe.count_exts("SCI") == 0:
                log.warning("Could not find an appropriate fringe for %s" \
                            % (ad.filename))
                # Append a blank entry to the fringe list
                adoutput_list.append(None)
                continue

            # Call the scale_fringe_to_science user level function
            # (this returns the scaled fringe frame)
            ad = pp.scale_fringe_to_science(adinput=fringe, science=ad,
                                            stats_scale=rc["stats_scale"])

            # Append the output AstroData object (which is currently in the
            # form of a list) to the list of output AstroData objects
            adoutput_list.append(ad[0])
            count += 1

        # Report the list of output AstroData objects and the scaled fringe
        # frames to the reduction context

        rc.report_output(adoutput_list, stream="fringe")
        rc.report_output(rc.get_inputs(style="AD"))
        
        yield rc
    
    def setContext(self, rc):
        rc.update(rc.localparms)
        yield rc
    
    def showCals(self, rc):
        """
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to 
                        screen. OR the message level as a string (i.e.,
                        'critical', 'status', 'fullinfo'...)
        """
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        if str(rc["showcals"]).lower() == "all":
            num = 0
            # print "pG256: showcals=all", repr (rc.calibrations)
            for calkey in rc.calibrations:
                num += 1
                log.fullinfo(rc.calibrations[calkey], category="calibrations")
            if (num == 0):
                log.warning("There are no calibrations in the cache.")
        else:
            for adr in rc.inputs:
                sid = IDFactory.generate_astro_data_id(adr.ad)
                num = 0
                for calkey in rc.calibrations:
                    if sid in calkey :
                        num += 1
                        log.fullinfo(rc.calibrations[calkey], 
                                     category="calibrations")
            if (num == 0):
                log.warning("There are no calibrations in the cache.")
        
        yield rc
    ptusage_showCals="Used to show calibrations currently in cache for inputs."
    
    def showInputs(self, rc):
        """
        A simple primitive to show the filenames for the current inputs to 
        this primitive.
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to 
                        screen. OR the message level as a string (i.e.,
                        'critical', 'status', 'fullinfo'...)
        """
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        log.fullinfo("Inputs:", category="inputs")
        if "stream" in rc:
            stream = rc["stream"]
        else:
            stream = "main"
            
        log.fullinfo("stream: "+stream)
        for inf in rc.inputs:
            log.fullinfo("  %s" % inf.filename, category="inputs")
        
        yield rc
    showFiles = showInputs
    
    def showList(self, rc):
        """
        This primitive will log the list of files in the stacking list matching
        the current inputs and 'purpose' value.
        
        :param purpose: 
        :type purpose: string, either: '' for regular image stacking, 
                       or 'fringe' for fringe stacking.
                       
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to 
                        screen. OR the message level as a string (i.e.,
                        'critical', 'status', 'fullinfo'...)
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
                    log.status("    %s" % os.path.basename(f), category="list")
            else:
                log.status("no datasets in list", category="list")
        
        yield rc
    
    def showParameters(self, rc):
        """
        A simple primitive to log the currently set parameters in the 
        reduction context dictionary.
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to 
                        screen. OR the message level as a string (i.e.,
                        'critical', 'status', 'fullinfo'...)
        """
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        rcparams = rc.param_names()
        if (rc["show"]):
            toshows = rc["show"].split(":")
            for toshow in toshows:
                if toshow in rcparams:
                    log.fullinfo("%s = %s" % (toshow, repr(rc[toshow])),
                                 category="parameters")
                else:
                    log.fullinfo("%s is not set" % (toshow),
                                 category="parameters")
        else:
            for param in rcparams:
                log.fullinfo("%s = %s" % (param, repr(rc[param])),
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
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to 
                        screen. OR the message level as a string (i.e.,
                        'critical', 'status', 'fullinfo'...)
        """
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "stackFrames", "starting"))
        # Call the stack_frames user level function
        adinput = rc.get_inputs(style='AD')
        if len(adinput)<2:
            log.warning("At least two frames must be provided to " +
                        "stackFrames")
            # Report input to RC without change
            adoutput_list = adinput
        else:
            adoutput_list = sk.stack_frames(adinput=rc.get_inputs(style="AD"),
                                   suffix=rc["suffix"],
                                   operation=rc["operation"],
                                   reject_method=rc["reject_method"],
                                   mask_type=rc["mask_type"])
        # Report the list containing a single AstroData object to the reduction
        # context
        rc.report_output(adoutput_list)
        
        yield rc
    
    def storeProcessedBias(self, rc):
        """
        This should be a primitive that interacts with the calibration system 
        (MAYBE) but that isn't up and running yet. Thus, this will just strip 
        the extra postfixes to create the 'final' name for the 
        makeProcessedBias outputs and write them to disk in a storedcals
        folder.
        
        :param clob: Write over any previous file with the same name that
                     all ready exists?
        :type clob: Python boolean (True/False)
                    default: False
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to 
                        screen. OR the message level as a string (i.e.,
                        'critical', 'status', 'fullinfo'...)
        """
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "storeProcessedBias",
                                 "starting"))
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs(style="AD"):
            # Updating the file name with the suffix for this primitive and
            # then report the new file to the reduction context
            ad.filename = gt.fileNameUpdater(adIn=ad, suffix="_bias",
                                             strip=True)
            log.status("File name of stored bias is %s" % ad.filename)
            # Adding a GBIAS time stamp to the PHU
            ad.history_mark(key="GBIAS", comment="fake key to trick CL that " \
                            "GBIAS was used to create this bias")
            # Write the bias frame to disk
            ad.write(os.path.join(rc["storedbiases"], ad.filename), 
                     clobber=rc["clob"])
            log.fullinfo("Bias written to %s" % (rc["storedbiases"]))
        
        yield rc
    
    def storeProcessedDark(self, rc):
        """
        This should be a primitive that interacts with the calibration system 
        (MAYBE) but that isn't up and running yet. Thus, this will just strip 
        the extra postfixes to create the 'final' name for the 
        makeProcessedDark outputs and write them to disk in a storedcals
        folder.
        
        :param clob: Write over any previous file with the same name that
                     all ready exists?
        :type clob: Python boolean (True/False)
                    default: False
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to 
                        screen. OR the message level as a string (i.e.,
                        'critical', 'status', 'fullinfo'...)
        """
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "storeProcessedDark",
                                 "starting"))
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs(style="AD"):
            # Updating the file name with the suffix for this primitive and
            # then report the new file to the reduction context
            ad.filename = gt.fileNameUpdater(adIn=ad, suffix="_dark",
                                             strip=True)
            log.status("File name of stored dark is %s" % ad.filename)
            # Write the dark frame to disk
            ad.write(os.path.join(rc["storeddarks"], ad.filename), 
                     clobber=rc["clob"])
            log.fullinfo("Dark written to %s" % (rc["storeddarks"]))
        
        yield rc
    
    def storeProcessedFlat(self, rc):
        """
        This should be a primitive that interacts with the calibration 
        system (MAYBE) but that isn't up and running yet. Thus, this will 
        just strip the extra postfixes to create the 'final' name for the 
        makeProcessedFlat outputs and write them to disk in a storedcals
        folder.
        
        :param clob: Write over any previous file with the same name that
                     all ready exists?
        :type clob: Python boolean (True/False)
                    default: False
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to 
                        screen. OR the message level as a string (i.e.,
                        'critical', 'status', 'fullinfo'...)
        """
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "storeProcessedFlat",
                                 "starting"))
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs(style="AD"):
            # Updating the file name with the suffix for this primitive and
            # then report the new file to the reduction context
            ad.filename = gt.fileNameUpdater(adIn=ad, suffix="_flat",
                                             strip=True)
            log.status("File name of stored flat is %s" % ad.filename)
            # Write the flat frame to disk
            ad.write(os.path.join(rc["storedflats"], ad.filename), 
                     clobber=rc["clob"])
            log.fullinfo("Flat written to %s" % (rc["storedflats"])),
        
        yield rc
    
    def storeProcessedFringe(self, rc):
        """
        This should be a primitive that interacts with the calibration 
        system (MAYBE) but that isn't up and running yet. Thus, this will 
        just strip the extra postfixes to create the 'final' name for the 
        makeProcessedFringe outputs and write them to disk in a storedcals
        folder.
        
        :param clob: Write over any previous file with the same name that
                     all ready exists?
        :type clob: Python boolean (True/False)
                    default: False
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to 
                        screen. OR the message level as a string (i.e.,
                        'critical', 'status', 'fullinfo'...)
        """
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "storeProcessedFringe",
                                 "starting"))
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs(style="AD"):
            # Updating the file name with the suffix for this primitive and
            # then report the new file to the reduction context
            ad.filename = gt.fileNameUpdater(adIn=ad, suffix="_fringe",
                                             strip=True)
            log.status("File name of stored fringe is %s" % ad.filename)
            # Write the fringe frame to disk
            ad.write(os.path.join(rc["storedfringes"], ad.filename), 
                     clobber=rc["clob"])
            log.fullinfo("Fringe written to %s" % (rc["storedfringes"])),
        
        yield rc
    
    def subtractDark(self, rc):
        """
        This primitive will subtract each SCI extension of the inputs by those
        of the corresponding dark. If the inputs contain VAR or DQ frames,
        those will also be updated accordingly due to the subtraction on the 
        data.
        
        This is all conducted in pure Python through the arith 'toolbox' of 
        astrodata. 
        
        It is currently assumed that the same dark file will be applied to all
        input images.
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to 
                        screen. OR the message level as a string (i.e.,
                        'critical', 'status', 'fullinfo'...)
        """
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "subtractDark", "starting"))
        # Initialize the list of output AstroData objects
        adoutput_list = []
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs(style="AD"):
            # Check whether the subtractDark primitive has been run previously
            if ad.phu_get_key_value("SUBDARK"):
                log.warning("%s has already been processed by " \
                            "subtractDark" % (ad.filename))
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue
            # Get the appropriate dark for this AstroData object
            dark = AstroData(rc.get_cal(ad, "dark"))
            # Call the subtract_dark user level function
            ad = pp.subtract_dark(adinput=ad, dark=dark)
            # Append the output AstroData object (which is currently in the
            # form of a list) to the list of output AstroData objects
            adoutput_list.append(ad[0])
        # Report the list of output AstroData objects to the reduction
        # context
        rc.report_output(adoutput_list)
        
        yield rc
    
    def subtractFringe(self, rc):
        """
        This primitive will subtract each SCI extension of the inputs by those
        of the corresponding fringe. If the inputs contain VAR or DQ frames,
        those will also be updated accordingly due to the subtraction on the 
        data.
        
        This is all conducted in pure Python through the arith 'toolbox' of 
        astrodata. 
        
        It is currently assumed that the same fringe file will be applied to
        all input images.
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to 
                        screen. OR the message level as a string (i.e.,
                        'critical', 'status', 'fullinfo'...)
        """
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "subtractFringe", "starting"))

        # Initialize the list of output AstroData objects
        adoutput_list = []

        # Get inputs from their streams
        adinput = rc.get_inputs(style="AD")
        fringes = rc.get_stream(stream="fringe",style="AD")

        # Check that there are as many fringes as inputs
        if len(adinput)!=len(fringes):
            log.warning("Fringe input list does not match science input list;" +
                        "no fringe-correction will be performed.")
            adoutput_list = adinput
        else:

            # Loop over input science and fringe AstroData inputs
            for i in range(0,len(adinput)):

                ad = adinput[i]
                fringe = fringes[i]

                # Check whether the subtractFringe primitive has been run
                # previously
                if ad.phu_get_key_value("SUBFRING"):
                    log.warning("%s has already been processed by " +
                                "subtractFringe" % (ad.filename))
                    # Append the input AstroData object to the list of output
                    # AstroData objects without further processing
                    adoutput_list.append(ad)
                    continue

                # Check for valid fringe
                if fringe is None or fringe.count_exts("SCI") == 0:
                    log.warning("Could not find an appropriate fringe for %s" %
                                (ad.filename))
                    # Append the input AstroData object to the list of output
                    # AstroData objects without further processing
                    adoutput_list.append(ad)
                    continue

                # Call the subtract_fringe user level function
                ad = pp.subtract_fringe(adinput=ad, fringe=fringe) 
                
                adoutput_list.append(ad[0])

        # Report the output of the user level function to the reduction
        # context
        rc.report_output(adoutput_list, stream="main")
        
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
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to 
                        screen. OR the message level as a string (i.e.,
                        'critical', 'status', 'fullinfo'...)
        """
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        # Logging current values of suffix and prefix
        log.info("suffix = %s" % str(rc["suffix"]))
        log.info("prefix = %s" % str(rc["prefix"]))
        log.info("strip = %s" % str(rc["strip"]))
        
        if rc["suffix"] and rc["prefix"]:
            log.critical("The input will have %s pre pended and %s post " +
                         "pended onto it" % (rc["prefix"], rc["suffix"]))
        for ad in rc.get_inputs(style="AD"):
            # If the value of "suffix" was set, then set the file name 
            # to be written to disk to be postpended by it
            if rc["suffix"]:
                log.debug("calling gt.fileNameUpdater on %s" % ad.filename)
                ad.filename = gt.fileNameUpdater(adIn=ad,
                                                 suffix=rc["suffix"],
                                                 strip=rc["strip"])
                log.info("File name updated to %s" % ad.filename)
                outfilename = os.path.basename(ad.filename)
            # If the value of "prefix" was set, then set the file name 
            # to be written to disk to be pre pended by it
            if rc["prefix"]:
                infilename = os.path.basename(ad.filename)
                outfilename = "%s%s" % (rc["prefix"], infilename)
            # If the "outfilename" was set, set the file name of the file 
            # file to be written to this
            elif rc["outfilename"]:
                # Check that there is not more than one file to be written
                # to this file name, if so throw exception
                if len(rc.get_inputs(style="AD")) > 1:
                    message = """
                        More than one file was requested to be written to
                        the same name %s""" % (rc["outfilename"])
                    log.critical(message)
                    raise Errors.PrimitiveError(message)
                else:
                    outfilename = rc["outfilename"]
            # If no changes to file names are requested then write inputs
            # to their current file names
            else:
                outfilename = os.path.basename(ad.filename) 
                log.info("not changing the file name to be written " +
                "from its current name") 
            # Finally, write the file to the name that was decided 
            # upon above
            log.stdinfo("Writing to file %s" % outfilename)
            # AstroData checks if the output exists and raises an exception
            ad.write(filename=outfilename, clobber=rc["clobber"])
            rc.report_output(ad)
        yield rc
