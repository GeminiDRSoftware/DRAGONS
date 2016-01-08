import os
import numpy as np
from copy import deepcopy

import pifgemini.gmos_image as gmi

from astrodata import AstroData
from astrodata.utils import Errors
from astrodata.utils import Lookups
from astrodata.utils import logutils
from astrodata.utils.gemutil import pyrafLoader

from gempy.gemini import gemini_tools as gt

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
        log = logutils.get_logger(__name__)

        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "fringeCorrect", "starting"))
        
        # Loop over each input AstroData object in the input list to
        # test whether it's appropriate to try to remove the fringes
        rm_fringe = True
        for ad in rc.get_inputs_as_astrodata():
            
            # Test the filter and exposure time to see if we 
            # need to fringeCorrect at all
            filter = ad.filter_name(pretty=True)
            tel = ad.telescope().as_pytype()
            exposure = ad.exposure_time()
            if filter not in ["i","z","Z","Y"]:
                log.stdinfo("No fringe correction necessary for filter " +
                            filter)
                rm_fringe = False
                break
            elif filter=="i" and "Gemini-North" in tel:
                if "qa" in rc.context:
                    log.stdinfo("No fringe correction necessary for filter " +
                                filter + " with GMOS-N")
                    rm_fringe = False
                    break
                else:
                    # Allow it in the science case, but warn that it
                    # may not be helpful.
                    log.warning("Data uses filter " + filter +
                                "with GMOS-N. Fringe " +
                                "correction is not recommended.")
            if exposure<60.0:
                log.stdinfo("No fringe correction necessary with "\
                            "exposure time %.1fs" %
                            exposure)
                rm_fringe = False
                break

        if rm_fringe:
            # Retrieve processed fringes for the input
            
            # Initialize output list
            adoutput_list = []

            # Check for a fringe in the "fringe" stream first; the makeFringe
            # primitive, if it was called, would have added it there;
            # this avoids the latency involved in storing and retrieving
            # a calibration in the central system
            fringes = rc.get_stream("fringe",empty=True,style="AD")
            adinput = rc.get_inputs_as_astrodata()
            if fringes is None or len(fringes)!=1:
                rc.run("getProcessedFringe")
                for ad in adinput:
                    fringe = rc.get_cal(ad, "processed_fringe")
                    if fringe is None:
                        log.warning("Could not find an appropriate fringe "\
                                    "for %s" % (ad.filename))
                        adoutput_list.append(ad)
                        continue

                    # For generic fringes, scale by statistics
                    fringe = gmi.scale_fringe_to_science(
                        fringe, science=ad, stats_scale=True,
                        copy_input=False,index=rc["index"])

                    # Subtract the fringe
                    ad = gmi.subtract_fringe(
                        ad, fringe=fringe, copy_input=False, index=rc["index"])
                    
                    adoutput_list.append(ad)

            else:
                log.stdinfo("Using fringe: %s" % fringes[0].filename)

                # If fringe was created from science, don't scale it,
                # just subtract
                adoutput_list = gmi.subtract_fringe(
                        adinput, fringe=fringes[0], 
                        copy_input=False, index=rc["index"])

            rc.report_output(adoutput_list)            
        
        yield rc
    
    def makeFringe(self, rc):

        # Instantiate the log
        log = logutils.get_logger(__name__)


        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "makeFringe", 
                                 "starting"))

        # Get input, initialize output
        orig_input = rc.get_inputs_as_astrodata()
        adoutput_list = []

        # Check that filter is i, z, Z, or Y; this step doesn't
        # help data taken in other filters
        # Also check that exposure time is not too short;
        # there isn't much fringing for the shortest exposure times
        red = True
        long_exposure = True
        all_filter = None
        tel = [ad.telescope().as_pytype() for ad in orig_input]
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
            if filter not in ["i","z","Z","Y"]:
                red = False
                log.stdinfo("No fringe necessary for filter " +
                            filter)
                adoutput_list = orig_input
                break
            elif filter=="i" and "Gemini-North" in tel:
                if "qa" in rc.context:
                    red = False
                    log.stdinfo("No fringe necessary for filter " +
                                filter + " with GMOS-N")
                    adoutput_list = orig_input
                    break
                else:
                    # Allow it in the science case, but warn that it
                    # may not be helpful.
                    log.warning("Data uses filter " + filter +
                                "with GMOS-N. Fringe " +
                                "frame generation is not recommended.")

            # Check for long exposure times
            if exposure<60.0:
                long_exposure=False
                log.stdinfo("No fringe necessary with exposure time %.1fs" %
                            exposure)
                adoutput_list = orig_input
                break

        # ***** From KL:  Before we start measuring BG and detect sources
        # *****        how about we check if we have the minimum required
        # *****        number of frames!  Geez!  Trac #527

        enough = False
        if red and long_exposure:
            # Fringing on Cerro Pachon is generally stronger than
            # on Mauna Kea. A sextractor mask alone is usually
            # sufficient for GN data, but GS data needs to be median-
            # subtracted to distinguish fringes from objects
            sub_med = rc["subtract_median_image"]
            if sub_med is None:
                if "Gemini-South" in tel:
                    sub_med = True
                else:
                    sub_med = False

            if not sub_med:
                # Detect sources to get an object mask
                # Do it before the list for efficiency in
                # the case where it's possible
                rc.run("detectSources")

            # Measure BG levels (for zero-level removal, later)
            # This is done before the addToList so that it
            # doesn't have to be repeated on all frames later
            if rc["logLevel"]=="stdinfo":
                # Turn down logging from measureBG - no need
                # to see the numbers in the console at this
                # point in processing
                log.changeLevels(logLevel="status")
                rc.run("measureBG(separate_ext=True,remove_bias=False)")
                log.changeLevels(logLevel=rc["logLevel"])
            else:
                rc.run("measureBG(separate_ext=True,remove_bias=False)")

            # Add the current frame to a list
            rc.run("addToList(purpose=forFringe)")

            # Get other frames from the list
            rc.run("getList(purpose=forFringe)")

            # Check that there are enough input files
            adinput = rc.get_inputs_as_astrodata()
            
            # ****** From KL:  This check needs to be done BEFORE we start 
            # ****** measuring BG and detecting sources!!!  Trac #527
            # ****** Apparently, it isn't possible...
            if len(adinput)<3:
                # Can't make a useful fringe frame without at least
                # three input frames.
                enough = False
                log.stdinfo("Fewer than 3 frames provided as input. " +
                            "Not making fringe frame.")
                adoutput_list = orig_input

            elif "Gemini-North" in tel and len(adinput)<5:
                if "qa" in rc.context:
                    # If fewer than 5 frames and in QA context, don't
                    # bother making a fringe -- it'll just make the data
                    # look worse.
                    enough = False
                    log.stdinfo("Fewer than 5 frames provided as input " +
                                "for GMOS-N data. Not making fringe frame.")
                    adoutput_list = orig_input
                else:
                    # Allow it in the science case, but warn that it
                    # may not be helpful.
                    enough = True
                    log.warning("Fewer than 5 frames " +
                                "provided as input for GMOS-N data. Fringe " +
                                "frame generation is not recommended.")
            else:
                # Gemini South can use 4 fringes and above
                enough = True

        if enough:

            # Forward the science input to the fringe stream
            rc.run("forwardInput(to_stream=fringe)")

            # Call the makeFringeFrame primitive
            rc.run("makeFringeFrame(stream=fringe,subtract_median_image=%s)" %
                   str(sub_med))

            # Store the generated fringe
            rc.run("storeProcessedFringe(stream=fringe)")

            # Get the list of science frames back into the main stream
            adoutput_list = adinput

        # Report files back to the reduction context: if no fringe
        # is needed, this is the original input file. If fringe
        # processing is needed, this is the list of associated inputs.
        rc.report_output(adoutput_list)
        yield rc

    def makeFringeFrame(self,rc):

        # Instantiate the log
        log = logutils.get_logger(__name__)


        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "makeFringeFrame", 
                                 "starting"))

        # Initialize the list of output AstroData objects
        adoutput_list = []

        # Check for at least 3 input frames
        adinput = rc.get_inputs_as_astrodata()
        if len(adinput)<3:
            log.stdinfo('Fewer than 3 frames provided as input. ' +
                        'Not making fringe frame.')

            # Report the empty list to the reduction context
            rc.report_output(adoutput_list)
        
        else:
            rc.run("correctBackgroundToReferenceImage"\
                       "(remove_zero_level=True)")

            # If needed, do a rough median on all frames, subtract,
            # and then redetect to help distinguish sources from fringes
            sub_med = rc["subtract_median_image"]
            if sub_med:
                adinput = rc.get_inputs_as_astrodata()

                # Get data by science extension
                data = {}
                for ad in adinput:
                    for sciext in ad["SCI"]:
                        key = (sciext.extname(),sciext.extver())
                        if data.has_key(key):
                            data[key].append(sciext.data)
                        else:
                            data[key] = [sciext.data]


                # Make a median image for each extension
                import pyfits as pf
                median_ad = AstroData()
                median_ad.filename = gt.filename_updater(
                    adinput=adinput[0], suffix="_stack_median", strip=True)
                for key in data:
                    med_data = np.median(np.dstack(data[key]),axis=2)
                    hdr = pf.Header()
                    ext = AstroData(data=med_data, header=hdr)
                    ext.rename_ext(key)
                    median_ad.append(ext)

                # Subtract the median image
                rc["operand"] = median_ad
                rc.run("subtract")

                # Redetect to get a good object mask
                rc.run("detectSources")

                # Add the median image back in to the input
                rc.run("add")

            # Add the object mask into the DQ plane
            rc.run("addObjectMaskToDQ")
            
            # Stack frames with masking from DQ plane
            rc.run("stackFrames(operation=%s)" % rc["operation"])

        yield rc

    def normalizeFlat(self, rc):
        """
        This primitive will calculate a normalization factor from statistics
        on CCD2, then divide by this factor and propagate variance accordingly.
        CCD2 is used because of the dome-like shape of the GMOS detector
        response: CCDs 1 and 3 have lower average illumination than CCD2, 
        and that needs to be corrected for by the flat.
        """
        
        # Instantiate the log
        log = logutils.get_logger(__name__)

        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "normalizeFlat", "starting"))
        
        # Define the keyword to be used for the time stamp for this primitive
        timestamp_key = self.timestamp_keys["normalizeFlat"]

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
            ad.filename = gt.filename_updater(adinput=ad, suffix=rc["suffix"], 
                                              strip=True)
            
            # Append the output AstroData object to the list
            # of output AstroData objects
            adoutput_list.append(ad)

        # Report the list of output AstroData objects to the reduction
        # context
        rc.report_output(adoutput_list)
        
        yield rc
    
    def scaleByIntensity(self, rc):
        """
        This primitive scales input images to the mean value of the first
        image.  It is intended to be used to scale flats to the same
        level before stacking.
        """
        # Instantiate the log
        log = logutils.get_logger(__name__)


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
            this_mean = np.mean(stat_region, dtype=np.float64)

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
            ad.filename = gt.filename_updater(adinput=ad, suffix=rc["suffix"], 
                                              strip=True)

            # Append the output AstroData object to the list
            # of output AstroData objects
            adoutput_list.append(ad)

        # Report the list of output AstroData objects to the reduction
        # context
        rc.report_output(adoutput_list)
        
        yield rc

    def scaleFringeToScience(self, rc):
        """
        This primitive will scale the fringes to their matching science data
        The fringes should be in the stream this primitive is called on,
        and the reference science frames should be loaded into the RC,
        as, eg. rc["science"] = adinput.
        
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
        log = logutils.get_logger(__name__)

        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "scaleFringeToScience",
                                 "starting"))
        
        # Define the keyword to be used for the time stamp for this primitive
        timestamp_key = self.timestamp_keys["scaleFringeToScience"]

        # Check for user-supplied science frames
        fringe = rc.get_inputs_as_astrodata()
        science_param = rc["science"]
        fringe_dict = None
        if science_param is not None:
            # The user supplied an input to the science parameter
            if not isinstance(science_param, list):
                science_list = [science_param]
            else:
                science_list = science_param

            # If there is one fringe and multiple science frames,
            # the fringe must be deepcopied to allow it to be
            # scaled separately for each frame
            if len(fringe)==1 and len(science_list)>1:
                fringe = [deepcopy(fringe[0]) for img in science_list]

            # Convert filenames to AD instances if necessary
            tmp_list = []
            for science in science_list:
                if type(science) is not AstroData:
                    science = AstroData(science)
                tmp_list.append(science)
            science_list = tmp_list
            
            fringe_dict = gt.make_dict(key_list=science_list, 
                                       value_list=fringe)
            fringe_output = []
        else:
            log.warning("No science frames specified; no scaling will be done")
            science_list = []
            fringe_output = fringe

        # Loop over each AstroData object in the science list
        for ad in science_list:
            
            # Retrieve the appropriate fringe
            fringe = fringe_dict[ad]

            # Check the inputs have matching filters, binning and SCI shapes.
            try:
                gt.check_inputs_match(ad1=ad, ad2=fringe)
            except Errors.ToolboxError:
                # If not, try to clip the fringe frame to the size of the
                # science data
                # For a GMOS example, this allows a full frame fringe to
                # be used for a CCD2-only science frame. 
                fringe = gt.clip_auxiliary_data(
                    adinput=ad, aux=fringe, aux_type="cal")[0]

                # Check again, but allow it to fail if they still don't match
                gt.check_inputs_match(ad1=ad, ad2=fringe)

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

                # Use statistics to calculate the scaling factor
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
                    dqdata = []
                    for i in range(nsciext):

                        # Get the next extension in physical order
                        sciext = statsad["SCI",sci_info["amps_order"][i]]
                        frngext = statsfringe["SCI",frng_info["amps_order"][i]]

                        # Check to see if it is on CCD2; if so, keep it
                        if sci_info[
                            "array_number"][("SCI",sciext.extver())]==2:

                            scidata.append(sciext.data)

                            dqext = statsad["DQ",sci_info["amps_order"][i]]
                            maskext = statsad["OBJMASK",
                                              sci_info["amps_order"][i]]
                            if dqext is not None and maskext is not None:
                                dqdata.append(dqext.data | maskext.data)
                            elif dqext is not None:
                                dqdata.append(dqext.data)
                            elif maskext is not None:
                                dqdata.append(maskext.data)

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
                    if len(dqdata)>0:
                        if len(dqdata)>1:
                            dqdata = np.hstack(dqdata)
                        else:
                            dqdata = dqdata[0]
                    else:
                        dqdata = None
                else:
                    scidata = statsad["SCI"].data
                    frngdata = statsfringe["SCI"].data

                    dqext = statsad["DQ"]
                    maskext = statsad["OBJMASK"]
                    if dqext is not None and maskext is not None:
                        dqdata = dqext.data | maskext.data
                    elif dqext is not None:
                        dqdata = dqext.data
                    elif maskext is not None:
                        dqdata = maskext.data
                    else:
                        dqdata = None

                if dqdata is not None:
                    # Replace any DQ-flagged data with the median value
                    smed = np.median(scidata[dqdata==0])
                    scidata = np.where(dqdata!=0,smed,scidata)

                # Calculate the maximum and minimum in a box centered on 
                # each data point.  The local depth of the fringe is
                # max - min.  The overall fringe strength is the median
                # of the local fringe depths.

                # Width of the box is binning and
                # filter dependent, determined by experimentation
                # Results don't seem to depend heavily on the box size
                if ad.filter_name(pretty=True).as_pytype=="i":
                    size = 20
                else:
                    size = 40
                size /= ad.detector_x_bin().as_pytype()
                
                # Use ndimage maximum_filter and minimum_filter to
                # get the local maxima and minima
                import scipy.ndimage as ndimage
                sci_max = ndimage.filters.maximum_filter(scidata,size)
                sci_min = ndimage.filters.minimum_filter(scidata,size)


                # Take off 5% of the width as a border
                xborder = int(0.05 * scidata.shape[1])
                yborder = int(0.05 * scidata.shape[0])
                if xborder<20:
                    xborder = 20
                if yborder<20:
                    yborder = 20
                sci_max = sci_max[yborder:-yborder,xborder:-xborder]
                sci_min = sci_min[yborder:-yborder,xborder:-xborder]

                # Take the median difference
                sci_df = np.median(sci_max - sci_min)

                # Do the same for the fringe
                frn_max = ndimage.filters.maximum_filter(frngdata,size)
                frn_min = ndimage.filters.minimum_filter(frngdata,size)
                frn_max = frn_max[yborder:-yborder,xborder:-xborder]
                frn_min = frn_min[yborder:-yborder,xborder:-xborder]
                frn_df = np.median(frn_max - frn_min)

                # Scale factor
                # This tends to overestimate the factor, but it is
                # at least in the right ballpark, unlike the estimation
                # used in girmfringe (masked_sci.std/fringe.std)
                scale = sci_df / frn_df

            log.fullinfo("Scale factor found = "+str(scale))
                
            # Use mult from the arith toolbox to perform the scaling of 
            # the fringe frame
            scaled_fringe = fringe.mult(float(scale))
            
            # Add the appropriate time stamps to the PHU
            gt.mark_history(adinput=scaled_fringe, keyword=timestamp_key)

            # Change the filename
            scaled_fringe.filename = gt.filename_updater(
                adinput=ad, suffix=rc["suffix"], strip=True)
            
            fringe_output.append(scaled_fringe)
            
        # Report the list of output AstroData objects to the reduction context
        rc.report_output(fringe_output)
        yield rc
    
    def stackFlats(self, rc):
        """
        This primitive will combine the input flats with rejection
        parameters set appropriately for GMOS imaging twilight flats.
        """
        
        # Instantiate the log
        log = logutils.get_logger(__name__)

        
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
                reject_method = None
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
                         (nframes, reject_method, nlow, nhigh))

            # Run the scaleByIntensity primitive to scale flats to the
            # same level
            rc.run("scaleByIntensity")

            # Run the stackFrames primitive with the defined parameters
            prim_str = "stackFrames(suffix=%s,operation=%s,mask=%s," \
                       "reject_method=%s,nlow=%s,nhigh=%s)" % \
                       (rc["suffix"],rc["operation"],rc["mask"],
                        reject_method,nlow,nhigh)
            rc.run(prim_str)
        
        yield rc

    def subtractFringe(self, rc):
        
        # Instantiate the log
        log = logutils.get_logger(__name__)

        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "subtractFringe",
                                 "starting"))
        
        # Define the keyword to be used for the time stamp for this primitive
        timestamp_key = self.timestamp_keys["subtractFringe"]

        # Initialize the list of output AstroData objects
        adoutput_list = []
        
        # Check for a user-supplied fringe
        adinput = rc.get_inputs_as_astrodata()
        fringe_param = rc["fringe"]
        fringe_dict = None
        if fringe_param is not None:
            # The user supplied an input to the fringe parameter
            if not isinstance(fringe_param, list):
                fringe_list = [fringe_param]
            else:
                fringe_list = fringe_param

            # Convert filenames to AD instances if necessary
            tmp_list = []
            for fringe in fringe_list:
                if type(fringe) is not AstroData:
                    fringe = AstroData(fringe)
                tmp_list.append(fringe)
            fringe_list = tmp_list
            
            fringe_dict = gt.make_dict(key_list=adinput, value_list=fringe_list)
        

        # Loop over each input AstroData object in the input list
        for ad in adinput:
            
            # Check whether the subtractFringe primitive has been run
            # previously
            if ad.phu_get_key_value(timestamp_key):
                log.warning("No changes will be made to %s, since it has " \
                            "already been processed by subtractFringe" \
                            % (ad.filename))
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue
            
            # Retrieve the appropriate fringe
            if fringe_dict is not None:
                fringe = fringe_dict[ad]
            else:
                fringe = rc.get_cal(ad, "processed_fringe")
            
                # Take care of the case where there was no fringe 
                if fringe is None:
                    log.warning("Could not find an appropriate fringe for %s" \
                                % (ad.filename))
                    # Append the input to the output without further processing
                    adoutput_list.append(ad)
                    continue
                else:
                    fringe = AstroData(fringe)

            # Check the inputs have matching filters, binning and SCI shapes.
            try:
                gt.check_inputs_match(ad1=ad, ad2=fringe)
            except Errors.ToolboxError:
                # If not, try to clip the fringe frame to the size of the
                # science data
                # For a GMOS example, this allows a full frame fringe to
                # be used for a CCD2-only science frame. 
                fringe = gt.clip_auxiliary_data(
                    adinput=ad, aux=fringe, aux_type="cal")[0]

                # Check again, but allow it to fail if they still don't match
                gt.check_inputs_match(ad1=ad, ad2=fringe)


            # Subtract the fringe from the science
            ad = ad.sub(fringe)
            
            # Record the fringe file used
            ad.phu_set_key_value("FRINGEIM", 
                                 os.path.basename(fringe.filename),
                                 comment=self.keyword_comments["FRINGEIM"])

            # Add the appropriate time stamps to the PHU
            gt.mark_history(adinput=ad, keyword=timestamp_key)

            # Change the filename
            ad.filename = gt.filename_updater(adinput=ad, suffix=rc["suffix"], 
                                              strip=True)
            
            # Append the output AstroData object to the list 
            # of output AstroData objects
            adoutput_list.append(ad)
            
        # Report the list of output AstroData objects to the reduction context
        rc.report_output(adoutput_list)
        yield rc
    
