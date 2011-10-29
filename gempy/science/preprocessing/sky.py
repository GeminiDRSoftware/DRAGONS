# This module contains user level functions related to the preprocessing of
# the input dataset with a sky or fringe frame

import sys
from copy import deepcopy
import numpy as np
from astrodata import Errors
from astrodata import Lookups
from astrodata.adutils import gemLog
from astrodata.adutils.gemutil import pyrafLoader
from gempy import geminiTools as gt
from gempy import managers as mgr
from gempy.geminiCLParDicts import CLDefaultParamsDict
from gempy.science import resample as rs
from gempy.science import qa

# Load the timestamp keyword dictionary that will be used to define the keyword
# to be used for the time stamp for the user level function
timestamp_keys = Lookups.get_lookup_table("Gemini/timestamp_keywords",
                                          "timestamp_keys")

def correct_background_to_reference_image(adinput=None):
    """
    This function does an additive correction to a set
    of images to put their sky background at the same level
    as the reference image before stacking.
    """

    # Instantiate log
    log = gemLog.getGeminiLog()
    
    # Ensure that adinput is not None and return
    # a list containing one or more AstroData objects
    adinput = gt.validate_input(adinput=adinput)
    
    # Keyword to be used for time stamp
    timestamp_key = timestamp_keys["correct_background_to_reference_image"]
    
    adoutput_list = []
    try:        
        # Check that there are at least 2 images provided
        if len(adinput)<2:
            raise Errors.InputError("At least two images must be provided.")

        # Check that all images have the same number of science extensions
        next = np.array([ad.count_exts("SCI") for ad in adinput])
        if not np.all(next==next[0]):
            raise Errors.InputError("Number of science extensions in input "\
                                    "images do not match")

        ref_bg = None
        for ad in adinput:

            ref_bg_dict = {}
            diff_dict = {}
            for sciext in ad["SCI"]:
                # Get background value from header if it exists
                bg = sciext.get_key_value("SKYLEVEL")

                # Run measure_bg if it doesn't
                if bg is None:
                    log.fullinfo("SKYLEVEL not found, measuring background")
                    ad = qa.measure_bg(ad,separate_ext=True)[0]
                    sciext = ad["SCI",sciext.extver()]
                    bg = sciext.get_key_value("SKYLEVEL")
                    if bg is None:
                        raise Errors.ScienceError(
                            "Could not get background level from %s[SCI,%d]" %
                            (sciext.filename,sciext.extver))
                
                if ref_bg is None:
                    ref_bg_dict[(sciext.extname(),sciext.extver())]=bg
                else:
                    ref = ref_bg[(sciext.extname(),sciext.extver())]
                    difference = ref - bg
                    sciext.add(difference)
                    sciext.set_key_value("SKYLEVEL",bg+difference)

            # Store background level of first image
            if ref_bg is None:
                ref_bg = ref_bg_dict

            gt.mark_history(adinput=ad, keyword=timestamp_key)
            adoutput_list.append(ad)
        
        return adoutput_list
    
    except:
        # Log the message from the exception
        log.error(repr(sys.exc_info()[1]))
        raise



def make_fringe_image_gmos(adinput=None, operation="median", 
                           reject_method="avsigclip", suffix="_fringe"):
    """
    This function will create and return a single fringe image from all the 
    inputs. It uses the CL script gifringe to create the fringe image.
    
    NOTE: The inputs to this function MUST be prepared.
    
    Either a 'main' type logger object, if it exists, or a null logger 
    (ie, no log file, no messages to screen) will be retrieved/created
    and used within this function.
    
    :param adinput: Astrodata inputs to be combined
    :type adinput: Astrodata objects, either a single or a list of objects
    
    :param operation: type of combining operation to use.
    :type operation: string, options: 'average', 'median'.
    
    :param reject_method: type of combining operation to use.
    :type reject_method: string, options: 'avsigclip', 'minmax', etc.
    
    :param suffix: string to add on the end of the first input filename 
                   to make the output filename.
    :type suffix: string
    """
    
    # instantiate log
    log = gemLog.getGeminiLog()
    
    # ensure that adinput is not None and make it into a list
    # if it is not one already
    adinput = gt.validate_input(adinput=adinput)
    
    # time stamp keyword
    timestamp_key = timestamp_keys["make_fringe_image_gmos"]
    
    # initialize output list
    adoutput_list = []
    
    try:
        
        # Ensure there is more than one input to make a fringe frame from
        if (len(adinput)<2):
            raise Errors.InputError("Only one input was passed in for " +
                                    "adinput. At least two frames are " +
                                    "required to make a fringe" +
                                    "frame.")
        
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
            raise Errors.ScienceError("One of the inputs has not been " +
                                      "prepared,the combine function " + 
                                      "can only work on prepared data.")
        
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
        
        # Log the values in the soft and prim parameter dictionaries
        log.fullinfo("\nParameters set by the CLManager or "+
                     "dictated by the definition of the primitive:\n", 
                     category="parameters")
        mgr.logDictParams(clPrimParams)
        log.fullinfo("\nUser adjustable parameters in the "+
                     "parameters file:\n", category="parameters")
        mgr.logDictParams(clSoftcodedParams)
        
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

        # Change type of DQ plane back to int16 (gemcombine sets it to int32)
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
            gt.update_key_value(adinput=ad_out, function="bunit",
                                value="%s*%s" % (bunit,bunit),
                                extname="VAR")
       


        # Update GEM-TLM (automatic) and COMBINE time stamps to the PHU
        # and update logger with updated/added time stamps
        gt.mark_history(adinput=ad_out, keyword=timestamp_key)
        
        adoutput_list.append(ad_out)
        return adoutput_list
    except:
        # log the exact message from the actual exception that was raised
        # in the try block. Then raise a general ScienceError with message.
        log.critical(repr(sys.exc_info()[1]))
        raise 

def remove_fringe_image_gmos(adinput=None, fringe=None, 
                             stats_section=None, stats_scale=True):
    """
    This function will scale the fringe extensions to the science
    extensions, then subtract them.
    
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
    
    :param adinput: Astrodata input science data
    :type adinput: Astrodata objects, either a single or a list of objects
    
    :param fringe: The fringe(s) to be scaled and subtracted from the input(s).
    :type fringe: AstroData objects in a list, or a single instance.
                Note: If there are multiple inputs and one fringe provided, 
                then the same fringe will be applied to all inputs; else the 
                fringe list must match the length of the inputs.
    
    :param stats_scale: Use statistics to calculate the scale values?
    :type stats_scale: Python boolean (True/False). Default, True.
    """
    
    # Instantiate log
    log = gemLog.getGeminiLog()
    
    # Ensure that adinput and fringe are not None and make 
    # them into lists if they are not already
    adinput = gt.validate_input(adinput=adinput)
    fringe = gt.validate_input(adinput=fringe)
    
    # Create a dictionary that has the AstroData objects specified by adinput
    # as the key and the AstroData objects specified by fringe as the value
    fringe_dict = gt.make_dict(key_list=adinput, value_list=fringe)

    # Time stamp keyword
    timestamp_key = timestamp_keys["remove_fringe_image_gmos"]
    
    # Initialize output list
    adoutput_list = []
    
    try:
        
        # Loop through the inputs to perform scaling of fringes to the science 
        for ad in adinput:
            
            # Check whether the user level function has been
            # run previously
            if ad.phu_get_key_value(timestamp_key):
                raise Errors.InputError("%s has already been processed by " \
                                        "remove_fringe_image_gmos" % 
                                        (ad.filename))

            # Get matching fringe
            this_fringe = fringe_dict[ad]

            # Clip the fringe frame to the size of the science data
            # For a GMOS example, this allows a full frame fringe to
            # be used for a CCD2-only science frame. 
            this_fringe = gt.clip_auxiliary_data(adinput=ad, 
                                                 aux=this_fringe, 
                                                 aux_type="cal")[0]

            
            # Check the inputs have matching filters, binning and SCI shapes.
            gt.checkInputsMatch(adInsA=ad, adInsB=this_fringe)
        
            
            scale = 1.0
            if not stats_scale:
                # Use the exposure times to calculate the scale
                log.fullinfo("Using exposure times to calculate the scaling"+
                             " factor")
                try:
                    scale = ad.exposure_time() / this_fringe.exposure_time()
                except:
                    raise Errors.InputError("Could not get exposure times " +
                                            "for %s, %s. Try stats_scale=True" %
                                            (ad.filename,this_fringe.filename))
            else:

                # Use statistics to calculate the scaling factor, following
                # masked_sci = where({where[sciExt < 
                #                    (sciExt.median+2.5*sciExt.std)]} 
                #                 > [sciExt.median-3*sciExt.std])
                # scale = masked_sci.std / fringeExt.std
                log.fullinfo("Using statistics to calculate the " +
                             "scaling factor")

                # Get CCD2 data for statistics
                if ad.count_exts("SCI")==1:
                    # Only one CCD present, assume it is CCD2
                    sciext = ad["SCI",1]
                    frngext = this_fringe["SCI",1]
                else:
                    # Otherwise, take the second science extension

                    # Tile the data into one CCD per science extension,
                    # reordering if necessary
                    temp_ad = deepcopy(ad)
                    temp_ad = rs.tile_arrays(adinput=temp_ad)[0]
                    sciext = temp_ad["SCI",2]

                    temp_fr = deepcopy(this_fringe)
                    temp_fr = rs.tile_arrays(adinput=temp_fr)[0]
                    frngext = temp_fr["SCI",2]

                scidata = sciext.data
                frngdata = frngext.data

                # Take off 5% of the width as a border
                xborder = int(0.05 * scidata.shape[1])
                yborder = int(0.05 * scidata.shape[0])
                if xborder<20:
                    xborder = 20
                if yborder<20:
                    yborder = 20
                log.fullinfo("Using CCD2 data section [%i:%i,%i:%i] for " \
                             "statistics" %
                             (xborder,scidata.shape[1]-xborder,
                              yborder,scidata.shape[0]-yborder))

                s = scidata[yborder:-yborder,xborder:-xborder]
                f = frngdata[yborder:-yborder,xborder:-xborder]

                # Get median and standard deviation
                # (Must flatten for compatibility with 
                # older versions of numpy)
                smed = np.median(s.flatten()) 
                sstd = s.std()
                  
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
            scaled_fringe = this_fringe.mult(scale)
            
            # Subtract the scaled fringe from the science
            ad_out = ad.sub(scaled_fringe)
            
            # Update GEM-TLM (automatic) and RMFRINGE time stamps to the PHU
            # and update logger with updated/added time stamps
            gt.mark_history(adinput=ad_out, keyword=timestamp_key)
            
            # Append to output list
            adoutput_list.append(ad_out)
            
        # Return the output list
        # These are the scaled fringe ad's
        return adoutput_list
    except:
        # log the exact message from the actual exception that was raised
        # in the try block. Then raise a general ScienceError with message.
        log.critical(repr(sys.exc_info()[1]))
        raise 
