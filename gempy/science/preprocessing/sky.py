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

# Load the timestamp keyword dictionary that will be used to define the keyword
# to be used for the time stamp for the user level function
timestamp_keys = Lookups.get_lookup_table("Gemini/timestamp_keywords",
                                          "timestamp_keys")

def make_fringe_image_gmos(adinput=None, operation="median", 
                                 suffix="_fringe"):
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
            "reject"        :"none",
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

def remove_fringe(adinput=None, fringe=None, 
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
    
    A section of the SCI arrays to use for calculating these statistics can
    be defined with stats_section, or the default; the default is the
    original SCI data excluding the outer 100 pixels on all 4 sides (so 
    less 200 pixels in width and height).
    
    2. If stats_scale=False, then scale will be calculated using:
    exposure time of science / exposure time of fringe
    
    :param adinput: Astrodata input science data
    :type adinput: Astrodata objects, either a single or a list of objects
    
    :param fringe: The fringe(s) to be scaled and subtracted from the input(s).
    :type fringe: AstroData objects in a list, or a single instance.
                Note: If there are multiple inputs and one fringe provided, 
                then the same fringe will be applied to all inputs; else the 
                fringe list must match the length of the inputs.
    
    :param stats_section: sections of detectors to use for calculating
                          the statistics
    :type stats_section: Dictionary of the format:
                         {('SCI',1):[x1,x2,y1,y2], ('SCI',2):[x1,x2,y1,y2], ...}
                         with every SCI extension having a data section 
                         defined. Default is the inner region 100 pixels
                         from all 4 sides of SCI data.
    
    :param stats_scale: Use statistics to calculate the scale values?
    :type stats_scale: Python boolean (True/False). Default, True.
    """
    
    # instantiate log
    log = gemLog.getGeminiLog()
    
    # ensure that adinput and fringe are not None and make 
    # them into lists if they are not already
    adinput = gt.validate_input(adinput=adinput)
    fringe = gt.validate_input(adinput=fringe)
    
    # time stamp keyword
    timestamp_key = timestamp_keys["remove_fringe"]
    
    # initialize output list
    adoutput_list = []
    
    try:
        
        # if only one fringe is provided, make it into a list that
        # matches adinput
        if len(fringe)!=len(adinput):
            if len(fringe)==1:
                fringe = [fringe[0] for ad in adinput]
            else:
                raise Errors.inputError("Length of fringe list is not 1 and " +
                                        "does not match length of adinput " +
                                        "list.")
        
        # check the inputs have matching filters, binning and SCI shapes.
        gt.checkInputsMatch(adInsA=adinput, adInsB=fringe)
        
        # Loop through the inputs to perform scaling of fringes to the science 
        count=0
        for ad in adinput:
            
            # set up empty dict to hold scale vals for each extension
            scaleDict = {}
            # get matching fringe
            this_fringe = fringe[count]
            log.fullinfo("Scaling this fringe to input (%s):\n%s" %
                        (ad.filename,this_fringe.filename))
            
            for sciExt in ad["SCI"]:
                # Grab the fringe and science SCI extensions to operate on
                curExtver = sciExt.extver()
                fringeExt = this_fringe[("SCI", curExtver)]
                
                log.fullinfo("Scaling SCI extension "+str(curExtver))
                
                if stats_scale:
                    # use statistics to calculate the scaling factor, following
                    # masked_sci = where({where[sciExt < 
                    #                    (sciExt.median+2.5*sciExt.std)]} 
                    #                 > [sciExt.median-3*sciExt.std])
                    # scale = masked_sci.std / fringeExt.std
                    log.fullinfo("Using statistics to calculate the " +
                                 "scaling factor")
                    # Get current SCI's stats_section
                    if stats_section is None:
                        # use default inner region
                        
                        # Get the data section as a int list of form:
                        # [y1, y2, x1, x2] 0-based and non-inclusive
                        sds = sciExt.data_section().as_pytype()
                        # Take 100 pixels off each side
                        curStatsecList = [sds[0]+100,sds[1]-100,
                                          sds[2]+100,sds[3]-100]
                    else:
                        # pull value from stats_section dict provided
                        if isinstance(stats_section,dict):
                            curStatsecList = stats_section[("SCI",curExtver)]
                        else:
                            raise Errors.InputError("stats_section must be " +
                                                    "a dictionary. "+
                                                    "It was found to be a "+
                                                    str(type(stats_section)))
                    
                    cl = curStatsecList
                    log.fullinfo("Using section "+repr(cl)+" of data to "+
                                 "calculate the scaling factor")
                    # pull the data arrays from the extensions, 
                    # for the stats_section region
                    s = sciExt.data[cl[0]:cl[1],cl[2]:cl[3]]
                    f = fringeExt.data[cl[0]:cl[1],cl[2]:cl[3]]
                    # Must flatten for compatibility with older
                    # versions of numpy
                    
                    # science median
                    smed = np.median(s.flatten()) 
                    # science standard deviation
                    sstd = s.std()
                    
                    # make an array of all the points where the pixel value is 
                    # less than the median value + 2.5 x the standard deviation.
                    sbelow = s[np.where(s<(smed+(2.5*sstd)))]
                    
                    # make an array from the previous one where all the pixels
                    # in it have a value greater than the median -3 x the 
                    # standard deviation. Thus a final array of all the pixels 
                    # with values between (median + 2.5xstd) and (median -3xstd)
                    smiddle = sbelow[np.where(sbelow>(smed-(3.*sstd)))]
                    
                    ######## NOTE: kathleen believes the median should #########
                    ########       be used below instead of the std    #########
                    ### This needs real scientific review and discussion with ##
                    ### DA's to make a decision as to what is appropriate/works#
                    curScale = smiddle.std() / f.std() 
                
                else:
                    # use the exposure times to calculate the scale
                    log.fullinfo("Using exposure times to calculate the scaling"+
                                 " factor")
                    curScale = sciExt.exposure_time()/fringeExt.exposure_time()
                
                log.fullinfo("Scale factor found = "+str(curScale))
                
                # load determined scale for this extension into scaleDict
                scaleDict[("SCI",sciExt.extver())] = curScale
            
            # Use mult from the arith toolbox to perform the scaling of 
            # the fringe frame
            scaled_fringe = this_fringe.mult(scaleDict)
            
            # Subtract the scaled fringe from the science
            ad_out = ad.sub(scaled_fringe)
            
            # Update GEM-TLM (automatic) and RMFRINGE time stamps to the PHU
            # and update logger with updated/added time stamps
            gt.mark_history(adinput=ad_out, keyword=timestamp_key)
            
            # Append to output list
            adoutput_list.append(ad_out)
            
            count+=1
        
        # Return the output list
        # These are the scaled fringe ad's
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
    timestamp_key = timestamp_keys["remove_fringe"]
    
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

            # Check for the case that the science data is a CCD2-only
            # frame and the fringe is a full frame                
            if ad.count_exts("SCI")==1 and this_fringe.count_exts("SCI")>1:
                new_fringe = None
                sciext = ad["SCI",1]
                for fringeext in this_fringe["SCI"]:
                    # Use this extension if the fringe detector section
                    # matches the science detector section
                    if (str(fringeext.detector_section()) == 
                        str(sciext.detector_section())):
                        
                        extver = fringeext.extver()
                        log.fullinfo("Using fringe extension [SCI,%i]" % 
                                     extver)

                        varext = this_fringe["VAR",extver]
                        dqext = this_fringe["DQ",extver]

                        new_fringe = deepcopy(fringeext)
                        new_fringe.rename_ext(name="SCI",ver=1)
                        if varext is not None:
                            newvar = deepcopy(varext)
                            newvar.rename_ext(name="VAR",ver=1)
                            new_fringe.append(newvar)
                        if dqext is not None:
                            newdq = deepcopy(dqext)
                            newdq.rename_ext(name="DQ",ver=1)
                            new_fringe.append(newdq)

                        this_fringe = new_fringe
                        break
                if new_fringe is None:
                    raise Errors.InputError("Fringe %s does not match " \
                                            "science %s" % 
                                            (this_fringe.filename,ad.filename))
            
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
