# This module contains user level functions related to the preprocessing of
# the input dataset with a bias frame

import os
import sys
import numpy as np
from astrodata import Errors
from astrodata import Lookups
from astrodata.adutils import gemLog
from astrodata.adutils.gemutil import pyrafLoader
from astrodata.AstroData import AstroData
from gempy import geminiTools as gt
from gempy import managers as mgr
from gempy import string
from gempy.geminiCLParDicts import CLDefaultParamsDict

# Load the timestamp keyword dictionary that will be used to define the keyword
# to be used for the time stamp for the user level function
timestamp_keys = Lookups.get_lookup_table("Gemini/timestamp_keywords",
                                          "timestamp_keys")

def subtract_bias(adinput=None, bias=None):
    """
    This function will subtract the biases from the inputs using the 
    CL script gireduce.
    
    WARNING: The gireduce script used here replaces the previously 
    calculated DQ frames with its own versions. This may be corrected 
    in the future by replacing the use of the gireduce
    with a Python routine to do the bias subtraction.
    
    NOTE: The inputs to this function MUST be prepared.
    
    Either a 'main' type logger object, if it exists, or a null logger 
    (ie, no log file, no messages to screen) will be retrieved/created
    and used within this function.
    
    :param adinput: Astrodata inputs to be bias subtracted
    :type adinput: Astrodata objects, either a single or a list of objects
    
    :param bias: The bias(es) to divide the input(s) by.
    :type bias: 
        AstroData objects in a list, or a single instance.
        Note: If there is multiple inputs and one bias provided, then the
        same bias will be applied to all inputs; else the bias
        list must match the length of the inputs.
    """
    
    # instantiate log
    log = gemLog.getGeminiLog()
    
    # ensure that adinput and bias are not None and make 
    # them into lists if they are not already
    adinput = gt.validate_input(adinput=adinput)
    bias = gt.validate_input(adinput=bias)
    
    # time stamp keyword
    timestamp_key = timestamp_keys["subtract_bias"]
    
    # initialize output list
    adoutput_list = []
     
    try:
        
        # check the inputs have matching binning and SCI shapes.
        gt.checkInputsMatch(adInsA=bias, adInsB=adinput, check_filter=False) 
        
        # load and bring the pyraf related modules into the name-space
        pyraf, gemini, yes, no = pyrafLoader()
            
        # Perform work in a loop, so that different biases may be
        # used for each input as gireduce only allows one bias input per run.
        count=0
        for ad in adinput:
            
            # Determine whether VAR/DQ needs to be propagated
            if (ad.count_exts("VAR") == 
                ad.count_exts("DQ") == 
                ad.count_exts("SCI")):
                fl_vardq=yes
            else:
                fl_vardq=no
            
            # Get the right bias frame for this input
            if len(bias)>1:
                this_bias = bias[count]
            else:
                this_bias = bias[0]
            log.fullinfo("Subtracting this bias from the input " \
                         "AstroData object (%s):\n%s" % (ad.filename, 
                                                         this_bias.filename))
            
            # Prepare input files, lists, parameters... for input to 
            # the CL script
            clm = mgr.CLManager(imageIns=ad, suffix="_out",
                                refIns=this_bias,
                                funcName="biasCorrect", log=log)
            
            # Check the status of the CLManager object, 
            # True=continue, False=issue warning
            if not clm.status:
                raise Errors.ScienceError("One of the inputs has not " +
                                          "been prepared, the combine " +
                                          "function can only work on " +
                                          "prepared data.")
            
            # Parameters set by the mgr.CLManager or the 
            # definition of the function 
            clPrimParams = {
                "inimages"    :clm.imageInsFiles(type="string"),
                "gp_outpref"  :clm.prefix,
                "outimages"   :clm.imageOutsFiles(type="string"),
                # This returns a unique/temp log file for IRAF 
                "logfile"     :clm.templog.name,
                "fl_bias"     :yes,
                # Possibly add this to the params file so the user can override
                # this input file
                "bias"        :clm.refInsFiles(type="string"),
                "outpref"    :"",
                "fl_over"    :no,
                "fl_trim"    :no,
                "fl_vardq"   :mgr.pyrafBoolean(fl_vardq)
                }
            
            # Parameters from the Parameter file adjustable by the user
            # (none, currently)
            clSoftcodedParams = {}
            
            # Grab the default params dict and update it 
            # with the two above dicts
            clParamsDict = CLDefaultParamsDict("gireduce")
            clParamsDict.update(clPrimParams)
            clParamsDict.update(clSoftcodedParams)
            
            # Log the parameters that were not defaults
            log.fullinfo("\nParameters set automatically:", 
                         category="parameters")
            # Loop through the parameters in the clPrimParams
            # dictionary and log them
            mgr.logDictParams(clPrimParams)
            
            log.fullinfo("\nParameters adjustable by the user:", 
                         category="parameters")
            # Loop through the parameters in the clSoftcodedParams 
            # dictionary and log them
            mgr.logDictParams(clSoftcodedParams)
            
            log.debug("calling the gireduce CL script for inputs "+
                      clm.imageInsFiles(type="string"))
            
            gemini.gmos.gireduce(**clParamsDict)
            
            if gemini.gmos.gireduce.status:
                raise Errors.ScienceError("gireduce failed for inputs "+
                                          clm.imageInsFiles(type="string"))
            else:
                log.fullinfo("Exited the gireduce CL script successfully")
            
            # Rename CL outputs and load them back into memory 
            # and clean up the intermediate temp files written to disk
            # refOuts and arrayOuts are None here
            imageOuts, refOuts, arrayOuts = clm.finishCL() 
            
            # There is only one at this point so no need to perform a loop
            # CLmanager outputs a list always, so take the 0th
            ad_out = imageOuts[0]
            ad_out.filename = ad.filename
            
            # Verify gireduce was actually ran on the file
            # then log file names of successfully reduced files
            if ad_out.phu_get_key_value("GIREDUCE"): 
                log.fullinfo("File "+ad_out.filename+
                             " was successfully bias-subtracted.")
            
            # Update GEM-TLM (automatic) and BIASCORR time stamps to the PHU
            # and update logger with updated/added time stamps
            gt.mark_history(adinput=ad_out, keyword=timestamp_key)
            
            # Reset the value set by gireduce to just the filename
            # for clarity
            ad_out.phu_set_key_value("BIASIM", 
                                    os.path.basename(this_bias.filename)) 
            
            # Update log with new BIASIM header key
            log.fullinfo("Another PHU keyword added:", "header")
            log.fullinfo("BIASIM = "+ad_out.phu_get_key_value("BIASIM")+"\n", 
                         category="header")
            
            # Append to output list
            adoutput_list.append(ad_out)
            
            count = count+1
        
        log.fullinfo("The CL script gireduce REPLACED any previously "+
                    "calculated DQ frames")
        # Return the outputs list, even if there is only one output
        return adoutput_list
    except:
        # log the exact message from the actual exception that was raised
        # in the try block. Then raise a general ScienceError with message.
        log.critical(repr(sys.exc_info()[1]))
        raise 

def subtract_overscan_gmos(adinput=None, overscan_section=None):
    """
    This function uses the CL script gireduce to subtract the overscan 
    from the input images.
    
    Variance and DQ planes, if they exist, will be saved and restored
    after gireduce has been run.
    
    NOTE:
    The inputs to this function MUST be prepared.
    
    Either a 'main' type logger object, if it exists, or a null logger 
    (ie, no log file, no messages to screen) will be retrieved/created 
    and used within this function.
    
    FOR FUTURE
    This function has many GMOS dependencies that would be great to work out
    so that this could be made a more general function (say at the Gemini
    level). In the future the parameters can be looked into and the CL 
    script can be upgraded to handle things like row based overscan
    calculations/fitting/modeling... vs the column based used right now, 
    add the model, nbiascontam,... params to the functions inputs so the 
    user can choose them for themselves.
    
    :param adinput: Astrodata inputs to be converted to Electron pixel units
    :type adinput: Astrodata objects, either a single or a list of objects
    
    :param overscan_section: overscan_section parameter of format 
                    '[x1:x2,y1:y2],[x1:x2,y1:y2],[x1:x2,y1:y2]'
    :type overscan_section: string. 
                   eg: '[2:25,1:2304],[2:25,1:2304],[1032:1055,1:2304]' 
                   is ideal for 2x2 GMOS data. Default is '', which
                   causes default nbiascontam=4 columns to be used.
    """
    
    # instantiate log
    log = gemLog.getGeminiLog()
    
    # ensure that adinput is not None and make it into a list
    # if it is not one already
    adinput = gt.validate_input(adinput=adinput)
    
    # time stamp keyword
    timestamp_key = timestamp_keys["subtract_overscan_gmos"]
    
    # initialize output list
    adoutput_list = []
    
    try: 
        # load and bring the pyraf related modules into the name-space
        pyraf, gemini, yes, no = pyrafLoader() 
        
        for ad in adinput:
            
            # Save VAR and DQ extensions
            var_ext = ad["VAR"]
            dq_ext = ad["DQ"]
            
            # Prepare input files, lists, parameters... for input to 
            # the CL script
            clm=mgr.CLManager(imageIns=ad, suffix="_out",
                              funcName="subtractOverscan", log=log)
            
            # Check the status of the CLManager object, 
            # True=continue, False= issue warning
            if not clm.status:
                raise Errors.ScienceError("One of the inputs has not been " +
                                          "prepared, the " + 
                                          "subtract_overscan_gmos function " +
                                          "can only work on prepared data.")
            
            # Take care of the overscan_section->nbiascontam param
            if overscan_section is not None:
                nbiascontam = clm.nbiascontam(adinput, overscan_section)
                log.fullinfo("nbiascontam parameter was updated to = "+
                             str(nbiascontam))
            else: 
                # Do not try to calculate it, just use default value of 4.
                log.fullinfo("Using default nbiascontam parameter = 4")
                nbiascontam = 4
            
            # Parameters set by the mgr.CLManager or the definition 
            # of the primitive 
            clPrimParams = {
              "inimages"    :clm.imageInsFiles(type="string"),
              "gp_outpref"  :clm.prefix,
              "outimages"   :clm.imageOutsFiles(type="string"),
              # This returns a unique/temp log file for IRAF
              "logfile"     :clm.templog.name,
              "fl_over"     :yes, 
              "fl_trim"     :no,
              "outpref"     :"",
              "fl_vardq"    :no,
                          }
            
            # Parameters from the Parameter file that are adjustable by the user
            clSoftcodedParams = {
               "nbiascontam":nbiascontam
                               }
            # Grab the default params dict and update it with 
            # the two above dicts
            clParamsDict = CLDefaultParamsDict("gireduce")
            clParamsDict.update(clPrimParams)
            clParamsDict.update(clSoftcodedParams)
            
            # Log the parameters that were not defaults
            log.fullinfo("\nParameters set automatically:", 
                         category="parameters")
            # Loop through the parameters in the clPrimParams dictionary
            # and log them
            mgr.logDictParams(clPrimParams)
            
            log.fullinfo("\nParameters adjustable by the user:", 
                         category="parameters")
            # Loop through the parameters in the clSoftcodedParams 
            # dictionary and log them
            mgr.logDictParams(clSoftcodedParams)
            
            log.debug("Calling the gireduce CL script for inputs "+
                      clm.imageInsFiles(type="string"))
            
            gemini.gmos.gireduce(**clParamsDict)
            
            if gemini.gmos.gireduce.status:
                raise Errors.ScienceError("gireduce failed for inputs "+
                             clm.imageInsFiles(type="string"))
            else:
                log.fullinfo("Exited the gireduce CL script successfully")
            
            # Rename CL outputs and load them back into memory, and 
            # clean up the intermediate tmp files written to disk
            # refOuts and arrayOuts are None here
            imageOuts, refOuts, arrayOuts = clm.finishCL() 
            
            ad_out = imageOuts[0]
            ad_out.filename = ad.filename
            
            # Restore VAR/DQ planes; no additional propagation 
            # should be needed
            if dq_ext is not None:
                ad_out.append(dq_ext)
            if var_ext is not None:
                ad_out.append(var_ext)
            
            # Verify gireduce was actually run on the file
            if ad_out.phu_get_key_value("GIREDUCE"): 
                # If gireduce was ran, then log the changes to the files 
                # it made
                log.fullinfo("File "+ad_out.filename+
                             " was successfully overscan-subtracted.")
            
            # Update GEM-TLM (automatic) and BIASCORR time stamps to the PHU
            # and update logger with updated/added time stamps
            gt.mark_history(adinput=ad_out, keyword=timestamp_key)
            
            adoutput_list.append(ad_out)
        
        # Return the outputs list, even if there is only one output
        return adoutput_list
    except:
        # Log the message from the exception
        log.critical(repr(sys.exc_info()[1]))
        raise 

def trim_overscan(adinput=None):
    """
    This function uses AstroData to trim the overscan region 
    from the input images and update their headers.
    
    NOTE: The inputs to this function MUST be prepared. 
    
    Either a 'main' type logger object, if it exists, or a null logger 
    (ie, no log file, no messages to screen) will be retrieved/created 
    and used within this function.
    
    :param adinput: Astrodata inputs to have DQ extensions added to
    :type adinput: Astrodata objects, either a single or a list of objects
    """
    
    # instantiate log
    log = gemLog.getGeminiLog()
    
    # ensure that adinput is not None and make it into a list
    # if it is not one already
    adinput = gt.validate_input(adinput=adinput)
    
    # time stamp keyword
    timestamp_key = timestamp_keys["trim_overscan"]
    
    # initialize output list
    adoutput_list = []
    
    try:
        
        # Loop through the inputs
        for ad in adinput:
            
            for sciExt in ad["SCI"]:
                
                # get matching VAR and DQ planes if present
                extver = sciExt.extver()
                varExt = ad["VAR",extver]
                dqExt = ad["DQ",extver]
                
                # Get the data section 
                # as a direct string from header
                datasecStr = str(sciExt.data_section(pretty=True))
                # int list of form [x1, x2, y1, y2] 0-based and non-inclusive
                dsl = sciExt.data_section().as_pytype()
                
                # Update logger with the section being kept
                log.fullinfo("For "+ad.filename+" extension "+
                             str(sciExt.extver())+
                             ", keeping the data from the section "+
                             datasecStr,"science")
                
                # Trim the data section from input SCI array
                # and make it the new SCI data
                sciExt.data=sciExt.data[dsl[2]:dsl[3],dsl[0]:dsl[1]]
                
                # Update header keys to match new dimensions
                newDataSecStr = "[1:"+str(dsl[1]-dsl[0])+",1:"+\
                                str(dsl[3]-dsl[2])+"]" 
                sciExt.header["NAXIS1"] = dsl[1]-dsl[0]
                sciExt.header["NAXIS2"] = dsl[3]-dsl[2]
                sciExt.header["DATASEC"]=newDataSecStr
                sciExt.header.update("TRIMSEC", datasecStr, 
                                   "Data section prior to trimming")
                
                # Update WCS reference pixel coordinate
                crpix1 = sciExt.get_key_value("CRPIX1") - dsl[0]
                crpix2 = sciExt.get_key_value("CRPIX2") - dsl[2]
                sciExt.header["CRPIX1"] = crpix1
                sciExt.header["CRPIX2"] = crpix2
                
                # If VAR and DQ planes present, update them to match
                if varExt is not None:
                    varExt.data=varExt.data[dsl[2]:dsl[3],dsl[0]:dsl[1]]
                    varExt.header["NAXIS1"] = dsl[1]-dsl[0]
                    varExt.header["NAXIS2"] = dsl[3]-dsl[2]
                    varExt.header.update("DATASEC", newDataSecStr,
                                        "Data section(s)")
                    varExt.header.update("TRIMSEC", datasecStr, 
                                         "Data section prior to trimming")
                    varExt.header.update("CRPIX1", crpix1,
                                         "RA at Ref pix in decimal degrees")
                    varExt.header.update("CRPIX2", crpix2,
                                         "DEC at Ref pix in decimal degrees")
                
                if dqExt is not None:
                    # gireduce DQ planes do not include
                    # overscan region, so don't trim DQ if it
                    # already matches the science
                    if dqExt.data.shape!=sciExt.data.shape:
                        dqExt.data=dqExt.data[dsl[2]:dsl[3],dsl[0]:dsl[1]]
                    dqExt.header["NAXIS1"] = dsl[1]-dsl[0]
                    dqExt.header["NAXIS2"] = dsl[3]-dsl[2]
                    dqExt.header.update("DATASEC", newDataSecStr,
                                        "Data section(s)")
                    dqExt.header.update("TRIMSEC", datasecStr, 
                                        "Data section prior to trimming")
                    dqExt.header.update("CRPIX1", crpix1,
                                         "RA at Ref pix in decimal degrees")
                    dqExt.header.update("CRPIX2", crpix2,
                                         "DEC at Ref pix in decimal degrees")
                
                # Update logger with updated/added keywords to each SCI frame
                log.fullinfo("*"*50, category="header")
                log.fullinfo("File = "+ad.filename, category="header")
                log.fullinfo("~"*50, category="header")
                log.fullinfo("SCI extension number "+str(sciExt.extver())+
                             " keywords updated/added:\n", "header")
                log.fullinfo("NAXIS1= "+str(sciExt.get_key_value("NAXIS1")),
                            category="header")
                log.fullinfo("NAXIS2= "+str(sciExt.get_key_value("NAXIS2")),
                             category="header")
                log.fullinfo("DATASEC= "+newDataSecStr, category="header")
                log.fullinfo("TRIMSEC= "+datasecStr, category="header")
            
            # Update GEM-TLM (automatic) and BIASCORR time stamps to the PHU
            # and update logger with updated/added time stamps
            gt.mark_history(adinput=ad, keyword=timestamp_key)
            
            # Set 'TRIMMED' to 'yes' in the PHU and update the log
            ad.phu_set_key_value("TRIMMED","yes","Overscan section trimmed")
            log.fullinfo("Another PHU keyword added:", "header")
            log.fullinfo("TRIMMED = "+ad.phu_get_key_value("TRIMMED")+"\n", 
                         category="header")
            
            # Append to output list
            adoutput_list.append(ad)
        
        # Return the outputs list, even if there is only one output
        return adoutput_list
    except:
        # log the exact message from the actual exception that was raised
        # in the try block. Then raise a general ScienceError with message.
        log.critical(repr(sys.exc_info()[1]))
        raise 
