# This module contains user level functions related to the preprocessing of
# the input dataset with a bias frame

import os
import sys
from copy import deepcopy
import numpy as np
from astrodata import Errors
from astrodata import Lookups
from astrodata.adutils import gemLog
from astrodata.adutils.gemutil import pyrafLoader
from astrodata.AstroData import AstroData
from gempy import geminiTools as gt
from gempy import managers as mgr
from gempy.geminiCLParDicts import CLDefaultParamsDict

# Load the timestamp keyword dictionary that will be used to define the keyword
# to be used for the time stamp for the user level function
timestamp_keys = Lookups.get_lookup_table("Gemini/timestamp_keywords",
                                          "timestamp_keys")

def subtract_bias(adinput=None, bias=None):
    """
    This function will subtract the biases from the inputs using the 
    AstroData arith module.
    
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
    
    # Instantiate log
    log = gemLog.getGeminiLog()
    
    # Ensure that adinput and bias are not None and make 
    # them into lists if they are not already
    adinput = gt.validate_input(adinput=adinput)
    bias = gt.validate_input(adinput=bias)
    
    # Make a dictionary with adinputs as keys and bias(es) as values
    bias_dict = gt.make_dict(key_list=adinput, value_list=bias)
    
    # Time stamp keyword
    timestamp_key = timestamp_keys["subtract_bias"]
    
    # Initialize output list
    adoutput_list = []
     
    try:
        
        # Loop through AD inputs, subtracting the appropriate biases
        count=0
        for ad in adinput:
            
            # Check whether the subtract_bias user level function has been
            # run previously
            if ad.phu_get_key_value(timestamp_key):
                raise Errors.InputError("%s has already been processed by " \
                                        "subtract_bias" % (ad.filename))

            # Get the right bias frame for this input
            this_bias = bias_dict[ad]

            # Check for the case that the science data is a CCD2-only
            # frame and the bias is a full frame                
            if ad.count_exts("SCI")==1 and this_bias.count_exts("SCI")>1:
                new_bias = None
                sciext = ad["SCI",1]
                for biasext in this_bias["SCI"]:
                    # Use this extension if the bias detector section
                    # matches the science detector section
                    if (str(biasext.detector_section()) == 
                        str(sciext.detector_section())):
                        
                        extver = biasext.extver()
                        log.fullinfo("Using bias extension [SCI,%i]" % 
                                     extver)

                        varext = this_bias["VAR",extver]
                        dqext = this_bias["DQ",extver]

                        new_bias = deepcopy(biasext)
                        new_bias.rename_ext(name="SCI",ver=1)
                        if varext is not None:
                            newvar = deepcopy(varext)
                            newvar.rename_ext(name="VAR",ver=1)
                            new_bias.append(newvar)
                        if dqext is not None:
                            newdq = deepcopy(dqext)
                            newdq.rename_ext(name="DQ",ver=1)
                            new_bias.append(newdq)

                        this_bias = new_bias
                        break

                if new_bias is None:
                    raise Errors.InputError("Bias %s does not match " \
                                            "science %s" % 
                                            (this_bias.filename,ad.filename))

            # Check the inputs have matching binning and SCI shapes.
            gt.checkInputsMatch(adInsA=ad, adInsB=this_bias, 
                                check_filter=False) 

            log.fullinfo("Subtracting this bias from the input " \
                         "AstroData object (%s):\n%s" % (ad.filename, 
                                                         this_bias.filename))
            
            # Subtract the bias and handle VAR/DQ appropriately
            ad = ad.sub(this_bias)
            
            # Add the appropriate time stamps to the PHU
            gt.mark_history(adinput=ad, keyword=timestamp_key)
            
            # Record the bias file used
            ad.phu_set_key_value("BIASIM", 
                                 os.path.basename(this_bias.filename),
                                 "Bias image subtracted") 
            
            # Update log with new BIASIM header key
            log.fullinfo("PHU keyword added:", "header")
            log.fullinfo("BIASIM = "+ad.phu_get_key_value("BIASIM")+"\n", 
                         category="header")
            
            # Append to output list
            adoutput_list.append(ad)
            
            count = count+1
        
        # Return the outputs list
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
                   is ideal for 2x2 GMOS data. Default is None, which
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
            
            # Check whether this user level function has been
            # run previously
            if ad.phu_get_key_value(timestamp_key):
                raise Errors.InputError("%s has already been processed by " \
                                        "trim_overscan" % (ad.filename))

            for sciExt in ad["SCI"]:
                
                # get matching VAR and DQ planes if present
                extver = sciExt.extver()
                varExt = ad["VAR",extver]
                dqExt = ad["DQ",extver]
                
                # Get the data section from the descriptor
                try:
                    # as a string for printing
                    datasecStr = str(sciExt.data_section(pretty=True))
                    # int list of form [x1, x2, y1, y2] 0-based and non-inclusive
                    dsl = sciExt.data_section().as_pytype()
                except:
                    raise Errors.ScienceError("No data section defined; " +
                                              "cannot trim overscan")

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
                try:
                    crpix1 = sciExt.get_key_value("CRPIX1") - dsl[0]
                    crpix2 = sciExt.get_key_value("CRPIX2") - dsl[2]
                except:
                    log.warning("Could not access WCS keywords; using dummy " +
                                "CRPIX1 and CRPIX2")
                    crpix1 = 0
                    crpix2 = 0
                sciExt.header.update("CRPIX1",crpix1,"Ref pix of axis 1")
                sciExt.header.update("CRPIX2",crpix2,"Ref pix of axis 2")
                

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
                log.fullinfo("\n", category="header")
                log.fullinfo("File = "+ad.filename, category="header")
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
