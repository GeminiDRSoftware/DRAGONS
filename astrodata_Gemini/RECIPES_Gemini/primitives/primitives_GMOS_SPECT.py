import numpy as np
from astrodata import Errors
from astrodata.adutils import gemLog
from astrodata.adutils.gemutil import pyrafLoader
from primitives_GMOS import GMOSPrimitives
from gempy import geminiTools as gt
from gempy import managers as mgr
from gempy.geminiCLParDicts import CLDefaultParamsDict

class GMOS_SPECTPrimitives(GMOSPrimitives):
    """
    This is the class containing all of the primitives for the GMOS_SPECT 
    level of the type hierarchy tree. It inherits all the primitives from the
    level above, 'GMOSPrimitives'.
    """
    astrotype = "GMOS_SPECT"
    
    def init(self, rc):
        GMOSPrimitives.init(self, rc)
        return rc

    def makeFlat(self,rc):

        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        
        # Define the keyword to be used for the time stamp
        timestamp_key = self.timestamp_keys["makeFlat"]

        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "makeFlat", "starting"))
                
        # Initialize the list of output AstroData objects
        adoutput_list = []

        # Get the list of input files
        adinput = rc.get_inputs_as_astrodata()

        # Load PyRAF
        pyraf, gemini, yes, no = pyrafLoader()

        # Test whether to propagate VAR/DQ planes
        fl_vardq = no
        for ad in adinput:
            if ad["DQ"]:
                if ad["VAR"]:
                    fl_vardq = yes


        # Prepare input files, lists, parameters... for input to 
        # the CL script
        clm=mgr.CLManager(imageIns=adinput, suffix=rc["suffix"], 
                          funcName="makeFlat", combinedImages=True, log=log)

            
        # Check the status of the CLManager object, 
        # True=continue, False= issue warning
        if not clm.status: 
            raise Errors.InputError("Input files must be prepared")


        # Get parameters for gsflat
        prim_params = {
            # Retrieve the input/output as a string of filenames
            "inflats"    :clm.imageInsFiles(type="string"),
            "specflat"   :clm.imageOutsFiles(type="string"),
            "fl_vardq"    :fl_vardq,
            # This returns a unique/temp log file for IRAF 
            "logfile"     :clm.templog.name,
            }

        # Get the default parameters for IRAF and update them
        # using the above dictionary
        cl_params = CLDefaultParamsDict("gsflat")
        cl_params.update(prim_params)

        # Log the parameters
        mgr.logDictParams(cl_params)

        # Call gsflat
        gemini.gsflat(**cl_params)
        if gemini.gsflat.status:
            raise Errors.OutputError("The IRAF task gsflat failed")
        else:
            log.fullinfo("The IRAF task gsflat completed sucessfully")

        # Create the output AstroData object by loading the output file from
        # gemcombine into AstroData, remove intermediate temporary files from
        # disk 
        adstack, junk, junk = clm.finishCL()
        adout = adstack[0]

        # Set any zero-values to 1 (to avoid dividing by zero)
        for sciext in adout["SCI"]:
            sciext.data[sciext.data==0] = 1.0

        # Blank out any position or program information in the
        # header (spectroscopy flats are often taken with science data)
        adout = gt.convert_to_cal_header(adinput=adout,caltype="flat")[0]

        # Add the appropriate time stamps to the PHU
        gt.mark_history(adinput=adout, keyword=timestamp_key)

        adoutput_list.append(adout)

        # Report the list of output AstroData objects to the reduction
        # context
        rc.report_output(adoutput_list)
        
        yield rc
