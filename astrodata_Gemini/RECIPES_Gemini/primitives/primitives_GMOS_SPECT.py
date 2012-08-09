import numpy as np
from astrodata import AstroData
from astrodata import Errors
from astrodata.adutils import logutils
from astrodata.adutils.gemutil import pyrafLoader
from primitives_GMOS import GMOSPrimitives
from gempy import gemini_tools as gt
from gempy import managers as mgr
from gempy.geminiCLParDicts import CLDefaultParamsDict
from gempy import eti

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

    def attachWavelengthSolution(self,rc):

        # Instantiate the log
        log = logutils.get_logger(__name__)
        
        # Define the keyword to be used for the time stamp
        timestamp_key = self.timestamp_keys["attachWavelengthSolution"]

        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "attachWavelengthSolution",
                                 "starting"))

        # Initialize the list of output AstroData objects
        adoutput_list = []

        # Check for a user-supplied arc
        adinput = rc.get_inputs_as_astrodata()
        arc_param = rc["arc"]
        arc_dict = None
        if arc_param is not None:
            # The user supplied an input to the arc parameter
            if not isinstance(arc_param, list):
                arc_list = [arc_param]
            else:
                arc_list = arc_param

            # Convert filenames to AD instances if necessary
            tmp_list = []
            for arc in arc_list:
                if type(arc) is not AstroData:
                    arc = AstroData(arc)
                tmp_list.append(arc)
            arc_list = tmp_list
            
            arc_dict = gt.make_dict(key_list=adinput, value_list=arc_list)

        for ad in adinput:
            if arc_dict is not None:
                arc = arc_dict[ad]
            else:
                arc = rc.get_cal(ad, "processed_arc")
            
                # Take care of the case where there was no arc 
                if arc is None:
                    log.warning("Could not find an appropriate arc for %s" \
                                % (ad.filename))
                    adoutput_list.append(ad)
                    continue
                else:
                    arc = AstroData(arc)

            wavecal = arc["WAVECAL"]
            if wavecal is not None:
                # Remove old versions
                if ad["WAVECAL"] is not None:
                    for wc in ad["WAVECAL"]:
                        ad.remove((wc.extname(),wc.extver()))
                # Append new solution
                ad.append(wavecal)

                # Add the appropriate time stamps to the PHU
                gt.mark_history(adinput=ad, keyword=timestamp_key)

                # Change the filename
                ad.filename = gt.filename_updater(adinput=ad, 
                                                  suffix=rc["suffix"], 
                                                  strip=True)
                adoutput_list.append(ad)
            else:
                log.warning("No wavelength solution found for %s" % ad.filename)
                adoutput_list.append(ad)

        # Report the list of output AstroData objects to the reduction
        # context
        rc.report_output(adoutput_list)
      
        yield rc

    def determineWavelengthSolution(self,rc):

        # Instantiate the log
        log = logutils.get_logger(__name__)
        
        # Define the keyword to be used for the time stamp
        timestamp_key = self.timestamp_keys["determineWavelengthSolution"]

        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "determineWavelengthSolution",
                                 "starting"))
                
        # Initialize the list of output AstroData objects
        adoutput_list = []

        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():

            # Instantiate ETI and then run the task 
            # Run in a try/except because gswavelength sometimes fails
            # badly, and we want to be able to continue without
            # wavelength calibration in the QA case
            gswavelength_task = eti.gswavelengtheti.GswavelengthETI(rc,ad)
            try:
                adout = gswavelength_task.run()
            except Errors.OutputError:
                gswavelength_task.clean()
                if "qa" in rc.context:
                    log.warning("gswavelength failed for input " + ad.filename)
                    adoutput_list.append(ad)
                    continue
                else:
                    raise Errors.ScienceError("gswavelength failed for input "+
                                              ad.filename + ". Try interactive"+
                                              "=True")

            # Add the appropriate time stamps to the PHU
            gt.mark_history(adinput=adout, keyword=timestamp_key)

            # Change the filename
            adout.filename = gt.filename_updater(
                adinput=adout, suffix=rc["suffix"], strip=True)
            
            # Append the output AstroData object to the list
            # of output AstroData objects
            adoutput_list.append(adout)
        
        # Report the list of output AstroData objects to the reduction
        # context
        rc.report_output(adoutput_list)

        yield rc
    

    def extract1DSpectra(self,rc):

        # Instantiate the log
        log = logutils.get_logger(__name__)
        
        # Define the keyword to be used for the time stamp
        timestamp_key = self.timestamp_keys["extract1DSpectra"]

        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "extract1DSpectra", "starting"))
                
        # Initialize the list of output AstroData objects
        adoutput_list = []

        # Load PyRAF
        pyraf, gemini, yes, no = pyrafLoader()

        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():
            
            # Test whether to propagate VAR/DQ planes
            fl_vardq = no
            weights = "none"
            if ad["DQ"]:
                if ad["VAR"]:
                    fl_vardq = yes
                    weights = "variance"

            # Prepare input files, lists, parameters... for input to
            # the CL script
            clm=mgr.CLManager(imageIns=ad, suffix=rc["suffix"],
                              funcName="extract1DSpectra", log=log)
            
            # Check the status of the CLManager object, 
            # True=continue, False= issue warning
            if not clm.status:
                raise Errors.InputError("Inputs must be prepared")
            
            # Parameters set by the mgr.CLManager or the definition 
            # of the primitive 
            clPrimParams = {
              "inimages": clm.imageInsFiles(type="string"),
              "outimages": clm.imageOutsFiles(type="string"),
              "fl_vardq": fl_vardq,
              "weights": weights,
              # This returns a unique/temp log file for IRAF
              "logfile"     :clm.templog.name,
                          }
            
            # Grab the default params dict and update it with 
            # the above dict
            clParamsDict = CLDefaultParamsDict("gsextract")
            clParamsDict.update(clPrimParams)
            
            # Log the parameters
            mgr.logDictParams(clParamsDict)
            
            log.debug("Calling the gsextract CL script for inputs "+
                      clm.imageInsFiles(type="string"))
            
            gemini.gmos.gsextract(**clParamsDict)
            
            if gemini.gmos.gsextract.status:
                raise Errors.ScienceError("gsextract failed for inputs "+
                             clm.imageInsFiles(type="string"))
            else:
                log.fullinfo("Exited the gsextract CL script successfully")
            
            # Rename CL outputs and load them back into memory, and 
            # clean up the intermediate tmp files written to disk
            # refOuts and arrayOuts are None here
            imageOuts, refOuts, arrayOuts = clm.finishCL() 
            ad = imageOuts[0]

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
    

    def makeFlat(self,rc):

        # Instantiate the log
        log = logutils.get_logger(__name__)
        
        # Define the keyword to be used for the time stamp
        timestamp_key = self.timestamp_keys["makeFlat"]

        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "makeFlat", "starting"))
                
        # Initialize the list of output AstroData objects
        adoutput_list = []

        # Check if inputs prepared
        for ad in rc.get_inputs_as_astrodata():
            if "PREPARED" not in ad.types:
                raise Errors.InputError("%s must be prepared" % ad.filename)

        # Instantiate ETI and then run the task 
        gsflat_task = eti.gsflateti.GsflatETI(rc)
        adout = gsflat_task.run()

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

    def rejectCosmicRays(self,rc):

        # Instantiate the log
        log = logutils.get_logger(__name__)
        
        # Define the keyword to be used for the time stamp
        timestamp_key = self.timestamp_keys["rejectCosmicRays"]

        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "rejectCosmicRays", "starting"))
                
        # Initialize the list of output AstroData objects
        adoutput_list = []

        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():
            
            # Instantiate ETI and then run the task 
            gscrrej_task = eti.gscrrejeti.GscrrejETI(rc,ad)
            adout = gscrrej_task.run()

            # Add the appropriate time stamps to the PHU
            gt.mark_history(adinput=adout, keyword=timestamp_key)

            # Change the filename
            adout.filename = gt.filename_updater(
                adinput=adout, suffix=rc["suffix"], strip=True)
            
            # Append the output AstroData object to the list
            # of output AstroData objects
            adoutput_list.append(adout)
        
        # Report the list of output AstroData objects to the reduction
        # context
        rc.report_output(adoutput_list)
        
        yield rc

    def resampleToLinearCoords(self,rc):

        # Instantiate the log
        log = logutils.get_logger(__name__)
        
        # Define the keyword to be used for the time stamp
        timestamp_key = self.timestamp_keys["resampleToLinearCoords"]

        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "resampleToLinearCoords", 
                                 "starting"))
                
        # Initialize the list of output AstroData objects
        adoutput_list = []

        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():

            # Check for a wavelength solution
            if ad["WAVECAL"] is None:
                if "qa" in rc.context:
                    log.warning("No wavelength solution found for %s" %
                                ad.filename)

                    # Use gsappwave to do a rough wavelength calibration
                    log.stdinfo("Applying rough wavelength calibration")
                    gsappwave_task = eti.gsappwaveeti.GsappwaveETI(rc,ad)
                    adout = gsappwave_task.run()
                else:
                    raise Errors.InputError("No wavelength solution found "\
                                            "for %s" % ad.filename)
            else:
                # Wavelength solution found, use gstransform to apply it
                gstransform_task = eti.gstransformeti.GstransformETI(rc,ad)
                adout = gstransform_task.run()

            # Add the appropriate time stamps to the PHU
            gt.mark_history(adinput=adout, keyword=timestamp_key)

            # Change the filename
            adout.filename = gt.filename_updater(
                adinput=adout, suffix=rc["suffix"], strip=True)
            
            # Append the output AstroData object to the list
            # of output AstroData objects
            adoutput_list.append(adout)
        
        # Report the list of output AstroData objects to the reduction
        # context
        rc.report_output(adoutput_list)
        
        yield rc

    def skyCorrectFromSlit(self,rc):

        # Instantiate the log
        log = logutils.get_logger(__name__)
        
        # Define the keyword to be used for the time stamp
        timestamp_key = self.timestamp_keys["skyCorrectFromSlit"]

        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "skyCorrectFromSlit", "starting"))
                
        # Initialize the list of output AstroData objects
        adoutput_list = []

        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():
            
            # Instantiate ETI and then run the task 
            gsskysub_task = eti.gsskysubeti.GsskysubETI(rc,ad)
            adout = gsskysub_task.run()

            # Add the appropriate time stamps to the PHU
            gt.mark_history(adinput=adout, keyword=timestamp_key)

            # Change the filename
            adout.filename = gt.filename_updater(
                adinput=adout, suffix=rc["suffix"], strip=True)
            
            # Append the output AstroData object to the list
            # of output AstroData objects
            adoutput_list.append(adout)
        
        # Report the list of output AstroData objects to the reduction
        # context
        rc.report_output(adoutput_list)
        
        yield rc

