import numpy as np
from copy import deepcopy 

from astrodata import AstroData
from astrodata.utils import Errors
from astrodata.utils import logutils
from astrodata.utils.gemutil import pyrafLoader

from gempy.gemini import gemini_tools as gt
from gempy.gemini import eti

from primitives_GMOS import GMOSPrimitives

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

        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():
            
            # Instantiate ETI and then run the task 
            gsextract_task = eti.gsextracteti.GsextractETI(rc,ad)
            adout = gsextract_task.run()

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

                    # Flip the data left-to-right to put longer wavelengths
                    # on the right
                    for ext in adout:
                        if ext.extname() in ["SCI","VAR","DQ"]:
                            # Reverse the data in the rows
                            ext.data = ext.data[:,::-1]
                            # Change the WCS to match 
                            cd11 = ext.get_key_value("CD1_1")
                            crpix1 = ext.get_key_value("CRPIX1")
                            if cd11 and crpix1:
                                # Flip the sign on the first matrix element
                                ext.set_key_value("CD1_1",-cd11)
                                # Add one to the reference point
                                # (because the origin is 1, not 0)
                                ext.set_key_value("CRPIX1",crpix1+1)
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

            try:
                xbin = ad.detector_x_bin().as_pytype()
                ybin = ad.detector_y_bin().as_pytype()
                bin_factor = xbin*ybin
                roi = ad.detector_roi_setting().as_pytype()
            except:
                bin_factor = 1
                roi = "unknown"

            if bin_factor<=2 and roi=="Full Frame" and "qa" in rc.context:
                log.warning("Frame is too large to subtract sky efficiently; not "\
                            "subtracting sky for %s" % ad.filename)
                adoutput_list.append(ad)
                continue

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

