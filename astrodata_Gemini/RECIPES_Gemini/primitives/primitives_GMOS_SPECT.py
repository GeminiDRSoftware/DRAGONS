import numpy as np
from copy import deepcopy 

from astrodata import AstroData
from astrodata.utils import Errors
from astrodata.utils import logutils
from astrodata.utils.gemutil import pyrafLoader
from astrodata.interface.slices import pixel_exts
from astrodata.utils.gemconstants import SCI, DQ

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
            gt.mark_history(adinput=adout, primname=self.myself(), keyword=timestamp_key)

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
            gt.mark_history(adinput=adout, primname=self.myself(), keyword=timestamp_key)

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
        
    def findAcquisitionSlits(self,rc):
        
        # GMOS plate scale
        GMOS_ARCSEC_PER_MM = 1.611444
        
        # Instantiate the log
        log = logutils.get_logger(__name__)
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "findAcquisitionSlits", 
                                 "starting"))
                
        # Define the keyword to be used for the time stamp
        timestamp_key = self.timestamp_keys["findAcquisitionSlits"]

        # Initialize the list of output AstroData objects
        adoutput_list = []

        adinput = rc.get_inputs_as_astrodata()
        orig_input = [deepcopy(ad) for ad in adinput]
        rc.run("tileArrays(tile_all=True)")
        acq_star_list = []
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():
            
            # Check whether this primitive has been run previously
            if ad.phu_get_key_value(timestamp_key):
                log.warning("No changes will be made to %s, since it has "
                            "already been processed by findAcqusitionSlits"
                            % ad.filename)
            
                # Set the list of stars to None so no further processing
                acq_star_list.append(None)
                continue
            
            # Check whether MDF exists
            if ad['MDF'] is None:
                log.warning("No MDF associated with %s" % ad.filename)
                acq_star_list.append(None)
                continue
            
            # Check whether there's a "priority" column in the MDF
            if 'priority' not in ad['MDF'].data.columns.names:
                log.warning("No acquisition slits in image %s"
                            % ad.filename)
                acq_star_list.append(None)
                continue
            
            mask_data = ad['MDF'].data
            # Collapse the tiled image along the wavelength direction
            # accounting for bad pixels/CCD regions from DQ
            okdata = np.where(ad[DQ].data==0, ad[SCI].data,0)
            spatial_profile = np.sum(okdata, axis=1)
            
            # Construct a theoretical illumination map from the MDF data
            slits_profile = np.zeros_like(spatial_profile)
            image_pix_scale = ad.pixel_scale()
  
            shuffle = ad.nod_pixels() // ad.detector_y_bin()
            # It is possible to use simply the MDF information in mm to get
            # the necessary slit position data, but this relies on knowing
            # the distortion correction. It seems better to use the MDF
            # pixel information, if it exists.
            try:
                mdf_pix_scale = ad['MDF'].header['PIXSCALE']
            except KeyError:
                mdf_pix_scale = ad.pixel_scale() / ad.detector_y_bin()
            # -1 because python is zero-indexed (see +1 later)
            # There was a problem with the mdf_pix_scale for GMOS-S pre-2009B
            # Work around this because the two pixel scales should be in a
            # simple ratio (3:2, 2:1, etc.)
            ratios = np.array([1.*a/b for a in range(1,6) for b in range(1,6)])
            # Here we have to account for the EEV->Hamamatsu change
            # (I've future-proofed this for the same event on GMOS-N)
            ratios = np.append(ratios,[ratios*0.73/0.8,ratios*0.727/0.8])
            nearest_ratio = ratios[np.argmin(abs(mdf_pix_scale /
                                                 image_pix_scale - ratios))]
            slits_y = mask_data['y_ccd'] * nearest_ratio - 1
            #slits_my = mask_data['slitpos_my']
            #if ad.instrument() == 'GMOS-S':
            #    slits_y = slits_my*(0.99911+slits_my*(3.0494e-7*slits_my-1.7465e-5))
            #else:
            #    slits_y = slits_my*(0.99591859227+slits_my*(1.7447902551997e-7*slits_my+5.304221133343e-7))
            #slits_y = (slits_y + 105) * GMOS_ARCSEC_PER_MM / image_pix_scale

            try:
                    slits_width = mask_data['slitsize_y']
            except KeyError:
                    slits_width = mask_data['slitsize_my'] * GMOS_ARCSEC_PER_MM
                    
            for (slit, width) in zip(slits_y, slits_width):
                slit_ymin = slit - 0.5*width/image_pix_scale
                slit_ymax = slit + 0.5*width/image_pix_scale
                # Only add slit if it wasn't shuffled off top of CCD
                if slit < ad['SCI'].data.shape[0]-shuffle:
                    slits_profile[max(int(slit_ymin),0):
                                  min(int(slit_ymax+1),len(slits_profile))] = 1
                    if slit_ymin > shuffle:
                        slits_profile[slit_ymin-shuffle:slit_ymax-shuffle+1] = 1

            # Cross-correlate collapsed image with theoretical profile
            c = np.correlate(spatial_profile, slits_profile, mode='full')
            slit_offset = np.argmax(c)-len(spatial_profile) + 1
          
            # Work out where the alignment slits actually are!
            # NODAYOFF should possibly be incorporated here, to better estimate
            # the locations of the positive traces, but I see inconsistencies
            # in the sign (direction of +ve values) for different datasets.
            acq_slits = np.logical_and(mask_data['priority']=='0',
                                       slits_y<ad['SCI'].data.shape[0]-shuffle)
            acq_slits_y = (slits_y[acq_slits] + slit_offset + 0.5).astype(int)
            acq_slits_width = (slits_width[acq_slits] / image_pix_scale +
                               0.5).astype(int)
            star_list = ' '.join(str(y)+':'+str(w) for y,w in 
                                 zip(acq_slits_y,acq_slits_width))
            acq_star_list.append(star_list)
            
        for ad,acq_stars in zip(orig_input,acq_star_list):
            if acq_stars is not None:
                ad.phu_set_key_value('ACQSLITS', acq_stars,
                                 comment='Locations of acquisition slits')
             
                # Add the appropriate time stamps to the PHU
                gt.mark_history(adinput=ad, primname=self.myself(), keyword=timestamp_key)

            # Change the filename
            ad.filename = gt.filename_updater(adinput=ad, 
                             suffix=rc["suffix"], strip=True)
            
            # Append the output AstroData object to the list of output
            # AstroData objects
            adoutput_list.append(ad)
            
        # Report the list of output AstroData objects to the reduction context    
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
        adout = gt.convert_to_cal_header(adinput=adout, caltype="flat",
                                         keyword_comments=self.keyword_comments)[0]

        # Add the appropriate time stamps to the PHU
        gt.mark_history(adinput=adout, primname=self.myself(), keyword=timestamp_key)

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
            gt.mark_history(adinput=adout, primname=self.myself(), keyword=timestamp_key)

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
            gt.mark_history(adinput=adout, primname=self.myself(), keyword=timestamp_key)

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
            gt.mark_history(adinput=adout, primname=self.myself(), keyword=timestamp_key)

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

    def skyCorrectNodShuffle(self, rc):
        
        # Instantiate the log
        log = logutils.get_logger(__name__)
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "skyCorrectNodShuffle", 
                                 "starting"))
                
        # Define the keyword to be used for the time stamp
        timestamp_key = self.timestamp_keys["skyCorrectNodShuffle"]

        # Initialize the list of output AstroData objects
        adoutput_list = []

        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():
            # Check whether the myScienceStep primitive has been run previously
            if ad.phu_get_key_value(timestamp_key):
                log.warning("No changes will be made to %s, since it has "
                            "already been processed by skyCorrectNodShuffle"
                            % ad.filename)
            
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue
            
            # Determine N&S offset in (binned) pixels
            shuffle = ad.nod_pixels() // ad.detector_y_bin()
            
            a_nod_count, b_nod_count = ad.nod_count().as_pytype()
            
            ad_nodded = deepcopy(ad)
            
            # Shuffle B position data up for all extensions (SCI, DQ, VAR)
            for ext, ext_nodded in zip(ad[pixel_exts], ad_nodded[pixel_exts]):
                # Make image equal to bottom row of data (imshift "nearest")
                ext_nodded.data = np.tile(ext.data[0,:], (ext.data.shape[0],1))
                # Then shift upwards
                ext_nodded.data[shuffle:,:] = ext.data[:-shuffle,:]
            
            # Normalize if the A and B nod counts differ
            if a_nod_count != b_nod_count:
                log.stdinfo("%s A and B nod counts differ...normalizing"
                            % ad.filename)
                ad.mult(0.5*(a_nod_count+b_nod_count)/a_nod_count)
                ad_nodded.mult(0.5*(a_nod_count+b_nod_count)/b_nod_count)
                
            # Subtract nodded image from image to complete the process
            ad.sub(ad_nodded)
            del ad_nodded
            
            # Add the appropriate time stamps to the PHU
            gt.mark_history(adinput=ad, primname=self.myself(), keyword=timestamp_key)

            # Change the filename
            ad.filename = gt.filename_updater(adinput=ad, 
                             suffix=rc["suffix"], strip=True)
            
            # Append the output AstroData object to the list of output
            # AstroData objects
            adoutput_list.append(ad)
            
        # Report the list of output AstroData objects to the reduction context    
        rc.report_output(adoutput_list)
        
        yield rc
