#
#                                                                     QAP Gemini
#
#                            RECIPES_Gemini.primitives.primitives_standardize.py
#                                                                        08-2013
# ------------------------------------------------------------------------------
# $Id$
# ------------------------------------------------------------------------------
__version__      = '$Revision$'[11:-2]
__version_date__ = '$Date$'[7:-2]
# ------------------------------------------------------------------------------
import os
import numpy as np

from astrodata import AstroData
from astrodata import Errors
from astrodata import Lookups

from astrodata.adutils import logutils
from astrodata.ConfigSpace  import lookup_path
from astrodata.gemconstants import SCI, VAR, DQ

from gempy.gemini import gemini_tools as gt

from primitives_GENERAL import GENERALPrimitives
# ------------------------------------------------------------------------------

class StandardizePrimitives(GENERALPrimitives):
    """
    This is the class containing all of the primitives used to standardize an
    AstroData object. It inherits all the primitives from the
    'GENERALPrimitives' class.
    """
    astrotype = "GENERAL"
    
    def init(self, rc):
        GENERALPrimitives.init(self, rc)
        return rc

    init.pt_hide = True 
    
    def addDQ(self, rc):
        """
        This primitive is used to add a DQ extension to the input AstroData
        object. The value of a pixel in the DQ extension will be the sum of the
        following: (0=good, 1=bad pixel (found in bad pixel mask), 2=pixel is
        in the non-linear regime, 4=pixel is saturated). This primitive will
        trim the BPM to match the input AstroData object(s).
        
        :param bpm: The file name, including the full path, of the BPM(s) to be
                    used to flag bad pixels in the DQ extension. If only one
                    BPM is provided, that BPM will be used to flag bad pixels
                    in the DQ extension for all input AstroData object(s). If
                    more than one BPM is provided, the number of BPMs must
                    match the number of input AstroData objects. If no BPM is
                    provided, the primitive will attempt to determine an
                    appropriate BPM.
        :type bpm: string or list of strings
        """
        # Instantiate the log
        log = logutils.get_logger(__name__)
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "addDQ", "starting"))
        
        # Define the keyword to be used for the time stamp for this primitive
        timestamp_key = self.timestamp_keys["addDQ"]
        
        # Initialize the list of output AstroData objects
        adoutput_list = []
        
        # Set the data type of the data quality array
        # It can be uint8 for now, it will get converted up as we assign higher bit values
        # shouldn't need to force it up to 16bpp yet.
        dq_dtype = np.dtype(np.uint8)
        #dq_dtype = np.dtype(np.uint16)
        
        # Get the input AstroData objects
        adinput = rc.get_inputs_as_astrodata()
        
        # Loop over each input AstroData object in the input list
        for ad in adinput:
            
            # Check whether the addDQ primitive has been run previously
            if ad.phu_get_key_value(timestamp_key):
                log.warning("No changes will be made to %s, since it has "
                            "already been processed by addDQ" % ad.filename)
                
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue
            
            # Parameters specified on the command line to reduce are converted
            # to strings, including None
            ##M What about if a user doesn't want to add a BPM at all?
            ##M Are None's not converted to Nonetype from the command line?
            if rc["bpm"] and rc["bpm"] != "None":
                # The user supplied an input to the bpm parameter
                bpm = rc["bpm"]
            else:
                # The user did not supply an input to the bpm parameter, so try
                # to find an appropriate one. Get the dictionary containing the
                # list of BPMs for all instruments and modes.
                all_bpm_dict = Lookups.get_lookup_table("Gemini/BPMDict",
                                                        "bpm_dict")
                
                # Call the _get_bpm_key helper function to get the key for the
                # lookup table 
                key = self._get_bpm_key(ad)
                
                # Get the appropriate BPM from the look up table
                if key in all_bpm_dict:
                    bpm = lookup_path(all_bpm_dict[key])
                else:
                    bpm = None
                    log.warning("No BPM found for %s, no BPM will be "
                                "included" % ad.filename)

            # Ensure that the BPMs are AstroData objects
            bpm_ad = None
            if bpm is not None:
                log.fullinfo("Using %s as BPM" % str(bpm))
                if isinstance(bpm, AstroData):
                    bpm_ad = bpm
                else:
                    bpm_ad = AstroData(bpm)
                    ##M Do we want to fail here depending on context?
                    if bpm_ad is None:
                        log.warning("Cannot convert %s into an AstroData "
                                    "object, no BPM will be added" % bpm)

            final_bpm = None
            if bpm_ad is not None:
                # Clip the BPM data to match the size of the input AstroData
                # object science and pad with overscan region, if necessary
                final_bpm = gt.clip_auxiliary_data(adinput=ad, aux=bpm_ad,
                                                   aux_type="bpm")[0]

            # Get the non-linear level and the saturation level using the
            # appropriate descriptors - Individual values get checked in the
            # next loop 
            non_linear_level_dv = ad.non_linear_level()
            saturation_level_dv = ad.saturation_level()

            # Loop over each science extension in each input AstroData object
            for ext in ad[SCI]:
                
                # Retrieve the extension number for this extension
                extver = ext.extver()
                
                # Check whether an extension with the same name as the DQ
                # AstroData object already exists in the input AstroData object
                if ad[DQ, extver]:
                    log.warning("A [%s,%d] extension already exists in %s"
                                % (DQ, extver, ad.filename))
                    continue
                
                # Get the non-linear level and the saturation level for this
                # extension
                non_linear_level = non_linear_level_dv.get_value(extver=extver)
                saturation_level = saturation_level_dv.get_value(extver=extver)

                # To store individual arrays created for each of the DQ bit
                # types
                dq_bit_arrays = []

                # Create an array that contains pixels that have a value of 2
                # when that pixel is in the non-linear regime in the input
                # science extension
                if non_linear_level is not None:
                    non_linear_array = None
                    if saturation_level is not None:
                        # Test the saturation level against non_linear level
                        # They can be the same or the saturation level can be
                        # greater than but not less than the non-linear level.
                        # If they are the same then only flag saturated pixels
                        # below. This just means not creating an unneccessary
                        # intermediate array.
                        if saturation_level > non_linear_level:
                            log.fullinfo("Flagging pixels in the DQ extension "
                                         "corresponding to non linear pixels "
                                         "in %s[%s,%d] using non linear "
                                         "level = %.2f" % (ad.filename, SCI,
                                                           extver,
                                                           non_linear_level))

                            non_linear_array = np.where(
                                ((ext.data >= non_linear_level) &
                                (ext.data < saturation_level)), 2, 0)
                            
                        elif saturation_level < non_linear_level:
                            log.warning("%s[%s,%d] saturation_level value is"
                                        "less than the non_linear_level not"
                                        "flagging non linear pixels" %
                                        (ad.filname, SCI, extver))
                        else:
                            log.fullinfo("Saturation and non-linear values "
                                         "for %s[%s,%d] are the same. Only "
                                         "flagging saturated pixels."
                                         % (ad.filename, SCI, extver))
                            
                    else:
                        log.fullinfo("Flagging pixels in the DQ extension "
                                     "corresponding to non linear pixels "
                                     "in %s[%s,%d] using non linear "
                                     "level = %.2f" % (ad.filename, SCI, extver,
                                                       non_linear_level))

                        non_linear_array = np.where(
                            (ext.data >= non_linear_level), 2, 0)
                    
                    dq_bit_arrays.append(non_linear_array)

                # Create an array that contains pixels that have a value of 4
                # when that pixel is saturated in the input science extension
                if saturation_level is not None:
                    saturation_array = None
                    log.fullinfo("Flagging pixels in the DQ extension "
                                 "corresponding to saturated pixels in "
                                 "%s[%s,%d] using saturation level = %.2f" %
                                 (ad.filename, SCI, extver, saturation_level))
                    saturation_array = np.where(
                        ext.data >= saturation_level, 4, 0)
                    dq_bit_arrays.append(saturation_array)
                
                # BPMs have an EXTNAME equal to DQ
                bpmname = None
                if final_bpm is not None:
                    bpm_array = None
                    bpmname = os.path.basename(final_bpm.filename)
                    log.fullinfo("Flagging pixels in the DQ extension "
                                 "corresponding to bad pixels in %s[%s,%d] "
                                 "using the BPM %s[%s,%d]" %
                                 (ad.filename, SCI, extver, bpmname, DQ, extver))
                    bpm_array = final_bpm[DQ, extver].data
                    dq_bit_arrays.append(bpm_array)
                
                # Create a single DQ extension from the three arrays (BPM,
                # non-linear and saturated)
                if not dq_bit_arrays:
                    # The BPM, non-linear and saturated arrays were not
                    # created. Create a single DQ array with all pixels set
                    # equal to 0 
                    log.fullinfo("The BPM, non-linear and saturated arrays "
                                 "were not created. Creating a single DQ "
                                 "array with all the pixels set equal to zero")
                    final_dq_array = np.zeros(ext.data.shape).astype(dq_dtype)

                else:
                    final_dq_array = self._bitwise_OR_list(dq_bit_arrays)
                    final_dq_array = final_dq_array.astype(dq_dtype)
                
                # Create a data quality AstroData object
                dq = AstroData(data=final_dq_array)
                dq.rename_ext(DQ, ver=extver)
                dq.filename = ad.filename
                
                # Call the _update_dq_header helper function to update the
                # header of the data quality extension with some useful
                # keywords
                dq = self._update_dq_header(sci=ext, dq=dq, bpmname=bpmname)
                
                # Append the DQ AstroData object to the input AstroData object
                log.fullinfo("Adding extension [%s,%d] to %s"
                             % (DQ, extver, ad.filename))
                ad.append(moredata=dq)
            
            # Add the appropriate time stamps to the PHU
            gt.mark_history(adinput=ad, keyword=timestamp_key)
            
            # Change the filename
            ad.filename = gt.filename_updater(adinput=ad, suffix=rc["suffix"],
                                              strip=True)
            
            # Append the output AstroData object to the list of output
            # AstroData objects
            adoutput_list.append(ad)

        # Report the list of output AstroData objects to the reduction context
        rc.report_output(adoutput_list)
        
        yield rc
    
    def addMDF(self, rc):
        """
        This primitive is used to add an MDF extension to the input AstroData
        object. If only one MDF is provided, that MDF will be add to all input
        AstroData object(s). If more than one MDF is provided, the number of
        MDF AstroData objects must match the number of input AstroData objects.
        If no MDF is provided, the primitive will attempt to determine an
        appropriate MDF.
        
        :param mdf: The file name of the MDF(s) to be added to the input(s)
        :type mdf: string
        
        """
        # Instantiate the log
        log = logutils.get_logger(__name__)
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "addMDF", "starting"))
        
        # Define the keyword to be used for the time stamp for this primitive
        timestamp_key = self.timestamp_keys["addMDF"]
        
        # Initialize the list of output AstroData objects
        adoutput_list = []
        
        # Get the input AstroData objects
        adinput = rc.get_inputs_as_astrodata()
        
        # Loop over each input AstroData object in the input list
        for ad in adinput:
            
            # Check whether the addMDF primitive has been run previously
            if ad.phu_get_key_value(timestamp_key):
                log.warning("No changes will be made to %s, since it has "
                            "already been processed by addMDF" % ad.filename)
                
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue
            
            # Check whether the input is spectroscopic data
            if "SPECT" not in ad.types:
                log.stdinfo("%s is not spectroscopic data, so no MDF will be "
                            "added" % ad.filename)
                
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue
            
            # Check whether an MDF extension already exists in the input
            # AstroData object
            if ad["MDF"]:
                log.warning("An MDF extension already exists in %s, so no MDF "
                            "will be added" % ad.filename)
                
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue
            
            # Parameters specified on the command line to reduce are converted
            # to strings, including None
            if rc["mdf"] and rc["mdf"] != "None":
                # The user supplied an input to the mdf parameter
                mdf = rc["mdf"]
            else:
                # The user did not supply an input to the mdf parameter, so try
                # to find an appropriate one. Get the dictionary containing the
                # list of MDFs for all instruments and modes.
                all_mdf_dict = Lookups.get_lookup_table("Gemini/MDFDict",
                                                        "mdf_dict")
                
                # The MDFs are keyed by the instrument and the MASKNAME. Get
                # the instrument and the MASKNAME values using the appropriate
                # descriptors 
                instrument = ad.instrument()
                mask_name = ad.phu_get_key_value("MASKNAME")
                
                # Create the key for the lookup table
                if instrument is None or mask_name is None:
                    log.warning("Unable to create the key for the lookup "
                                "table (%s), so no MDF will be added"
                                % ad.exception_info)
                    
                    # Append the input AstroData object to the list of output
                    # AstroData objects without further processing
                    adoutput_list.append(ad)
                    continue
                
                key = "%s_%s" % (instrument, mask_name)
                
                # Get the appropriate MDF from the look up table
                if key in all_mdf_dict:
                    mdf = lookup_path(all_mdf_dict[key])
                else:
                    # The MASKNAME keyword defines the actual name of an MDF
                    if not mask_name.endswith(".fits"):
                        mdf = "%s.fits" % mask_name
                    else:
                        mdf = str(mask_name)
                    
                    # Check if the MDF exists in the current working directory
                    if not os.path.exists(mdf):
                        log.warning("The MDF %s was not found in the current "
                                    "working directory, so no MDF will be "
                                    "added" % mdf)
                    
                    # Append the input AstroData object to the list of output
                    # AstroData objects without further processing
                    adoutput_list.append(ad)
                    continue
            
            # Ensure that the MDFs are AstroData objects
            if not isinstance(mdf, AstroData):
                mdf_ad = AstroData(mdf)
            
            if mdf_ad is None:
                log.warning("Cannot convert %s into an AstroData object, so "
                            "no MDF will be added" % mdf)
                
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue
            
            # Check if the MDF is a single extension fits file
            if len(mdf_ad) > 1:
                log.warning("The MDF %s is not a single extension fits file, "
                            "so no MDF will be added" % mdf)
                
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue
                
            # Name the extension appropriately
            mdf_ad.rename_ext("MDF", 1)
            
            # Append the MDF AstroData object to the input AstroData object
            log.fullinfo("Adding the MDF %s to the input AstroData object "
                         "%s" % (mdf_ad.filename, ad.filename))
            ad.append(moredata=mdf_ad)
            
            # Add the appropriate time stamps to the PHU
            gt.mark_history(adinput=ad, keyword=timestamp_key)
            
            # Change the filename
            ad.filename = gt.filename_updater(adinput=ad, suffix=rc["suffix"],
                                              strip=True)
            
            # Append the output AstroData object to the list of output
            # AstroData objects
            adoutput_list.append(ad)
        
        # Report the list of output AstroData objects to the reduction context
        rc.report_output(adoutput_list)
        
        yield rc
    
    def addVAR(self, rc):
        """
        This primitive calculates the variance of each science extension in the
        input AstroData object and adds the variance as an additional
        extension. This primitive will determine the units of the pixel data in
        the input science extension and calculate the variance in the same
        units. The two main components of the variance can be calculated and
        added separately, if desired, using the following formula:
        
        variance(read_noise) [electrons] = (read_noise [electrons])^2 
        variance(read_noise) [ADU] = ((read_noise [electrons]) / gain)^2
        
        variance(poisson_noise) [electrons] =
            (number of electrons in that pixel)
        variance(poisson_noise) [ADU] =
            ((number of electrons in that pixel) / gain)
        
        The pixel data in the variance extensions will be the same size as the
        pixel data in the science extension.
        
        The read noise component of the variance can be calculated and added to
        the variance extension at any time, but should be done before
        performing operations with other datasets.
        
        The Poisson noise component of the variance can be calculated and added
        to the variance extension only after any bias levels have been
        subtracted from the pixel data in the science extension. 
        
        The variance of a raw bias frame contains only a read noise component
        (which represents the uncertainty in the bias level of each pixel),
        since the Poisson noise component of a bias frame is meaningless.
        
        :param read_noise: set to True to add the read noise component of the
                           variance to the variance extension
        :type read_noise: Python boolean
        
        :param poisson_noise: set to True to add the Poisson noise component
                              of the variance to the variance extension
        :type poisson_noise: Python boolean
        
        """
        # Instantiate the log
        log = logutils.get_logger(__name__)
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "addVAR", "starting"))
        
        # Define the keyword to be used for the time stamp for this primitive
        timestamp_key = self.timestamp_keys["addVAR"]
        
        # Initialize the list of output AstroData objects
        adoutput_list = []
        
        # Check to see what component of variance will be added and whether it
        # is sensible to do so
        read_noise = rc["read_noise"]
        poisson_noise = rc["poisson_noise"]
        
        if read_noise and poisson_noise:
            log.stdinfo("Adding the read noise component and the poisson "
                        "noise component of the variance")
        if read_noise and not poisson_noise:
            log.stdinfo("Adding the read noise component of the variance")
        if not read_noise and poisson_noise:
            log.stdinfo("Adding the poisson noise component of the variance")
        if not read_noise and not poisson_noise:
            log.warning("Cannot add a variance extension since no variance "
                        "component has been selected")
        
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():
            
            if poisson_noise and "BIAS" in ad.types:
                log.warning("It is not recommended to add a poisson noise "
                            "component to the variance of a bias frame")
            if (poisson_noise and "GMOS" in ad.types and not
                ad.phu_get_key_value(self.timestamp_keys["subtractBias"])):
                
                log.warning("It is not recommended to calculate a poisson "
                            "noise component of the variance using data that "
                            "still contains a bias level")
            
            # Call the _calculate_var helper function to calculate and add the
            # variance extension to the input AstroData object
            ad = self._calculate_var(adinput=ad, add_read_noise=read_noise,
                                     add_poisson_noise=poisson_noise)
            
            # Add the appropriate time stamps to the PHU
            gt.mark_history(adinput=ad, keyword=timestamp_key)
            
            # Change the filename
            ad.filename = gt.filename_updater(adinput=ad, suffix=rc["suffix"],
                                              strip=True)
            
            # Append the output AstroData object to the list of output
            # AstroData objects
            adoutput_list.append(ad)
        
        # Report the list of output AstroData objects to the reduction context
        rc.report_output(adoutput_list)
        
        yield rc
    
    def markAsPrepared(self, rc):
        """
        This primitive is used to add a time stamp keyword to the PHU of the
        AstroData object and update the AstroData type, allowing the output
        AstroData object to be recognised as PREPARED.
        
        """
        # Instantiate the log
        log = logutils.get_logger(__name__)
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "markAsPrepared", "starting"))
        
        # Define the keyword to be used for the time stamp for this primitive
        timestamp_key = self.timestamp_keys["prepare"]
        
        # Initialize the list of output AstroData objects
        adoutput_list = []
        
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():
            
            # Add the appropriate time stamps to the PHU
            gt.mark_history(adinput=ad, keyword=timestamp_key)
            
            # Update the AstroData type so that the AstroData object is
            # recognised as being prepared
            ad.refresh_types()
            
            # Change the filename
            ad.filename = gt.filename_updater(adinput=ad, suffix=rc["suffix"],
                                              strip=True)
            
            # Append the output AstroData object to the list of output
            # AstroData objects
            adoutput_list.append(ad)
        
        # Report the list of output AstroData objects to the reduction context
        rc.report_output(adoutput_list)
        
        yield rc
    
    ##########################################################################
    # Below are the helper functions for the primitives in this module       #
    ##########################################################################
    
    def _get_bpm_key(self, adinput=None):
        # The BPMs are keyed by the instrument and the binning. Get the
        # instrument, the x binning and the y binning values using the 
        # appropriate descriptors
        ad = adinput
        instrument = ad.instrument()
        detector_x_bin = ad.detector_x_bin()
        detector_y_bin = ad.detector_y_bin()
        
        if (instrument is None or detector_x_bin is None or
            detector_y_bin is None):
            
            raise Errors.Error("Input parameters")
        
        key = "%s_%s_%s" % (instrument, detector_x_bin, detector_y_bin)
        
        if "GMOS" in ad.types:
            # Note: it would probably be better to make this into
            # a primitive and put it into the type specific files.
            #
            # GMOS BPMs are keyed by:
            # GMOS-(N/S)_(EEV/e2v/HAM)_(binning)_(n)amp_v(#)(_mosaic),
            # to correspond to BPMs named, eg.:
            # gmos-n_bpm_e2v_22_6amp_v1.fits
            # gmos-s_bpm_EEV_11_3amp_v1_mosaic.fits
            #
            # Format binning
            binning = "%s%s" % (detector_x_bin, detector_y_bin)
            detector_type = ad.phu_get_key_value("DETTYPE")

            if detector_type   == "SDSU II CCD":
                det = "EEV"
            elif detector_type == "SDSU II e2v DD CCD42-90":
                det = "e2v"
            elif detector_type == "S10892-01":
                det = "HAM"
            else:
                det = None
            
            # Check the number of amps used
            namps = ad.phu_get_key_value("NAMPS")

            if   namps == 2: 
                amp = "6amp"
            elif namps == 1: 
                amp = "3amp"
            else: 
                amp = None
            
            # Check whether data is mosaicked
            mosaicked = (
                (ad.phu_get_key_value(
                        self.timestamp_keys["mosaicDetectors"]) is not None)
                or
                (ad.phu_get_key_value(
                        self.timestamp_keys["tileArrays"]) is not None))

            if mosaicked:
                mos = "_mosaic"
            else:
                mos = ""
            
            # Get version required
            # So far, there is only one version. This may change someday.
            ver = "v1"
            
            # Create the key
            key = "%s_%s_%s_%s_%s%s" % (instrument, det, binning, amp, ver, mos)
        
        return key


    def _calculate_var(self, adinput=None, add_read_noise=False,
                       add_poisson_noise=False):
        """
        The _calculate_var helper function is used to calculate the variance
        and add a variance extension to the single input AstroData object.
        
        """
        # Instantiate the log
        log = logutils.get_logger(__name__)
        
        # Get the gain and the read noise using the appropriate descriptors.
        gain_dv = adinput.gain()
        read_noise_dv = adinput.read_noise()

        # Only check read_noise here as gain descriptor is only used if units
        # are in ADU
        if read_noise_dv.is_none() and add_read_noise:
            # The descriptor functions return None if a value cannot be found
            # and stores the exception info. Re-raise the exception.
            if hasattr(adinput, "exception_info"):
                raise adinput.exception_info
            else:
                raise Errors.InputError("read_noise descriptor "
                                        "returned None...\n%s"
                                        % (read_noise_dv.info()))
            
        # Set the data type of the final variance array
        var_dtype = np.dtype(np.float32)
        
        # Loop over the science extensions in the dataset
        for ext in adinput[SCI]:
            extver = ext.extver()
            bunit  = ext.get_key_value("BUNIT")

            if bunit == "adu":
                # Get the gain value using the appropriate descriptor. The gain
                # is only used if the units are in ADU. Raise if gain is None
                gain = gain_dv.get_value(extver=extver)
                if gain is not None:
                    log.fullinfo("Gain for %s[%s,%d] = %f"
                                 % (adinput.filename, SCI, extver, gain))
                elif add_read_noise or add_poisson_noise:
                    err_msg = ("Gain for %s[%s,%d] is None. Cannot calculate "
                                "variance properly. Setting to zero."
                                % (adinput.filename, SCI, extver))
                    raise Errors.InputError(err_msg)
                
                units = "ADU"
            elif bunit == "electron" or bunit == "electrons":
                units = "electrons"
            else:
                # Perhaps something more sensible should be done here?
                raise Errors.InputError("No units found. Not calculating "
                                        "variance.")
            
            if add_read_noise:
                # Get the read noise value (in units of electrons) using the
                # appropriate descriptor. The read noise is only used if
                # add_read_noise is True
                read_noise = read_noise_dv.get_value(extver=extver)
                if read_noise is not None:
                    log.fullinfo("Read noise for %s[%s,%d] = %f"
                                 % (adinput.filename, SCI, extver, read_noise))
                    
                    # Determine the variance value to use when calculating the
                    # read noise component of the variance.
                    read_noise_var_value = read_noise
                    if units == "ADU":
                        read_noise_var_value = read_noise / gain
                    
                    # Add the read noise component of the variance to a zeros
                    # array that is the same size as the pixel data in the
                    # science extension
                    log.fullinfo("Calculating the read noise component of the "
                                 "variance in %s" % units)
                    var_array_rn = np.add(
                      np.zeros(ext.data.shape), (read_noise_var_value)**2)
                else:
                    logwarning("Read noise for %s[%s,%d] is None. Setting to "
                               "zero" % (adinput.filename, SCI, extver))
                    var_array_rn = np.zeros(ext.data.shape)
                    
            if add_poisson_noise:
                # Determine the variance value to use when calculating the
                # poisson noise component of the variance
                poisson_noise_var_value = ext.data
                if units == "ADU":
                    poisson_noise_var_value = ext.data / gain
                
                # Calculate the poisson noise component of the variance. Set
                # pixels that are less than or equal to zero to zero.
                log.fullinfo("Calculating the poisson noise component of "
                             "the variance in %s" % units)
                var_array_pn = np.where(
                  ext.data > 0, poisson_noise_var_value, 0)
            
            # Create the final variance array
            if add_read_noise and add_poisson_noise:
                var_array_final = np.add(var_array_rn, var_array_pn)
            
            if add_read_noise and not add_poisson_noise:
                var_array_final = var_array_rn
            
            if not add_read_noise and add_poisson_noise:
                var_array_final = var_array_pn
            
            var_array_final = var_array_final.astype(var_dtype)
            
            # If the read noise component and the poisson noise component are
            # calculated and added separately, then a variance extension will
            # already exist in the input AstroData object. In this case, just
            # add this new array to the current variance extension
            if adinput[VAR, extver]:
                
                # If both the read noise component and the poisson noise
                # component have been calculated, don't add to the variance
                # extension
                if add_read_noise and add_poisson_noise:
                    raise Errors.InputError(
                        "Cannot add read noise component and poisson noise "
                        "component to variance extension as the variance "
                        "extension already exists")
                else:
                    log.fullinfo("Combining the newly calculated variance "
                                 "with the current variance extension "
                                 "%s[%s,%d]" % (adinput.filename, VAR, extver))
                    adinput[VAR, extver].data = np.add(
                      adinput[VAR, extver].data,
                      var_array_final).astype(var_dtype)
            else:
                # Create the variance AstroData object
                var = AstroData(data=var_array_final)
                var.rename_ext(VAR, ver=extver)
                var.filename = adinput.filename
                
                # Call the _update_var_header helper function to update the
                # header of the variance extension with some useful keywords
                var = self._update_var_header(sci=ext, var=var, bunit=bunit)
                
                # Append the variance AstroData object to the input AstroData
                # object. 
                log.fullinfo("Adding the [%s,%d] extension to the input "
                             "AstroData object %s" % (VAR, extver,
                                                      adinput.filename))
                adinput.append(moredata=var)
        
        return adinput


    def _update_dq_header(self, sci=None, dq=None, bpmname=None):
        # Add the physical units keyword
        gt.update_key(adinput=dq, keyword="BUNIT", value="bit", comment=None,
                      extname=DQ)
        
        # Add the name of the bad pixel mask
        if bpmname is not None:
            gt.update_key(adinput=dq, keyword="BPMNAME", value=bpmname,
                          comment=None, extname=DQ)
        
        # These should probably be done using descriptors (?)
        keywords_from_sci = [
          "AMPNAME", "BIASSEC", "CCDNAME", "CCDSEC", "CCDSIZE", "CCDSUM",
          "CD1_1", "CD1_2", "CD2_1", "CD2_2", "CRPIX1", "CRPIX2", "CRVAL1",
          "CRVAL2", "CTYPE1", "CTYPE2", "DATASEC", "DETSEC", "EXPTIME", "GAIN",
          "GAINSET", "NONLINEA", "RDNOISE", "SATLEVEL", "LOWROW", "LOWCOL", "HIROW", "HICOL"] 
        dq_comment = "Copied from ['%s',%d]" % (SCI, sci.extver())
        
        for keyword in keywords_from_sci:
            # Check if the keyword exists in the header of the input science
            # extension
            keyword_value = sci.get_key_value(key=keyword)
            if keyword_value is not None:
                gt.update_key(adinput=dq, keyword=keyword, value=keyword_value,
                              comment=dq_comment, extname=DQ)
        
        return dq


    def _update_var_header(self, sci=None, var=None, bunit=None):
        # Add the physical units keyword
        if bunit is not None:
            gt.update_key(adinput=var, keyword="BUNIT", value="%s*%s"
                          % (bunit, bunit), comment=None, extname=VAR)
        
        # These should probably be done using descriptors (?)
        keywords_from_sci = [
          "AMPNAME", "BIASSEC", "CCDNAME", "CCDSEC", "CCDSIZE", "CCDSUM",
          "CD1_1", "CD1_2", "CD2_1", "CD2_2", "CRPIX1", "CRPIX2", "CRVAL1",
          "CRVAL2", "CTYPE1", "CTYPE2", "DATASEC", "DETSEC", "EXPTIME", "GAIN",
          "GAINSET", "NONLINEA", "RDNOISE", "SATLEVEL", "LOWROW", "LOWCOL", "HIROW", "HICOL"]
        var_comment = "Copied from ['%s',%d]" % (SCI, sci.extver())
        
        for keyword in keywords_from_sci:
            # Check if the keyword exists in the header of the input science
            # extension
            keyword_value = sci.get_key_value(key=keyword)
            if keyword_value is not None:
                gt.update_key(adinput=var, keyword=keyword,
                              value=keyword_value, comment=var_comment,
                              extname=VAR)
        
        return var


    def _bitwise_OR_list(self, input_list=None):
        """
        ##M This should get moved in the trunk to a library function
        Return a numpy array consisting of the bitwise OR combination of
        the input numpy arrays within the input list. List entries can be the
        None type.

        :param input_list: default = None: Numpy arrays to be bitwise OR'ed
                                           together. List can contain None
                                           types. No check of the type of the
                                           Numpy array is performed by this
                                           method.
        :type input_list: Python list
        
        :returns: Numpy array
        
        """
        # Test input_list is a list and not empty 
        if not isinstance(input_list, list):
            raise Errors.InputError("input_list is not a list")
        elif not input_list:
            raise Errors.InputError("input_list is empty")

        # Bitwise or arrays
        return_array = None
        for in_array in input_list:
            if in_array is not None:
                if return_array is None:
                    return_array = in_array
                else:
                    # Let numpy catch the case where in_array is not a numpy
                    # array
                    return_array = np.bitwise_or(return_array, in_array)

        # Test for all Nones
        if return_array is None:
            raise Errors.InputError("All input list values are None")

        return return_array
            
