from astrodata.utils import logutils
from astrodata.utils import Errors

from gempy.gemini import gemini_tools as gt
from gempy.adlibrary.mosaicAD import MosaicAD
from gempy.gemini.gemMosaicFunction import gemini_mosaic_function
from gempy.adlibrary.extract import trace_footprints, cut_footprints

from primitives_bookkeeping import BookkeepingPrimitives
from primitives_calibration import CalibrationPrimitives
from primitives_display import DisplayPrimitives
from primitives_mask import MaskPrimitives
from primitives_photometry import PhotometryPrimitives
from primitives_preprocess import PreprocessPrimitives
from primitives_qa import QAPrimitives
from primitives_register import RegisterPrimitives
from primitives_resample import ResamplePrimitives
from primitives_stack import StackPrimitives
from primitives_standardize import StandardizePrimitives


class GEMINIPrimitives(BookkeepingPrimitives,CalibrationPrimitives,
                       DisplayPrimitives, MaskPrimitives,
                       PhotometryPrimitives,PreprocessPrimitives,
                       QAPrimitives,RegisterPrimitives,
                       ResamplePrimitives,StackPrimitives,
                       StandardizePrimitives):
    """
    This is the class containing all of the primitives for the GEMINI level of
    the type hierarchy tree. It inherits all the primitives from the level
    above, 'GENERALPrimitives'.
    """
    astrotype = "GEMINI"

    def init(self, rc):
        BookkeepingPrimitives.init(self, rc)
        CalibrationPrimitives.init(self, rc)
        DisplayPrimitives.init(self, rc)
        MaskPrimitives.init(self, rc)
        PhotometryPrimitives.init(self, rc)
        PreprocessPrimitives.init(self, rc)
        QAPrimitives.init(self, rc)
        RegisterPrimitives.init(self, rc)
        ResamplePrimitives.init(self, rc)
        StackPrimitives.init(self, rc)
        StandardizePrimitives.init(self, rc)
        return rc
    init.pt_hide = True

    def standardizeGeminiHeaders(self, rc):
        """
        This primitive is used to make the changes and additions to the
        keywords in the headers of Gemini data.
        """
        # Instantiate the log
        log = logutils.get_logger(__name__)

        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "standardizeGeminiHeaders",
                                 "starting"))

        # Define the keyword to be used for the time stamp for this primitive
        timestamp_key = self.timestamp_keys["standardizeGeminiHeaders"]

        # Initialize the list of output AstroData objects
        adoutput_list = []

        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():

            # Check whether the standardizeGeminiHeaders primitive has been run
            # previously
            if ad.phu_get_key_value(timestamp_key):
                log.warning("No changes will be made to %s, since it has "
                            "already been processed by "
                            "standardizeGeminiHeaders" % ad.filename)

                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue

            # Standardize the headers of the input AstroData object. Update the
            # keywords in the headers that are common to all Gemini data.
            log.status("Updating keywords that are common to all Gemini data")

            # Original name
            ad.store_original_name()

            # Number of science extensions
            gt.update_key(adinput=ad, keyword="NSCIEXT", value=ad.count_exts("SCI"),
                comment=None, extname="PHU", keyword_comments=self.keyword_comments)

            # Number of extensions
            gt.update_key(adinput=ad, keyword="NEXTEND", value=len(ad), comment=None,
                          extname="PHU", keyword_comments=self.keyword_comments)

            # Physical units (assuming raw data has units of ADU)
            gt.update_key(adinput=ad, keyword="BUNIT", value="adu", comment=None,
                          extname="SCI", keyword_comments=self.keyword_comments)

            # Add the appropriate time stamps to the PHU
            gt.mark_history(adinput=ad, primname=self.myself(), keyword=timestamp_key)

            # Change the filename
            ad.filename = gt.filename_updater(adinput=ad, suffix=rc["suffix"],
                                              strip=True)

            # Append the output AstroData object to the list of output
            # AstroData objects
            adoutput_list.append(ad)

        # Report the list of output AstroData objects to the reduction context
        rc.report_output(adoutput_list)

        yield rc

    def mosaicADdetectors(self,rc):      # Uses python MosaicAD script
        """
        This primitive will mosaic the SCI frames of the input images, along
        with the VAR and DQ frames if they exist.

        :param tile: tile images instead of mosaic
        :type tile: Python boolean (True/False), default is False

        """
        log = logutils.get_logger(__name__)

        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "mosaicADdetectors", "starting"))

        # Define the keyword to be used for the time stamp for this primitive
        timestamp_key = self.timestamp_keys["mosaicADdetectors"]

        # Initialize the list of output AstroData objects
        adoutput_list = []

        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():

            # Validate Data
            #if (ad.phu_get_key_value('GPREPARE')==None) and \
            #    (ad.phu_get_key_value('PREPARE')==None):
            #    raise Errors.InputError("%s must be prepared" % ad.filename)

            # Check whether the mosaicDetectors primitive has been run
            # previously
            if ad.phu_get_key_value(timestamp_key):
                log.warning("No changes will be made to %s, since it has " \
                            "already been processed by mosaicDetectors" \
                            % (ad.filename))
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue

            # If the input AstroData object only has one extension, there is no
            # need to mosaic the detectors
            if ad.count_exts("SCI") == 1:
                log.stdinfo("No changes will be made to %s, since it " \
                            "contains only one extension" % (ad.filename))
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue

            # Get the necessary parameters from the RC
            tile = rc["tile"]

            log.stdinfo("Mosaicking %s ..."%ad.filename)
            log.stdinfo("MosaicAD: Using tile: %s ..."%tile)
            #t1 = time.time()
            mo = MosaicAD(ad,
                            mosaic_ad_function=gemini_mosaic_function,
                            dq_planes=rc['dq_planes'])

            adout = mo.as_astrodata(tile=tile)
            #t2 = time.time()
            #print '%s took %0.3f ms' % ('as_astrodata', (t2-t1)*1000.0)

            # Verify mosaicAD was actually run on the file
            # then log file names of successfully reduced files
            if adout.phu_get_key_value("MOSAIC"):
                log.fullinfo("File "+adout.filename+\
                            " was successfully mosaicked")

            # Add the appropriate time stamps to the PHU
            gt.mark_history(adinput=adout, primname=self.myself(), keyword=timestamp_key)

            # Change the filename
            adout.filename = gt.filename_updater(
                adinput=ad, suffix=rc["suffix"], strip=True)

            # Append the output AstroData object to the list
            # of output AstroData objects
            adoutput_list.append(adout)

        # Report the list of output AstroData objects to the reduction
        # context
        rc.report_output(adoutput_list)

        yield rc


    def traceFootprints(self, rc):

        """
        This primitive will create and append a 'TRACEFP' Bintable HDU to the
        AD object. The content of this HDU is the footprints information
        from the espectroscopic flat in the SCI array.

        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to
                        screen. OR the message level as a string (i.e.,
                        'critical', 'status', 'fullinfo'...)
        """
        # Instantiate the log
        log = logutils.get_logger(__name__)

        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "", "starting"))

        # Initialize the list of output AstroData objects
        adoutput_list = []

        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():
            # Check whether this primitive has been run previously
            if ad.phu_get_key_value("TRACEFP"):
                log.warning("%s has already been processed by traceSlits" \
                            % (ad.filename))
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue
            # Call the  user level function
            try:
               adout = trace_footprints(ad,function=rc["function"],
                                  order=rc["order"],
                                  trace_threshold=rc["trace_threshold"])
            except:
               log.warning("Error in traceFootprints with file: %s"%ad.filename)

            # Change the filename
            adout.filename = gt.filename_updater(adinput=ad,
                                                 suffix=rc["suffix"],
                                                 strip=True)

            # Append the output AstroData object to the list of output
            # AstroData objects.
            adoutput_list.append(adout)

        # Report the list of output AstroData objects to the reduction
        # context
        rc.report_output(adoutput_list)

        yield rc


    def cutFootprints(self, rc):

        """
        This primitive will create and append multiple HDU to the output
        AD object. Each HDU correspond to a rectangular cut containing a
        slit from a MOS Flat exposure or a XD flat exposure as in the
        Gnirs case.

        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to
                        screen. OR the message level as a string (i.e.,
                        'critical', 'status', 'fullinfo'...)
        """

        # Instantiate the log
        log = logutils.get_logger(__name__)

        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "cutFootprints", "starting"))

        # Initialize the list of output AstroData objects
        adoutput_list = []

        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():
            # Call the  user level function

            # Check that the input ad has the TRACEFP extension,
            # otherwise, create it.
            if ad['TRACEFP'] == None:
                ad = trace_footprints(ad)

            log.stdinfo("Cutting_footprints for: %s"%ad.filename)
            try:
                adout = cut_footprints(ad)
            except:
                log.error("Error in cut_slits with file: %s"%ad.filename)
                # DO NOT add this input ad to the adoutput_lis
                continue


            # Change the filename
            adout.filename = gt.filename_updater(adinput=ad,
                                                 suffix=rc["suffix"],
                                                 strip=True)

            # Append the output AstroData object to the list of output
            # AstroData objects.
            adoutput_list.append(adout)

        # Report the list of output AstroData objects to the reduction
        # context
        rc.report_output(adoutput_list)

        yield rc

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
                gt.mark_history(adinput=ad, primname=self.myself(), keyword=timestamp_key)

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
