from astrodata.adutils import gemLog
from gempy import geminiTools as gt
from gempy.science import preprocessing as pp
from primitives_GMOS import GMOSPrimitives

class GMOS_IMAGEPrimitives(GMOSPrimitives):
    """
    This is the class containing all of the primitives for the GMOS_IMAGE
    level of the type hierarchy tree. It inherits all the primitives from the
    level above, 'GMOSPrimitives'.
    """
    astrotype = "GMOS_IMAGE"
    
    def init(self, rc):
        GMOSPrimitives.init(self, rc)
        return rc
    
    def normalize(self, rc):
        """
        This primitive will combine the input flats and then normalize them
        using the CL script giflat.
        
        Warning: giflat calculates its own DQ frames and thus replaces the
        previously produced ones in calculateDQ. This may be fixed in the
        future by replacing giflat with a Python equivilent with more
        appropriate options for the recipe system. 
        
        :param overscan: Subtract the overscan level from the frames?
        :type overscan: Python boolean (True/False)
        
        :param trim: Trim the overscan region from the frames?
        :type trim: Python boolean (True/False)
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to 
                        screen. OR the message level as a string (i.e.,
                        'critical', 'status', 'fullinfo'...)
        """
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "normalize", "starting"))
        adoutput_list = []
        for ad in rc.get_inputs(style='AD'):
            if ad.phu_get_key_value('GIFLAT'):
                log.warning('%s has already been processed by normalize' %
                            (ad.filename))
                adoutput_list.append(ad)
                continue
            
            ad = pp.normalize_flat_image_gmos(adinput=ad,
                                              trim=rc["trim"],
                                              overscan=rc["overscan"])
            adoutput_list.append(ad[0])

        rc.report_output(adoutput_list)
        yield rc
    

    def makeFringe(self, rc):
        """
        This primitive makes a fringe frame by masking out sources
        in the science frames and stacking them together.  It calls 
        gifringe to do so, so works only for GMOS imaging currently.
        When called in the context of a science reduction script,
        separateFringe should be run first, then makeFringe run with 
        the parameter stream="fringe".
        """
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])

        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "makeFringe", "starting"))

        adinput = rc.get_inputs(style="AD")
        if len(adinput)<2:
            log.warning('Only one frame provided as input; at least two ' +
                        'frames are required. Not making fringe frame.')
            adoutput = adinput
        else:
            # Call the make_fringe_image_gmos user level function
            adoutput = pp.make_fringe_image_gmos(adinput=adinput,
                                                 suffix=rc["suffix"],
                                                 operation=rc["operation"])

        # Report the list of output AstroData objects to the reduction
        # context
        rc.report_output(adoutput)
        
        yield rc

    def separateFringe(self,rc):
        """
        This primitive sends images to be used for making the fringe
        frame into the "fringe" stream.  Currently, all it does is report
        the main inputs (all science frames) to the fringe stream.  It may
        be desirable later to include handling for frames taken solely
        for the purpose of making fringe frames (similar to the way sky
        frames are handled).
        """

        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])

        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "separateFringe", "starting"))

        # Get input frames
        adinput = rc.get_inputs(style="AD")

        # Report them to both the fringe stream and the main stream
        rc.report_output(adinput,stream="fringe")
        rc.report_output(adinput,stream="main")
        
        yield rc
