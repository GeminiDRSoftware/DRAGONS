import os
import numpy as np
from astrodata import AstroData
from astrodata import Errors
from astrodata import IDFactory
from astrodata.adutils import logutils
from astrodata.adutils.reduceutils.prsproxyutil import upload_calibration
from gempy.gemini import gemini_tools as gt
from primitives_GENERAL import GENERALPrimitives

class CalibrationPrimitives(GENERALPrimitives):
    astrotype = "GEMINI"
    
    def init(self, rc):
        GENERALPrimitives.init(self, rc)
        return rc
    init.pt_hide = True
    
    def failCalibration(self,rc):
        # Mark a given calibration "fail" and upload it 
        # to the system. This is intended to be used to mark a 
        # calibration file that has already been uploaded, so that
        # it will not be returned as a valid match for future data.
        
        # Instantiate the log
        log = logutils.get_logger(__name__)
        
        # Initialize the list of output AstroData objects 
        adoutput_list = []
        
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():
            
            # Change the two keywords -- BAD and NO = Fail
            ad.phu_set_key_value("RAWGEMQA","BAD",
                                 comment=self.keyword_comments["RAWGEMQA"])
            ad.phu_set_key_value("RAWPIREQ","NO",
                                 comment=self.keyword_comments["RAWPIREQ"])
            log.fullinfo("%s has been marked %s" % (ad.filename,ad.qa_state()))
            
            # Append the output AstroData object to the list
            # of output AstroData objects
            adoutput_list.append(ad)
        
        # Report the list of output AstroData objects to the 
        # reduction context
        rc.report_output(adoutput_list)
        
        # Run the storeCalibration primitive, so that the 
        # failed file gets re-uploaded
        rc.run("storeCalibration")
        
        yield rc
    
    def getCalibration(self, rc):
        
        # Instantiate the log
        log = logutils.get_logger(__name__)
        
        # Retrieve type of calibration requested
        caltype = rc["caltype"]
        if caltype == None:
            log.error("getCalibration: caltype not set")
            raise Errors.PrimitiveError("getCalibration: caltype not set")
        
        # Retrieve source of calibration
        source = rc["source"]
        if source == None:
            source = "all"
            
        # Check whether calibrations are already available
        calibrationless_adlist = []
        adinput = rc.get_inputs_as_astrodata()
        
        
        #print "70: WRITE ALL CALIBRATION SOURCES\n"*10
        #for ad in adinput:
        #    ad.write(clobber=True)

        #for ad in adinput:
        #    ad.mode = "update"
        #    calurl = rc.get_cal(ad,caltype)
        #    if not calurl:
        #        calibrationless_adlist.append(ad)
        calibrationless_adlist = adinput
        # Request any needed calibrations
        if len(calibrationless_adlist) ==0:
            # print "pG603: calibrations for all files already present"
            pass
        else:
            rc.rq_cal(caltype, calibrationless_adlist, source=source)
        
        yield rc
    
    def getProcessedArc(self, rc):
        # Instantiate the log
        log = logutils.get_logger(__name__)
        
        caltype = "processed_arc"
        source = rc["source"]
        if source == None:
            rc.run("getCalibration(caltype=%s)" % caltype)
        else:
            rc.run("getCalibration(caltype=%s, source=%s)" % (caltype,source))
        
        # List calibrations found
        first = True
        for ad in rc.get_inputs_as_astrodata():
            calurl = rc.get_cal(ad, caltype) #get from cache
            if calurl:
                cal = AstroData(calurl)
                if cal.filename is None:
                    if "qa" not in rc.context:
                        raise Errors.InputError("Calibration not found for " \
                                                "%s" % ad.filename)
                else:
                    if first:
                        log.stdinfo("getCalibration: Results")
                        first = False
                    log.stdinfo("   %s\n      for %s" % (cal.filename,
                                                         ad.filename))
            else: 
                if "qa" not in rc.context:
                    raise Errors.InputError("Calibration not found for %s" % 
                                            ad.filename)
        
        yield rc
    
    def getProcessedBias(self, rc):
        # Instantiate the log
        log = logutils.get_logger(__name__)
        
        caltype = "processed_bias"
        source = rc["source"]
        if source == None:
            rc.run("getCalibration(caltype=%s)" % caltype)
        else:
            rc.run("getCalibration(caltype=%s, source=%s)" % (caltype,source))
        
        # List calibrations found
        first = True
        for ad in rc.get_inputs_as_astrodata():
            calurl = rc.get_cal(ad, caltype) #get from cache
            if calurl:
                cal = AstroData(calurl)
                if cal.filename is None:
                    if "qa" not in rc.context:
                        raise Errors.InputError("Calibration not found for " \
                                                "%s" % ad.filename)
                else:
                    if first:
                        log.stdinfo("getCalibration: Results")
                        first = False
                    log.stdinfo("   %s\n      for %s" % (cal.filename,
                                                         ad.filename))
            else: 
                if "qa" not in rc.context:
                    raise Errors.InputError("Calibration not found for %s" % 
                                            ad.filename)
        
        yield rc
    
    def getProcessedDark(self, rc):
        # Instantiate the log
        log = logutils.get_logger(__name__)
        
        caltype = "processed_dark"
        source = rc["source"]
        if source == None:
            rc.run("getCalibration(caltype=%s)" % caltype)
        else:
            rc.run("getCalibration(caltype=%s, source=%s)" % (caltype,source))
            
        # List calibrations found
        first = True
        for ad in rc.get_inputs_as_astrodata():
            calurl = rc.get_cal(ad, caltype) #get from cache
            if calurl:
                cal = AstroData(calurl)
                if cal.filename is None:
                    if "qa" not in rc.context:
                        raise Errors.InputError("Calibration not found for " \
                                                "%s" % ad.filename)
                else:
                    if first:
                        log.stdinfo("getCalibration: Results")
                        first = False
                    log.stdinfo("   %s\n      for %s" % (cal.filename,
                                                         ad.filename))
            else: 
                if "qa" not in rc.context:
                    raise Errors.InputError("Calibration not found for %s" % 
                                            ad.filename)
        
        yield rc
    
    def getProcessedFlat(self, rc):
        # Instantiate the log
        log = logutils.get_logger(__name__)
        
        caltype = "processed_flat"
        source = rc["source"]
        if source == None:
            rc.run("getCalibration(caltype=%s)" % caltype)
        else:
            rc.run("getCalibration(caltype=%s, source=%s)" % (caltype,source))
        
        # List calibrations found
        first = True
        for ad in rc.get_inputs_as_astrodata():
            calurl = rc.get_cal(ad, caltype) #get from cache
            if calurl:
                cal = AstroData(calurl)
                if cal.filename is None:
                    if "qa" not in rc.context:
                        raise Errors.InputError("Calibration not found for " \
                                                "%s" % ad.filename)
                else:
                    if first:
                        log.stdinfo("getCalibration: Results")
                        first = False
                    log.stdinfo("   %s\n      for %s" % (cal.filename,
                                                         ad.filename))
            else: 
                if "qa" not in rc.context:
                    raise Errors.InputError("Calibration not found for %s" % 
                                            ad.filename)
        
        yield rc
    
    def getProcessedFringe(self, rc):
        # Instantiate the log
        log = logutils.get_logger(__name__)
        
        caltype = "processed_fringe"
        source = rc["source"]
        if source == None:
            rc.run("getCalibration(caltype=%s)" % caltype)
        else:
            rc.run("getCalibration(caltype=%s, source=%s)" % (caltype,source))
            
        # List calibrations found
        # Fringe correction is always optional, so don't raise errors if fringe
        # not found
        first = True
        for ad in rc.get_inputs_as_astrodata():
            calurl = rc.get_cal(ad, caltype) #get from cache
            if calurl:
                cal = AstroData(calurl)
                if cal.filename is not None:
                    if first:
                        log.stdinfo("getCalibration: Results")
                        first = False
                    log.stdinfo("   %s\n      for %s" % (cal.filename,
                                                     ad.filename))
        
        yield rc
    
    def showCals(self, rc):
        # Instantiate the log
        log = logutils.get_logger(__name__)
        
        if str(rc["showcals"]).lower() == "all":
            num = 0
            # print "pG256: showcals=all", repr (rc.calibrations)
            for calkey in rc.calibrations:
                num += 1
                log.stdinfo(rc.calibrations[calkey], category="calibrations")
            if (num == 0):
                log.stdinfo("There are no calibrations in the cache.")
        else:
            for adr in rc.inputs:
                sid = IDFactory.generate_astro_data_id(adr.ad)
                num = 0
                for calkey in rc.calibrations:
                    if sid in calkey :
                        num += 1
                        log.stdinfo(rc.calibrations[calkey], 
                                     category="calibrations")
            if (num == 0):
                log.stdinfo("There are no calibrations in the cache.")
        
        yield rc
    ptusage_showCals="Used to show calibrations currently in cache for inputs."
    
    def storeCalibration(self, rc):
        # Instantiate the log
        log = logutils.get_logger(__name__)
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "storeCalibration", "starting"))
        
        # Determine the path where the calibration will be stored
        storedcals = rc["cachedict"]["storedcals"]
        
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():
            
            # Construct the filename of the calibration, including the path
            fname = os.path.join(storedcals, os.path.basename(ad.filename))
            
            # Write the calibration to disk. Use rename=False so that
            # ad.filename does not change (i.e., does not include the
            # calibration path)
            ad.write(filename=fname, rename=False, clobber=True)
            log.stdinfo("Calibration stored as %s" % fname)
            
            if "upload" in rc.context:
                try:
                    upload_calibration(fname)
                except:
                    log.warning("Unable to upload file to calibration system")
                else:
                    log.stdinfo("File %s uploaded to fitsstore." % 
                                os.path.basename(ad.filename))
            yield rc
        
        yield rc
    
    def storeProcessedArc(self, rc):
        # Instantiate the log
        log = logutils.get_logger(__name__)
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "storeProcessedArc",
                                 "starting"))
        
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():
            
            # Updating the file name with the suffix for this primitive and
            # then report the new file to the reduction context
            ad.filename = gt.filename_updater(adinput=ad, suffix=rc["suffix"],
                                              strip=True)
            
            # Adding a PROCARC time stamp to the PHU
            gt.mark_history(adinput=ad, keyword="PROCARC")
            
            # Refresh the AD types to reflect new processed status
            ad.refresh_types()
        
        # Upload arc(s) to cal system
        rc.run("storeCalibration")
        
        yield rc
    
    def storeProcessedBias(self, rc):
        # Instantiate the log
        log = logutils.get_logger(__name__)
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "storeProcessedBias",
                                 "starting"))
        
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():
            
            # Updating the file name with the suffix for this primitive and
            # then report the new file to the reduction context
            ad.filename = gt.filename_updater(adinput=ad, suffix=rc["suffix"],
                                              strip=True)
            
            # Adding a PROCBIAS time stamp to the PHU
            gt.mark_history(adinput=ad, keyword="PROCBIAS")
            
            # Refresh the AD types to reflect new processed status
            ad.refresh_types()
        
        # Upload bias(es) to cal system
        rc.run("storeCalibration")
        
        yield rc
    
    def storeProcessedDark(self, rc):
        # Instantiate the log
        log = logutils.get_logger(__name__)
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "storeProcessedDark",
                                 "starting"))
        
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():
            
            # Updating the file name with the suffix for this primitive and
            # then report the new file to the reduction context
            ad.filename = gt.filename_updater(adinput=ad, suffix=rc["suffix"],
                                              strip=True)
            
            # Adding a PROCDARK time stamp to the PHU
            gt.mark_history(adinput=ad, keyword="PROCDARK")
            
            # Refresh the AD types to reflect new processed status
            ad.refresh_types()
        
        # Upload to cal system
        rc.run("storeCalibration")
        
        yield rc
    
    def storeProcessedFlat(self, rc):
        # Instantiate the log
        log = logutils.get_logger(__name__)
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "storeProcessedFlat",
                                 "starting"))
        
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():
            
            # Updating the file name with the suffix for this primitive and
            # then report the new file to the reduction context
            ad.filename = gt.filename_updater(adinput=ad, suffix=rc["suffix"],
                                              strip=True)
            
            # Adding a PROCFLAT time stamp to the PHU
            gt.mark_history(adinput=ad, keyword="PROCFLAT")
            
            # Refresh the AD types to reflect new processed status
            ad.refresh_types()
        
        # Upload to cal system
        rc.run("storeCalibration")
        
        yield rc
    
    def storeProcessedFringe(self, rc):
        # Instantiate the log
        log = logutils.get_logger(__name__)
        
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "storeProcessedFringe",
                                 "starting"))
        
        # Initialize the list of output AstroData objects
        adoutput_list = []
        
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():
            
            # Updating the file name with the suffix for this primitive and
            # then report the new file to the reduction context
            ad.filename = gt.filename_updater(adinput=ad, suffix=rc["suffix"],
                                              strip=True)
            
            # Sanitize the headers of the file so that it looks like
            # a public calibration file rather than a science file
            ad = gt.convert_to_cal_header(adinput=ad, caltype="fringe")[0]
            
            # Adding a PROCFRNG time stamp to the PHU
            gt.mark_history(adinput=ad, keyword="PROCFRNG")
            
            # Refresh the AD types to reflect new processed status
            ad.refresh_types()
            
            adoutput_list.append(ad)
        
        # Upload to cal system
        rc.run("storeCalibration")
        
        # Report the list of output AstroData objects to the reduction
        # context
        rc.report_output(adoutput_list)
        
        yield rc

    def separateLampOff(self, rc):
        """
        This primitive is intended to run on gcal imaging flats. 
        It goes through the input list and figures out which ones are lamp-on
        and which ones are lamp-off
        """
        # Instantiate the log
        log = logutils.get_logger(__name__)

        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "separateLampOff", "starting"))

        # Initialize the list of output AstroData objects
        lampon_list = []
        lampoff_list = []

        # Loop over the input frames
        for ad in rc.get_inputs_as_astrodata():
            if('GCAL_IR_ON' in ad.types):
                log.stdinfo("%s is a lamp-on flat" % ad.data_label())
                #rc.run("addToList(purpose=lampOn)")
                lampon_list.append(ad)
            elif('GCAL_IR_OFF' in ad.types):
                log.stdinfo("%s is a lamp-off flat" % ad.data_label())
                #rc.run("addToList(purpose=lampOff)")
                lampoff_list.append(ad)
            else:
                log.warning("Not a GCAL flatfield? Cannot tell if it is lamp-on or lamp-off for %s" % ad.data_label())

        rc.report_output(lampon_list, stream="lampOn")
        rc.report_output(lampoff_list, stream="lampOff")

        yield rc


    def stackLampOnLampOff(self, rc):
        """
        This primitive stacks the Lamp On flats and the LampOff flats, then subtracts the two stacks
        """
        # Instantiate the log
        log = logutils.get_logger(__name__)

        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "stackLampOnLampOff", "starting"))

        # Initialize the list of output AstroData objects
        adoutput_list = []

        # Get the lamp on list, stack it, and add the stack to the lampOnStack stream
        rc.run("showInputs(stream=lampOn)")
        rc.run("stackFrames(stream=lampOn)")

        # Get the lamp off list, stack it, and add the stack to the lampOnStack stream
        rc.run("showInputs(stream=lampOff)")
        rc.run("stackFrames(stream=lampOff)")

        yield rc

    def subtractLampOnLampOff(self, rc):
        """
        This primitive subtracts the lamp off stack from the lampon stack. It expects there to be only
        one file (the stack) on each stream - call stackLampOnLampOff to do the stacking before calling this
        """

        # Instantiate the log
        log = logutils.get_logger(__name__)

        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "subtractLampOnLampOff", "starting"))

        # Initialize the list of output AstroData objects
        adoutput_list = []

        lampon_list = rc.get_stream(stream="lampOn", style="AD")
        lampoff_list = rc.get_stream(stream="lampOff", style="AD")

        if((len(lampon_list) > 0) and (len(lampoff_list) > 0)):
            lampon = lampon_list[0]
            lampoff = lampoff_list[0]
 

            log.stdinfo("Lamp ON is: %s %s" % (lampon.data_label(), lampon.filename))
            log.stdinfo("Lamp OFF is: %s %s" % (lampoff.data_label(), lampoff.filename))
            lampon.sub(lampoff)
            lampon.filanme = gt.filename_updater(adinput=lampon, suffix="lampOnOff")

            adoutput_list.append(lampon)

        else:
            log.warning("Cannot subtract lamp on - lamp off flats as do not have some of each")

        rc.report_output(adoutput_list)
        yield rc
