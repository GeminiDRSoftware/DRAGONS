import os
from datetime import datetime
import shutil
import time

from astrodata import AstroData
from astrodata import Errors
from astrodata import IDFactory
from astrodata.adutils import gemLog
from gempy import geminiTools as gt
from gempy.science import preprocessing as pp
from gempy.science import qa
from gempy.science import stack as sk
from gempy.science import standardization as sdz
from primitives_GENERAL import GENERALPrimitives

class GEMINIPrimitives(GENERALPrimitives):
    """
    This is the class containing all of the primitives for the GEMINI level of
    the type hierarchy tree. It inherits all the primitives from the level
    above, 'GENERALPrimitives'.
    """
    astrotype = "GEMINI"
    
    def init(self, rc):
        GENERALPrimitives.init(self, rc)
        return rc
    init.pt_hide = True
    
    def addDQ(self, rc):
        """
        This primitive will create a numpy array for the data quality 
        of each SCI frame of the input data. This will then have a 
        header created and append to the input using AstroData as a DQ 
        frame. The value of a pixel will be the sum of the following: 
        (0=good, 1=bad pixel (found in bad pixel mask), 
        2=value is non linear, 4=pixel is saturated)
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to 
                        screen. OR the message level as a string (i.e.,
                        'critical', 'status', 'fullinfo'...)
        """
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "addDQ", "starting"))
        # Initialize the list of output AstroData objects
        adoutput_list = []
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs(style="AD"):
            # Check whether the addDQ primitive has been run previously
            if ad.phu_get_key_value("ADDDQ"):
                log.warning("%s has already been processed by addDQ" \
                            % (ad.filename))
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue
            # Call the add_dq user level function
            ad = sdz.add_dq(adinput=ad)
            # Append the output AstroData object (which is currently in the
            # form of a list) to the list of output AstroData objects
            adoutput_list.append(ad[0])
        # Report the list of output AstroData objects to the reduction
        # context
        rc.report_output(adoutput_list)
        
        yield rc
    
    def addToList(self, rc):
        """
        This primitive will update the lists of files to be stacked
        that have the same observationID with the current inputs.
        This file is cached between calls to reduce, thus allowing
        for one-file-at-a-time processing.
        
        :param purpose: 
        :type purpose: string, either: "" for regular image stacking, 
                       or 'fringe' for fringe stacking.
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to 
                        screen. OR the message level as a string (i.e.,
                        'critical', 'status', 'fullinfo'...)
        """
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        # Perform an update to the stack cache file (or create it) using the
        # current inputs in the reduction context
        purpose = rc["purpose"]
        if purpose is None:
            purpose = ""
        # Call the rq_stack_update method
        rc.rq_stack_update(purpose=purpose)
        # Write the files in the stack to disk if they do not already exist
        for ad in rc.get_inputs(style="AD"):
            if not os.path.exists(ad.filename):
                log.fullinfo("writing %s to disk" % ad.filename,
                             category="list")
                ad.write(ad.filename)
        
        yield rc
    
    def addVAR(self, rc):
        """
        This primitive uses numpy to calculate the variance of each SCI frame
        in the input files and appends it as a VAR frame using AstroData.
        
        The calculation will follow the formula:
        variance = (read noise/gain)2 + max(data,0.0)/gain
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to 
                        screen. OR the message level as a string (i.e.,
                        'critical', 'status', 'fullinfo'...)
        """
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "addVAR", "starting"))
        # Initialize the list of output AstroData objects
        adoutput_list = []
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs(style="AD"):
            # Check whether the addVAR primitive has been run previously
            if ad.phu_get_key_value("ADDVAR"):
                log.warning("%s has already been processed by addVAR" \
                            % (ad.filename))
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue
            # Call the add_var user level function
            ad = sdz.add_var(adinput=ad)
            # Append the output AstroData object (which is currently in the
            # form of a list) to the list of output AstroData objects
            adoutput_list.append(ad[0])
        # Report the list of output AstroData objects to the reduction
        # context
        rc.report_output(adoutput_list)
        
        yield rc 
    
    def aduToElectrons(self, rc):
        """
        This primitive will convert the inputs from having pixel 
        units of ADU to electrons.
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to 
                        screen. OR the message level as a string (i.e.,
                        'critical', 'status', 'fullinfo'...)
        """
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "aduToElectrons", "starting"))
        # Initialize the list of output AstroData objects
        adoutput_list = []
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs(style="AD"):
            # Check whether the aduToElectrons primitive has been run
            # previously
            if ad.phu_get_key_value("ADU2ELEC"):
                log.warning("%s has already been processed by aduToElectrons" \
                            % (ad.filename))
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue
            # Call the adu_to_electrons user level function
            ad = pp.adu_to_electrons(adinput=ad)
            # Append the output AstroData object (which is currently in the
            # form of a list) to the list of output AstroData objects
            adoutput_list.append(ad[0])
        # Report the list of output AstroData objects to the reduction
        # context
        rc.report_output(adoutput_list)
        
        yield rc
    
    def clearCalCache(self, rc):
        # print "pG61:", rc.calindfile
        rc.persist_cal_index(rc.calindfile, newindex={})
        scals = rc["storedcals"]
        if scals:
            if os.path.exists(scals):
                shutil.rmtree(scals)
            cachedict = rc["cachedict"]
            for cachename in cachedict:
                cachedir = cachedict[cachename]
                if not os.path.exists(cachedir):
                    os.mkdir(cachedir)
        
        yield rc
     
    def crashReduce(self, rc):
        raise "Crashing"
        yield rc
    
    def display(self, rc):
        rc.rq_display(display_id=rc["display_id"])
        
        yield rc
    
    def divideByFlat(self, rc):
        """
        This primitive will divide each SCI extension of the inputs by those
        of the corresponding flat. If the inputs contain VAR or DQ frames,
        those will also be updated accordingly due to the division on the data.
        
        This is all conducted in pure Python through the arith 'toolbox' of 
        astrodata. 
        
        It is currently assumed that the same flat file will be applied to all
        input images.
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to 
                        screen. OR the message level as a string (i.e.,
                        'critical', 'status', 'fullinfo'...)
        """
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "divideByFlat", "starting"))
        # Retrieving the appropriate flat for the first of the inputs
        adOne = rc.get_inputs(style="AD")[0]
        #processedFlat = AstroData(rc.get_cal(adOne,"flat"))
        ###################BULL CRAP FOR TESTING ######################### 
        from copy import deepcopy
        processedFlat = deepcopy(adOne)
        processedFlat.filename = "TEMPNAMEforFLAT.fits"
        processedFlat.phu_set_key_value("ORIGNAME","TEMPNAMEforFLAT.fits")
        ###################################################################
        
        # Taking care of the case where there was no, or an invalid flat 
        if processedFlat.count_exts("SCI") == 0:
            raise Errors.PrimitiveError("Invalid processed flat " +
                                        "retrieved")
        # Call the divide_by_flat user level function
        output = pp.divide_by_flat(adinput=rc.get_inputs(style="AD"),
                                   flats=processedFlat)
        # Report the list of output AstroData objects to the reduction
        # context
        rc.report_output(output)
        
        yield rc
     
    def getCal(self, rc):
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        caltype = rc["caltype"]
        if caltype is None:
            log.critical("Requested a calibration no particular " +
                         "calibration type")
            raise Errors.PrimitiveError("get_cal: %s was None" % caltype)
        source = rc["source"]
        if source is None:
            source = "all"
        
        centralSource = False
        localSource = False
        if source == "all":
            centralSource = True
            localSource = True
        if source == "central":
            centralSource = True
        if source == "local":
            localSource = True
        
        inps = rc.get_inputs_as_astro_data()
        
        if localSource:
            rc.rq_cal(caltype, inps, source="local")
            for ad in inps:
                cal = rc.get_cal(ad, caltype)
                if cal is None:
                    print "get central"
                else:
                    print "got local", cal
            
            yield rc
    
    def getList(self, rc):
        """
        This primitive will check the files in the stack lists are on disk,
        and then update the inputs list to include all members of the stack 
        for stacking.
        
        :param purpose: 
        :type purpose: string, either: '' for regular image stacking, 
                       or 'fringe' for fringe stacking.
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to 
                        screen. OR the message level as a string (i.e.,
                        'critical', 'status', 'fullinfo'...)
        """
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        sidset = set()
        purpose=rc["purpose"]
        if purpose is None:
            purpose = ""
        for inp in rc.inputs:
            sidset.add(purpose+IDFactory.generate_stackable_id(inp.ad))
        for sid in sidset:
            stacklist = rc.get_stack(sid) #.filelist
            log.fullinfo("List for stack id=%s" % sid, category="list")
            for f in stacklist:
                rc.report_output(f)
                log.fullinfo("   %s" % os.path.basename(f),
                             category="list")
        
        yield rc
    
    def getProcessedBias(self, rc):
        """
        This primitive will check the files in the lists that are on disk,
        and then update the inputs list to include all members of the list.
        """
        rc.rq_cal("bias", rc.get_inputs(style="AD"))
        yield rc
    
    def getProcessedDark(self, rc):
        """
        A primitive to search and return the appropriate calibration dark from
        a server for the given inputs.
        """
        rc.rq_cal("dark", rc.get_inputs(style="AD"))
        yield rc
    
    def getProcessedFlat(self, rc):
        """
        A primitive to search and return the appropriate calibration flat from
        a server for the given inputs.
        
        """
        rc.rq_cal("flat", rc.get_inputs(style="AD"))
        yield rc
    
    def measureIQ(self, rc):
        """
        This primitive will detect the sources in the input images and fit
        both Gaussian and Moffat models to their profiles and calculate the 
        Image Quality and seeing from this.
        
        :param function: Function for centroid fitting
        :type function: string, can be: 'moffat','gauss' or 'both'; 
                        Default 'both'
                        
        :param display: Flag to turn on displaying the fitting to ds9
        :type display: Python boolean (True/False)
                       Default: True
        
        :param qa: flag to use a grid of sub-windows for detecting the sources
                   in the image frames, rather than the entire frame all at
                   once.
        :type qa: Python boolean (True/False)
                  default: True
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to 
                        screen. OR the message level as a string (i.e.,
                        'critical', 'status', 'fullinfo'...)
        """
        #@@FIXME: Detecting sources is done here as well. This should
        # eventually be split up into separate primitives, i.e. detectSources
        # and measureIQ.
        
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "measureIQ", "starting"))
        # Initialize the list of output AstroData objects
        adoutput_list = []
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs(style="AD"):
            # Check whether the measureIQ primitive has been run previously
            if ad.phu_get_key_value("MEASREIQ"):
                log.warning("%s has already been processed by measureIQ" \
                            % (ad.filename))
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue
            # Call the measure_iq user level function
            ad = qa.measure_iq(adinput=ad, function=rc["function"],
                               display=rc["display"], qa=rc["qa"])
            # Append the output AstroData object (which is currently in the
            # form of a list) to the list of output AstroData objects
            adoutput_list.append(ad[0])
        # Report the list of output AstroData objects to the reduction
        # context
        rc.report_output(adoutput_list)
        
        yield rc
    
    def nonlinearityCorrect(self, rc):
        """
        This primitive corrects the input for non-linearity
        """
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "nonlinearityCorrect",
                                "starting"))
        # Initialize the list of output AstroData objects
        adoutput_list = []
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs(style="AD"):
            # Check whether the nonlinearityCorrect primitive has been run
            # previously
            if ad.phu_get_key_value("LINCOR"):
                log.warning("%s has already been processed by " \
                            "nonlinearityCorrect" % (ad.filename))
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue
            # Call the nonlinearity_correct user level function
            ad = pp.nonlinearity_correct(adinput=ad)
            # Append the output AstroData object (which is currently in the
            # form of a list) to the list of output AstroData objects
            adoutput_list.append(ad[0])
        # Report the list of output AstroData objects to the reduction
        # context
        rc.report_output(adoutput_list)
        
        yield rc
    
    def pause(self, rc):
        rc.request_pause()
        yield rc
    
    def scaleFringeToScience(self, rc):
        """
        This primitive will scale the fringes to their matching science data
        in the inputs.
        The primitive getProcessedFringe must have been run prior to this in 
        order to find and load the matching fringes into memory.
        
        :param stats_scale: Use statistics to calculate the scale values?
        :type stats_scale: Python boolean (True/False)
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to 
                        screen. OR the message level as a string (i.e.,
                        'critical', 'status', 'fullinfo'...)
        """
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "scaleFringeToScience",
                                 "starting"))
        # Initialize the list of output AstroData objects
        adoutput_list = []
        # Get the fringes. Make this better.
        inputs = rc.get_inputs(style="AD", category="main")
        fringes = []
        for input in inputs:
            fringes.append(AstroData(rc.get_cal(input, "fringe")))
        # Loop over each input AstroData object in the input list
        count = 0
        for ad in rc.get_inputs(style="AD"):
            fringe = fringes[count]
            # Check whether the scaleFringeToScience primitive has been run
            # previously
            if ad.phu_get_key_value("SCALEFRG"):
                log.warning("%s has already been processed by " \
                            "scaleFringeToScience" % (ad.filename))
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue
            # Call the scale_fringe_to_science user level function
            ad = pp.scale_fringe_to_science(adinput=fringe, science=ad,
                                            stats_scale=rc["stats_scale"])
            # Append the output AstroData object (which is currently in the
            # form of a list) to the list of output AstroData objects
            adoutput_list.append(ad[0])
            count += 1
        # Report the list of output AstroData objects and the scaled fringe
        # frames to the reduction context
        rc.report_output(adoutput_list, category="fringe")
        rc.report_output(rc.get_inputs(style="AD"), category="main")
        
        yield rc
    
    def setContext(self, rc):
        rc.update(rc.localparms)
        yield rc
    
    def showCals(self, rc):
        """
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to 
                        screen. OR the message level as a string (i.e.,
                        'critical', 'status', 'fullinfo'...)
        """
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        if str(rc["showcals"]).lower() == "all":
            num = 0
            # print "pG256: showcals=all", repr (rc.calibrations)
            for calkey in rc.calibrations:
                num += 1
                log.fullinfo(rc.calibrations[calkey], category="calibrations")
            if (num == 0):
                log.warning("There are no calibrations in the cache.")
        else:
            for adr in rc.inputs:
                sid = IDFactory.generate_astro_data_id(adr.ad)
                num = 0
                for calkey in rc.calibrations:
                    if sid in calkey :
                        num += 1
                        log.fullinfo(rc.calibrations[calkey], 
                                     category="calibrations")
            if (num == 0):
                log.warning("There are no calibrations in the cache.")
        
        yield rc
    ptusage_showCals="Used to show calibrations currently in cache for inputs."
    
    def showInputs(self, rc):
        """
        A simple primitive to show the filenames for the current inputs to 
        this primitive.
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to 
                        screen. OR the message level as a string (i.e.,
                        'critical', 'status', 'fullinfo'...)
        """
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        log.fullinfo("Inputs:", category="inputs")
        for inf in rc.inputs:
            log.fullinfo("  %s" % inf.filename, category="inputs")
        
        yield rc
    showFiles = showInputs
    
    def showList(self, rc):
        """
        This primitive will log the list of files in the stacking list matching
        the current inputs and 'purpose' value.
        
        :param purpose: 
        :type purpose: string, either: '' for regular image stacking, 
                       or 'fringe' for fringe stacking.
                       
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to 
                        screen. OR the message level as a string (i.e.,
                        'critical', 'status', 'fullinfo'...)
        """
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        sidset = set()
        purpose = rc["purpose"]
        if purpose is None:
            purpose = ""
        # print "pG710"
        if purpose == "all":
            allsids = rc.get_stack_ids()
            # print "pG713:", repr(allsids)
            for sid in allsids:
                sidset.add(sid)
        else:
            for inp in rc.inputs:
                sidset.add(purpose+IDFactory.generate_stackable_id(inp.ad))
        for sid in sidset:
            stacklist = rc.get_stack(sid) #.filelist
            log.status("List for stack id=%s" % sid, category="list")
            if len(stacklist) > 0:
                for f in stacklist:
                    log.status("    %s" % os.path.basename(f), category="list")
            else:
                log.status("no datasets in list", category="list")
        
        yield rc
    
    def showParameters(self, rc):
        """
        A simple primitive to log the currently set parameters in the 
        reduction context dictionary.
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to 
                        screen. OR the message level as a string (i.e.,
                        'critical', 'status', 'fullinfo'...)
        """
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        rcparams = rc.param_names()
        if (rc["show"]):
            toshows = rc["show"].split(":")
            for toshow in toshows:
                if toshow in rcparams:
                    log.fullinfo("%s = %s" % (toshow, repr(rc[toshow])),
                                 category="parameters")
                else:
                    log.fullinfo("%s is not set" % (toshow),
                                 category="parameters")
        else:
            for param in rcparams:
                log.fullinfo("%s = %s" % (param, repr(rc[param])),
                             category="parameters")
        # print "all",repr(rc.parm_dict_by_tag("showParams", "all"))
        # print "iraf",repr(rc.parm_dict_by_tag("showParams", "iraf"))
        # print "test",repr(rc.parm_dict_by_tag("showParams", "test"))
        # print "sdf",repr(rc.parm_dict_by_tag("showParams", "sdf"))
        # print repr(dir(rc.ro.primDict[rc.ro.curPrimType][0]))
        
        yield rc
    
    def sleep(self, rc):
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        if rc["duration"]:
            dur = float(rc["duration"])
        else:
            dur = 5.
        log.status("Sleeping for %f seconds" % dur)
        time.sleep(dur)
        
        yield rc
    
    def stackFrames(self, rc):
        """
        This primitive will stack each science extension in the input dataset.
        New variance extensions are created from the stacked science extensions
        and the data quality extensions are propagated through to the final
        file.
        
        :param method: type of combining method to use. The options are
                       'average' or 'median'.
        :type method: string
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to 
                        screen. OR the message level as a string (i.e.,
                        'critical', 'status', 'fullinfo'...)
        """
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "stackFrames", "starting"))
        # Call the stack_frames user level function
        adoutput = sk.stack_frames(adinput=rc.get_inputs(style="AD"),
                                   suffix=rc["suffix"],
                                   operation=rc["operation"])
        # Report the list containing a single AstroData object to the reduction
        # context
        rc.report_output(adoutput)
        
        yield rc
    
    def storeProcessedBias(self, rc):
        """
        This should be a primitive that interacts with the calibration system 
        (MAYBE) but that isn't up and running yet. Thus, this will just strip 
        the extra postfixes to create the 'final' name for the 
        makeProcessedBias outputs and write them to disk in a storedcals
        folder.
        
        :param clob: Write over any previous file with the same name that
                     all ready exists?
        :type clob: Python boolean (True/False)
                    default: False
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to 
                        screen. OR the message level as a string (i.e.,
                        'critical', 'status', 'fullinfo'...)
        """
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "storeProcessedBias",
                                 "starting"))
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs(style="AD"):
            # Updating the file name with the suffix for this primitive and
            # then report the new file to the reduction context
            ad.filename = gt.fileNameUpdater(adIn=ad, suffix="_bias",
                                             strip=True)
            log.status("File name of stored bias is %s" % ad.filename)
            # Adding a GBIAS time stamp to the PHU
            ad.history_mark(key="GBIAS", comment="fake key to trick CL that " \
                            "GBIAS was used to create this bias")
            # Write the bias frame to disk
            ad.write(os.path.join(rc["storedbiases"], ad.filename), 
                     clobber=rc["clob"])
            log.fullinfo("Bias written to %s" % (rc["storedbiases"]))
        
        yield rc
    
    def storeProcessedDark(self, rc):
        """
        This should be a primitive that interacts with the calibration system 
        (MAYBE) but that isn't up and running yet. Thus, this will just strip 
        the extra postfixes to create the 'final' name for the 
        makeProcessedDark outputs and write them to disk in a storedcals
        folder.
        
        :param clob: Write over any previous file with the same name that
                     all ready exists?
        :type clob: Python boolean (True/False)
                    default: False
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to 
                        screen. OR the message level as a string (i.e.,
                        'critical', 'status', 'fullinfo'...)
        """
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "storeProcessedDark",
                                 "starting"))
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs(style="AD"):
            # Updating the file name with the suffix for this primitive and
            # then report the new file to the reduction context
            ad.filename = gt.fileNameUpdater(adIn=ad, suffix="_dark",
                                             strip=True)
            log.status("File name of stored dark is %s" % ad.filename)
            # Write the dark frame to disk
            ad.write(os.path.join(rc["storeddarks"], ad.filename), 
                     clobber=rc["clob"])
            log.fullinfo("Dark written to %s" % (rc["storeddarks"]))
        
        yield rc
    
    def storeProcessedFlat(self, rc):
        """
        This should be a primitive that interacts with the calibration 
        system (MAYBE) but that isn't up and running yet. Thus, this will 
        just strip the extra postfixes to create the 'final' name for the 
        makeProcessedFlat outputs and write them to disk in a storedcals
        folder.
        
        :param clob: Write over any previous file with the same name that
                     all ready exists?
        :type clob: Python boolean (True/False)
                    default: False
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to 
                        screen. OR the message level as a string (i.e.,
                        'critical', 'status', 'fullinfo'...)
        """
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "storeProcessedFlat",
                                 "starting"))
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs(style="AD"):
            # Updating the file name with the suffix for this primitive and
            # then report the new file to the reduction context
            ad.filename = gt.fileNameUpdater(adIn=ad, suffix="_flat",
                                             strip=True)
            log.status("File name of stored flat is %s" % ad.filename)
            # Write the flat frame to disk
            ad.write(os.path.join(rc["storedflats"], ad.filename), 
                     clobber=rc["clob"])
            log.fullinfo("Flat written to %s" % (rc["storedflats"])),
        
        yield rc
    
    def subtractDark(self, rc):
        """
        This primitive will subtract each SCI extension of the inputs by those
        of the corresponding dark. If the inputs contain VAR or DQ frames,
        those will also be updated accordingly due to the subtraction on the 
        data.
        
        This is all conducted in pure Python through the arith 'toolbox' of 
        astrodata. 
        
        It is currently assumed that the same dark file will be applied to all
        input images.
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to 
                        screen. OR the message level as a string (i.e.,
                        'critical', 'status', 'fullinfo'...)
        """
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "subtractDark", "starting"))
        # Initialize the list of output AstroData objects
        adoutput_list = []
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs(style="AD"):
            # Check whether the subtractDark primitive has been run previously
            if ad.phu_get_key_value("SUBDARK"):
                log.warning("%s has already been processed by " \
                            "subtractDark" % (ad.filename))
                # Append the input AstroData object to the list of output
                # AstroData objects without further processing
                adoutput_list.append(ad)
                continue
            # Get the appropriate dark for this AstroData object
            dark = AstroData(rc.get_cal(ad, "dark"))
            # Call the subtract_dark user level function
            ad = pp.subtract_dark(adinput=ad, dark=dark)
            # Append the output AstroData object (which is currently in the
            # form of a list) to the list of output AstroData objects
            adoutput_list.append(ad[0])
        # Report the list of output AstroData objects to the reduction
        # context
        rc.report_output(adoutput_list)
        
        yield rc
    
    def subtractFringe(self, rc):
        """
        This primitive will subtract each SCI extension of the inputs by those
        of the corresponding fringe. If the inputs contain VAR or DQ frames,
        those will also be updated accordingly due to the subtraction on the 
        data.
        
        This is all conducted in pure Python through the arith 'toolbox' of 
        astrodata. 
        
        It is currently assumed that the same fringe file will be applied to
        all input images.
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to 
                        screen. OR the message level as a string (i.e.,
                        'critical', 'status', 'fullinfo'...)
        """
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "subtractFringe", "starting"))
        # Retrieving the appropriate fringe for the first of the inputs
        adOne = rc.get_inputs(style="AD")[0]
        #fringe=rc.get_inputs(style="AD", category="fringe")
        ###################BULL CRAP FOR TESTING ######################### 
        from copy import deepcopy
        fringe = deepcopy(adOne)
        fringe.filename = "TEMPNAMEforFRINGE.fits"
        fringe.phu_set_key_value("ORIGNAME","TEMPNAMEforFRINGE.fits")
        ##################################################################
        # Call the subtract_fringe user level function
        output = pp.subtract_fringe(adinput=rc.get_inputs(style="AD"),
                                    fringe=fringe) 
        # Report the output of the user level function to the reduction
        # context
        rc.report_output(output, category="standard")
        
        yield rc
    
    def time(self, rc):
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        cur = datetime.now()
        elap = ""
        if rc["lastTime"] and not rc["start"]:
            td = cur - rc["lastTime"]
            elap = " (%s)" % str(td)
        log.fullinfo("Time: %s %s" % (str(datetime.now()), elap))
        rc.update({"lastTime":cur})
        
        yield rc
    
    def writeOutputs(self, rc):
        """
        A primitive that may be called by a recipe at any stage to
        write the outputs to disk.
        If suffix is set during the call to writeOutputs, any previous 
        suffixs will be striped and replaced by the one provided.
        examples: 
        writeOutputs(suffix= '_string'), writeOutputs(prefix= '_string') 
        or if you have a full file name in mind for a SINGLE file being 
        ran through Reduce you may use writeOutputs(outfilename='name.fits').
        
        :param strip: Strip the previously suffixed strings off file name?
        :type strip: Python boolean (True/False)
                     default: False
        
        :param clobber: Write over any previous file with the same name that
                        all ready exists?
        :type clobber: Python boolean (True/False)
                       default: False
        
        :param suffix: Value to be post pended onto each input name(s) to 
                       create the output name(s).
        :type suffix: string
        
        :param prefix: Value to be post pended onto each input name(s) to 
                         create the output name(s).
        :type prefix: string
        
        :param outfilename: The full filename you wish the file to be written
                            to. Note: this only works if there is ONLY ONE file
                            in the inputs.
        :type outfilename: string
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to 
                        screen. OR the message level as a string (i.e.,
                        'critical', 'status', 'fullinfo'...)
        """
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        # Logging current values of suffix and prefix
        log.status("suffix = %s" % str(rc["suffix"]))
        log.status("prefix = %s" % str(rc["prefix"]))
        log.status("strip = %s" % str(rc["strip"]))
        
        if rc["suffix"] and rc["prefix"]:
            log.critical("The input will have %s pre pended and %s post " +
                         "pended onto it" % (rc["prefix"], rc["suffix"]))
        for ad in rc.get_inputs(style="AD"):
            # If the value of "suffix" was set, then set the file name 
            # to be written to disk to be postpended by it
            if rc["suffix"]:
                log.debug("calling gt.fileNameUpdater on %s" % ad.filename)
                ad.filename = gt.fileNameUpdater(adIn=ad,
                                                 suffix=rc["suffix"],
                                                 strip=rc["strip"])
                log.status("File name updated to %s" % ad.filename)
                outfilename = os.path.basename(ad.filename)
            # If the value of "prefix" was set, then set the file name 
            # to be written to disk to be pre pended by it
            if rc["prefix"]:
                infilename = os.path.basename(ad.filename)
                outfilename = "%s%s" % (rc["prefix"], infilename)
            # If the "outfilename" was set, set the file name of the file 
            # file to be written to this
            elif rc["outfilename"]:
                # Check that there is not more than one file to be written
                # to this file name, if so throw exception
                if len(rc.get_inputs(style="AD")) > 1:
                    message = """
                        More than one file was requested to be written to
                        the same name %s""" % (rc["outfilename"])
                    log.critical(message)
                    raise Errors.PrimitiveError(message)
                else:
                    outfilename = rc["outfilename"]
            # If no changes to file names are requested then write inputs
            # to their current file names
            else:
                outfilename = os.path.basename(ad.filename) 
                log.status("not changing the file name to be written " +
                "from its current name") 
            # Finally, write the file to the name that was decided 
            # upon above
            log.status("writing to file %s" % outfilename)
            # AstroData checks if the output exists and raises an exception
            ad.write(filename=outfilename, clobber=rc["clobber"])
        
        yield rc
