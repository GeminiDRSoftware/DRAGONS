# Author: Kyle Mede. 2010
# Skeleton originally written by Craig Allen, callen@gemini.edu
import os, sys
from datetime import datetime
import shutil
import time

from astrodata import Errors
from astrodata import IDFactory
from astrodata.adutils import gemLog
from gempy import geminiTools as gt
from gempy.science import calibrate as cal
from gempy.science import geminiScience as gs
from gempy.science import preprocessing as pp
from primitives_GENERAL import GENERALPrimitives

class GEMINIPrimitives(GENERALPrimitives):
    """
    This is the class containing all of the primitives for the GEMINI level of
    the type hierarchy tree. It inherits all the primitives from the level
    above, 'GENERALPrimitives'.
    """
    astrotype = "GEMINI"
    
    def init(self, rc):
        return 
    init.pt_hide = True
    
    def addDQ(self, rc):
        """
        This primitive will create a numpy array for the data quality 
        of each SCI frame of the input data. This will then have a 
        header created and append to the input using AstroData as a DQ 
        frame. The value of a pixel will be the sum of the following: 
        (0=good, 1=bad pixel (found in bad pixel mask), 
        2=value is non linear, 4=pixel is saturated)
        
        
        :param suffix: Value to be post pended onto each input name(s) to 
                       create the output name(s).
        :type suffix: string
        
        :param fl_nonlinear: Flag to turn checking for nonlinear pixels on/off
        :type fl_nonLinear: Python boolean (True/False), default is True
        
        :param fl_saturated: Flag to turn checking for saturated pixels on/off
        :type fl_saturated: Python boolean (True/False), default is True
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to 
                        screen. OR the message level as a string (i.e.,
                        'critical', 'status', 'fullinfo'...)
        """
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        # Log the standard "starting primitive" debug message
        log.debug(gt.logMessage("primitive", "addDQ", "starting"))
        try:
            # Call the addBPM primitive
            rc.run("addBPM")
            # Call the add_dq user level function
            output = gs.add_dq(adInputs=rc.getInputs(style="AD"),
                               fl_nonlinear=rc["fl_nonlinear"],
                               fl_saturated=rc["fl_saturated"],
                               suffix=rc["suffix"])
            # Report the output of the user level function to the reduction
            # context
            rc.reportOutput(output)
        except:
            # Log the message from the exception
            log.critical(repr(sys.exc_info()[1]))
            raise
        
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
        try:
            # Perform an update to the stack cache file (or create it) using 
            # the current inputs in the reduction context
            purpose = rc["purpose"]
            if purpose is None:
                purpose = ""
            # Call the rqStackUpdate method
            rc.rqStackUpdate(purpose=purpose)
            # Write the files in the stack to disk if they do not already exist
            for ad in rc.getInputs(style="AD"):
                if not os.path.exists(ad.filename):
                    log.fullinfo("writing %s to disk" % ad.filename,
                                 category="list")
                    ad.write(ad.filename)
        except:
            # Log the message from the exception
            log.critical(repr(sys.exc_info()[1]))
            raise
        
        yield rc
    
    def addVAR(self, rc):
        """
        This primitive uses numpy to calculate the variance of each SCI frame
        in the input files and appends it as a VAR frame using AstroData.
        
        The calculation will follow the formula:
        variance = (read noise/gain)2 + max(data,0.0)/gain
        
        :param suffix: Value to be post pended onto each input name(s) to 
                       create the output name(s).
        :type suffix: string
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to 
                        screen. OR the message level as a string (i.e.,
                        'critical', 'status', 'fullinfo'...)
        """
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        # Log the standard "starting primitive" debug message
        log.debug(gt.logMessage("primitive", "addVAR", "starting"))
        try:
            # Call the add_var user level function
            output = gs.add_var(adInputs=rc.getInputs(style="AD"),
                                   suffix=rc["suffix"])
            # Report the output of the user level function to the reduction
            # context
            rc.reportOutput(output)
        except:
            # Log the message from the exception
            log.critical(repr(sys.exc_info()[1]))
            raise
        
        yield rc 
    
    def aduToElectrons(self, rc):
        """
        This primitive will convert the inputs from having pixel 
        units of ADU to electrons.
        
        :param suffix: Value to be post pended onto each input name(s) to 
                         create the output name(s).
        :type suffix: string
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to 
                        screen. OR the message level as a string (i.e.,
                        'critical', 'status', 'fullinfo'...)
        """
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        # Log the standard "starting primitive" debug message
        log.debug(gt.logMessage("primitive", "aduToElectrons", "starting"))
        try:
            # Call the adu_to_electrons user level function
            output = gs.adu_to_electrons(adInputs=rc.getInputs(style="AD"),
                                         suffix=rc["suffix"])
            # Report the output of the user level function to the reduction
            # context
            rc.reportOutput(output)
        except:
            # Log the message from the exception
            log.critical(repr(sys.exc_info()[1]))
            raise
        
        yield rc
    
    def crashReduce(self, rc):
        raise "Crashing"
        yield rc
    
    def clearCalCache(self, rc):
        # print "pG61:", rc.calindfile
        rc.persistCalIndex(rc.calindfile, newindex={})
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
    
    def display(self, rc):
        """
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to 
                        screen. OR the message level as a string (i.e.,
                        'critical', 'status', 'fullinfo'...)
        """
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        try:
            rc.rqDisplay(displayID=rc["displayID"])
        except:
            # Log the message from the exception
            log.critical(repr(sys.exc_info()[1]))
            raise
        
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
        
        :param suffix: Value to be post pended onto each input name(s) to 
                         create the output name(s).
        :type suffix: string
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to 
                        screen. OR the message level as a string (i.e.,
                        'critical', 'status', 'fullinfo'...)
        """
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        # Log the standard "starting primitive" debug message
        log.debug(gt.logMessage("primitive", "divideByFlat", "starting"))
        try:
            # Retrieving the appropriate flat for the first of the inputs
            adOne = rc.getInputs(style="AD")[0]
            #processedFlat = AstroData(rc.getCal(adOne,"flat"))
            ###################BULL CRAP FOR TESTING ######################### 
            from copy import deepcopy
            processedFlat = deepcopy(adOne)
            processedFlat.filename = "TEMPNAMEforFLAT.fits"
            processedFlat.phuSetKeyValue("ORIGNAME","TEMPNAMEforFLAT.fits")
            ###################################################################
            
            # Taking care of the case where there was no, or an invalid flat 
            if processedFlat.countExts("SCI") == 0:
                raise Errors.PrimitiveError("Invalid processed flat " +
                                            "retrieved")
            # Call the divide_by_flat user level function
            output = cal.divide_by_flat(adInputs=rc.getInputs(style="AD"),
                                        flats=processedFlat,
                                        suffix=rc["suffix"])
            # Report the output of the user level function to the reduction
            # context
            rc.reportOutput(output)
        except:
            # Log the message from the exception
            log.critical(repr(sys.exc_info()[1]))
            raise
        
        yield rc
     
    def getCal(self, rc):
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        
        caltype = rc["caltype"]
        if caltype is None:
            log.critical("Requested a calibration no particular " +
                         "calibration type")
            raise Errors.PrimitiveError("getCal: %s was None" % caltype)
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
        
        inps = rc.getInputsAsAstroData()
        
        if localSource:
            rc.rqCal(caltype, inps, source="local")
            for ad in inps:
                cal = rc.getCal(ad, caltype)
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
        try:
            for inp in rc.inputs:
                sidset.add(purpose+IDFactory.generateStackableID(inp.ad))
            for sid in sidset:
                stacklist = rc.getStack(sid) #.filelist
                log.fullinfo("List for stack id=%s" % sid, category="list")
                for f in stacklist:
                    rc.reportOutput(f)
                    log.fullinfo("   %s" % os.path.basename(f),
                                 category="list")
        except:
            # Log the message from the exception
            log.critical(repr(sys.exc_info()[1]))
            raise
        
        yield rc
    
    def getProcessedBias(self, rc):
        """
        This primitive will check the files in the lists that are on disk,
        and then update the inputs list to include all members of the list.
        """
        rc.rqCal("bias", rc.getInputs(style="AD"))
        yield rc
    
    def getProcessedDark(self, rc):
        """
        A primitive to search and return the appropriate calibration dark from
        a server for the given inputs.
        """
        rc.rqCal("dark", rc.getInputs(style="AD"))
        yield rc
    
    def getProcessedFlat(self, rc):
        """
        A primitive to search and return the appropriate calibration flat from
        a server for the given inputs.
        
        """
        rc.rqCal("flat", rc.getInputs(style="AD"))
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
        #@@FIXME: Detecting sources is done here as well. This 
        # should eventually be split up into
        # separate primitives, i.e. detectSources and measureIQ.
        
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        # Log the standard "starting primitive" debug message
        log.debug(gt.logMessage("primitive", "measureIQ", "starting"))
        try:
            # Call the measure_iq user level function
            output = gs.measure_iq(adInputs=rc.getInputs(style="AD"),
                                   function=rc["function"],
                                   display=rc["display"],
                                   qa=rc["qa"])
            # Report the output of the user level function to the reduction
            # context
            rc.reportOutput(output)
        except:
            # Log the message from the exception
            log.critical(repr(sys.exc_info()[1]))
            raise
        
        yield rc
    
    def nonlinearityCorrect(self, rc):
        """
        This primitive corrects the input for non-linearity
        """
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        # Log the standard "starting primitive" debug message
        log.debug(gt.logMessage("primitive", "nonlinearityCorrect",
                                "starting"))
        try:
            # Call the nonlinearity_correct user level function
            output = pp.nonlinearity_correct(input=rc.getInputs(style="AD"),
                                             suffix=rc["suffix"])
            # Report the output of the user level function to the reduction
            # context
            rc.reportOutput(output)
        except:
            # Log the message from the exception
            log.critical(repr(sys.exc_info()[1]))
            raise
        
        yield rc
    
    def normalizeFlat(self, rc):
        """
        This primitive normalises the input flat
        """
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        # Log the standard "starting primitive" debug message
        log.debug(gt.logMessage("primitive", "normalizeFlat", "starting"))
        try:
            # Call the normalize_flat user level function
            output = cal.normalize_flat_image(
                adInputs=rc.getInputs(style="AD"),
                suffix=rc["suffix"])
            # Report the output of the user level function to the reduction
            # context
            rc.reportOutput(output)
        except:
            # Log the message from the exception
            log.critical(repr(sys.exc_info()[1]))
            raise
        
        yield rc
    
    def pause(self, rc):
        rc.requestPause()
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
                sid = IDFactory.generateAstroDataID(adr.ad)
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
    
    def scaleFringeToScience(self,rc):
        """
        This primitive will scale the fringes to their matching science data
        in the inputs.
        The primitive getProcessedFringe must have been ran prior to this in 
        order to find and load the matching fringes into memory.
        
        :param suffix: Value to be post pended onto each input name(s) to 
                       create the output name(s).
        :type suffix: string
        
        :param statScale: Use statistics to calculate the scale values?
        :type statScale: Python boolean (True/False)
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to 
                        screen. OR the message level as a string (i.e.,
                        'critical', 'status', 'fullinfo'...)
        """
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        # Log the standard "starting primitive" debug message
        log.debug(gt.logMessage("primitive", "scaleFringeToScience",
                                "starting"))
        try:
            inputs = rc.getInputs(style="AD", category="standard")
            fringes = []
            for input in inputs:
                fringes.append(AstroData(rc.getCal(input, "fringe")))
            # Call the scale_fringe_to_science user level function
            output = cal.scale_fringe_to_science(fringes=fringes,
                                                 sciInputs=inputs,
                                                 statScale=rc["statScale"],
                                                 suffix=rc["suffix"])
            # Report the output of the user level function (the scaled fringes)
            # and the original science inputs to the reduction context. 
            rc.reportOutput(output, category="fringe")
            rc.reportOutput(inputs, category="standard")
        except:
            # Log the message from the exception
            log.critical(repr(sys.exc_info()[1]))
            raise
        
        yield rc
    
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
            allsids = rc.getStackIDs()
            # print "pG713:", repr(allsids)
            for sid in allsids:
                sidset.add(sid)
        else:
            for inp in rc.inputs:
                sidset.add(purpose+IDFactory.generateStackableID(inp.ad))
        for sid in sidset:
            stacklist = rc.getStack(sid) #.filelist
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
        rcparams = rc.paramNames()
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
        # print "all",repr(rc.parmDictByTag("showParams", "all"))
        # print "iraf",repr(rc.parmDictByTag("showParams", "iraf"))
        # print "test",repr(rc.parmDictByTag("showParams", "test"))
        # print "sdf",repr(rc.parmDictByTag("showParams", "sdf"))
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
        
        :param suffix: Value to be post pended onto each input name(s) to 
                       create the output name(s).
        :type suffix: string
        
        :param fl_vardq: Create variance and data quality frames?
        :type fl_vardq: Python boolean (True/False), OR string 'AUTO' to do 
                        it automatically if there are VAR and DQ frames in the
                        inputs. NOTE: 'AUTO' uses the first input to determine
                        if VAR and DQ frames exist, so, if the first does, then
                        the rest MUST also have them as well.
        
        :param fl_dqprop: propogate the current DQ values?
        :type fl_dqprop: Python boolean (True/False)
        
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
        log.debug(gt.logMessage("primitive", "stackFrames", "starting"))
        try:
            # Call the stack_frames user level function
            output = gs.stack_frames(adInputs=rc.getInputs(style="AD"),
                                     fl_vardq=rc["fl_vardq"],
                                     fl_dqprop=rc["fl_dqprop"],
                                     method=rc["method"],
                                     suffix=rc["suffix"])
            # Report the output of the user level function to the reduction
            # context
            rc.reportOutput(output)
        except:
            # Log the message from the exception
            log.critical(repr(sys.exc_info()[1]))
            raise
        
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
        try:
            for ad in rc.getInputs(style="AD"):
                # Updating the file name with the suffix for this primitive and
                # then reporting the new file to the reduction context
                log.debug("Calling gt.fileNameUpdater on %s" % ad.filename)
                ad.filename = gt.fileNameUpdater(adIn=ad,
                                                 suffix="_bias",
                                                 strip=True)
                log.status("File name updated to %s" % ad.filename)
                
                # Adding a GBIAS time stamp to the PHU
                ad.historyMark(key="GBIAS",
                               comment="fake key to trick CL that GBIAS " +
                               "was ran")
                log.fullinfo("File written to %s/%s" % (rc["storedbiases"],
                                                        ad.filename))
                ad.write(os.path.join(rc["storedbiases"], ad.filename), 
                         clobber=rc["clob"])
        except:
            # Log the message from the exception
            log.critical(repr(sys.exc_info()[1]))
            raise
        
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
        try:
            for ad in rc.getInputs(style="AD"):
                # Updating the file name with the suffix for this primitive and
                # then reporting the new file to the reduction context
                log.debug("Calling gt.fileNameUpdater on %s" % ad.filename)
                ad.filename = gt.fileNameUpdater(adIn=ad,
                                                 suffix="_flat",
                                                 strip=True)
                log.status("File name updated to %s" % ad.filename)
                log.fullinfo("File written to %s/%s" % (rc["storedflats"],
                                                        ad.filename))
                ad.write(os.path.join(rc["storedflats"], ad.filename),
                         clobber=rc["clob"])
        except:
            # Log the message from the exception
            log.critical(repr(sys.exc_info()[1]))
            raise
        
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
        
        :param suffix: Value to be post pended onto each input name(s) to 
                         create the output name(s).
        :type suffix: string
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to 
                        screen. OR the message level as a string (i.e.,
                        'critical', 'status', 'fullinfo'...)
        """
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        # Log the standard "starting primitive" debug message
        log.debug(gt.logMessage("primitive", "subtractDark", "starting"))
        try:
            # Retrieving the appropriate dark for the first of the inputs
            adOne = rc.getInputs(style="AD")[0]
            #processedDark = AstroData(rc.getCal(adOne,"dark"))
            ###################BULL CRAP FOR TESTING ######################### 
            from copy import deepcopy
            processedDark = deepcopy(adOne)
            processedDark.filename = "TEMPNAMEforDARK.fits"
            processedDark.phuSetKeyValue("ORIGNAME","TEMPNAMEforDARK.fits")
            ###################################################################
            # Taking care of the case where there was no, or an invalid flat 
            if processedDark.countExts("SCI") == 0:
                raise Errors.PrimitiveError("Invalid processed dark " +
                                            "retrieved")
            # Call the subtract_dark user level function
            output = cal.subtract_dark(adInputs=rc.getInputs(style="AD"),
                                       darks=processedDark,
                                       suffix=rc["suffix"])
            # Report the output of the user level function to the reduction
            # context
            rc.reportOutput(output)
        except:
            # Log the message from the exception
            log.critical(repr(sys.exc_info()[1]))
            raise
        
        yield rc
    
    def subtractFringe(self,rc):
        """
        This primitive will subtract each SCI extension of the inputs by those
        of the corresponding fringe. If the inputs contain VAR or DQ frames,
        those will also be updated accordingly due to the subtraction on the 
        data.
        
        This is all conducted in pure Python through the arith 'toolbox' of 
        astrodata. 
        
        It is currently assumed that the same fringe file will be applied to
        all input images.
        
        :param suffix: Value to be post pended onto each input name(s) to 
                         create the output name(s).
        :type suffix: string
        
        :param logLevel: Verbosity setting for log messages to the screen.
        :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to 
                        screen. OR the message level as a string (i.e.,
                        'critical', 'status', 'fullinfo'...)
        """
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        # Log the standard "starting primitive" debug message
        log.debug(gt.logMessage("primitive", "subtractFringe", "starting"))
        try:
            # Retrieving the appropriate fringe for the first of the inputs
            adOne = rc.getInputs(style="AD")[0]
            #fringes=rc.getInputs(style="AD", category="fringe")
            ###################BULL CRAP FOR TESTING ######################### 
            from copy import deepcopy
            fringes = deepcopy(adOne)
            fringes.filename = "TEMPNAMEforFRINGE.fits"
            fringes.phuSetKeyValue("ORIGNAME","TEMPNAMEforFRINGE.fits")
            ##################################################################
            # Call the subtract_fringe user level function
            output = cal.subtract_fringe(adInputs=rc.getInputs(style="AD"),
                                         fringes=fringes, 
                                         suffix=rc["suffix"])
            # Report the output of the user level function to the reduction
            # context
            rc.reportOutput(output, category="standard")
        except:
            # Log the message from the exception
            log.critical(repr(sys.exc_info()[1]))
            raise
        
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
        try:
            # Logging current values of suffix and prefix
            log.status("suffix = %s" % str(rc["suffix"]))
            log.status("prefix = %s" % str(rc["prefix"]))
            log.status("strip = %s" % str(rc["strip"]))
            
            if rc["suffix"] and rc["prefix"]:
                log.critical("The input will have %s pre pended and %s post " +
                             "pended onto it" % (rc["prefix"], rc["suffix"]))
            for ad in rc.getInputs(style="AD"):
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
                    if len(rc.getInputs(style="AD")) > 1:
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
        except:
            # Log the message from the exception
            log.critical(repr(sys.exc_info()[1]))
            raise
        
        yield rc
