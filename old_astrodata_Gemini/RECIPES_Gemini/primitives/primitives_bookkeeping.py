import os
import time
import shutil

from datetime import datetime

from astrodata.utils import Errors
from astrodata.utils import logutils
from gempy.gemini import gemini_tools as gt
from recipe_system.reduction import IDFactory

from primitives_GENERAL import GENERALPrimitives

class BookkeepingPrimitives(GENERALPrimitives):
    """
    This is the class containing all of the bookkeeping primitives
    for the GEMINI level of the type hierarchy tree. It inherits all
    the primitives from the level above, 'GENERALPrimitives'.
    """
    astrotype = "GEMINI"
    
    def init(self, rc):
        GENERALPrimitives.init(self, rc)
        return rc
    init.pt_hide = True
    
    def addToList(self, rc):
        """
        This primitive will update the lists of files to be stacked
        that have the same observationID with the current inputs.
        This file is cached between calls to reduce, thus allowing
        for one-file-at-a-time processing.
        
        :param purpose: 
        :type purpose: string
        
        """
        # Instantiate the log
        log = logutils.get_logger(__name__)
        
        # Perform an update to the stack cache file (or create it) using the
        # current inputs in the reduction context
        purpose = rc["purpose"]
        if purpose is None:
            purpose = ""
        if purpose == "":
            suffix = "_list"
        else:
            suffix = "_%s" % purpose
        
        # Update file names and write the files to disk to ensure the right
        # version is stored before adding it to the list.
        adoutput = []
        for ad in rc.get_inputs_as_astrodata():
            ad.filename = gt.filename_updater(adinput=ad, suffix=suffix,
                                              strip=True)
            log.stdinfo("Writing %s to disk" % ad.filename)
            ad.write(clobber=True)
            adoutput.append(ad)
        
        rc.report_output(adoutput)
        
        # Call the rq_stack_update method
        rc.rq_stack_update(purpose=purpose)
        
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
    
    def contextReport(self, rc):
        # Instantiate the log
        log = logutils.get_logger(__name__)

        log.fullinfo(rc.report(report_history=rc["report_history"],
                               internal_dict=rc["internal_dict"],
                               context_vars=rc["context_vars"],
                               report_inputs=rc["report_inputs"],
                               report_parameters=rc["report_parameters"],
                               showall=rc["showall"]))
        
        yield rc
    
    def crashReduce(self, rc):
        raise "Crashing"
        yield rc
    
    def getList(self, rc):
        """
        This primitive will check the files in the stack lists are on disk,
        and then update the inputs list to include all members of the stack 
        for stacking.
        
        :param purpose: 
        :type purpose: string
        """
        
        # Instantiate the log
        log = logutils.get_logger(__name__)
        
        # Get purpose of list
        sidset = set()
        purpose = rc["purpose"]
        if purpose is None:
            purpose = ""
        max_frames = rc["max_frames"]
        
        # Get ID for all inputs
        for inp in rc.inputs:
            sidset.add(purpose+IDFactory.generate_stackable_id(inp.ad))
        
        # Import inputs from all lists
        for sid in sidset:
            stacklist = rc.get_list(sid) #.filelist
            log.stdinfo("List for stack id %s(...):" % sid[0:35])
            # Limit length of stacklist
            if len(stacklist)>max_frames and max_frames is not None:
                stacklist = sorted(stacklist)[-max_frames:]
            for f in stacklist:
                rc.report_output(f, stream=rc["to_stream"])
                log.stdinfo("   %s" % os.path.basename(f))
        
        yield rc
    
    def pause(self, rc):
        rc.request_pause()
        yield rc
    
    def setContext(self, rc):
        rc.update(rc.localparms)
        yield rc
    
    def showInputs(self, rc):
        """
        A simple primitive to show the filenames for the current inputs to 
        this primitive.
        """
        
        # Instantiate the log
        log = logutils.get_logger(__name__)
        log.stdinfo("Inputs:")
        #print "pG977:", id(rc), repr(rc.inputs)
        #if "stream" in rc:
        #    stream = rc["stream"]
        #else:
        #    stream = "main"
        
        log.stdinfo("stream: %s" % (rc._current_stream))
        for inf in rc.inputs:
            log.stdinfo("  %s" % inf.filename)
        
        yield rc
    showFiles = showInputs
    
    def showList(self, rc):
        """
        This primitive will log the list of files in the stacking list matching
        the current inputs and 'purpose' value.
        
        :param purpose: 
        :type purpose: string
        """
        
        # Instantiate the log
        log = logutils.get_logger(__name__)
        
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
            stacklist = rc.get_list(sid) #.filelist
            log.status("List for stack id=%s" % sid)
            if len(stacklist) > 0:
                for f in stacklist:
                    log.status("   %s" % os.path.basename(f))
            else:
                log.status("No datasets in list")
        
        yield rc
    
    def showParameters(self, rc):
        """
        A simple primitive to log the currently set parameters in the 
        reduction context dictionary.
        """
        
        # Instantiate the log
        log = logutils.get_logger(__name__)
        
        rcparams = rc.param_names()
        if (rc["show"]):
            toshows = rc["show"].split(":")
            for toshow in toshows:
                if toshow in rcparams:
                    log.stdinfo("%s = %s" % (toshow, repr(rc[toshow])))
                else:
                    log.stdinfo("%s is not set" % (toshow))
        else:
            for param in rcparams:
                log.stdinfo("%s = %s" % (param, repr(rc[param])))
        # print "all",repr(rc.parm_dict_by_tag("showParams", "all"))
        # print "iraf",repr(rc.parm_dict_by_tag("showParams", "iraf"))
        # print "test",repr(rc.parm_dict_by_tag("showParams", "test"))
        # print "sdf",repr(rc.parm_dict_by_tag("showParams", "sdf"))
        # print repr(dir(rc.ro.primDict[rc.ro.curPrimType][0]))
        
        yield rc
    
    def sleep(self, rc):
        # Instantiate the log
        log = logutils.get_logger(__name__)

        if rc["duration"]:
            dur = float(rc["duration"])
        else:
            dur = 5.
        log.status("Sleeping for %f seconds" % dur)
        time.sleep(dur)
        
        yield rc
    
    def time(self, rc):
        # Instantiate the log
        log = logutils.get_logger(__name__)
        cur = datetime.now()
        elap = ""
        if rc["lastTime"] and not rc["start"]:
            td = cur - rc["lastTime"]
            elap = " (%s)" % str(td)
        log.stdinfo("Time: %s %s" % (str(datetime.now()), elap))
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
        """
        
        # Instantiate the log
        log = logutils.get_logger(__name__)
        # Logging current values of suffix and prefix
        log.fullinfo("suffix = %s" % str(rc["suffix"]))
        log.fullinfo("prefix = %s" % str(rc["prefix"]))
        log.fullinfo("strip = %s" % str(rc["strip"]))
        
        # Initialize the list of output AstroData objects
        adoutput_list = []
        
        for ad in rc.get_inputs_as_astrodata():
            if rc["suffix"] and rc["prefix"]:
                ad.filename = gt.filename_updater(adinput=ad,
                                                  prefix=rc["prefix"],
                                                  suffix=rc["suffix"],
                                                  strip=rc["strip"])
                log.fullinfo("File name updated to %s" % ad.filename)
                outfilename = os.path.basename(ad.filename)
            
            elif rc["suffix"]:
                # If the value of "suffix" was set, then set the file name 
                # to be written to disk to be postpended by it
                ad.filename = gt.filename_updater(adinput=ad,
                                                  suffix=rc["suffix"],
                                                  strip=rc["strip"])
                log.fullinfo("File name updated to %s" % ad.filename)
                outfilename = os.path.basename(ad.filename)
            
            elif rc["prefix"]:
                # If the value of "prefix" was set, then set the file name 
                # to be written to disk to be pre pended by it
                ad.filename = gt.filename_updater(adinput=ad,
                                                  prefix=rc["prefix"],
                                                  strip=rc["strip"])
                log.fullinfo("File name updated to %s" % ad.filename)
                outfilename = os.path.basename(ad.filename)
            
            elif rc["outfilename"]:
                # If the "outfilename" was set, set the file name of the file 
                # file to be written to this
                
                # Check that there is not more than one file to be written
                # to this file name, if so throw exception
                if len(rc.get_inputs_as_astrodata()) > 1:
                    message = """
                        More than one file was requested to be written to
                        the same name %s""" % (rc["outfilename"])
                    log.critical(message)
                    raise Errors.PrimitiveError(message)
                else:
                    outfilename = rc["outfilename"]
            else:
                # If no changes to file names are requested then write inputs
                # to their current file names
                outfilename = os.path.basename(ad.filename) 
                log.fullinfo("not changing the file name to be written " \
                             "from its current name")
            
            # Finally, write the file to the name that was decided 
            # upon above
            log.stdinfo("Writing to file %s" % outfilename)
            
            # AstroData checks if the output exists and raises an exception
            ad.write(filename=outfilename, clobber=rc["clobber"])
            
            # Append the output AstroData object to the list of output
            # AstroData objects
            adoutput_list.append(ad)
        
        # Report the list of output AstroData objects to the
        # reduction context
        rc.report_output(adoutput_list)
        
        yield rc
