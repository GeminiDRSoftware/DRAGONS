import os
import glob as gl
from copy import deepcopy
from inspect import stack
from astrodata import AstroData
from astrodata.utils import Lookups
from astrodata.utils import logutils

from gempy.gemini import gemini_tools as gt

from recipe_system.reduction.reductionObjects import PrimitiveSet

from astrodata_Gemini.ADCONFIG_Gemini.lookups import calurl_dict
from astrodata_Gemini.ADCONFIG_Gemini.lookups import keyword_comments
from astrodata_Gemini.ADCONFIG_Gemini.lookups import timestamp_keywords
from astrodata_Gemini.ADCONFIG_Gemini.lookups.source_detection import sextractor_default_dict

# ------------------------------------------------------------------------------
class GENERALPrimitives(PrimitiveSet):
    """
    This is the class containing all of the primitives for the GENERAL level of
    the type hierarchy tree. It inherits all the primitives from the level
    above, 'PrimitiveSet'.
    """
    astrotype = "GENERAL"
    
    def init(self, rc):
        # Load the timestamp keyword dictionary that will be used to define the
        # keyword to be used for the time stamp for all the primitives and user
        # level function. This only needs to be done once in the highest level
        # primitive due to primitive inheritance.
        self.timestamp_keys = timestamp_keywords.timestamp_keys

        # Also load the standard comments for header keywords that will be
        # updated in the primitives
        self.keyword_comments = keyword_comments.keyword_comments
        self.sx_default_dict = sextractor_default_dict.sextractor_default_dict
        self.calurl_dict = calurl_dict.calurl_dict
        # This lambda will return the name of the current caller.
        self.myself = lambda: stack()[1][3]
        return 
    init.pt_hide = True
    
    def addInputs(self, rc):
        # Instantiate the log
        log = logutils.get_logger(__name__)
        
        if rc["files"] == None:
            glob = "./*.fits"
        else:
            glob = rc["files"]
        log.status("Listing for: "+ glob)
        files = gl.glob(glob)
        files.sort()
        if len(files) == 0:
            log.status("No files")
        else:
            log.status("\t"+"\n\t".join(files))
        yield rc
        add = True # rc["inputs"]
        if add:
            rc.add_input(files)
        
        yield rc
    
    def clearInputs(self, rc):
        rc.clear_input()
        
        yield rc
    
    def copy(self, rc):
        for ad in rc.get_inputs_as_astro_data():
            nd = deepcopy(ad)
            nd.filename = "copy_"+os.path.basename(ad.filename)
            rc.report_output(nd)
        
        yield rc

    def inputInfo(self, rc):
        for ad in rc.get_inputs_as_astro_data():
            ad.info()            
        yield rc


    def listDir(self, rc):
        # Instantiate the log
        log = logutils.get_logger(__name__)

        
        if rc["dir"] == None:
            thedir = "."
        else:
            thedir = rc["dir"]
        log.status("Listing for: "+ thedir)
        files = os.listdir(thedir)
        sfiles = []
        for f in files:
            if f[-5:].lower() == ".fits":
                sfiles.append(f)
        sfiles.sort()
        if len(sfiles) == 0:
            log.status("No FITS files")
        else:
            log.status("\n\t".join(sfiles))
        
        yield rc
    
    def setInputs(self, rc):
        files = rc["files"]
        if files != None:
            a = files.split(" ")
            if len(a)>0:
                rc.add_input(a)
        
        yield rc
    
    def clearStream(self, rc):
        # print repr(rc)
        if "stream" in rc:
            stream = rc["stream"]
        else:
            stream = "main"
        
        rc.get_stream(stream, empty=True)
        
        yield rc
    
    def contextTest(self, rc):
        print rc.context
        yield rc
        print rc.inContext("QA")
        yield rc
    
    def forwardInput(self, rc):
        log = logutils.get_logger(__name__)

        
        if rc["to_stream"] != None:
            stream = rc["to_stream"]
        else:
            stream = "main"
        prefix = rc["prefix"];
        do_deepcopy = rc["deepcopy"]
        if do_deepcopy is None:
            do_deepcopy = True

        if "by_token" in rc:
            bt = rc["by_token"]
            for ar in rc.inputs:
                if bt in ar.filename:
                    rc.report_output(ar.ad, stream = stream)
            #print "pG110:",repr(rc.outputs)
        else:
            inputs = rc.get_inputs_as_astrodata()
            inputs_copy = []
            for ad in inputs:
                if do_deepcopy:
                    ad_copy = deepcopy(ad)
                else:
                    ad_copy = ad
                if prefix:
                    ad_copy.filename = os.path.join(
                                        prefix+os.path.basename(ad.filename))
                else:
                    ad_copy.filename = ad.filename
                inputs_copy.append(ad_copy)

            log.fullinfo("Reporting Output: "+ \
                             ", ".join([ ad.filename for ad in inputs_copy]))
            rc.report_output(inputs_copy, stream = stream, )
        
        yield rc
    forwardStream = forwardInput
    
    def showOutputs(self, rc):
        log = logutils.get_logger(__name__)

        streams = rc.outputs.keys()
        streams.sort()
        streams.remove("main")
        streams.insert(0,"main")
        tstream = rc["streams"]
        
        for stream in streams:
            if tstream == None or stream in tstream:
                log.stdinfo("stream: "+stream)
                if len(rc.outputs[stream])>0:
                    for adr in rc.outputs[stream]:
                        log.stdinfo(str(adr))
                else:
                    log.stdinfo("    empty")
        
        yield rc
    
    def change(self, rc):
        inputs = rc.get_inputs_as_astrodata()
        # print "pG140:", repr(rc.current_stream), repr(rc._nonstandard_stream)
        
        if rc["changeI"] == None:
            rc.update({"changeI":0})
        
        changeI = rc["changeI"]
        ci = "_"+str(changeI)
        
        rc.update({"changeI":changeI+1})
        for ad in inputs:
            ad.filename = gt.filename_updater(adinput=ad, suffix=ci,
                                              strip=False)
            # print "pG152:", ad.filename
        rc.report_output(inputs)
        
        yield rc
    
    def passOutputToInput(self,rc):
        yield rc

    def log(self, rc):
        log = logutils.get_logger(__name__)

        
        msg = rc["msg"]
        if msg == None:
            msg = "..."
        log.fullinfo(msg)
        
        yield rc
        
    def returnFromRecipe(self, rc):
        rc.return_from_recipe()
        
        yield rc

    def add(self,rc):
        # This is a bare-bones primitive interface to the ad add
        # function from the arith module.  The number, dictionary,
        # or AD instance to be added to the input is stored in
        # rc["operand"]

        # Instantiate the log
        log = logutils.get_logger(__name__)

        # the function name, passed to mark_history()
        # primname = __str__().split('.')[-1].replace(">","")

        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "add", "starting"))

        # Define the keyword to be used for the time stamp for this primitive
        timestamp_key = self.timestamp_keys["add"]

        # Initialize the list of output AstroData objects
        adoutput_list = []
        
        # Get data to be added from the RC
        operand = rc["operand"]
        if operand is None:
            log.stdinfo("No operand to add; no changes will be "
                        "made to input")
        elif type(operand)==AstroData:
            log.stdinfo("Adding %s to input" % 
                        (operand.filename))
        else:
            log.stdinfo("Adding %s to input" % 
                        (repr(operand)))
  
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():

            if operand is not None:
                # Add operand to data
                ad.add(operand)

                # Add the appropriate time stamps to the PHU
                gt.mark_history(adinput=ad, primname=self.myself(), keyword=timestamp_key)

                # Change the filename
                ad.filename = gt.filename_updater(
                    adinput=ad, suffix=rc["suffix"], strip=True)

            # Append the output AstroData object to the list
            # of output AstroData objects
            adoutput_list.append(ad)

        # Report the list of output AstroData objects to the reduction
        # context
        rc.report_output(adoutput_list)
        
        yield rc

    def divide(self,rc):
        # This is a bare-bones primitive interface to the ad div
        # function from the arith module.  The value, dictionary,
        # or AD instance to be divided into the input is stored in
        # rc["operand"]

        # Instantiate the log
        log = logutils.get_logger(__name__)


        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "divide", "starting"))

        # Define the keyword to be used for the time stamp for this primitive
        timestamp_key = self.timestamp_keys["divide"]

        # Initialize the list of output AstroData objects
        adoutput_list = []
        
        # Get data to be divided from the RC
        operand = rc["operand"]
        if operand is None:
            log.stdinfo("No operand to divide; no changes will be "\
                        "made to input")
        elif type(operand)==AstroData:
            log.stdinfo("Dividing input by %s" % 
                        (operand.filename))
        else:
            log.stdinfo("Dividing input by %s" % 
                        (repr(operand)))

        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():

            if operand is not None:
                # Divide ad by operand
                ad.div(operand)

                # Add the appropriate time stamps to the PHU
                gt.mark_history(adinput=ad, primname=self.myself(), keyword=timestamp_key)

                # Change the filename
                ad.filename = gt.filename_updater(
                    adinput=ad, suffix=rc["suffix"], strip=True)

            # Append the output AstroData object to the list
            # of output AstroData objects
            adoutput_list.append(ad)

        # Report the list of output AstroData objects to the reduction
        # context
        rc.report_output(adoutput_list)
        
        yield rc

    def multiply(self,rc):
        # This is a bare-bones primitive interface to the ad mult
        # function from the arith module.  The value, dictionary,
        # or AD instance to be multiplied into the input is stored in
        # rc["operand"]

        # Instantiate the log
        log = logutils.get_logger(__name__)


        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "multiply", "starting"))

        # Define the keyword to be used for the time stamp for this primitive
        timestamp_key = self.timestamp_keys["multiply"]

        # Initialize the list of output AstroData objects
        adoutput_list = []
        
        # Get data to be multiplied from the RC
        operand = rc["operand"]
        if operand is None:
            log.stdinfo("No operand to multiply; no changes will be "\
                            "made to input")
        elif type(operand)==AstroData:
            log.stdinfo("Multiplying input by %s" % 
                        (operand.filename))
        else:
            log.stdinfo("Multiplying input by %s" % 
                        (repr(operand)))

        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():

            if operand is not None:
                # Multiply ad by operand
                ad.mult(operand)

                # Add the appropriate time stamps to the PHU
                gt.mark_history(adinput=ad, primname=self.myself(), keyword=timestamp_key)

                # Change the filename
                ad.filename = gt.filename_updater(
                    adinput=ad, suffix=rc["suffix"], strip=True)

            # Append the output AstroData object to the list
            # of output AstroData objects
            adoutput_list.append(ad)

        # Report the list of output AstroData objects to the reduction
        # context
        rc.report_output(adoutput_list)
        
        yield rc

    def subtract(self,rc):
        # This is a bare-bones primitive interface to the ad sub
        # function from the arith module.  The value, dictionary,
        # or AD instance to be subtracted from the input is stored in
        # rc["operand"]

        # Instantiate the log
        log = logutils.get_logger(__name__)


        # Log the standard "starting primitive" debug message
        log.debug(gt.log_message("primitive", "subtract", "starting"))

        # Define the keyword to be used for the time stamp for this primitive
        timestamp_key = self.timestamp_keys["subtract"]

        # Initialize the list of output AstroData objects
        adoutput_list = []
        
        # Get data to be subtracted from the RC
        operand = rc["operand"]
        if operand is None:
            log.stdinfo("No operand to subtract; no changes will be "\
                            "made to input")
        elif type(operand)==AstroData:
            log.stdinfo("Subtracting %s from input" % 
                        (operand.filename))
        else:
            log.stdinfo("Subtracting %s from input" % 
                        (repr(operand)))
            
        # Loop over each input AstroData object in the input list
        for ad in rc.get_inputs_as_astrodata():

            if operand is not None:
                # Subtract operand from data
                ad.sub(operand)

                # Add the appropriate time stamps to the PHU
                gt.mark_history(adinput=ad, primname=self.myself(), keyword=timestamp_key)

                # Change the filename
                ad.filename = gt.filename_updater(
                    adinput=ad, suffix=rc["suffix"], strip=True)

            # Append the output AstroData object to the list
            # of output AstroData objects
            adoutput_list.append(ad)

        # Report the list of output AstroData objects to the reduction
        # context
        rc.report_output(adoutput_list)
        
        yield rc
    
    def ls(self, rc):
        from astrodata.eti.lseti import LSETI
        print "PRIMITIVE: instantiate LSETI(rc) object..."
        lspopen_external_task = LSETI(rc)
        print "PRIMITIVE: LSETI.run()..."
        lspopen_external_task.run()
        print "PRIMITIVE: yield rc"
        
        yield rc
