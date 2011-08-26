import os
from astrodata.adutils import gemLog
from astrodata.ReductionObjects import PrimitiveSet

from gempy import geminiTools as gt

class GENERALPrimitives(PrimitiveSet):
    """
    This is the class containing all of the primitives for the GENERAL level of
    the type hierarchy tree. It inherits all the primitives from the level
    above, 'PrimitiveSet'.
    """
    astrotype = "GENERAL"
    
    def init(self, rc):
        return 
    init.pt_hide = True
    
    def addInputs(self, rc):
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        import glob as gl
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
            from copy import deepcopy
            nd = deepcopy(ad)
            nd.filename = "copy_"+os.path.basename(ad.filename)
            rc.report_output(nd)
        
        yield rc
    
    def listDir(self, rc):
        # Instantiate the log
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
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
            stream = rc['stream']
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
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        
        if rc["to_stream"] != None:
            stream = rc["to_stream"]
        else:
            stream = "main"
        prefix = rc["prefix"];
        
        if "by_token" in rc:
            bt = rc["by_token"]
            for ar in rc.inputs:
                if bt in ar.filename:
                    rc.report_output(ar.ad, stream = stream)
            # print "pG110:",repr(rc.outputs)
        else:
            inputs = rc.get_inputs_as_astrodata()
                
            log.info("Reporting Output: "+", ".join([ ad.filename for ad in inputs]))
            if prefix:
                for inp in inputs:
                    inp.filename = os.path.join(
                                        os.path.dirname(ad.filename),
                                        prefix+os.path.basename(ad.filename))
            rc.report_output(inputs, stream = stream, )
        yield rc
    forwardStream = forwardInput
    
    def showOutputs(self, rc):
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        streams = rc.outputs.keys()
        streams.sort()
        streams.remove("main")
        streams.insert(0,"main")
        tstream = rc["streams"]
        
        for stream in streams:
            if tstream == None or stream in tstream:
                log.info("stream: "+stream)
                if len(rc.outputs[stream])>0:
                    for adr in rc.outputs[stream]:
                        log.info(str(adr))
                else:
                    log.info("    empty")                
    
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
            ad.filename = gt.fileNameUpdater(adIn=ad, suffix=ci,
                                             strip=False)
            # print "pG152:", ad.filename
        rc.report_output(inputs)
        
        yield rc

    def log(self, rc):
        log = gemLog.getGeminiLog(logType=rc["logType"],
                                  logLevel=rc["logLevel"])
        msg = rc["msg"]
        if msg == None:
            msg = "..."
        log.info(msg)
        yield rc
        
    def returnFromRecipe(self, rc):
        rc.return_from_recipe()
        yield rc
