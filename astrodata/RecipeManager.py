# This module operates like a singleton
from copy import deepcopy, copy
from datetime import datetime
import new
import pickle # for persisting the calibration index
import socket # to get host name for local statistics
#------------------------------------------------------------------------------ 
from astrodata.AstroData import AstroData
import traceback
import AstroDataType
from AstroDataType import getClassificationLibrary
from CalibrationDefinitionLibrary import CalibrationDefinitionLibrary # For xml calibration requests
import ConfigSpace
import Descriptors
import gdpgutil
from gdpgutil import pickConfig
import IDFactory as idFac # id hashing functions
from ParamObject import PrimitiveParameter
from ReductionContextRecords import CalibrationRecord, StackableRecord, AstroDataRecord, FringeRecord
import ReductionObjects
from ReductionObjects import ReductionObject
from ReductionObjectRequests import UpdateStackableRequest, GetStackableRequest, DisplayRequest, \
    ImageQualityRequest
from StackKeeper import StackKeeper, FringeKeeper

#------------------------------------------------------------------------------ 
centralPrimitivesIndex = {}
centralRecipeIndex = {}
centralReductionMap = {}
centralAstroTypeRecipeIndex = {}
centralParametersIndex = {}
centralAstroTypeParametersIndex = {}
#------------------------------------------------------------------------------ 

class RecipeExcept:
    """ This is the general exception the classes and functions in the
    Structures.py module raise.
    """
    def __init__(self, msg="Exception Raised in Recipe System", **argv):
        """This constructor takes a message to print to the user."""
        self.message = msg
        for arg in argv.keys():
            exec("self."+arg+"="+repr(argv[arg]))
            
            
    def __str__(self):
        """This str conversion member returns the message given by the user (or the default message)
        when the exception is not caught."""
        return self.message
        
class SettingFixedParam(RecipeExcept):
    pass
    
class UserParam(object):
    astrotype = None
    primname = None
    param = None
    value = None
    def __init__(self, astrotype, primname, param, value):
        self.astrotype = astrotype
        self.primname = primname
        self.param = param
        self.value = value
        
class UserParams(object):
    userParamDict = None
    def getUserParam(self, astrotype, primname):
        if self.userParamDict == None:
            return None
        if astrotype not in self.userParamDict:
            return None
        if primname not in self.userParamDict[astrotype]:
            return None
        return self.userParamDict[astrotype][primname]
        
    def addUserParam(self, userparam):
        up = userparam
        if userparam == None:
            return
        if self.userParamDict == None:
            self.userParamDict = {}
            
        if up.astrotype not in self.userParamDict:
            self.userParamDict.update({up.astrotype: {}})
        
        if up.primname not in self.userParamDict[up.astrotype]:
            self.userParamDict[up.astrotype].update({up.primname: {}})
            
        if up.param in self.userParamDict[up.astrotype][up.primname]:
            raise RecipeExcept("Parameter (%s.%s%s) already set by user" % (up.astrotype, up.primname, up.param))
        else:
            self.userParamDict[up.astrotype][up.primname].update({up.param:up.value})
            
class ReductionContext(dict):
    """The ReductionContext is used by primitives and recipiesen, hidden in the later case,
    to get input and report output. This allows primitives to be controlled in many different
    running environments, from pipelines to command line interactive reduction.
    """
    inputs = None
    originalInputs = None
    inputsHistory = None
    outputs = None
    calibrations = None
    rorqs = None
    status = "EXTANT"
    reason = "EXTANT"
    cmdRequest = "NONE"
    hostname = None
    displayName = None
    stephistory = None
    stackeep = None
    displayMode = None
    displayID = None
    irafstdout = None
    irafstderr = None
    callbacks = None
    arguments = None
    _localparms = None # dictionary with local args (given in recipe as args, generally)
    userParams = None # meant to be UserParams instance
    
    #------------------------------------------------------------------------------ 
 
     
    def __init__(self):
        """The ReductionContext constructor creates empty dictionaries and lists, members set to
        None in the class."""
        self.inputs = []
        self.callbacks = {}
        self.inputsHistory = []
        self.calibrations = {}
        self.rorqs = []
        self.outputs = {"standard":[]}
        self.stephistory = {}
        self.hostname = socket.gethostname()
        self.displayName = None
        self.arguments = []
        # TESTING
        self.cdl = CalibrationDefinitionLibrary()
        # undeclared
        self.indent = 0 
        
        # Stack Keep is a resource for all RecipeManager functions... one shared StackKeeper to simulate the shared ObservationServie
        # used in PRS mode.
        self.stackeep = StackKeeper()
        self.fringes = FringeKeeper()
        
    def __getitem__(self, arg):
        """Note, the ReductionContext version of __getitem__ returns None instead of throwing a KeyError.
        """
        if self.localparms and arg in self.localparms:
            return self.localparms[arg]
        else:
            try:
                return dict.__getitem__(self, arg)
            except KeyError:
                return None
    
    def __str__(self):
        """Used to dump Reduction Context(co) into file for test system
        """
        tempStr = ""
        tempStr = tempStr + "REDUCTION CONTEXT OBJECT (CO)\n" + \
            "inputs = " + str(self.inputs) + \
            "\ninputsHistory =  " + str(self.inputsHistory) + \
            "\ncalibrations = \n" + self.calsummary() + \
            "\nrorqs = " 
        if self.rorqs != []:
            for rq_obj in self.rorqs:            
                tempStr = tempStr + str(rq_obj)
        else:
            tempStr = tempStr + str(self.rorqs)
        
        #no loop initiated for stkrqs object printouts yet
        tempStr = tempStr + "\noutputs = " 
        
        if self.outputs["standard"] != []:
            for out_obj in self.outputs["standard"]:
                tempStr = tempStr + str(out_obj)
        else:
            tempStr = tempStr + str(self.outputs)
        #"stephistory = " + str( self.stephistory ) + \
        tempStr = tempStr + "\nhostname = " + str(self.hostname) + \
            "\ndisplayName = " + str(self.displayName) + \
            "\ncdl = " + str(self.cdl) + \
            "\nindent = " + str(self.indent) + \
            "\nstackeep = " + str(self.stackeep)
        for param in self.values():
            tempStr += "\n" + self.paramsummary()
        return tempStr   
    
    def addCal(self, data, caltyp, calname, timestamp=None):
        '''
        Add a calibration to the calibration index with a key (DATALAB, caltype).
        
        @param data: The path or AstroData for which the calibration will be applied to.
        @type data: str or AstroData instance
        
        @param caltyp: The type of calibration. For example, 'bias' and 'flat'.
        @type caltyp: str
        
        @param calname: The URI for the MEF calibration file.
        @type calname: str
        
        @param timestamp: Default= None. Timestamp for when calibration was added. The format of time is
        taken from datetime.datetime.
        @type timestamp: str
        '''
        adID = idFac.generateAstroDataID(data)
        calname = os.path.abspath(calname)
        
        if timestamp == None:
            timestamp = datetime.now()
        else:
            timestamp = timestamp
        
        if self.calibrations == None:
            self.calibrations = {}
        
        calrec = CalibrationRecord(data.filename, calname, caltyp, timestamp)
        key = (adID, caltyp)
        #print "RM542:", key, calrec
        self.calibrations.update({key: calrec})
        
    def addCallback(self, name, function):
        callbacks = self.callbacks
        if name in callbacks:
            l = callbacks[name]
        else:
            l = []
            callbacks.update({name:l})
        l.append(function)
    
    def addInput(self, filenames):
        '''
        Add input to be processed the next batch around. If this is the first input being added,
        it is also added to originalInputs.
        
        @param filenames: Inputs you want added.
        @type filenames: list, AstroData, str 
        '''
        if type(filenames) != list:
            filenames = [filenames]
        
        ##@@TODO: Approve that this is acceptable. (i.e. should it be done here or after the first 
        ## round is complete?)
        origFlag = False
        if self.originalInputs is None or self.originalInputs == []:
            self.originalInputs = []
            origFlag = True
        
        for filename in filenames:
            if type(filename) == str:
                filename = AstroDataRecord(filename) # filename converted from str -> AstroData 
            elif type(filename) == AstroData:
                filename = AstroDataRecord(filename)
            elif type(filename) == AstroDataRecord:
                pass
            else:
                raise("BadArgument: '%(name)s' is an invalid type '%(type)s'. Should be str, AstroData, AstroDataRecord." 
                      % {'name':str(filename), 'type':str(type(filename))})
            
            self.inputs.append(filename)
            if origFlag:
                self.originalInputs.append(filename)        
       
    def addRq(self, rq):
        '''
        Add a request to be evaluated by the control loop.
        
        @param rq: The request.
        @type rq: ReductionObjectRequests instance
        '''
        if self.rorqs == None:
            self.rorqs = []
        self.rorqs.append(rq)
        
    def begin(self, stepname):
        key = datetime.now()
        # value = dictionary
        val = self.stepMoment(stepname, "begin")
        self.indent += 1
        self.stephistory.update({key: val}) 
        self.lastBeginDt = key
        return self
        
    def getBeginMark(self, stepname, indent=None):
        for time in self.stephistory.keys():
            if     self.stephistory[time]["stepname"] == stepname \
               and self.stephistory[time]["mark"] == "begin":
                    if indent != None:
                        if self.stephistory[time]["indent"] == indent:
                            return (time, self.stephistory[time])
                    else:
                        return (time, self.stephistory[time])    
        return None
    
    def calFilename(self, caltype):
        """returns a local filename for a retrieved calibration"""
        if self.originalInputs == None:
            self.originalInputs = deepcopy(self.inputs)
        if len(self.originalInputs) == 0:
            return None
        #elif len(self.originalInputs) == 1:
        #    adID = idFac.generateAstroDataID(self.inputs[0].ad)
        #    key = (adID, caltype)
        #    infile = os.path.basename(self.inputs[0].filename)
        #    if key in self.calibrations:
        #        return {self.calibrations[key].filename:[infile]}
        #    else:
        #        return None
        else:
            retl = {}
            for inp in self.originalInputs:
                key = (idFac.generateAstroDataID(inp.ad), caltype)
                calfile = self.calibrations[key].filename
                infile = os.path.basename(inp.filename)
                if retl.has_key(calfile):
                    retl.update({calfile:retl[calfile] + [infile]})
                else:
                    retl.update({calfile:[infile]})
            return retl
                     
    def callCallbacks(self, name, **params):
        callbacks = self.callbacks
        if name in callbacks:
            for f in callbacks[name]:
                f(**params)
                    
    def calSummary(self, mode="text"):
        rets = ""
        for key in self.calibrations.keys():
            rets += str(key)
            rets += str(self.calibrations[key])
        return rets
    
    def checkControl(self):
        return self.cmdRequest
    
    def clearRqs(self, rtype=None):
        '''
        Clear all requests.
        '''
        if rtype == None:
            self.rorqs = []
        else:
            rql = copy(self.rorqs)
            for rq in rql:
                if type(rq) == type(rtype):
                    self.rorqs.remove(rq)
    
    def control(self, cmd="NONE"):
        self.cmdRequest = cmd
    
    def end(self, stepname):
        key = datetime.now()
        self.indent -= 1
        val = self.stepMoment(stepname, "end")
        # this step saves inputs
        self.stephistory.update({key: val})
        # this step moves outputs["standard"] to inputs
        # and clears outputs
        self.finalizeOutputs()
        self.localparms = None
        return self
    
    def finalizeOutputs(self):
        """ This function means there are no more outputs, generally called
        in a control loop when a generator function primitive ends.  Standard
        outputs become the new inputs. Calibrations and non-standard output
        is not affected.
        """
        # only push is outputs is filled
        if len(self.outputs["standard"]) != 0:
            # don't do this if the set is empty, it's a non-IO primitive
            ##@@TODO: The below if statement could be redundant because this is done
            # in addInputs
            if self.originalInputs == None:
                self.originalInputs = deepcopy(self.inputs)
            
            #print "OUTPUTS:", self.outputs["standard"]
            newinputlist = []
            for out in self.outputs['standard']:
                if type(out) == AstroDataRecord:
                    newinputlist.append(out)
                else:
                    raise RuntimeError("Bad Argument: Wrong Type '%(val)s' '%(typ)s'." 
                                       % {'val':str(out), 'typ':str(type(out))})
            
            self.inputs = newinputlist
            self.outputs.update({"standard":[]})
            
#------------------ FINISH ----------------------------------------------------   
    def isFinished(self, arg=None):
        if arg == None:
            return self.status == "FINISHED"
        else:
            if arg == True:
                self.status = "FINISHED"
            elif self.status != "FINISHED":
                raise RecipeExcept("Attempt to change status from %s to FINISHED" % self.status)
        return self.isFinished()
    def finish(self):
        self.isFinished(True)
    finished = property(isFinished, isFinished)
#------------------------------------------------------------------------------
    
    def getCal(self, data, caltype):
        '''
        Retrieve calibration.
        
        @param data: File for which calibration will be applied.
        @type data: str or AstroData instance
        
        @param caltype: The type of calibration. For example, 'bias' and 'flat'.
        @type caltype: str
        
        @return: The URI of the currently stored calibration or None.
        @rtype: str or None 
        '''
        #print "RM551:", data, type( data )
        adID = idFac.generateAstroDataID(data)
        #filename = os.path.abspath(filename)
        key = (adID, caltype)
        if key in self.calibrations.keys():
            return self.calibrations[(adID, caltype)].filename
        return None
    
    def getEndMark(self, stepname, indent=None):
        for time in self.stephistory.keys():
            if     self.stephistory[time]["stepname"] == stepname \
               and self.stephistory[time]["mark"] == "end":
                if indent != None:
                    if self.stephistory[time]["indent"] == indent:
                        return (time, self.stephistory[time])
                else:
                    return (time, self.stephistory[time])
        return None    
    
    def getInputs(self, style=None):
        if style==None:
            return inputs
        elif style == "AD": #@@HARDCODED: means "as AstroData instances"
            retl = []
            for inp in self.inputs:
                if inp.ad == None:
                    inp.load()
                retl.append(inp.ad)
            return retl
        elif style == "FN": #@@HARDCODED: means "as Filenames"
            retl = [inp.filename for inp in self.inputs]
            return retl
        else:
            return None # this should not happen, but given a mispelled style arg

    def getInputsAsAstroData(self):
        return self.getInputs(style="AD")
        
    def getInputsAsFilenames(self):
        return self.getInputs(style="FN")

    def getInputFromParent(self, parent):
        '''
        Very inefficient.
        '''
        # @@CLEAN: I don't know what this is
        for inp in self.inputs:
            if inp.parent == parent:
                return inp.filename
           
    def getIrafStderr(self):
        if self.irafstderr != None:
            return self.irafstderr
        else:
            return sys.stderr
        
    def getIrafStdout(self):
        if self.irafstdout != None:
            return self.irafstdout
        else:
            return sys.stdout
        
    def getReferenceImage(self):
        if len(self.inputs) == 0:
            return None
        if self.inputs[0].ad == None:
            return None # @@NOTE: return none if reference image not loaded, reconsider
            # raise RecipeExcept("AstroData instance not loaded for input %s" % self.inputs[0].filename)
        return self.inputs[0].ad
    
    def getStack(self, ID):
        return self.stackeep.get(ID)
 
    def inputsAsStr(self, strippath=True):
        if self.inputs == None:
            return ""
        else:
            inputlist = []
            for inp in self.inputs:
                inputlist.append(inp.filename)

            if strippath == False:
                return ",".join(inputlist)                
            else:
                return ",".join([os.path.basename(path) for path in inputlist])

    def localparmsSet(self, lpd):
        self._localparms = lpd
        
    def localparmsGet(self):
        if self._localparms == None:
            self._localparms = {}
        return self._localparms 
    localparms = property(localparmsGet, localparmsSet)
    
    def makeInlistFile(self, filename, filelist):
        try:
            fh = open(filename, 'w')
            for item in filelist:
                fh.writelines(item + '\n')
        except:
            raise "Could not write inlist file for stacking." 
        finally:
            fh.close()
        return "@" + filename
        
    def parameterCollate(self, astrotype, primset, primname):
        """This function looks at the default primset paramaters for primname
        and sets the localparms member."""
        
        # @@HERE: is where parameter metadata is respected, or not
        if primname in primset.paramDict:
            # localparms should always be defined by here
            
            # users can never override argument in recipes (too confusing)
            correctUPD = None
            if self.userParams != None:
                correctUPD = self.userParams.getUserParam(astrotype, primname)
                if correctUPD != None:
                    for param in correctUPD.keys():
                        if param in self.localparms:
                            exs  = "User attempting to override parameter set in recipe\n"
                            exs += "\tastrotype = %s\n" % astrotype
                            exs += "\tprimitive = %s\n" % primname
                            exs += "\tparameter = %s\n" % str(param)
                            exs += "\t\tattempt to set to = %s\n" % correctUPD[param]
                            exs += "\t\trecipe setting = %s\n" % self.localparms[param]
                            raise SettingFixedParam(exs)
                            
            # use primset.paramDict to update self.localparms
            for param in primset.paramDict[primname].keys():
                # @@NAMING: naming of default value in parameter dictionary hardcoded
                if param in self.localparms:
                    repOvrd = ("recipeOverride" not in primset.paramDict[primname][param])\
                                 or primset.paramDict[primname][param]["recipeOverride"]
                    # then it's already in there, check metadata
                    # @@NAMING: "recipeOverride" used in RecipeManager code
                    if not repOvrd:
                        exs =  "Recipe attempts to set fixed parameter\n"
                        exs += "\tastrotype = %s\n" % astrotype
                        exs += "\tprimitive = %s\n" % primname
                        exs += "\tparameter = %s\n" % str(param)
                        exs += "\t\tattempt to set to = %s\n" % self.localparms[param]
                        exs += "\t\tfixed setting = %s\n" % primset.paramDict[primname][param]["default"]
                        raise SettingFixedParam(exs)
                    else:
                        # don't update this one, because the recipe is allowed to override
                        continue
                if "default" in primset.paramDict[primname][param]:
                    self.localparms.update({param:primset.paramDict[primname][param]["default"]})
                
            # users override everything else if  it gets here... and is allowed
            if correctUPD:
                for param in correctUPD:
                    if param in self.localparms:
                        userOvrd = ("userOverride" not in primset.paramDict[primname][param])\
                                      or primset.paramDict[primname][param]["userOverride"]
                        if not userOvrd:
                            exs =  "User attempted to set fixed parameter\n"
                            exs += "\tastrotype = %s\n" % astrotype
                            exs += "\tprimitive = %s\n" % primname
                            exs += "\tparameter = %s\n" % str(param)
                            exs += "\t\tattempt to set to = %s\n" % correctUPD[param]
                            exs += "\t\tfixed setting = %s\n" % primset.paramDict[primname][param]["default"]
                            
                            raise SettingFixedParam(exs, astrotype = astrotype)
                        else:
                            self.localparms.update({param:correctUPD[param]})
                    else:
                        self.localparms.update({param:correctUPD[param]})
              
    def paramNames(self, subset = None):
        if subset == "local":
            return self.localparms.keys()
        else:
            lpkeys = set(self.localparms.keys())
            rckeys = set(self.keys())
            retl = list(lpkeys | rckeys)
            return retl
                                                                
    
    def outputsAsStr(self, strippath=True):
        if self.outputs == None:
            return ""
        else:
            outputlist = []
            for inp in self.outputs:
                outputlist.append(inp.filename)
            #print "RM289:", outputlist
            #"""
            if strippath == False:
                # print self.inputs
                return ", ".join(outputlist)
            else:
                return ", ".join([os.path.basename(path) for path in outputlist])
            
    #------------------ PAUSE ---------------------------------------------------- 
    def isPaused(self, bpaused=None):
        if bpaused == None:
            return self.status == "PAUSED"
        else:
            if bpaused:
                self.status = "PAUSED"
            else:
                self.status = "RUNNING"
        
        return self.isPaused()
    def pause(self):
        self.callCallbacks("pause")
        self.isPaused(True)
    def unpause (self):
        self.isPaused(False)
    paused = property(isPaused, isPaused)
    def requestPause(self):
        self.control("pause") 
    def pauseRequested(self):
        return self.cmdRequest == "pause"
    #--------------------------------------------------------------------------- 
    #------------------ PAUSE ----------------------------------------------------
    def paramsummary(self):
        '''
        A util function for printing out all the parameters for this reduction 
        context in a semi-organized fashion.
        
        @return: The formatted message for all the current parameters.
        @rtype: str
        '''
        char = "-"
        rets = '\n' + char * 40 + "\n"
        rets += '''------Global Parameters------\n'''
        
        globval = "global"
        
        def printParam(val, param):
            # This temp function prints out the stuff inside an individual parameter.
            # I have a feeling this and paramsummary will be moved to a util function.
            tempStr = ""
            list_of_params = param.keys()
            list_of_params.sort()
            tempStr += char * 40 + "\n"
            for pars in list_of_params:
                tempStr += str(param[pars]) + "\n"
                tempStr += char * 40 + "\n"
            return tempStr
            
        rets += printParam(globval, self[globval])
        list_of_prims = self.keys()
        list_of_prims.sort()
        for primname in list_of_prims:
            if primname != globval:
                rets += '''------%s Parameters------\n''' % (primname)
                rets += printParam(primname, self[primname])
        
        return rets
    
    def persistCalIndex(self, filename):
        #print "Calibration List Before Persist:"
        #print self.calsummary()
        try:
            pickle.dump(self.calibrations, open(filename, "w"))
        except:
            print "Could not persist the calibration cache."
            raise 
    
    def persistFringeIndex(self, filename):
        try:
            pickle.dump(self.fringes.stackLists, open(filename, "w"))
        except:
            print 'Could not persist the fringe cache.'
            raise
            
    def persistStkIndex(self, filename):
        try:
            #print "RM80:", self.stackeep
            pickle.dump(self.stackeep.stackLists, open(filename, "w"))
        except:
            print "Could not persist the stackable cache."
            raise
    
    def prependNames(self, prepend, currentDir=True, filepaths=None):
        '''
        Prepend a string to a filename.
        
        @param prepend: The string to be put at the front of the file.
        @type prepend: string
        
        @param currentDir: Used if the filename (astrodata filename) is in the
        current working directory.
        @type currentDir: boolean
        
        @return: List of new prepended paths.
        @rtype: list  
        '''
        retlist = []
        if filepaths is None:
            dataset = self.inputs
        else:
            dataset = filepaths
            
        for data in dataset:
            parent = None
            if type(data) == AstroData:
                filename = data.filename
            elif type(data) == str:
                filename = data
            elif type(data) == AstroDataRecord:
                filename = data.filename
                parent = data.parent
            else:
                raise RecipeExcept("BAD ARGUMENT: '%(data)s'->'%(type)s'" % {'data':str(data), 'type':str(type(data))})
               
            if currentDir == True:
                root = os.getcwd()
            else:
                root = os.path.dirname(filename)

            bname = os.path.basename(filename)
            prependfile = os.path.join(root, prepend + bname)
            if parent is None:
                retlist.append(prependfile)
            else:
                retlist.append((prependfile, parent))
        
        return retlist
    
    def printHeaders(self):
        for inp in self.inputs:
            if type(inp) == str:
                ad = AstroData(inp)
            elif type(inp) == AstroData:
                ad = inp
            try:
                outfile = open(os.path.basename(ad.filename) + ".headers", 'w')
                for ext in ad.hdulist:
                    outfile.write("\n" + "*" * 80 + "\n")
                    outfile.write(str(ext.header))
                
            except:
                raise "Error writing headers for '%{name}s'." % {'name':ad.filename}
            finally:
                outfile.close()
    
    def processCmdReq(self):
        if self.cmdRequest == "pause":
            self.cmdRequest = "NONE"
            self.pause()
            
    def removeCallback(self, name, function):
        if name in self.callbacks:
            if function in self.callbackp[name]:
                self.callbacks[name].remove(function)
        else:
            return
    
    def reportHistory(self):
        
        sh = self.stephistory
        
        ks = self.stephistory.keys()
        
        ks.sort()
        
        # print sort(sh.keys())
        lastdt = None
        startdt = None
        enddt = None

        retstr = "RUNNING TIMES\n"
        retstr += "-------------\n"
        for dt in ks: # self.stephistory.keys():
            indent = sh[dt]["indent"]
            indentstr = "".join(["  " for i in range(0, indent)])
            
            mark = sh[dt]["mark"]
            if mark == "begin":
                elapsed = ""
                format = "%(indent)s%(stepname)s begin at %(time)s"
            elif mark == "end":
                elapsed = "(" + str(dt - lastdt) + ") "
                format = "\x1b[1m%(indent)s%(stepname)s %(elapsed)s \x1b[22mends at %(time)s"
            else:
                elapsed = ""
                format = "%(indent)s%(stepname)s %(elapsed)s%(mark)s at %(time)s"
                
            lastdt = dt
            if startdt == None:
                startdt = dt

            pargs = {  "indent":indentstr,
                        "stepname":str(sh[dt]['stepname']),
                        "mark":str(sh[dt]['mark']),
                        "inputs":str(",".join(sh[dt]['inputs'])),
                        "outputs":str(sh[dt]['outputs']),
                        "time":str(dt),
                        "elapsed":elapsed,
                        "runtime":str(dt - startdt),
                    }
            retstr += format % pargs + "\n"
            retstr += "%(indent)sTOTAL RUNNING TIME: %(runtime)s (MM:SS:ms)" % pargs + "\n"
       
        startdt = None
        lastdt = None
        enddt = None
        wide = 75
        retstr += "\n\n"
        retstr += "SHOW IO".center(wide) + "\n"
        retstr += "-------".center(wide) + "\n"
        retstr += "\n"
        for dt in ks: # self.stephistory.keys():
            indent = sh[dt]["indent"]
            indentstr = "".join(["  " for i in range(0, indent)])
            
            mark = sh[dt]["mark"]
            if mark == "begin":
                elapsed = ""
            elif mark == "end":
                elapsed = "(" + str(dt - lastdt) + ") "
                
            pargs = {  "indent":indentstr,
                        "stepname":str(sh[dt]['stepname']),
                        "mark":str(sh[dt]['mark']),
                        "inputs":str(",".join(sh[dt]['inputs'])),
                        "outputs":str(",".join(sh[dt]['outputs']['standard'])),
                        "time":str(dt),
                        "elapsed":elapsed,
                    }
            if startdt == None:
                retstr += ("%(inputs)s" % pargs).center(wide) + "\n"

            if (pargs["mark"] == "end"):
                retstr += " | ".center(wide) + "\n"
                retstr += "\|/".center(wide) + "\n"
                retstr += " ' ".center(wide) + "\n"
                
                line = ("%(stepname)s" % pargs).center(wide)
                line = "\x1b[1m" + line + "\x1b[22m" + "\n"
                retstr += line
                
            if len(sh[dt]["outputs"]["standard"]) != 0:
                retstr += " | ".center(wide) + "\n"
                retstr += "\|/".center(wide) + "\n"
                retstr += " ' ".center(wide) + "\n"
                retstr += ("%(outputs)s" % pargs).center(wide) + "\n"
                
                
            lastdt = dt
            if startdt == None:
                startdt = dt
        
        return retstr
        
    def reportOutput(self, inp, category="standard", load=True):
        ##@@TODO: Read the new way code is done.
        if category != "standard":
            raise RecipeExcept("You may only use " + 
                "'standard' category output at this time.")
        if type(inp) == str:
            self.outputs[category].append(AstroDataRecord(inp, self.displayID, load=load))
        elif isinstance(inp, AstroData):
            self.outputs[category].append(AstroDataRecord(inp))
        elif type(inp) == list:
            for temp in inp:
                # This is a good way to check if IRAF failed.
                
                if type(temp) == tuple:
                    #@@CHECK: seems bad to assume a tuple means it is from 
                    #@@.....: a primitive that needs it's output checked!
                    if not os.path.exists(temp[0]):
                        raise "LAST PRIMITIVE FAILED: %s does not exist" % temp[0]
                    orecord = AstroDataRecord(temp[0], self.displayID, parent=temp[1], load=load)
                    #print 'RM370:', orecord
                elif isinstance(temp, AstroData):
                    print "RM891:", type(temp)
                    orecord = AstroDataRecord(temp)
                elif type(temp) == str:
                    if not os.path.exists(temp):
                        raise "LAST PRIMITIVE FAILED."
                    orecord = AstroDataRecord(temp, self.displayID , load=load)
                else:
                    raise "RM292 type: " + str(type(temp))
                #print "RM344:", orecord
                self.outputs[category].append(orecord)
    
    def restoreCalIndex(self, filename):
        if os.path.exists(filename):
            self.calibrations = pickle.load(open(filename, 'r'))
        else:
            pickle.dump({}, open(filename, 'w'))
    
    def restoreFringeIndex(self, filename):
        '''
        
        '''
        if os.path.exists(filename):
            self.fringes.stackLists = pickle.load(open(filename, 'r'))
        else:
            pickle.dump({}, open(filename, 'w'))
                            
    def restoreStkIndex(self, filename):
        '''
        
        '''
        if os.path.exists(filename):
            self.stackeep.stackLists = pickle.load(open(filename, 'r'))
        else:
            pickle.dump({}, open(filename, 'w'))
    
    def rmCal(self, data, caltype):
        '''
        Remove a calibration. This is used in command line argument (rmcal). This may end up being used
        for some sort of TTL thing for cals in the future.
        
        @param data: Images who desire their cals to be removed.
        @type data: str, list or AstroData instance.
        
        @param caltype: Calibration type (e.g. 'bias').
        @type caltype: str
        '''
        datalist = gdpgutil.checkDataSet(data)
        
        for dat in datalist:
            datid = idFac.generateAstroDataID(data)
            key = (datid, caltype)
            if key in self.calibrations.keys():
                self.calibrations.pop(key)
            else:
                print "'%(tup)s', was not registered in the calibrations."
    
    def rqCal(self, caltype, inputs=None):
        '''
        Create calibration requests based on raw inputs.
        
        @param caltype: The type of calibration. For example, 'bias' and 'flat'.
        @type caltype: str
        '''
        if inputs is None:
            addToCmdQueue = self.cdl.getCalReq(self.originalInputs, caltype)
        else:
            addToCmdQueue = self.cdl.getCalReq(inputs, caltype)
        for re in addToCmdQueue:
            self.addRq(re)
    
    def rqDisplay(self, displayID=None):
        '''
        self, filename = None
        if None use self.inputs
        
        Create requests to display inputs.
        '''
        ver = "1_0"
        displayObject = DisplayRequest()
        if displayID:
            Did = displayID
        else:
            Did = idFac.generateDisplayID(self.inputs[0].filename, ver)
        displayObject.disID = Did
        displayObject.disList = self.inputs
        self.addRq(displayObject)
    
    def rqIQ(self, ad, eM, eS, fM, fS):
        iqReq = ImageQualityRequest(ad, eM, eS, fM, fS)
        self.addRq(iqReq)
        
    def rqStackGet(self):
        ver = "1_0"
        # Not sure how version stuff is going to be done. This version stuff is temporary.
        for orig in self.originalInputs:
            Sid = idFac.generateStackableID(orig.ad, ver)
            stackUEv = GetStackableRequest()
            stackUEv.stkID = Sid
            self.addRq(stackUEv)
                
    def rqStackUpdate(self):
        '''
        Create requests to update a stack list.
        '''
        ver = "1_0"
        # Not sure how version stuff is going to be done. This version stuff is temporary.
        for inp in self.inputs:
            stackUEv = UpdateStackableRequest()
            Sid = idFac.generateStackableID(inp.ad, ver)
            stackUEv.stkID = Sid
            stackUEv.stkList = inp.filename
            self.addRq(stackUEv)
                       
    def setIrafStderr(self, so):
        self.irafstderr = so
        return
    
    def setIrafStdout(self, so):
        self.irafstdout = so
        return
    
    def stackAppend(self, ID, files):
        self.stackeep.add(ID, files)
        
    def stack_inputsAsStr(self, ID):        
        #pass back the stack files as strings
        stack = self.stackeep.get(ID)
        return ",".join(stack.filelist)
    
    def stepMoment(self, stepname, mark):
        val = { "stepname"  : stepname,
                "indent"    : self.indent,
                "mark"      : mark,
                "inputs"    : copy(self.inputs),
                "outputs"   : copy(self.outputs),
                "processed" : False
                }
        return val
    
    def suffixNames(self, suffix, currentDir=True):
        '''
        
        '''
        newlist = []
        for nam in self.inputs:
            if currentDir == True:
                path = os.getcwd()
            else:
                path = os.path.dirname(nam.filename)
            
            fn = os.path.basename(nam.filename)
            finame, ext = os.path.splitext(fn)
            fn = finame + "_" + suffix + ext
            newpath = os.path.join(path, fn) 
            newlist.append(newpath)
        return newlist
    
def openIfName(dataset):
    """Utility function to handle accepting datasets as AstroData
    instances or string filenames. Works in conjunction with closeIfName.
    The way it works, openIfName opens returns an GeminiData isntance"""    
    bNeedsClosing = False    
    if type(dataset) == str:
        bNeedsClosing = True
        gd = AstroData(dataset)
    elif isinstance(dataset, AstroData):
        bNeedsClosing = False
        gd = dataset
    else:
        raise RecipeExcept("BadArgument in recipe utility function: openIfName(..)\n MUST be filename (string) or GeminiData instrument")
    return (gd, bNeedsClosing)
    
def closeIfName(dataset, bNeedsClosing):
    """Utility function to handle accepting datasets as AstroData
    instances or string filenames. Works in conjunction with openIfName."""

    if bNeedsClosing == True:
        dataset.close()
    
    return

class RecipeLibrary(object):

    primLoadTimes = {}
    
    def addLoadTime(self, source, start, end):
        key = datetime.now()
        pair = {key: {"source":source, "start":start, "end":end}}
        self.primLoadTimes.update(pair)

    def discoverCorrectPrimType(self, context):
        ref = context.getReferenceImage()
        if ref == None:
            return None
        val = pickConfig(ref, centralPrimitivesIndex)
        k = val.keys()
        if len(k) != 1:
                raise RecipeExcept("Can't discover correct primtype for %s, more than one (%s)" % (ref.filename, repr(k)))
        return k[0]
        
    def reportHistory(self):
        self.reportLoadTimes()
        
    def reportLoadTimes(self):
        skeys = self.primLoadTimes.keys()
        skeys.sort()
        
        for key in skeys:
            primrecord = self.primLoadTimes[key]
            source = primrecord["source"]
            start = primrecord["start"]
            end = primrecord["end"]
            duration = end - start
            
            pargs = {   "module":source,
                        "duration":duration,
                        }
            print "Module '%(module)s took %(duration)s to load'" % pargs

    def loadAndBindRecipe(self, ro, name, dataset=None, astrotype=None):
        """
        Will load a single recipe, compile and bind it to the given reduction objects
        """
        
        if astrotype != None:
            # get recipe source
            
            rec = self.retrieveRecipe(name, astrotype=astrotype)
            # print "RM1113:", name, rec, astrotype
            try:
                # print "RM1115: before"
                ps = ro.getPrimSet(name, astrotype=astrotype)
                # print "RM1117: after"
                if ps:
                    if rec == None:
                        return #not a recipe, but exists as primitive
                    else:
                        msg = "NAME CONFLICT: ASSIGNING RECIPE %s BUT EXISTS AS PRIMITIVE:\n\t%s" % rec, repr(ps)
                        raise RecipeExcept(msg)
            except ReductionObjects.ReductionExcept:
                 pass # just means there is no primset, that function throws
                
                
            if rec:
                # compose to python source
                prec = self.composeRecipe(name, rec)
                # compile to unbound function (using the python interpretor obviously)
                rfunc = self.compileRecipe(name, prec)
                # bind the recipe to the reduction object
                ro = self.bindRecipe(ro, name, rfunc)
            else:
                raise "can't find recipe!"
        elif dataset != None:
            gd, bnc = openIfName(dataset)
            types = gd.getTypes()
            rec = None
            for typ in types:
                rec = self.retrieveRecipe(name, astrotype=typ)
                if rec:
                    prec  = self.composeRecipe(name, rec)
                    rfunc = self.compileRecipe(name, prec)
                    ro = self.bindRecipe(ro, name, rfunc)
            # no recipe, see if there is a generic one
            if rec == None:
                rec = self.retrieveRecipe(name)
                if rec:
                    prec = self.composeRecipe(name, rec)
                    rfunc = self.compileRecipe(name, prec)
                    ro = self.bindRecipe(ro, name, rfunc)
            closeIfName(gd, bnc)
            

    def getApplicableRecipes(self, dataset= None, astrotype = None, collate=False):
        """
        Get list of recipes associated with all the types that apply to this dataset.
        """
        if dataset != None and astrotype != None:
            raise RecipeExcept("getApplicableRecipes cannot have dataset and astrotype set")
        if dataset == None and astrotype == None:
            raise RecipeExcept("getApplicableRecipes must have either a dataset or explicit astrotype set")
        byfname = False
        if dataset:
            if  type(dataset) == str:
                astrod = AstroData(dataset)
                byfname = True
            elif type(dataset) == AstroData:
                byfname = False
                astrod = dataset
            else:
                raise BadArgument()
            # get the types
            types = astrod.getTypes()
        else:
            types = [astrotype]
        # look up recipes, fill list
        reclist = []
        recdict = {}
        for typ in types:
            if typ in centralAstroTypeRecipeIndex.keys():
                recnames = centralAstroTypeRecipeIndex[typ]
                reclist.extend(recnames)
                recdict.update({typ: recnames})
            

        # if we opened the file we close it
        if byfname:
            astrod.close()
        
        if collate == False:
            return reclist
        else:
            return recdict
        
    
    
    def retrieveRecipe(self, name, astrotype=None):
        # @@NAMING: uses "recipe.TYPE" and recipe for recipe.ALL
        cri = centralRecipeIndex
        
        if astrotype:
            akey = name + "." + astrotype
            key = name 
        else:
            key = name
            akey = name + ".None"

        bdefRecipe = key in cri
        bastroRecipe = akey in cri
        
        fname = None
        if bastroRecipe:
            fname = cri[akey]
        elif bdefRecipe:
            if astrotype == None:
                fname = cri[key]
            else:
                # @@NOTE: OLD WAY: User must SPECIFY none to get the generic recipe
                # return None
                # @@....: new way: inherit generic recipe!
                fname = cri[key]        
        else:
            return None

        rfile = file(fname, "r")
        rtext = rfile.read()
        #print "RM718:", rtext
        return rtext
            
    def retrieveReductionObject(self, dataset=None, astrotype=None):
        a = datetime.now()
        
        # if astrotpye is None, but dataset is set, then we need to get the astrotype from the 
        # dataset.  For reduction objects, there can be only one assigned to a real object
        # if there are multiple reduction objects associated with type we must find out through
        # inheritance relationships which one applies. E.g. if a dataset is GMOS_SPEC and
        # GMOS_IFU, then an inheritance relationship is sought, and the child type has priority.
        # If they cannot be resolved, because there are unrelated types or through multiple
        # inheritance multiple ROs may apply, then we raise an exceptions, this is a configuration
        # problem.
        
        ro = ReductionObjects.ReductionObject()
        primsetlist = self.retrievePrimitiveSet(dataset=dataset, astrotype=astrotype)
        ro.recipeLib = self
        if primsetlist:
            ro.curPrimType = primsetlist[0].astrotype
        else:
            return None
        for primset in primsetlist:
            ro.addPrimSet(primset)
        
        b = datetime.now()
        if astrotype != None:
            source = "TYPE: " + astrotype
        elif dataset != None:
            source = "FILE: " + str(dataset)
        else:
            source = "UNKNOWN"
            
        #@@perform: monitory real performance loading primitives
        self.addLoadTime(source, a, b)
        return ro
        
    def retrievePrimitiveSet(self, dataset=None, astrotype=None):
        if (astrotype == None) and (dataset != None):
            val = pickConfig(dataset, centralPrimitivesIndex)
            k = val.keys()
            if len(k) != 1:
                raise RecipeExcept("CAN'T RESOLVE PRIMITIVE SET CONFLICT")
            astrotype = k[0]
        # print "RM1272:", astrotype
        primset = None
        # print "RM1275:", repr(centralPrimitivesIndex)
        if (astrotype != None) and (astrotype in centralPrimitivesIndex):
            primdeflist = centralPrimitivesIndex[astrotype]
            # print "RM1276:", repr(primdeflist)
            primlist = []
            for primdef in primdeflist:
                rfilename = primdef[0] # the first in the tuple is the primset file
                rpathname = centralReductionMap[rfilename]
                rootpath = os.path.dirname(rpathname)
                importname = os.path.splitext(rfilename)[0]
                a = datetime.now()
                try:
                    # print "RM1282: about to import", importname, primdef[1]
                    exec ("import " + importname)
                    # print ("RM1285: after import")
                except:
                    print traceback.format_exc()
                b = datetime.now()
                primset = eval (importname + "." + primdef[1] + "()")
                # set filename and directory name
                # used by other parts of the system for naming convention based retrieval
                # i.e. of parameters
                primset.astrotype = astrotype
                primset.acquireParamDict()
                primlist.append(primset)
            return primlist
        else:
            return None
        
    def composeRecipe(self, name, recipebuffer):
        templ = """
def %(name)s(self,cfgObj):
    print "${BOLD}RECIPE BEGINS: %(name)s${NORMAL}"
    recipeLocalParms = cfgObj.localparms
%(lines)s
    print "${BOLD}RECIPE ENDS:   %(name)s${NORMAL}"
    yield cfgObj
"""
        recipelines = recipebuffer.splitlines()
        lines = ""
        
        for line in recipelines:
            # remove comments
            line = re.sub("#.*?$", "",line)
            # strip whitespace
            line = line.strip()
            
            # PARSE PRIMITIVE ARGUMENT LIST
            # take parenthesis off, make arg dict with it
            m = re.match("(?P<prim>.*?)\((?P<args>.*?)\)$", line)
            d = {}
            if m:
                prim = m.group("prim")
                args = m.group("args")
                elems = args.split(",")
                for elem in elems:
                    selem = elem.strip()
                    if "=" in selem:
                        parmname, parmval = elem.split("=")
                        parmname = parmname.strip()
                        parmval = parmval.strip()
                        # remove quotes which are not needed but intuitive around strings
                        if parmval[0] == '"' or parmval[0] == "'":
                            parmval = parmval[1:]
                        if parmval[-1] == '"' or parmval [-1] == "'":
                            parmval = parmval[:-1]
                        d.update({parmname:parmval})
                    else:
                        if len(selem)>0:
                            d.update({selem:True})
                line = prim
            # need to add dictionary to context
            
            #print "RM778:", line
            if line == "" or line[0] == "#":
                continue
            newl = """
            
    if "%(line)s" in recipeLocalParms:
        dostep = (recipeLocalParms["%(line)s"].lower() != "false")
    else:
        dostep = True
    if dostep:
        cfgObj.localparms = eval('''%(parms)s''')
        #cfgObj.localparms.update(recipeLocalParms)
        # add parms specified
        for pkey in cfgObj.localparms:
            val = cfgObj.localparms[pkey]
            if val[0]=="[" and val[-1]=="]":
                vkey = val[1:-1]
                if vkey in recipeLocalParms:
                    cfgObj.localparms[pkey] = recipeLocalParms[vkey]
        for co in self.substeps('%(line)s', cfgObj):
            if (co.isFinished()):
                break
            yield co
    yield co""" % {"parms":repr(d),
                    "line":line}
            lines += newl
            
        rets = templ % {    "name" : name,
                            "lines" : lines,
                            }
        return rets
        
    def compileRecipe(self, name, recipeinpython):
        exec(recipeinpython)
        func = eval(name)
        return func
        
    def bindRecipe(self, redobj, name, recipefunc):
        rprimset = redobj.newPrimitiveSet(redobj.curPrimType, btype="RECIPE")
        bindstr = "rprimset.%s = new.instancemethod(recipefunc, redobj, None)" % name
        exec(bindstr)
        redobj.addPrimSet(rprimset)
        return redobj
    
    def checkMethod(self, redobj, primitivename):
        ps = redobj.getPrimSet(primitivename)
        if ps == None:
            # print "RM1382: %s doesn't exist" % primitivename
            # then this name doesn't exist
            return False
        else:
            # print "RM1382: %s does exist" % primitivename
            return True
        
    def checkAndBind(self, redobj, name, context=None):
        dir (redobj)
        # print "RM1389:", name
        if self.checkMethod(redobj, name):
            return False
        else:
            # print "RM1078:", str(dir(context.inputs[0]))
            self.loadAndBindRecipe(redobj, name, astrotype = redobj.curPrimType)
            return True

    def getApplicableParameters(self, dataset):
        '''
        
        '''
        explicitType = None
        if  type(dataset) == str:
            if os.path.exists(datset):
                # then it's a file
                astrod = AstroData(dataset)
                byfname = True
            else:
                explicitType = dataset
        elif type(dataset) == AstroData:
            byfname = False
            astrod = dataset
        else:
            raise BadArgument()
        
        # get the types
        if explicitType:
            types = [explicitType]
        else:
            types = astrod.getTypes()
            
        # look up recipes, fill list
        reclist = []
        recdict = {}
        #print "RM 695:", centralAstroTypeParametersIndex.keys()
        for typ in types:
            if typ in centralAstroTypeParametersIndex.keys():
                recnames = centralAstroTypeParametersIndex[typ]
                reclist.extend(recnames)
                recdict.update({typ: recnames})
        print reclist
        return reclist

    def retrieveParameters(self, dataset, contextobj, name):
        '''
        
        '''
        raise "this is old code which needs removing"
        # Load defaults
        print "RM1364: here"
        defaultParamFiles = self.getApplicableParameters(dataset)
        print "RM1365", defaultParamFiles
        #print "RM836:", defaultParamFiles
        for defaultParams in defaultParamFiles:
            contextobj.update(centralParametersIndex[defaultParams])
        
        """
        #print "RM841:", redobj.values()
        # Load local if it exists
        if centralParametersIndex.has_key( name ):
            for recKey in centralParametersIndex[name]:
                if recKey in contextobj.keys():
                    if contextobj[recKey].overwrite:
                        # This code looks a little confusing, but its purpose is to make sure
                        # everything in the default, except the value, is the same.
                        contextobj[recKey].value = centralParametersIndex[name][recKey].value
                    else:
                        print "Attempting to overwrite Parameter '" + str(recKey) + "'. This is not allowed."
                else:
                    print "Parameter '"+ str(recKey) + "' was not found. Adding..."
                    userParam = centralParametersIndex[name][recKey]
                    updateParam = PrimitiveParameter( userParam.name, userParam.value, overwrite=True, help="User Defined.")
                    contextobj.update( {recKey:updateParam} )
        """
      

# CODE THAT RUNS ON IMPORT
# THIS MODULE ACTS AS A SINGLETON FOR RECIPE FEATURES

# NOTE: The issue of a central service for recipes implies a need for
# a singleton as with the ClassificationLibrary and the Descriptors.py module.
# I have adopted the module-as-singleton approach for Structures as it does
# not involve the message try-instantiate-except block used in the 
# ClassificationLibrary.  I'm checking into
# possible complications but it seems acceptable python.

#: recipeIndexREMask used to identify which files by filename
#: are those with tables relating type names to structure types
primitivesIndexREMask = r"primitivesIndex\.(?P<modname>.*?)\.py$"
recipeIndexREMask = r"recipeIndex\.(?P<modname>.*?)\.py$"
parameterIndexREMask = r"parametersIndex\.(?P<modname>.*?)\.py$"
#theorectically could be automatically correlated by modname

reductionObjREMask = r"primitives_(?P<redname>.*?)\.py$"


recipeREMask = r"recipe\.(?P<recipename>.*?)$"
recipeAstroTypeREMask = r"(?P<recipename>.*?)\.(?P<astrotype>.*?)$"

parameterREMask = r"parameters\.(?P<recipename>.*?)\.py$"


import os, sys, re

if True: # was firstrun logic... python interpreter makes sure this module only runs once already

    # WALK the directory structure
    # add each directory to the sytem path (from which import can be done)
    # and exec the structureIndex.***.py files
    # These indexes are meant to append it to the centralDescriptorIndex
            
    for root, dirn, files in ConfigSpace.configWalk("recipes"):
        root = os.path.abspath(root)
        #print "RM840:", root
        sys.path.append(root)
        for sfilename in files:
            m = re.match(recipeREMask, sfilename)
            mpI = re.match(primitivesIndexREMask, sfilename)
            mri = re.match(recipeIndexREMask, sfilename)
            mro = re.match(reductionObjREMask, sfilename) 
            mpa = re.match(parameterREMask, sfilename)
            mpaI = re.match(parameterIndexREMask, sfilename)
            fullpath = os.path.join(root, sfilename)
            #print "RM1026 FULLPATH", fullpath 
            if m:
                # this is a recipe file
                recname = m.group("recipename")
                if False:
                    print sfilename
                    print "complete recipe name(%s)" % m.group("recipename")
                # For duplicate recipe names, add extras.
                if centralRecipeIndex.has_key(recname):
                    # check if the paths are really the same file
                    if os.path.abspath(fullpath) != os.path.abspath(centralRecipeIndex[recname]):

                        print "-" * 35 + " WARNING " + "-" * 35
                        print "There are two recipes with the same name."
                        print "The duplicate:"
                        print fullpath
                        print "The Original:"
                        print centralRecipeIndex[recname]
                        print
                        raise RecipeExcept("Two Recipes with the same name.")
                
                centralRecipeIndex.update({recname: fullpath})
                
                am = re.match(recipeAstroTypeREMask, m.group("recipename"))
                # print str(am)
                if False: # am:
                    print "recipe:(%s) for type:(%s)" % (am.group("recipename"), am.group("astrotype"))
            elif mpI: # this is an primitives index
                efile = open(fullpath, "r")
                exec (efile)
                efile.close()
                cpis = set(centralPrimitivesIndex.keys())
                cpi = centralPrimitivesIndex
                try:
                    lpis = set(localPrimitiveIndex.keys())
                    lpi = localPrimitiveIndex
                except NameError:
                    print "WARNING: localPrimitiveIndex not found in %s" % fullpath
                    continue
                intersect = cpis & lpis
                if  intersect:
                    for typ in intersect:
                        # we'll allow this
                        # @@NOTE: there may be a conflict, in which case order is used to give preference
                        # @@..    we should have a tool to check this, because really it's only OK
                        # @@..    if none of the members of the primitive set have the same name
                        # @@..    which we don't know until later, if we actually load and use the primtiveset
                        if False:
                            rs = "Multiple Primitive Sets Found for Type %s" % typ
                            rs += "\n  Primitive Index Entry from %s" % fullpath
                            rs += "\n  adds ... %s" % repr(localPrimitiveIndex[typ])
                            rs += "\n  conflicts with already present setting ... %s" % repr(centralPrimitivesIndex[typ])
                            print "${RED}WARNING:${NORMAL}\n" + rs
                for key in lpis:
                    if key not in cpis:
                        centralPrimitivesIndex.update({key:[]})
                    plist = centralPrimitivesIndex[key]
                    val = lpi[key]
                    if type(val) == tuple:
                        plist.append(localPrimitiveIndex[key])
                    else:
                        plist.extend(val)
                           
            elif mro: # reduction object file... contains  primitives as members
                centralReductionMap.update({sfilename: fullpath})
            elif mri: # this is a recipe index
                efile = open(fullpath, "r")
                # print "RM1559:", fullpath
                # print "RM1560:before: cri", centralRecipeIndex
                # print "RM1561:before: catri,", centralAstroTypeRecipeIndex
                # print fullpath
                exec efile
                efile.close()
                for key in localAstroTypeRecipeIndex.keys():
                    if centralRecipeIndex.has_key(key):
                        curl = centralRecipeIndex[key]
                        curl.append(localAstroTypeRecipeIndex[key])
                        localAstroTypeRecipeIndex.update({key: curl})
                    if key in centralAstroTypeRecipeIndex:
                        ls = centralAstroTypeRecipeIndex[key]
                    else:
                        ls = []
                        centralAstroTypeRecipeIndex.update({key:ls})
                        
                    ls.extend(localAstroTypeRecipeIndex[key])
                # print "RM1570:after: cri", centralRecipeIndex
                # print "RM1571:after: catri,", centralAstroTypeRecipeIndex
            elif mpa: # Parameter file
                efile = open(fullpath, "r")
                exec(efile)
                efile.close()
                recname = mpa.group("recipename")
                centralParametersIndex.update({recname:localParameterIndex})
            elif mpaI: # ParameterIndex file
                efile = open(fullpath, "r")
                exec(efile)
                efile.close()
                #for key in localparameterTypeIndex.keys():
                #    if centralParametersIndex.has_key(key):
                #        curl = centralParametersIndex[key]
                #        curl.append( localparameterTypeIndex[key])
                #        localparameterTypeIndex.update({key: curl})
                 
                centralAstroTypeParametersIndex.update(localparameterTypeIndex)
                
                
            # look for recipe
            # 
        
    if False:
        print "----- DICTIONARIES -----"
        print str(centralRecipeIndex)
        print str(centralAstroTypeRecipeIndex)
        print str(centralPrimitivesIndex)
        print str(centralReductionMap)
        print "--EOF DICTIONARIES EOF--"
    
        
        
    if False:
            # (re.match(structureIndexREMask, sfilename)):
                fullpath = os.path.join(root, sfilename)
                siFile = open(fullpath)
                exec siFile
                siFile.close()
                # file must declare structureIndex = {...}, keys are types, 
                # values are string names of structure classes that can
                # be instantiated when needed (should refer to modules
                # and classes in structures subdirectory, all of which is
                # in the import path.
                
                # note: make sure one index does not stomp another
                # Means misconfigured structureIndex.
                
                for key in structureIndex.keys():
                    if centralStructureIndex.has_key(key):
                        # @@log
                        msg = "Scructure Index CONFLICT\n"
                        msg += "... structure for type %s\n" % key
                        msg += "redefined in\n" 
                        msg += "... %s\n" % fullpath
                        msg += "... was already set to %s\n" % centralStructureIndex[key]
                        msg += "... this is a fatal error"
                        raise StructureExcept(msg)
                        
                centralStructureIndex.update(structureIndex)


