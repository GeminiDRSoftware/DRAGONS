from AstroData import AstroData
import AstroDataType

import new
import socket # to get host name for local statistics
import ReductionObjects
from ReductionObjects import ReductionObject
import ConfigSpace
from gdpgutil import pickConfig

from datetime import datetime
from copy import deepcopy, copy

from CalibrationDefinitionLibrary import CalibrationDefinitionLibrary
from ReductionContextRecord import CalibrationRecord, StackableRecord, OutputRecord
import pickle # for persisting the calibration index

import IDFactory as idFac # id hashing functions
from ReductionObjectRequests import UpdateStackableRequest, GetStackableRequest, DisplayRequest
from StackKeeper import StackKeeper
from ParamObject import PrimitiveParameter
# this module operates like a singleton
centralPrimitivesIndex = {}
centralRecipeIndex = {}
centralReductionMap = { }
centralAstroTypeRecipeIndex = {}
centralParametersIndex = {}
centralAstroTypeParametersIndex = {}

class RecipeExcept:
    """ This is the general exception the classes and functions in the
    Structures.py module raise.
    """
    def __init__(self, msg="Exception Raised in Recipe System"):
        """This constructor takes a message to print to the user."""
        self.message = msg
    def __str__(self):
        """This str conversion member returns the message given by the user (or the default message)
        when the exception is not caught."""
        return self.message

class ReductionContext(dict):
    """The ReductionContext is used by primitives and recipies, hidden in the later case,
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
    
    def persistCalIndex(self, filename):
        #print "Calibration List Before Persist:"
        #print self.calsummary()
        try:
            pickle.dump(self.calibrations, open(filename, "w"))
        except:
            print "Could not persist the calibration cache."
            raise 
    
    def restoreCalIndex(self, filename):
        if os.path.exists( filename ):
            self.calibrations = pickle.load( open(filename, 'r') )
        else:
            pickle.dump( {}, open( filename, 'w' ) )
    
    def persistStkIndex(self, filename ):
        try:
            #print "RM80:", self.stackeep
            pickle.dump( self.stackeep.stackLists, open(filename, "w") )
        except:
            print "Could not persist the stackable cache."
            raise
    
    def restoreStkIndex( self, filename ):
        '''
        
        '''
        if os.path.exists( filename ):
            self.stackeep.stackLists = pickle.load( open(filename, 'r') )
        else:
            pickle.dump( {}, open( filename, 'w' ) )
            
    def calsummary(self, mode = "text"):
        rets = ""
        for key in self.calibrations.keys():
            rets += str(key)
            rets += str(self.calibrations[key])
        return rets
    
    def paramsummary(self):
        char = "-"
        rets = char*40+"\n"
        rets += '''------Global Parameters------\n'''
        
        globval = "global"
        
        def printParam( val, param ):
            # This temp function prints out the stuff inside an individual parameter.
            # I have a feeling this and paramsummary will be moved to a util function.
            tempStr = ""
            list_of_keys = param.keys()
            list_of_keys.sort()
            tempStr += char*40 + "\n"
            for pars in list_of_keys:
                tempStr += str(param[pars]) + "\n"
                tempStr += char*40 + "\n"
            return tempStr
            
        rets += printParam( globval, self[globval])
        sortkeys = self.keys().sort()
        for primname in self.keys():
            if primname != globval:
                rets += '''------%(prim)s Parameters------\n''' %{'prim':primname}
                rets += printParam( primname, self[primname] )
        
        return rets
    
    def prepDisplay(self):
        pass
    
    
    def stack_inputsAsStr(self):        
        #pass back the stack files as strings            
        ID = idFac.generateStackableID(self.inputs)        
        print 'ID: ', ID
        stack = self.stackeep.get(ID)
        print 'stack_inputsAsStr returns:  ', ", ".join(stack.filelist) 
        return ", ".join(stack.filelist)
    
    def makeInlistFile(self):
        fh = open( "inlist","w" )
        ID = idFac.generateStackableID(self.inputs)
        stack = self.stackeep.get(ID)
        for item in stack.filelist:
            fh.writelines(item + '\n')        
        return "@inlist"
 
 
 
 
 #################################################################################
 
 
 
    
    def __init__(self):
        """The ReductionContext constructor creates empty dictionaries and lists, members set to
        None in the class."""
        self.inputs = []
        self.inputsHistory = []
        self.calibrations = {}
        self.rorqs = []
        self.outputs = {"standard":[]}
        self.stephistory = {}
        self.hostname = socket.gethostname()
        self.displayName = None
        # TESTING
        self.cdl = CalibrationDefinitionLibrary()
        # undeclared
        self.indent=0 
        
        # Stack Keep is a resource for all RecipeManager functions... one shared StackKeeper to simulate the shared ObservationServie
        # used in PRS mode.
        self.stackeep = StackKeeper()
        
    def __str__(self):
        """Used to dump Reduction Context(co) into file for test system
        """
        tempStr = ""
        tempStr = tempStr + "REDUCTION CONTEXT OBJECT (CO)\n" + \
            "inputs = " + str( self.inputs ) + \
            "\ninputsHistory =  " + str( self.inputsHistory )+ \
            "\ncalibrations = \n" + self.calsummary() + \
            "\nrorqs = " 
        if self.rorqs != []:
            for rq_obj in self.rorqs:            
                tempStr = tempStr + str( rq_obj )
        else:
            tempStr = tempStr + str( self.rorqs )
        
        #no loop initiated for stkrqs object printouts yet
        tempStr = tempStr + "\noutputs = " 
        
        if self.outputs["standard"] != []:
            for out_obj in self.outputs["standard"]:
                tempStr = tempStr + str( out_obj )
        else:
            tempStr = tempStr + str( self.outputs )
        #"stephistory = " + str( self.stephistory ) + \
        tempStr = tempStr +  "\nhostname = " + str( self.hostname ) + \
            "\ndisplayName = " + str( self.displayName ) + \
            "\ncdl = " + str( self.cdl ) + \
            "\nindent = " + str( self.indent ) + \
            "\nstackeep = " + str( self.stackeep )
        for param in self.values():
            tempStr += "\n" + self.paramsummary()
             
        return tempStr
               
    
    def stackAppend(self, ID, files):
        self.stackeep.add( ID, files )
    
    def getStack(self, ID=None):
        if ID == None:
            ver = "1_0"
            # Not sure how version stuff is going to be done. This version stuff is temporary.
            ID = idFac.generateStackableID( self.inputs, ver )
        return self.stackeep.get(ID)
        
    def isFinished(self,arg = None):
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

    def isPaused(self, bpaused = None):
        if bpaused == None:
            return self.status == "PAUSED"
        else:
            if bpaused:
                self.status = "PAUSED"
            else:
                self.status = "RUNNING"
        
        return self.isPaused()

    def pause(self):
        self.isPaused(True)
    def unpause (self):
        self.isPaused(False)

    paused = property(isPaused, isPaused)

    def processCmdReq(self):
        if self.cmdRequest == "pause":
            self.cmdRequest = "NONE"
            self.pause()


    def getEndMark(self, stepname, indent= None):
        for time in self.stephistory.keys():
            if     self.stephistory[time]["stepname"] == stepname \
               and self.stephistory[time]["mark"] == "end":
                if indent != None:
                    if self.stephistory[time]["indent"] == indent:
                        return (time,self.stephistory[time])
                else:
                    return (time,self.stephistory[time])
                
    
        return None
        
    def getBeginMark(self, stepname, indent=None):
        for time in self.stephistory.keys():
            if     self.stephistory[time]["stepname"] == stepname \
               and self.stephistory[time]["mark"] == "begin":
                    if indent != None:
                        if self.stephistory[time]["indent"] == indent:
                            return (time,self.stephistory[time])
                    else:
                        return (time,self.stephistory[time])    
        return None

        
    def control(self, cmd = "NONE"):
        self.cmdRequest = cmd

    def requestPause(self):
        self.control("pause")

    def pauseRequested(self):
        return self.cmdRequest == "pause"
        
    def checkControl(self):
        return self.cmdRequest        
        
    def addInput(self, filename):
        if type(filename) == list:
            self.inputs.extend(filename)
        else:
            self.inputs.append(filename)
        # This is kluge and needs to change as we potentially deal with lists
        if self.originalInputs == None:
            self.originalInputs = self.inputs
        
        
    def reportOutput(self, inp, category="standard"):
        # note, other categories not supported yet
        if category != "standard":
            raise RecipeExcept("You may only use " +
                "'standard' category output at this time.")
        if type(inp) == str:
            self.outputs["standard"].append( OutputRecord(inp,self.displayID) )
        elif type(inp) == list:
            for temp in inp:
                #print "RM287:", type(temp), temp
                if type(temp) == AstroData:
                    orecord = OutputRecord( temp.filename, self.displayID, temp )
                elif type(temp) == str:
                    orecord = OutputRecord( temp, self.displayID )
                else:
                    raise "RM292 type: " + str(type(temp))
                self.outputs["standard"].append( orecord )
            
    
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
                if type( out ) == OutputRecord:
                    newinputlist.append( out.ad )
                else:
                    newinputlist.append( out )
            
            self.inputs = newinputlist
            #for temp in self.outputs["standard"]:
            #   print "TEMP:", temp
            self.outputs.update({"standard":[]})
            
    
    def prependNames(self, prepend, currentDir = True):
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
        newlist = []
        for nam in self.inputs:
            if type( nam ) == AstroData:
                prePath = nam.filename
            else:
                prePath = nam
            
            if currentDir == True:
                path = os.getcwd()
            else:
                path = os.path.dirname(prePath)
            # nam is an astrodata instance.
            #print "RM337:", type(prePath), prePath
            fn   = os.path.basename(prePath)
            newpath = path + "/" + prepend + fn
            newlist.append(newpath)
        return newlist
    
    def suffixNames(self, suffix, currentDir=True):
        '''
        
        '''
        newlist = []
        for nam in self.inputs:
            if currentDir == True:
                path = os.getcwd()
            else:
                path = os.path.dirname(nam.filename)
            
            fn   = os.path.basename(nam.filename)
            finame, ext = os.path.splitext(fn)
            fn = finame + "_" + suffix + ext
            newpath = os.path.join( path, fn ) 
            newlist.append(newpath)
        return newlist
    
    def stepMoment(self, stepname, mark):
        val = { "stepname"  : stepname,
                "indent"    : self.indent,
                "mark"      : mark,
                "inputs"    : copy(self.inputs),
                "outputs"   : copy(self.outputs),
                "processed" : False
                }
        return val       
        
    def begin(self, stepname):
        key = datetime.now()
        # value = dictionary
        val = self.stepMoment(stepname, "begin")
        self.indent += 1
        self.stephistory.update({key: val}) 
        self.lastBeginDt = key
        return self
                
    def end(self,stepname):
        key = datetime.now()
        self.indent -= 1
        val = self.stepMoment(stepname,"end")
        # this step saves inputs
        self.stephistory.update({key: val})
        # this step moves outputs["standard"] to inputs
        # and clears outputs
        self.finalizeOutputs()
        return self
                
    def inputsAsStr(self, strippath = True):
        if self.inputs == None:
            return ""
        else:
            #print "RM282:", self.inputs
            #@@TODO: Quick-fix for getting ad stuff working, may need re-visit.
            # This is a quick fix to deal with the initial string / 
            # OutputRecord stuff. The first input (which is str),
            # all ensuing output is OutputRecord [which has metadata]
            # -Riv
            #"""
            inputlist = []
            for inp in self.inputs:
                if type(inp) == str:
                    inputlist.append( inp )
                elif type(inp) == AstroData:
                    inputlist.append( inp.filename )
            #print "RM289:", inputlist
            #"""
            if strippath == False:
                # print "RM227:", self.inputs
                return ", ".join( inputlist )                
            else:
                return ", ".join([os.path.basename(path) for path in inputlist])
                                      

    def outputsAsStr(self, strippath = True):
        if self.outputs == None:
            return ""
        else:
            #print "RM282:", self.inputs
            #@@TODO: Quick-fix for getting ad stuff working, may need re-visit.
            # This is a quick fix to deal with the initial string / 
            # OutputRecord stuff. The first input (which is str),
            # all ensuing output is OutputRecord [which has metadata]
            # -Riv
            #"""
            outputlist = []
            for inp in self.outputs:
                if type(inp) == str:
                    outputlist.append( inp )
                elif type(inp) == OutputRecord:
                    outputlist.append( inp.filename )
            print "RM289:", outputlist
            #"""
            if strippath == False:
                # print self.inputs
                return ", ".join(outputlist)
            else:
                return ", ".join([os.path.basename(path) for path in outputlist])
        
    def addCal(self, fname, caltyp, calname, timestamp = None):
        fname = os.path.abspath(fname)
        calname = os.path.abspath(calname)
        
        if timestamp == None:
            timestamp = datetime.now()
        else:
            timestamp = timestamp
        
        if self.calibrations == None:
            self.calibrations = {}
        
        calrec = CalibrationRecord(fname, calname, caltyp, timestamp)
        key = (fname, caltyp)
        self.calibrations.update({key: calrec})
    
    def getCal(self, filename, caltype):
        filename = os.path.abspath(filename)
        key = (filename, caltype)
        if key in self.calibrations.keys():
            return self.calibrations[(filename,caltype)].filename
        return None
        
    def addRq(self, rq):
        if self.rorqs == None:
            self.rorqs = []
        self.rorqs.append(rq)
    
    def clearRqs(self):
        self.rorqs = []
        
    def rqCal(self, caltype): 
        addToCmdQueue = self.cdl.getCalReq( self.originalInputs, caltype )
        for re in addToCmdQueue:
            self.addRq(re)
        
    def rqStackUpdate(self):
        ver = "1_0"
        # Not sure how version stuff is going to be done. This version stuff is temporary.
        Sid = idFac.generateStackableID( self.originalInputs, ver )
        stackUEv = UpdateStackableRequest()
        stackUEv.stkID = Sid
        stackUEv.stkList = self.originalInputs
        self.addRq( stackUEv )
        
    def rqDisplay(self):
        ver = "1_0"
        Did = idFac.generateDisplayID( self.inputs,ver )
        displayObject = DisplayRequest()
        displayObject.disID = Did
        displayObject.disList = self.inputs
        self.addRq( displayObject )
        
        
    def rqStackGet(self):
        ver = "1_0"
        # Not sure how version stuff is going to be done. This version stuff is temporary.
        Sid = idFac.generateStackableID( self.originalInputs, ver )
        stackUEv = GetStackableRequest()
        stackUEv.stkID = Sid
        self.addRq( stackUEv )
    
    def calFilename(self, caltype):
        """returns a local filename for a retrieved calibration"""
        if self.originalInputs == None:
            self.originalInputs = deepcopy(self.inputs)
        if len(self.originalInputs) == 0:
            return None
        elif len(self.originalInputs) == 1:
            fname = os.path.abspath( self.originalInputs[0] )
            key = (fname, caltype)
            return self.calibrations[key].filename
        else:
            retl = []
            for inp in self.originalInputs:
                key = (inp, caltype)
                retl.append(self.calibrations[key])
            return retl
        
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
            indentstr = "".join(["  " for i in range(0,indent)])
            
            mark = sh[dt]["mark"]
            if mark == "begin":
                elapsed = ""
                format = "%(indent)s%(stepname)s begin at %(time)s"
            elif mark == "end":
                elapsed = "("+str(dt-lastdt)+") "
                format="\x1b[1m%(indent)s%(stepname)s %(elapsed)s \x1b[22mends at %(time)s"
            else:
                elapsed = ""
                format = "%(indent)s%(stepname)s %(elapsed)s%(mark)s at %(time)s"
                
            lastdt = dt
            if startdt== None:
                startdt = dt

            pargs =  {  "indent":indentstr,
                        "stepname":str(sh[dt]['stepname']), 
                        "mark":str(sh[dt]['mark']),
                        "inputs":str(",".join(sh[dt]['inputs'])),
                        "outputs":str(sh[dt]['outputs']),
                        "time":str(dt),
                        "elapsed":elapsed,
                        "runtime":str(dt-startdt),
                    }
            retstr += format % pargs + "\n"
            retstr += "%(indent)sTOTAL RUNNING TIME: %(runtime)s (MM:SS:ms)" % pargs  + "\n"
       
        startdt = None
        lastdt = None
        enddt = None
        wide = 75
        retstr +=  "\n\n"
        retstr +=  "SHOW IO".center(wide)  + "\n"
        retstr +=  "-------".center(wide) + "\n"
        retstr +=  "\n"
        for dt in ks: # self.stephistory.keys():
            indent = sh[dt]["indent"]
            indentstr = "".join(["  " for i in range(0,indent)])
            
            mark = sh[dt]["mark"]
            if mark == "begin":
                elapsed = ""
            elif mark == "end":
                elapsed = "("+str(dt-lastdt)+") "
                
            pargs =  {  "indent":indentstr,
                        "stepname":str(sh[dt]['stepname']), 
                        "mark":str(sh[dt]['mark']),
                        "inputs":str(",".join(sh[dt]['inputs'])),
                        "outputs":str(",".join(sh[dt]['outputs']['standard'])),
                        "time":str(dt),
                        "elapsed":elapsed,
                    }
            if startdt == None:
                retstr +=  ("%(inputs)s" % pargs).center(wide) + "\n"

            if (pargs["mark"] == "end"):
                retstr +=  " | ".center(wide) + "\n"
                retstr +=  "\|/".center(wide) + "\n"
                retstr +=  " ' ".center(wide) + "\n"
                
                line = ("%(stepname)s" % pargs).center(wide)
                line = "\x1b[1m" + line + "\x1b[22m"  + "\n"
                retstr +=  line
                
            if len(sh[dt]["outputs"]["standard"]) != 0:
                retstr +=  " | ".center(wide) + "\n"
                retstr +=  "\|/".center(wide) + "\n"
                retstr +=  " ' ".center(wide) + "\n"
                retstr +=  ("%(outputs)s" % pargs).center(wide) + "\n"
                
                
            lastdt = dt
            if startdt== None:
                startdt = dt
        
        return retstr

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
        pair = {key: {"source":source,"start":start, "end":end}}
        self.primLoadTimes.update(pair)
        
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

    def loadAndBindRecipe(self,ro, name, file=None, astrotype=None):
        """
        Will load a single recipe, compile and bind it to the given reduction objects
        """
        # NOTE: sort out precedence of one type over another
        # in all cases.
        #print "RM636: KAPLAH"
        if astrotype != None:
            # get recipe source
            rec = self.retrieveRecipe(name, astrotype= astrotype)
    
            if rec:
                # compose to python source
                prec = self.composeRecipe(name, rec)
                # compile to unbound function (using the python interpretor obviously)
                rfunc = self.compileRecipe(name, prec)
                # bind the recipe to the reduction object
                ro = self.bindRecipe(ro, name, rfunc)
        elif file != None:
            gd, bnc = openIfName(file)
            types = gd.getTypes()
            for typ in types:
                rec   = self.retrieveRecipe(name, astrotype= typ)
                if rec:
                    prec  = self.composeRecipe(name, rec)
                    rfunc = self.compileRecipe(name, prec)
                    ro = self.bindRecipe(ro, name, rfunc)
        
            closeIfName(gd, bnc)
            

    def getApplicableRecipes(self, dataset, collate = False):
        """
        Get list of recipes associated with all the types that apply to this dataset.
        """
        if  type(dataset) == str:
            astrod = AstroData(dataset)
            byfname = True
        elif type(dataset) == AstroData:
            byfname = false
            astrod = dataset
        else:
            raise BadArgument()

        # get the types
        types = astrod.getTypes()
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
        cri = centralRecipeIndex
        if astrotype:
            akey = name+"."+astrotype
            key = name 
        else:
            key = name
            akey = name+".None"

        bdefRecipe = key in cri
        bastroRecipe = akey in cri
        
        fname = None
        if bastroRecipe:
            fname = cri[akey]
        elif bdefRecipe:
            fname = cri[key]
        else:
            return None

        rfile = file(fname, "r")
        rtext = rfile.read()
        #print "RM718:", rtext
        return rtext
            
    def retrieveReductionObject(self, dataset = None, astrotype=None):
        a = datetime.now()
        
        # if astrotpye is None, but dataset is set, then we need to get the astrotype from the 
        # dataset.  For reduction objects, there can be only one assigned to a real object
        # if there are multiple reduction objects associated with type we must find out through
        # inheritance relationships which one applies. E.g. if a dataset is GMOS_SPEC and
        # GMOS_IFU, then an inheritance relationship is sought, and the child type has priority.
        # If they cannot be resolved, because there are unrelated types or through multiple
        # inheritance multiple ROs may apply, then we raise an exceptions, this is a configuration
        # problem.
        
        if (astrotype == None) and (dataset != None):
            val = pickConfig(dataset, centralPrimitivesIndex)
            k = val.keys()
            if len(k) > 1:
                raise RecipeExcept("CAN'T RESOLVE PRIMITIVE SET CONFLICT")
            astrotype = k[0]
            
        if (astrotype != None) and (astrotype in centralPrimitivesIndex):
            rfilename = centralPrimitivesIndex[astrotype][0]
            rpathname = centralReductionMap[rfilename]
            rootpath = os.path.dirname(rpathname)
            importname = os.path.splitext(rfilename)[0]
            a = datetime.now()
            exec ("import " + importname)
            b = datetime.now()
            ro = eval (importname+"."+centralPrimitivesIndex[astrotype][1]+"()")
            c = datetime.now()
        else:
            ro = ReductionObjects.ReductionObject()
            raise ("Tried to retrieve base Reduction Object,\n" + \
                   "not allowed at this time." )
            
        ro.recipeLib = self
        
        b = datetime.now()
        if astrotype != None:
            source = "TYPE: " + astrotype
        elif dataset != None:
            source = "FILE: " + dataset
        else:
            source = "UNKNOWN"
            
        #@@perform: monitory real performance loading primitives
        self.addLoadTime(source, a, b)
        return ro
        
    def composeRecipe(self, name, recipebuffer):
        templ = """
def %(name)s(self,cfgObj):
\tprint "${BOLD}RECIPE BEGINS: %(name)s${NORMAL}"
%(lines)s
\tprint "${BOLD}RECIPE ENDS:   %(name)s${NORMAL}"
\tyield cfgObj
"""
        recipelines = recipebuffer.splitlines()
        lines = ""
        
        for line in recipelines:
            line = line.strip()
            #print "RM778:", line
            if line == "" or line[0]=="#":
                continue
            newl =  """
\tfor co in self.substeps('%s', cfgObj):
\t\tyield co""" % line
            lines += newl
            
        rets = templ % {    "name" : name,
                            "lines" : lines }
        
        return rets
        
    def compileRecipe(self, name, recipeinpython):
        exec(recipeinpython)
        func = eval(name)
        return func
        
    def bindRecipe(self, redobj, name, recipefunc):
        bindstr = "redobj.%s = new.instancemethod(recipefunc, redobj, None)" % name
        exec(bindstr)
        return redobj
    
    def checkMethod(self, redobj, primitivename):
        methstr = "redobj.%s" % primitivename
        try:
            # print "RM647:",methstr ,"EORM647"
            func = eval(methstr)
        except AttributeError:
            # then it does not exist
            return False
        
        return True
        
    def checkAndBind(self, redobj, name, context = None):
        dir (redobj)
        if self.checkMethod(redobj, name):
            return False
        else:
            self.loadAndBindRecipe(redobj, name, file=context.inputs[0])
            return True

    def getApplicableParameters(self, dataset):
        '''
        
        '''
        if  type(dataset) == str:
            astrod = AstroData(dataset)
            byfname = True
        elif type(dataset) == AstroData:
            byfname = false
            astrod = dataset
        else:
            raise BadArgument()
        
        # get the types
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
    
        return reclist

    def retrieveParameters(self, dataset, contextobj, name):
        '''
        
        '''
        # Load defaults
        defaultParamFiles = self.getApplicableParameters(dataset)
        #print "RM836:", defaultParamFiles
        for defaultParams in defaultParamFiles:
            contextobj.update( centralParametersIndex[defaultParams] )
        
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



import os,sys,re

if True: # was firstrun logic... python interpreter makes sure this module only runs once already

    # WALK the directory structure
    # add each directory to the sytem path (from which import can be done)
    # and exec the structureIndex.***.py files
    # These indexes are meant to append it to the centralDescriptorIndex

    for root, dirn, files in ConfigSpace.configWalk("recipes"):
        #print "RM840:", root
        sys.path.append(root)
        for sfilename in files:
            m = re.match(recipeREMask, sfilename)
            mpI = re.match(primitivesIndexREMask, sfilename)
            mri = re.match(recipeIndexREMask, sfilename)
            mro = re.match(reductionObjREMask,sfilename) 
            mpa = re.match(parameterREMask, sfilename)
            mpaI = re.match(parameterIndexREMask, sfilename)
            fullpath = os.path.join(root, sfilename)
            
            if m:
                recname = m.group("recipename")
                if False:
                    print sfilename
                    print "complete recipe name(%s)" % m.group("recipename")
                # For duplicate recipe names, until another solution is decided upon.
                if centralRecipeIndex.has_key( recname ):
                    print "-"*35+" WARNING "+"-"*35
                    print "There are two recipes with the same name."
                    print "The duplicate:"
                    print fullpath
                    print "The Original:"
                    print centralRecipeIndex[recname]
                    print
                    raise RecipeExcept( "Two Recipes with the same name." )
                
                centralRecipeIndex.update({recname: fullpath})
                
                am = re.match(recipeAstroTypeREMask, m.group("recipename"))
                # print str(am)
                if False: # am:
                    print "recipe:(%s) for type:(%s)" % (am.group("recipename"), am.group("astrotype"))
            elif mpI: # this is an primitives index
                efile = open(fullpath,"r")
                exec (efile)
                efile.close()
                centralPrimitivesIndex.update(localPrimitiveIndex)
            elif mro: # reduction object file... contains  primitives as members
                centralReductionMap.update({sfilename: fullpath})
            elif mri: # this is a recipe index
                efile = open(fullpath, "r")
                # print fullpath
                exec efile
                efile.close()
                for key in localAstroTypeRecipeIndex.keys():
                    if centralRecipeIndex.has_key(key):
                        curl = centralRecipeIndex[key]
                        curl.append(localAstroTypeRecipeIndex[key])
                        localAstroTypeRecipeIndex.update({key: curl})
                
                centralAstroTypeRecipeIndex.update(localAstroTypeRecipeIndex)
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
                for key in localparameterTypeIndex.keys():
                    if centralParametersIndex.has_key(key):
                        curl = centralParametersIndex[key]
                        curl.append( localparameterTypeIndex[key])
                        localparameterTypeIndex.update({key: curl})
                
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
                        msg += "... was already set to %s\n" %centralStructureIndex[key]
                        msg += "... this is a fatal error"
                        raise StructureExcept(msg)
                        
                centralStructureIndex.update(structureIndex)


