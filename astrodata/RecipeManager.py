# This module operates like a singleton
import new
import os, sys
import inspect
import pickle # for persisting the calibration index
import socket # to get host name for local statistics
from copy import deepcopy, copy
from datetime import datetime
from astrodata.AstroData import AstroData
from astrodata import IDFactory
import traceback
import astrodata
import AstroDataType
import ConfigSpace
from ConfigSpace import RECIPEMARKER
# not needed and double import import Descriptors
import gdpgutil
from gdpgutil import inherit_index
import ReductionObjects
import IDFactory as idFac # id hashing functions
from Errors import ReduceError
from gdpgutil import pick_config
from ParamObject import PrimitiveParameter
from astrodata.adutils import gemLog
from AstroDataType import get_classification_library
from ReductionContextRecords import CalibrationRecord, \
    StackableRecord, AstroDataRecord, FringeRecord
from ReductionObjects import ReductionObject
from ReductionObjectRequests import UpdateStackableRequest, \
    GetStackableRequest, DisplayRequest, ImageQualityRequest
from StackKeeper import StackKeeper, FringeKeeper
# For xml calibration requests
from CalibrationDefinitionLibrary import CalibrationDefinitionLibrary
import eventsmanagers
import traceback
from primitivescat import PrimitivesCatalog

centralPrimitivesIndex = {}
centralRecipeIndex = {}
centralRecipeInfo = {}
centralReductionMap = {}
centralAstroTypeRecipeIndex = {}
centralParametersIndex = {}
centralAstroTypeParametersIndex = {}

centralPrimitivesCatalog = PrimitivesCatalog()
#------------------------------------------------------------------------------ 
MAINSTREAM = "main"
class RecipeExcept:
    """ This is the general exception the classes and functions in the
    Structures.py module raise.
    """
    def __init__(self, msg="Exception Raised in Recipe System", **argv):
        """This constructor takes a message to print to the user.
        """
        self.message = msg
        for arg in argv.keys():
            exec("self."+arg+"="+repr(argv[arg]))
            
            
    def __str__(self):
        """This str conversion member returns the message given by the user
        (or the default message) when the exception is not caught.
        """
        return self.message
        
class SettingFixedParam(RecipeExcept):
    pass

class RCBadParmValue(RecipeExcept):
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
    
    def __repr__(self):
        ret = "UserParam: adtype=%s primname=%s %s=%s" % (repr(self.astrotype),
                                          repr(self.primname),
                                          repr(self.param),
                                          repr(self.value),
                                          )
        return ret
class UserParams(object):
    user_param_dict = None
    
    def is_empty(self):
        if self.user_param_dict == None or len(self.user_param_dict.keys()) == 0:
            return True
        else:
            return False
            
    def get_user_param(self, astrotype, primname):
        if self.user_param_dict == None:
            return None
        if astrotype not in self.user_param_dict:
            return None
        if primname not in self.user_param_dict[astrotype]:
            return None
        return self.user_param_dict[astrotype][primname]
        
    def add_user_param(self, userparam):
        up = userparam
        if userparam == None:
            return
        if self.user_param_dict == None:
            self.user_param_dict = {}
            
        if up.astrotype not in self.user_param_dict:
            self.user_param_dict.update({up.astrotype: {}})
        
        if up.primname not in self.user_param_dict[up.astrotype]:
            self.user_param_dict[up.astrotype].update({up.primname: {}})
            
        if up.param in self.user_param_dict[up.astrotype][up.primname]:
            raise RecipeExcept("Parameter (%s.%s%s) already set by user" % \
            (up.astrotype, up.primname, up.param))
        else:
            self.user_param_dict[up.astrotype][up.primname].update({up.param:up.value})
    def __repr__(self):
        ret = "UserParams: "
        ret += repr(self.user_param_dict)
        return ret
        
    def __contains__(self, arg):
        if self.user_param_dict:
            return self.user_param_dict.__contains__(arg0)
            
class ReductionContext(dict):
    """The ReductionContext is used by primitives and recipies, implicitely in the
    later case, to get input and report output. This allows primitives to be
    controlled in many different running environments, from pipelines to command
    line interactive reduction.
    
    :sort: __init__,__contains__,__getitem__,__str__,
        _*,a*,b*,c*,d*,e*,f*,g*,h*,i*,j*,k*,l*,m*,n*,o*,p*,q*,r*,
        s*,t*,u*,v*,w*,x*,y*,z*
    """
    inputs = None
    _original_inputs = None
    inputs_history = None
    outputs = None
    calibrations = None
    rorqs = None
    status = "EXTANT"
    reason = "EXTANT"
    cmd_request = "NONE"
    hostname = None
    display_name = None
    stephistory = None
    stackeep = None
    calindfile = None
    display_mode = None
    display_id = None
    irafstdout = None
    irafstderr = None
    callbacks = None
    arguments = None
    cache_files = None
    
    # dictionary with local args (given in recipe as args, generally)
    _running_contexts = None
    _localparms = None 
    _nonstandard_stream = None
    _current_stream = None
    _output_streams = None
    user_params = None # meant to be UserParams instance
    proxy_id = 1 # used to ensure uniqueness
    ro = None
    cmd_history = None
    cmd_index = None
    # return behavior
    _return_command = None
    _metricEvents = None
    
    def __init__(self, adcc_mode = "start_early"):
        """The ReductionContext constructor creates empty dictionaries and
        lists, members set to None in the class.
        """
        self.running_contexts = []
        self.cmd_history = []
        self.cmd_index = {}
        self.inputs = []
        self.callbacks = {}
        self.inputs_history = []
        self.calibrations = {}
        self.rorqs = []
        self.outputs = {"main":[]}
        self.stephistory = {}
        self.hostname = socket.gethostname()
        self.display_name = None
        self.arguments = []
        self.cache_files = {}
        # TESTING
        self.cdl = CalibrationDefinitionLibrary()
        # undeclared
        self.indent = 0 
        self._metricEvents = eventsmanagers.EventsManager(rc = self)
        
        
        # Stack Keep is a resource for all RecipeManager functions... 
        # one shared StackKeeper to simulate the shared ObservationService
        # used in PRS mode.
        self.stackeep = StackKeeper(local= True if adcc_mode == "start_lazy" else False)
        self.stackKeeper = self.stackeep # "stackeep" is not a good name
        self.fringes = FringeKeeper()
        self._nonstandard_stream = []
        self._current_stream = MAINSTREAM
        self._output_streams = []
        self._adcc_mode = adcc_mode
        
    def __getitem__(self, arg):
        """Note, the ReductionContext version of __getitem__ returns None
        instead of throwing a KeyError.
        """
        if self.localparms and arg in self.localparms:
            value = self.localparms[arg]
        else:
            try:
                value = dict.__getitem__(self, arg)
            except KeyError:
                return None
        if value == None:
            retval = None
        else:
            retval = self.convert_parm_to_val(arg, value)
        return retval
    
    def __contains__(self, thing):
        """
        :param thing: A key to check for presences in the Reduction Context
        :type thing: str
        
        The __contains__ function implements the python 'in' operator. The 
        ReductionContext is a subclass of a ``dict``, but it also has a secondary
        dict of "local parameters" which are available to the current primitive \
        only, which are also tested by the ``__contains__(..)`` member.
        These parameters will generally be those passed in as arguments
        to a primitive call from a recipe.
        """
        if thing in self._localparms:
            return True
        return dict.__contains__(self, thing)
        
    def get_context(self):
        # print "RM209:",repr(self._running_contexts)
        return ":".join(self._running_contexts)
    getContext = get_context
    
    context = property(getContext)
    
    def in_context(self, context):
        context = context.lower()
        return context in self._running_contexts
    inContext = in_context
    
    def add_context(self, context):
        context = context.lower()
        if context not in self._running_contexts:
            self._running_contexts.append(context)
    addContext = add_context        
    
    def set_context(self, context):
        if type(context) == list:
            context = [cstr.lower() for cstr in context]
            self._running_contexts = context
        else:
            self._running_contexts = [context.lower()]
    setContext = set_context
    
    def clear_context(self, context = None):
        if context == None:
            self._running_contexts = []
        else:
            context = context.lower()
            if context in self._running_contexts:
                self._running_contexts.remove(context)
    clearContext = clear_context
    
    def convert_parm_to_val(self, parmname, value):
        legalvartypes = ["bool", "int", "str", "float", None]
        vartype = self.ro.parameter_prop( parmname, prop="type")
        if vartype not in legalvartypes:
            mes =  "TEMPORARY EXCEPTION: illegal type in parameter defintions"
            mes += " for %s." % str(value)
            raise reduceError(mes)
            return value
        if vartype:
            # bool needs special handling
            if vartype == "bool":
                if type(value) == str:
                    if (value.lower() == "true"):
                        value = True
                    elif (value.lower() == "false"):
                        value = False
                    else:
                        mes = "%s is not legal boolean setting " % value
                        mes += 'for "boolean %s"' % parmname
                        raise RCBadParmValue(mes)
            retval = eval("%s(value)"%(vartype))
        else:
            retval = value
        return retval
    
    def parm_dict_by_tag(self, primname, tag, **otherkv):
        rd = self.ro.parm_dict_by_tag(primname, tag)
        rd.update(otherkv)
        return rd
       
    def __str__(self):
        """Used to dump Reduction Context(co) into file for test system
        """
        # @@DEPRECATED: remove on review
        FatalDeprecation("ReductionContext.__str__() obsolete and non-functional, please report.")
        
        #tempStr = ""
#        tempStr = tempStr + "REDUCTION CONTEXT OBJECT (CO)\n" + \
#            "inputs = " + str(self.inputs) + \
#            "\ninputsHistory =  " + str(self.inputs_history) + \
#            "\ncalibrations = \n" + self.calsummary() + \
#            "\nrorqs = " 
#        if self.rorqs != []:
#            for rq_obj in self.rorqs:            
#                tempStr = tempStr + str(rq_obj)
#        else:
#            tempStr = tempStr + str(self.rorqs)
#        
#        #no loop initiated for stkrqs object printouts yet
#        tempStr = tempStr + "\noutputs = " 
#        if self.outputs[MAINSTREAM] != []:
#            for out_obj in self.outputs[MAINSTREAM]:
#                tempStr = tempStr + str(out_obj)
#        else:
#            tempStr = tempStr + str(self.outputs)
#        #"stephistory = " + str( self.stephistory ) + \
#        tempStr = tempStr + "\nhostname = " + str(self.hostname) + \
#            "\ndisplayName = " + str(self.display_name) + \
#            "\ncdl = " + str(self.cdl) + \
#            "\nindent = " + str(self.indent) + \
#            "\nstackeep = " + str(self.stackeep)
#        for param in self.values():
#            tempStr += "\n" + self.paramsummary()
#        return tempStr   
    
    def add_cal(self, data, caltyp, calname, timestamp=None):
        '''
        :param data: The path or AstroData for which the calibration will be applied to.
        :type data: str or AstroData instance
        
        :param caltyp: The type of calibration. For example, 'bias' and 'flat'.
        :type caltyp: str
        
        :param calname: The URI for the MEF calibration file.
        :type calname: str
        
        :param timestamp: Default= None. Timestamp for when calibration was added.
            The format of time is
        taken from datetime.datetime.
        :type timestamp: str
        
        Add a calibration to the calibration index with a key related to the
        dataset's "datalabel", so it will apply, generally to later, processed
        versions of the dataset, and thus allow retrieval of the same calibration.
        
        '''
        adID = idFac.generate_astro_data_id(data)
        calname = os.path.abspath(calname)
        
        if timestamp == None:
            timestamp = datetime.now()
        else:
            timestamp = timestamp
        if self.calibrations == None:
            self.calibrations = {}
        if isinstance(data, AstroData):
            filename = data.filename
        else:
            filename = data
        calrec = CalibrationRecord(filename, calname, caltyp, timestamp)
        key = (adID, caltyp)
        self.calibrations.update({key: calrec})
    
    def add_callback(self, name, function):
        callbacks = self.callbacks
        if name in callbacks:
            l = callbacks[name]
        else:
            l = []
            callbacks.update({name:l})
        l.append(function)
    
    def clear_input(self):
        self.inputs = []
        
    def add_inputs(self, filelist):
        for f in filelist:
            self.add_input(f)
            
    def addInputs(self, filelist):
        print "called addInputs: deprecated, to be removed !!!!please change to add_inputss!!!!!"
        import traceback
        traceback.print_exc()
        raise
        self.add_inputs(filelist)
        
    def add_input(self, filenames):
        '''
        Add input to be processed the next batch around. If this is the first
        input being added, it is also added to original_inputs.
        
        @param filenames: Inputs you want added.
        @type filenames: list, AstroData, str 
        '''
        if type(filenames) != list:
            filenames = [filenames]
        
        ##@@TODO: Approve that this is acceptable. 
        ##(i.e. should it be done here or after the first round is complete?)
        # origFlag = False
        # if self.original_inputs is None or self.original_inputs == []:
        #    self.original_inputs = []
        #    origFlag = True
        
        for filename in filenames:
            if filename == None:
                continue
            elif type(filename) == str:
                filename = AstroDataRecord(filename)  
            elif type(filename) == AstroData:
                filename = AstroDataRecord(filename)
            elif type(filename) == AstroDataRecord:
                pass
            else:
                m = "BadArgument: '%(name)s' is an invalid type '%(type)s'." \
                    % {'name':str(filename), 'type':str(type(filename))}
                m += "Should be str, AstroData, AstroDataRecord."
                raise ReduceError(m) 
            
            #@@CONFUSING: the word filename here is an AstroDataRecord!
            if filename not in self.inputs:
                self.inputs.append(filename)
            #if origFlag:
            #    if filename not in self.original_inputs:
            #        self.original_inputs.append(filename)    
            #print "RM393:", repr(self.inputs)
       
    def add_rq(self, rq):
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
        val = self.step_moment(stepname, "begin")
        self.indent += 1
        self.stephistory.update({key: val}) 
        self.lastBeginDt = key
        self.initialize_inputs()
        return self
        
    def get_begin_mark(self, stepname, indent=None):
        for time in self.stephistory.keys():
            if     self.stephistory[time]["stepname"] == stepname \
               and self.stephistory[time]["mark"] == "begin":
                    if indent != None:
                        if self.stephistory[time]["indent"] == indent:
                            return (time, self.stephistory[time])
                    else:
                        return (time, self.stephistory[time])    
        return None
    
    def cal_filename(self, caltype):
        """returns a local filename for a retrieved calibration
        """
        #if self.original_inputs == None:
        #    self.original_inputs = deepcopy(self.inputs)
        #if len(self.original_inputs) == 0:
        #    return None
        #elif len(self.original_inputs) == 1:
        #    adID = idFac.generate_astro_data_id(self.inputs[0].ad)
        #    key = (adID, caltype)
        #    infile = os.path.basename(self.inputs[0].filename)
        #    if key in self.calibrations:
        #        return {self.calibrations[key].filename:[infile]}
        #    else:
        #        return None
        #else:
        retl = {}
        for inp in self.get_inputs_as_astrodata(): #self.original_inputs:
            key = (idFac.generate_astro_data_id(inp.ad), caltype)
            calfile = self.calibrations[key].filename
            infile = os.path.basename(inp.filename)
            if retl.has_key(calfile):
                retl.update({calfile:retl[calfile] + [infile]})
            else:
                retl.update({calfile:[infile]})
        return retl
                     
    def call_callbacks(self, name, **params):
        callbacks = self.callbacks
        if name in callbacks:
            for f in callbacks[name]:
                f(**params)
                    
    def cal_summary(self, mode="text"):
        rets = ""
        for key in self.calibrations.keys():
            rets += str(key)
            rets += str(self.calibrations[key])
        return rets
    
    def check_control(self):
        return self.cmd_request
    
    def clear_rqs(self, rtype=None):
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
        self.cmd_request = cmd
    
    def end(self, stepname):
        key = datetime.now()
        self.indent -= 1
        val = self.step_moment(stepname, "end")
        # this step saves inputs
        self.stephistory.update({key: val})
        # this step moves outputs[MAINSTREAM] to inputs
        # and clears outputs
        self.finalize_outputs()
        self.localparms = None
        self._output_streams = []
        return self
    
    def finalize_outputs(self):
        """ This function means there are no more outputs, generally called
        in a control loop when a generator function primitive ends.  Standard
        outputs become the new inputs. Calibrations and non-standard output
        is not affected.
        """
        # print "finalize_outputs"
        # only push if outputs is filled
        if len(self.outputs[self._current_stream]) != 0:
            # don't do this if the set is empty, it's a non-IO primitive
            ##@@TODO: The below if statement could be redundant because this 
            # is done in addInputs
            #if self.original_inputs == None:
                # SAY WHAT?  why deepcopy?
                # ack!
            #    self.original_inputs = deepcopy(self.inputs)
            
            #print "OUTPUTS:", self.outputs[MAINSTREAM]
            newinputlist = []
            # this code to be executed in initialize_inputs
            
    def initialize_inputs(self):
        newinputlist = []
        # print "initialize_inputs"
        for out in self.outputs[self._current_stream]:
            if type(out) == AstroDataRecord:
                newinputlist.append(out)
            else:
                mes = "Bad Argument: Wrong Type '%(val)s' '%(typ)s'." \
                    % {'val':str(out), 'typ':str(type(out))}
                raise RuntimeError(mes)
            
        self.inputs = newinputlist
    
    # finish and is_finished is combined using property
    def is_finished(self, arg=None):
        if arg == None:
            return self.status == "FINISHED"
        else:
            if arg == True:
                self.status = "FINISHED"
            elif self.status != "FINISHED":
                mes = "Attempt to change status from %s to FINISHED" % \
                    self.status
                raise ReduceError(mes)
        return self.is_finished()
    def finish(self):
        self.is_finished(True)
    finished = property(is_finished, is_finished)
    
    def get_cal(self, data, caltype):
        '''
        Retrieve calibration.
        
        :param data: File for which calibration will be applied.
        :type data: str or AstroData instance
        
        :param caltype: The type of calibration (ex.'bias', 'flat').
        :type caltype: str
        
        :return: The URI of the currently stored calibration or None.
        :rtype: str or None 
        '''
        #print "RM467:"+ repr(data)+repr( type( data ))dd
        adID = idFac.generate_astro_data_id(data)
        #filename = os.path.abspath(filename)
        key = (adID, caltype)
        if key in self.calibrations.keys():
            return self.calibrations[(adID, caltype)].filename
        return None
    
    def get_end_mark(self, stepname, indent=None):
        for time in self.stephistory.keys():
            if     self.stephistory[time]["stepname"] == stepname \
               and self.stephistory[time]["mark"] == "end":
                if indent != None:
                    if self.stephistory[time]["indent"] == indent:
                        return (time, self.stephistory[time])
                else:
                    return (time, self.stephistory[time])
        return None    
    
    def get_inputs(self, style=None):
        """
        :param style: Controls the type of return value. Supported values are "AD"
            and "FN" for ``AstroData`` and ``string`` filenames respectively.
        :type style: string
        :return: a list of ``AstroData`` instances or ``string`` filenames
        :rtype: list
        
        Get inputs gets the current input datasets from the current stream. You cannot
        choose the stream, use ``get_stream(..)`` for that.  To report modified
        datasets back to the stream use ``report_output(..)``.
        """
        if style==None:
            return self.inputs
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
    def get_outputs(self, style = None):
        return self.get_stream(style = style, stream = MAINSTREAM, empty = False)
        
    def get_stream(self, stream=MAINSTREAM, empty=False, style = None):
        """
        :param stream: A string name for the stream in question.  
            To use the standard stream do not set.
        :type stream: str
        :param empty: Controls if the stream is
            emptied, defaults to "False".
        :type empty: bool
        :param style: controls the type of output. "AD" directs the function
            to return a list
            of AstroData instances. "FN" directs it to return a list of filenames.
            If left blank or set to ``None``, the AstroDataRecord structures used
            by the Reduction Context will be returned.
        :returns: a list of ``AstroDataRecord`` objects, ``AstroData`` objects or filenames.
        :rtype: list
        
        Get stream returns a list of AstroData instances in the given stream.
        """
        
        if stream in self.outputs:
            outputs = self.outputs[stream]
        else:
            return None
        if empty:
            self.outputs.update({stream:[]})
        
        if style == None:
            return outputs
        elif style == "AD":
            retl = []
            for adrec in outputs:
                if not adrec.is_loaded():
                    adrec.load()
                retl.append(adrec.ad)
            return retl
        elif style == "FN":
            retl = [ad.filename for ad in outputs]
            return retl
        else:
            raise Errors.ReduceError("get_outputs: BAD STYLE ARGUMENT")
            
    def get_inputs_as_astrodata(self):
        """
            This function is equivalent to::
            
                get_inputs(style="AD")
        """
        return self.get_inputs(style="AD")
    get_inputs_as_astro_data = get_inputs_as_astrodata
    
    def get_inputs_as_filenames(self):
        """
            This function is equivalent for::
            
                get_inputs(style="FN")
        """
        return self.get_inputs(style="FN")

    def get_input_from_parent(self, parent):
        '''
        Very inefficient.
        '''
        # @@CLEAN: I don't know what this is
        for inp in self.inputs:
            if inp.parent == parent:
                return inp.filename
           
    def get_iraf_stderr(self):
        if self.irafstderr != None:
            return self.irafstderr
        else:
            return sys.stderr
        
    def get_iraf_stdout(self):
        if self.irafstdout != None:
            return self.irafstdout
        else:
            return sys.stdout
        
    def get_reference_image(self):
        """
        This function returns the current reference image.  At the moment
        this is simply the first dataset in the current inputs.  However,
        use of this function allows us to evolve our concept of reference
        image for more complicated issues where choice of a "reference" image
        may be more complicated (e.g. require some data analysis to determine).
        """
        if len(self.inputs) == 0:
            return None
        if self.inputs[0].ad == None:
            # @@NOTE: return none if reference image not loaded, reconsider
            #         raise ReduceError
            return None
        return self.inputs[0].ad
    
    
    def get_stack_ids(self):
        cachefile = self.get_cache_file("stackIndexFile")
        retval = self.stackeep.get_stack_ids(cachefile )
        return retval
    
    def populate_stream(self, infiles, stream=None, load = True):
        self.report_output(infiles, stream = stream, load = load)
        #print repr(self._output_streams)
        if stream == None:
            stream = self._current_stream
        self._output_streams.remove(stream)
        return
    
    def get_stack(self, purpose=""):
        sidset = set()
        purpose=self["purpose"]
        if purpose is None:
            purpose = ""
        
        # Get ID for all inputs
        for inp in self.inputs:
            sidset.add(purpose+IDFactory.generate_stackable_id(inp.ad))
        wholelist = []
        for sid in sidset:
            stacklist = self.get_list(sid) #.filelist
            wholelist.extend(stacklist)
        return wholelist
        
    def get_list(self, id):
        """
        :param id: Lists are assiciated with arbitrary identifiers,
            passed as strings.  See IDFactory for ids built from
            standard astrodata characteristics.
        :type id: str
        
        The list functionality allows storing dataset names in a list
        which is shared by all instances of reduce running in a given
        directory.  The list is kept by an adcc instance in charge of that
        sub-directory.  The "get_list" function retrieves a list that has
        already been requested via "rq_stack_get()" which initiates the
        interprocess request.
        
        This function does not block, and if the stack was not requested
        prior to a yeild, prior to this call, then None or an out of date
        version of this list will be retrieved.
        
        :note: "get_stack" calls get_list but takes a "purpose" to which it adds
               a stackingID as a suffix to the list identifier.
        
        """
        cachefile = self.get_cache_file("stackIndexFile")
        retval = self.stackeep.get(id, cachefile )
        return retval
    
    def inputs_as_str(self, strippath=True):
        if self.inputs == None:
            return ""
        else:
            inputlist = []
            for inp in self.inputs:
                if inp.ad != None:
                    inputlist.append(inp.ad.filename)
                else:
                    inputlist.append(inp.filename)
            if strippath == False:
                return ",".join(inputlist)                
            else:
                return ",".join([os.path.basename(path) for path in inputlist])

    def localparms_set(self, lpd):
        self._localparms = lpd
        
    def localparms_get(self):
        if self._localparms == None:
            self._localparms = {}
        return self._localparms 
    localparms = property(localparms_get, localparms_set)
    
    def make_inlist_file(self, filename, filelist):
        try:
            fh = open(filename, 'w')
            for item in filelist:
                fh.writelines(item + '\n')
        except:
            raise "Could not write inlist file for stacking." 
        finally:
            fh.close()
        return "@" + filename
                
    def parameter_collate(self, astrotype, primset, primname):
        """This function looks at the default primset paramaters for primname
        and sets the localparms member."""
        # @@HERE: is where parameter metadata is respected, or not
        if primname in primset.param_dict:
            #print "RM818: %s in param_dict"% primname
            # localparms should always be defined by here
            # users can never override argument in recipes (too confusing)
            correctUPD = None
            #print "RM822: %s" % repr((self.user_params))
            if self.user_params and not self.user_params.is_empty():
                correctUPD = self.user_params.get_user_param(astrotype, primname)
                #print "rm832:" , repr(correctUPD)
                if correctUPD != None:
                    for param in correctUPD.keys():
                        if param in self.localparms:
                            exs  = "User attempting to override parameter set "
                            exs += "in recipe\n\tastrotype = %s\n" % astrotype
                            exs += "\tprimitive = %s\n" % primname
                            exs += "\tparameter = %s\n" % str(param)
                            exs += "\t\tattempt to set to = %s\n" % \
                                correctUPD[param]
                            exs += "\t\trecipe setting = %s\n" % \
                                self.localparms[param]
                            raise SettingFixedParam(exs)
                            
            # use primset.param_dict to update self.localparms
            # print "rm847: %s" % repr(primset.param_dict[primname].keys())
            for param in primset.param_dict[primname].keys():
                #print "RM848:", param
                # @@NAMING: naming of default value in parameter dictionary hardcoded
                # print "RM571:", param, repr(self.localparms), repr(self), param in self
                if param in self.localparms: #  or param in self:
                    repOvrd = ("recipeOverride" not in primset.param_dict[primname][param])\
                                 or primset.param_dict[primname][param]["recipeOverride"]
                    # then it's already in there, check metadata
                    # @@NAMING: "recipeOverride" used in RecipeManager code
                    if not repOvrd:
                        exs =  "Recipe attempts to set fixed parameter\n"
                        exs += "\tastrotype = %s\n" % astrotype
                        exs += "\tprimitive = %s\n" % primname
                        exs += "\tparameter = %s\n" % str(param)
                        exs += "\t\tattempt to set to = %s\n" % self[param]
                        exs += "\t\tfixed setting = %s\n" %  \
                            primset.param_dict[primname][param]["default"]
                        raise SettingFixedParam(exs)
                if param not in self.localparms and param not in self:
                    if "default" in primset.param_dict[primname][param]:
                        self.localparms.update({param:primset.param_dict[primname][param]["default"]})
            # about to add user paramets... some of which may be in the global
            # context (and not in correct UPD), strictly speaking these may 
            # not have been added by the user but we consider it user space
            # and at any rate expect it to not be overrided by ANY means (we
            # may want a different flag than userOverride
            # print "RM863: %s ... %s" %( primname,repr(primset.param_dict[primname]))
            for param in primset.param_dict[primname].keys():
                # if this param is already set in the context... there is a
                # problem, it's not to be set.
                userOvrd = ("userOverride" not in primset.param_dict[primname][param])\
                             or primset.param_dict[primname][param]["userOverride"]
                
                #print "rm869: ", repr(self.localparms)
                #print "RM869: param="+param
                #print repr(correctUPD)
                if dict.__contains__(self, param):
                    
                    # note: if it's in self.localparms, that's due to legal
                    # behavior above... primitives parameters (as passed in
                    # recipes) are always added to the localparms space
                    # thus, if a value is in the main context, it MUST be
                    # userOverridable
                    if not userOvrd:
                        exs =  "\nParm set in context when userOverride is False\n"
                        exs += "\tastrotype = %s\n" % astrotype
                        exs += "\tprimitive = %s\n" % primname
                        exs += "\tparameter = %s\n" % str(param)
                        exs += "\t\tattempt to set to = %s\n" % self[param]
                        exs += "\t\tfixed setting = %s\n" % \
                            primset.param_dict[primname][param]["default"]
                        raise SettingFixedParam(exs, astrotype=astrotype)
            
            # users override everything else if  it gets here... and is allowed
            if correctUPD:
                for param in correctUPD:
                    userOvrd = ("userOverride" not in primset.param_dict[primname][param])\
                                 or primset.param_dict[primname][param]["userOverride"]
                    if param in self.localparms or param in self:
                        
                        if not userOvrd:
                            exs =  "User attempted to set fixed parameter\n"
                            exs += "\tastrotype = %s\n" % astrotype
                            exs += "\tprimitive = %s\n" % primname
                            exs += "\tparameter = %s\n" % str(param)
                            exs += "\t\tattempt to set to = %s\n" % correctUPD[param]
                            exs += "\t\tfixed setting = %s\n" % \
                                primset.param_dict[primname][param]["default"]
                            raise SettingFixedParam(exs, astrotype = astrotype)
                        else:
                            self.localparms.update({param:correctUPD[param]})
                    else:
                        self.localparms.update({param:correctUPD[param]})              
    
    def param_names(self, subset = None):
        if subset == "local":
            return self.localparms.keys()
        else:
            lpkeys = set(self.localparms.keys())
            rckeys = set(self.keys())
            retl = list(lpkeys | rckeys)
            return retl
                                                                
    def outputs_as_str(self, strippath=True):
        if self.outputs == None:
            return ""
        else:
            outputlist = []
            for inp in self.outputs['main']: 
                outputlist.append(inp.filename)
            #print "RM289:", outputlist
            #"""
            if strippath == False:
                # print self.inputs
                return ", ".join(outputlist)
            else:
                return ", ".join([os.path.basename(path) for path in outputlist])
    
    def run(self, stepname):
        """ :param stepname: The primitive or recipe name to run. Note: this is 
                actually compiled as a recipe... proxy recipe names may appear
                in the logs.
            :type stepname: string
            
            The ``run(..)`` function allows a primitive to use the reduction
            context to execute another recipe or primitive.
        """
        a = stepname.split()
        cleanname = ""
        for line in a:
            cleanname = re.sub(r'\(.*?\).*?$', '', line)
            cleanname = re.sub(r'#.*?$', '', cleanname)
            if line != "":
                break;
        # cleanname not used!
        name = "proxy_recipe%d"%self.proxy_id
        self.proxy_id += 1
        #print "RM630:", stepname
        self.ro.recipeLib.load_and_bind_recipe(self.ro, name, src=stepname)
        ret = self.ro.runstep(name, self)
        self.initialize_inputs()
        return ret
    #------------------ PAUSE ---------------------------------------------------- 
    def is_paused(self, bpaused=None):
        if bpaused == None:
            return self.status == "PAUSED"
        else:
            if bpaused:
                self.status = "PAUSED"
            else:
                self.status = "RUNNING"
        
        return self.is_paused()
    
    def pause(self):
        self.call_callbacks("pause")
        self.is_paused(True)
    
    def unpause (self):
        self.is_paused(False)
    paused = property(is_paused, is_paused)
    
    def request_pause(self):
        self.control("pause") 
    
    def pause_requested(self):
        return self.cmd_request == "pause"
    #------------------ PAUSE ----------------------------------------------------
    
    def report(self, report_history=False, internal_dict=False, context_vars=False, \
                report_inputs=False, report_parameters=False, showall=False):
        """
        Prints out a report of the contents of the context object 
        
        @return: The formatted message for all the current parameters.
        @rtype: str
        """
        if showall:
            report_history = True
            internal_dict = True
            context_vars = True
            report_inputs = True
            report_parameters = True
    
        rets = "\n\n" + "-" * 50  + "\n\n\n"
        rets += " "*11 + "C O N T E X T  R E P O R T\n\n\n"
        rets += "-" * 50 + "\n" 
        #varlist = ["inputs", "original_inputs", "inputs_history", "outputs", 
        #    "calibrations", "rorqs", "status", "reason", "cmd_request",
        #    "hostname", "display_name", "stephistory", "stackeep", 
        #    "calindfile", "display_mode", "display_id", "irafstdout",
        #    "irafstderr", "callbacks", "arguments", "cache_files",
        #    "_localparms", "_nonstandard_stream", "_current_stream", 
        #    "user_params", "proxy_id", "ro", "cmd_history", "cmd_index"]
        
        varlist = ["inputs_history", "calibrations", "rorqs",
            "status", "reason", "cmd_request", "hostname", "display_name",
            "stackeep", "calindfile", "display_mode",
            "display_id", "callbacks", "arguments",
            "_nonstandard_stream",
            "_current_stream", "proxy_id", "cmd_history",
            "cmd_index"]
            # removed irafstdout, irafstderr, cache_files, ro
        
        # add in vars to show they are not there
        if not self._localparms:
            varlist.append("_localparms")
        if not self.stephistory:
            varlist.append("stephistory")
        if not self.user_params:
            varlist.append("user_params")
        if not self.inputs:
            varlist.append("inputs")
        #if not self.original_inputs:
        #    varlist.append("original_inputs")

        if report_inputs:
            # inputs
            if self.inputs:
                rets += "\nInput (self.inputs)"
                for rcr in self.inputs:
                    if isinstance(rcr, \
                        astrodata.ReductionContextRecords.AstroDataRecord):
                        rets += "\n    ReductionContextRecords.AstroDataRecord:"
                        rets += str(rcr)
                        rets += rcr.ad.infostr()
        
        if context_vars:
            # original_inputs
            #if self.original_inputs:
            #    rets += "\n\nOriginal Input (self.original_inputs,"
            #    rets += " ReductionContextRecords):"
            #    rets += "\n    %-20s : " % "RCR.filename"
            #    for rcr in self.original_inputs:
            #        rets += rcr.filename + "\n" + " "*23
           
            
            # context vars
            rets += "\nContext Variables (self.<var>):"
            varlist.sort()
            for var in varlist:
                rets += "\n    %-20s = %s" % (var, eval("self.%s" % var ))
        
        if report_parameters:
            # _localparms
            if self._localparms:
                rets += "\n\nLocal Parameters (self._localparms)"
                pkeys = self._localparms.keys()
                pkeys.sort
                for pkey in pkeys:
                    rets += "\n    %-13s : %s" % \
                        (str(pkey), str(self._localparms[pkey]))
            
            # user params (from original varlist)
            if self.user_params:
                rets += "User Parameters:"
                rets += repr(self.user_params.user_param_dict)
            rets += "\n"

        if self.stephistory and report_history == True:
            # stephistory
            
            rets += "\n\nStep History (self.stephistory):\n"
            rets += "    " + "-"*41 + "\n\n"
            rets += "Feature deprecated until memory issue is resolved \
(callen@gemini.edu)"
            
            """
            shkeys = self.stephistory.keys()
            shkeys.sort()
            count = 0
            for key in shkeys:
                sh_dict = self.stephistory[key]
                rets += "\n" + "          S T E P " + str(count+1) 
                rets += ": " + sh_dict["stepname"] 
                rets += "\n\n\n    " + "-"*41
                rets += "\n    " + str(key) + ":"
                sh_dictkeys = sh_dict.keys()
                sh_dictkeys.sort()
                if sh_dict.has_key("inputs"):
                    rets += "\n%s%-10s : " % (" "*8, "inputs (self.inputs)")
                    for rcr in sh_dict["inputs"]:
                        if isinstance(rcr, \
                astrodata.ReductionContextRecords.AstroDataRecord):
                            rets += "'%s':\n\n    %s" % \
                (str(jkey), "ReductionContextRecords.AstroDataRecord:")
                            rets += str(rcr)
                        else:
                            rets += str(rcr)
                    sh_dictkeys.remove("inputs")
                for ikey in sh_dictkeys:
                    if ikey != "outputs":
                        rets += "\n%s%-10s : %s" % \
                           (" "*8, str(ikey), sh_dict[ikey])
                if sh_dict.has_key("outputs"):
                    rets += "\n%s%-10s : " % (" "*8, "outputs (self.outputs)")
                    outputs_dict = sh_dict["outputs"]
                    outputs_dictkeys = outputs_dict.keys()
                    outputs_dictkeys.sort()
                    for jkey in outputs_dictkeys:
                        for rcr in outputs_dict[jkey]:
                            if isinstance(rcr, \
                    astrodata.ReductionContextRecords.AstroDataRecord):
                                rets += "'%s':\n\n    %s" % \
                    (str(jkey), "ReductionContextRecords.AstroDataRecord:")
                                rets += str(rcr)
                                rets += "\n    OUTPUT AD.INFO (rcr.ad.infostr())"
                                rets += rcr.ad.infostr() + "\n"
                            else:
                                rets += str(rcr)
                rets += "    " + "-"*41 + "\n\n"
                count += 1"""

        if internal_dict:
            # internal dictionary contents
            cokeys = self.keys()
            rets += "\n       I N T E R N A L  D I C T I O N A R Y\n"
            loglist = []
            cachedirs = []
            others = []
            for key in cokeys:
                if key  == "cachedict":
                    rets += "\nCached Files (self[cachedict:{}]):\n"
                    cache_dict = self[key]
                    cdkeys = cache_dict.keys()
                    cdkeys.remove("storedcals")
                    cdkeys.remove("reducecache")
                    cdkeys.sort()
                    for ikey in cdkeys:
                        dirfiles = os.listdir(cache_dict[ikey])
                        if len(dirfiles) == 0:
                            dirfiles = "None"
                        rets += "    %-20s : %s\n" %(ikey, dirfiles)
                elif key[:3] == "log":
                    loglist.append(key)
                elif key == "reducecache" or key[:9] == "retrieved" or \
                    key[:6] == "stored":
                    cachedirs.append(key)
                else:
                    others.append(key)

            rets += "\nCache Directories (self[<dir>]):"
            for dir_ in cachedirs:
                rets +="\n    %-20s : %s" % (dir_, str(self[dir_]))
            rets += "\n\nLogger Info (self[<log...>]):"
            for l in loglist:
                rets +="\n    %-20s : %s" % (l, str(self[l]))
            if len(others) > 0:
                rets += "\nOther (self[<Other>]):\n"
                for o in others:
                    rets +="\n    %-20s : %s" % (o, str(self[o]))
                
        rets += "\n\n" + "-" * 50  + "\n"
        return rets
    
    def persist_cal_index(self, filename = None, newindex = None):
        # should call PRS!
        #return
        #print "Calibration List Before Persist:"
        #print self.calsummary()
        if newindex != None:
            # print "P781:", repr(newindex)
            self.calibrations = newindex
        try:
            pickle.dump(self.calibrations, open(filename, "w"))
            self.calindfile = filename
        except:
            print "Could not persist the calibration cache."
            raise 
    
    def persist_fringe_index(self, filename):
        try:
            pickle.dump(self.fringes.stack_lists, open(filename, "w"))
        except:
            raise 'Could not persist the fringe cache.'
            
    def persist_stk_index(self, filename):
        self.stackKeeper.persist(filename)
        #try:
        #    #print "RM80:", self.stackeep
        #    pickle.dump(self.stackeep.stack_lists, open(filename, "w"))
        #except:
        #    print "Could not persist the stackable cache."
        #    raise
    
    def prepend_names(self, prepend, current_dir=True, filepaths=None):
        '''
        :param prepend: The string to be put at the front of the file.
        :type prepend: string
        
        :param current_dir: Used if the filename (astrodata filename) is in the
                            current working directory.
        :type current_dir: boolean
        
        :param filepaths: If present, these file paths will be modified, otherwise
                          the current inputs are modified.
        :type filepaths:
        
        :return: List of new prepended paths.
        :rtype: list  
        
        Prepends a prefix string to either the inputs or the given list of filenamesfilename.
        
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
               
            if current_dir == True:
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
    
    def print_headers(self):
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
    
    def process_cmd_req(self):
        if self.cmd_request == "pause":
            self.cmd_request = "NONE"
            self.pause()
            
    def remove_callback(self, name, function):
        if name in self.callbacks:
            if function in self.callbackp[name]:
                self.callbacks[name].remove(function)
        else:
            return
    
    def report_history(self):
        
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
                
            lastdt = dtpostpend
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
                        "outputs":str(",".join(sh[dt]['outputs']['main'])),
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
                
            if len(sh[dt]["outputs"][MAINSTREAM]) != 0:
                retstr += " | ".center(wide) + "\n"
                retstr += "\|/".center(wide) + "\n"
                retstr += " ' ".center(wide) + "\n"
                retstr += ("%(outputs)s" % pargs).center(wide) + "\n"
                
                
            lastdt = dt
            if startdt == None:
                startdt = dt
        
        return retstr
        
    def report_output(self, inp, stream=None, load=True):
        """
        :param inp: The inputs to report (add to the given or current stream).
            Input can be a string (filename), an AstroData instance, or a list of
            strings and/or AstroData instances.  Each individual dataset is
            wrapped in an AstroDataRecord and stored in the current stream.
        :type inp: str, AstroData instance, or list
        :param stream: If not specified the default ("main") stream is used.
            When specified the named stream is created if necessary.
        :type stream: str
        :param load: A boolean (default: True) which specifies whether string
            arguments (pathnames) should be loaded into AstroData instances
            or if it should be kept as a filename, unloaded.  This argument
            has no effect when "report"
            ``AstroData`` instances already in memory.
            
        This function, along with ``get_inputs(..)`` allows a primitive to
        interact with the datastream in which it was invoked (or access
        other streams).
        """
        ##@@TODO: Read the new way code is done.
        #if category != MAINSTREAM:
        #    raise RecipeExcept("You may only use " + 
        #        "'main' category output at this time.")
        # print "RM1101:", self.ro.curPrimName, "stream:", repr(stream)
        if stream == None:
            stream = self._current_stream
        #print "RM1105:", self.ro.curPrimName, "stream:", stream
        # this clause saves the output stream so we know when to 
        # the first report happens so we can clear the set at that time.
        if stream not in self._output_streams:
            self._output_streams.append(stream)
            self.outputs.update({stream:[]})
            
        # this clause makes sure there is a list in self.outputs
        if stream not in self.outputs:
            self.outputs.update({stream:[]})
            
        if type(inp) == str:
            self.outputs[stream].append(AstroDataRecord(inp, self.display_id, load=load))
        elif isinstance(inp, AstroData):
            self.outputs[stream].append(AstroDataRecord(inp))
        elif type(inp) == list:
            for temp in inp:
                # This is a good way to check if IRAF failed.
                
                if type(temp) == tuple:
                    #@@CHECK: seems bad to assume a tuple means it is from 
                    #@@.....: a primitive that needs it's output checked!
                    if not os.path.exists(temp[0]):
                        raise "LAST PRIMITIVE FAILED: %s does not exist" % temp[0]
                    orecord = AstroDataRecord(temp[0], self.display_id, parent=temp[1], load=load)
                    #print 'RM370:', orecord
                elif isinstance(temp, AstroData):
                    # print "RM891:", type(temp)
                    orecord = AstroDataRecord(temp)
                elif type(temp) == str:
                    if not os.path.exists(temp):
                        raise "LAST PRIMITIVE FAILED."
                    orecord = AstroDataRecord(temp, self.display_id , load=load)
                else:
                    raise "RM292 type: " + str(type(temp))
                #print "RM344:", orecord
                if stream not in self.outputs:
                    self.outputs.update({stream:[]})
                self.outputs[stream].append(orecord)
    
    def restore_cal_index(self, filename):
        raise "don't call restore_cal_index"
        if os.path.exists(filename):
            self.calibrations = pickle.load(open(filename, 'r'))
            self.calindfile = filename
        else:
            pickle.dump({}, open(filename, 'w'))
    
    def restore_fringe_index(self, filename):
        '''
        
        '''
        if os.path.exists(filename):
            self.fringes.stack_lists = pickle.load(open(filename, 'r'))
        else:
            pickle.dump({}, open(filename, 'w'))
                            
    def restore_stk_index(self, filename):
        '''
        Get the stack list from 
        '''
        
        if False:
            if os.path.exists(filename):
                self.stackeep.stackLists = pickle.load(open(filename, 'r'))
            else:
                pickle.dump({}, open(filename, 'w'))
    
    def rm_cal(self, data, caltype):
        '''
        Remove a calibration. This is used in command line argument (rmcal). This may end up being used
        for some sort of TTL thing for cals in the future.
        
        @param data: Images who desire their cals to be removed.
        @type data: str, list or AstroData instance.
        
        @param caltype: Calibration type (e.g. 'bias').
        @type caltype: str
        '''
        datalist = gdpgutil.check_data_set(data)
        
        for dat in datalist:
            datid = idFac.generate_astro_data_id(data)
            key = (datid, caltype)
            if key in self.calibrations.keys():
                self.calibrations.pop(key)
            else:
                print "'%(tup)s', was not registered in the calibrations."
    
    def rq_cal(self, caltype, inputs=None, source="all"):
        '''
        Create calibration requests based on raw inputs.
        
        :param caltype: The type of calibration. For example, 'bias' and 'flat'.
        :type caltype: str
        
        :param inputs: The datasets for which to find calibrations, if not present
                        or ``None`` current "inputs" are used.
        :type inputs: list of AstroData instances                
        :param source: Directs what calibration service to contact, for future
                        compatibility, surrently only "all" is supported.
        '''
        if type(caltype) != str:
            raise RecipeExcept("caltype not string, type = " + str( type(caltype)))
        if inputs is None:
            # note: this was using original inputs!
            addToCmdQueue = self.cdl.get_cal_req(self.get_inputs_as_astrodata(),
                                                 caltype)
        else:
            addToCmdQueue = self.cdl.get_cal_req(inputs, caltype)
            #print "RM1389:", repr(addToCmdQueue[0].as_dict())
        for re in addToCmdQueue:
            # print "RM1558:",repr(dir(re))
            re.calurl_dict = self["calurl_dict"]
            re.source = source
            self.add_rq(re)
            
    def save_cmd_history(self):
        print "RM1569:", repr(self.rorqs)
        print "RM1570:saveCmdHistorythis saves nothing atm! It's for the HTML iface"
        
    def return_from_recipe(self):
        self._return_command = "return_from_recipe"
    
    def terminate_primitive(self):
        self._return_command = "terminate_primitive"
    
    def pop_return_command(self):
        rc = self._return_command
        self._return_command = None
        return rc
        
    def report_qametric(self, ad=None, name=None, metric_report = None, metadata = None):
        # print "RM1575:"+repr(metric_report)
        self._metricEvents.append_event(ad, name, metric_report, metadata = metadata)
        
    
    def get_metric_list(self, clear = False):
        ml = self._metricEvents.get_list()
        if clear:
            self._metricEvents.clear_list()
        return ml
    
    def rq_display(self, display_id=None):
        '''
        self, filename = None
        if None use self.inputs
        
        Create requests to display inputs.
        '''
        ver = "1_0"
        displayObject = DisplayRequest()
        if display_id:
            Did = display_id
        else:
            Did = idFac.generate_display_id(self.inputs[0].filename, ver)
        displayObject.disID = Did
        displayObject.disList = self.inputs
        self.add_rq(displayObject)
    
    def rq_iq(self, ad, e_m, e_s, f_m, f_s):
        iqReq = ImageQualityRequest(ad, e_m, e_s, f_m, f_s)
        self.add_rq(iqReq)
    rq_iqput = rq_iq
        
    def rq_stack_get(self, purpose = ""):
        """
        :param purpose: The purpose is a string prepended to the stackingID
                        used to identify the list (see get_list).
        :type purpose: str
        
        The stackingID (see IDFactory module) is used to identify the list.
        The first input in the rc.inputs list is used as the reference image 
        to generate  
        the stackingID portion of the list identifier.
        
        The stackingID function in IDFactory is meant to produce identical
        stacking identifiers for different images which can/should be stacked 
        together, e.g. based
        on program id and/or other details.  Again, see IDFactory for the
        particular algorithm in use.
        
        :note: a versioning system is latent within the code, and is added
            to the id to allow adaptation in the future if identifer construction
            methods change.
        """
        ver = "1_0"
        # Not sure how version stuff is going to be done. This version stuff is temporary.
        for orig in self.get_inputs_as_astrodata():
            # print "RM1453: HERE!"
            Sid = purpose + idFac.generate_stackable_id(orig.ad, ver)
            stackUEv = GetStackableRequest()
            stackUEv.stk_id = Sid
            self.add_rq(stackUEv)
                
    def rq_stack_update(self, purpose = None):
        '''
        :param purpose: The purpose argument is a string prefixed to the
            generated stackingID.  This allows two images which would
            produce identical stackingIDs to go in different lists,
            i.e. such as a fringe frame which, which might be prepended with
            "fringe" as the purpose.
            
        :type purpose: str
        
        This function creates requests to update a stack list with the files
        in the current rc.inputs list.  Each will go in a stack based on its
        own stackingID (prepended with "purpose").
        
        :note: this function places a message on an outbound message queue
            which will not be sent until the next "yield", allowing the
            ReductionObject command clause to execute.
        '''
        if purpose == None:
            purpose = ""
        ver = "1_0"
        # Not sure how version stuff is going to be done. This version stuff is temporary.
        inputs = self.get_inputs_as_astrodata()
        for inp in inputs:
            stackUEv = UpdateStackableRequest()
#            print "RM1507:", repr(purpose), repr(idFac.generate_stackable_id(inp, ver))
            Sid = purpose + idFac.generate_stackable_id(inp, ver)
            stackUEv.stk_id = Sid
            stackUEv.stk_list = inp.filename
            self.add_rq(stackUEv)
    #better name?
    rq_stack_put = rq_stack_update
    
    def set_cache_file(self, key, filename):
        filename = os.path.abspath(filename)
        self.cache_files.update({key:filename})
        
    def get_cache_file(self, key):
        if key in self.cache_files:
            return self.cache_files[key]
        else:
            return None
            
    def set_iraf_stderr(self, so):
        self.irafstderr = so
        return
    
    def set_iraf_stdout(self, so):
        self.irafstdout = so
        return
    
    def list_append(self, id, files, cachefile = None):
        """
        :param id: A string which identifies the list to append the listed 
            filenames to.
        :type id: str
        :param files: A list of filenames to add to the list.
        :type files: list of str
        :param cachefile: The filename to use to store the list.
        :type cachefile: str
        
        The caller is expected to supply cachefile, though in principle
        a value of "None" could mean the "default cachefile" this is not
        supported by the adcc as of yet, since the desired behavior is for
        reduce instances running in the same directory to cooperate, and those
        running in separate directories be kept separate, and this is 
        implemented by providing an argument for cachefile which is in a 
        generated subdirectory (hidden) based on the startup directory
        for the reduce process.  
        
        The adcc will negotiate all contention and race conditions regarding
        multiple applications manipulating a list simultaneously in separate
        process.
        """
        self.stackeep.add(id, files, cachefile)
    stack_append = list_append
        
    def list_inputs_as_str(self, id):
        """
        :param id: The identifier of the list to return as a comma separated string w/ no whitespace
        :type id: str
        
        This is used to provide the list of names as a single string.
        """
        #pass back the stack files as strings
        stack = self.stackeep.get(id)
        return ",".join(stack.filelist)
    stack_inputs_as_str = list_inputs_as_str

    def step_moment(self, stepname, mark):
        val = { "stepname"  : stepname,
                "indent"    : self.indent,
                "mark"      : mark,
                "inputs"    : [inp.filename for inp in self.inputs],  #copy(self.inputs),
                "outputs"   : None,   #copy(self.outputs),
                "processed" : False
                }
        return val
    
    def suffix_names(self, suffix, current_dir=True):
        '''
        
        '''
        newlist = []
        for nam in self.inputs:
            if current_dir == True:
                path = os.getcwd()
            else:
                path = os.path.dirname(nam.filename)
            
            fn = os.path.basename(nam.filename)
            finame, ext = os.path.splitext(fn)
            fn = finame + "_" + suffix + ext
            newpath = os.path.join(path, fn) 
            newlist.append(newpath)
        return newlist
        
    def switch_stream(self, switch_to = None):
        """
        :param switch_to: The string name of the stream to switch to. The 
            named stream must already exist.
        :type switch_to: str
        
        :note: This function is used by the infrastructure (in an application
            such as reduce and in the ReductionContext) to switch the stream
            being used. Reported output then goes to the specified stream.
        """
        if switch_to not in self.outputs:
            #raise ReduceError(
            #            '"%s" stream does not exist, cannot switch to it' 
            #                % repr(switch_to))
            self.outputs.update({switch_to:[]})
        
        self._current_stream = switch_to
        self._nonstandard_stream.append(switch_to)
        for ad in self.outputs[switch_to]:
            self.add_input(ad)
        #print "RM1360:", repr(self._nonstandard_stream)
        return switch_to
        
    def restore_stream(self, from_stream = None):
        """
        :param from_stream: This is the stream being reverted from. It does not
            need to be passed in but can be used to ensure it is the same
            stream the rc thinks it is  popping off.
        :type from_stream: str
        
        Revert to the last stream prior to previous switch_stream(..) call.
        """
        #print "RM1391: restore_stream"
        
        if len(self._nonstandard_stream) > 0:
            prevstream = self._nonstandard_stream.pop()
            if from_stream and prevstream != from_stream:
                raise ReduceError("from_stream does not match last stream")
            # copy 
                
            if len(self._nonstandard_stream)>0:
                self._current_stream = self._nonstandard_stream[-1]
            else:
                self._current_stream = MAINSTREAM
            
        else:
            raise ReduceError("Can't revert stream because there is no stream on stream list. The switch_stream(..) function not called.")
                    
    def bad_call(self, arg=None):
        raise "DO NOT USE ORIGINAL INPUTS"
    original_inputs = property(bad_call, bad_call)
    
def open_if_name(dataset):
    """Utility function to handle accepting datasets as AstroData
    instances or string filenames. Works in conjunction with close_if_name.
    The way it works, open_if_name opens returns an GeminiData isntance"""    
    bNeedsClosing = False    
    if type(dataset) == str:
        bNeedsClosing = True
        gd = AstroData(dataset)
    elif isinstance(dataset, AstroData):
        bNeedsClosing = False
        gd = dataset
    else:
        raise RecipeExcept("BadArgument in recipe utility function: open_if_name(..)\n MUST be filename (string) or GeminiData instrument")
    return (gd, bNeedsClosing)
    
def close_if_name(dataset, b_needs_closing):
    """Utility function to handle accepting datasets as AstroData
    instances or string filenames. Works in conjunction with open_if_name."""

    if b_needs_closing == True:
        dataset.close()
    
    return

class RecipeLibrary(object):

    prim_load_times = {}
    
    def add_load_time(self, source, start, end):
        key = datetime.now()
        pair = {key: {"source":source, "start":start, "end":end}}
        self.prim_load_times.update(pair)

    def discover_correct_prim_type(self, context):
        ref = context.get_reference_image()
        if ref == None:
            return None
        val = pick_config(ref, centralPrimitivesIndex, "leaves")
        k = val.keys()
        if False: # we do allow this, have TO! for recipes and multiple packages len(k) != 1:
                raise RecipeExcept("Can't discover correct primtype for %s, more than one (%s)" % (ref.filename, repr(k)))
        return k[0]
        
    def report_history(self):
        self.report_load_times()
        
    def report_load_times(self):
        skeys = self.prim_load_times.keys()
        skeys.sort()
        
        for key in skeys:
            primrecord = self.prim_load_times[key]
            source = primrecord["source"]
            start = primrecord["start"]
            end = primrecord["end"]
            duration = end - start
            
            pargs = {   "module":source,
                        "duration":duration,
                        }
            print "Module '%(module)s took %(duration)s to load'" % pargs

    def load_and_bind_recipe(self, ro, name, dataset=None, astrotype=None, src = None):
        """
        Will load a single recipe, compile and bind it to the given reduction objects.
        If src is set, dataset and astrotype are ignored (no recipe lookup)
        """
        if src != None:
            rec = src
            # compose to python source
            prec = self.compose_recipe(name, rec)
            # print "RM1139:", prec
            # compile to unbound function (using the python interpretor obviously)
            rfunc = self.compile_recipe(name, prec)
            # bind the recipe to the reduction object
            ro = self.bind_recipe(ro, name, rfunc)
        elif astrotype != None:
            # get recipe source
            rec = self.retrieve_recipe(name, astrotype=astrotype)
            #p rint "RM1894:", name, rec, astrotype
            try:
                # print "RM1115: before"
                ps = ro.get_prim_set(name)
                # print "RM1117: after"
                if ps:
                    if rec == None:
                        return #not a recipe, but exists as primitive
                    else:
                        msg = "NAME CONFLICT: ASSIGNING RECIPE %s BUT EXISTS AS PRIMITIVE:\n\t%s" % rec, repr(ps)
                        raise RecipeExcept(msg)
            except ReductionObjects.ReductionExcept:
                 pass # just means there is no primset, that function throws
                
            print "RM1894: here with rec"    
            if rec:
                # compose to python source
                prec = self.compose_recipe(name, rec)
                #p rint "RM1912:", prec
                # compile to unbound function (using the python interpretor obviously)
                rfunc = self.compile_recipe(name, prec)
                #p rint "RM1915:", rfunc
                # bind the recipe to the reduction object
                ro = self.bind_recipe(ro, name, rfunc)
                #p rint "RM1918:", dir(ro)
            else:
                raise RecipeExcept("Error: Recipe Source for '%s' Not Found\n\ttype=%s, instruction_name=%s, src=%s"
                                    % (name, astrotype, name, src), name = name)
        elif dataset != None:
            gd, bnc = open_if_name(dataset)
            types = gd.get_types()
            rec = None
            for typ in types:
                rec = self.retrieve_recipe(name, astrotype=typ, inherit=False)
                if rec:
                    prec  = self.compose_recipe(name, rec)
                    rfunc = self.compile_recipe(name, prec)
                    ro = self.bind_recipe(ro, name, rfunc)
            # no recipe, see if there is a generic one
            if rec == None:
                rec = self.retrieve_recipe(name)
                if rec:
                    prec = self.compose_recipe(name, rec)
                    rfunc = self.compile_recipe(name, prec)
                    ro = self.bind_recipe(ro, name, rfunc)
            close_if_name(gd, bnc)
            

    def get_applicable_recipes(self, dataset= None, 
                                     astrotype = None, 
                                     collate=False,
                                     prune = False):
        """
        Get list of recipes associated with all the types that apply to this dataset.
        """
        if dataset != None and astrotype != None:
            raise RecipeExcept("get_applicable_recipes cannot have dataset and astrotype set")
        if dataset == None and astrotype == None:
            raise RecipeExcept("get_applicable_recipes must have either a dataset or explicit astrotype set")
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
            types = astrod.get_types(prune=True)
        else:
            types = [astrotype]
        # look up recipes, fill list
        reclist = []
        recdict = {}
        # print "RM1785:@@",types
        for typ in types:
            if False:
                if typ in centralAstroTypeRecipeIndex.keys():
                        recnames = centralAstroTypeRecipeIndex[typ]
                        
            
            recnames = inherit_index(typ, centralAstroTypeRecipeIndex)
            if recnames:
                reclist.extend(recnames[1])
                recdict.update({recnames[0]: recnames[1]})
        reclist = list(set(reclist))
        #print "RM:1798",repr(recdict)

        # if we opened the file we close it
        if byfname:
            astrod.close()
        
        if collate == False:
            return reclist
        else:
            return recdict
        
    def recipe_index(self, as_xml = False):
        cri = centralRecipeIndex
        
        if as_xml == False:
            return copy(cri)
        else:
            rs  = '<?xml version="1.0" encoding="UTF-8" ?>\n'
            rs += "<recipe_index>\n"
            for typ in cri.keys():
                recipe = cri[typ]
                rs += '\t<recipeAssignment type="%s" recipe="%s"/>\n' % (typ, recipe)
            rs += "</recipe_index>\n"
            return rs
        
    
    def list_recipes(self, name=None, astrotype=None, as_xml = False):
        
        cri = centralRecipeIndex
        
        recipelist = cri.keys()
            
        if as_xml==True:
            retxml  = '<?xml version="1.0" encoding="UTF-8" ?>\n'
            retxml += "<recipes>\n"
            for recipe in recipelist:
                retxml += """\t<recipe name="%s" path="%s"/>\n""" % (recipe, cri[recipe])
            retxml += "</recipes>\n"
            return retxml
        else:
            return recipelist
        
    def retrieve_recipe(self, name, astrotype=None, inherit= True):
        # @@NAMING: uses "recipe.TYPE" and recipe for recipe.ALL
        cri = centralRecipeIndex
        #print "RM1406:", repr(astrotype)
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
                if inherit == True:
                    fname = cri[key]
                else:
                    return None        
        else:
            return None

        rfile = file(fname, "r")
        rtext = rfile.read()
        # print "RM1433:", rtext
        return rtext
            
    def retrieve_reduction_object(self, dataset=None, astrotype=None):
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
        primsetlist = self.retrieve_primitive_set(dataset=dataset, astrotype=astrotype)
        # print "RM2071:",repr(primsetlist)
        ro.recipeLib = self
        if primsetlist:
            ro.curPrimType = primsetlist[0].astrotype
            #print "RM1916:", repr([ps.astrotype for ps in primsetlist])
        else:
            return None
        for primset in primsetlist:
            ro.add_prim_set(primset)
        
        b = datetime.now()
        if astrotype != None:
            source = "TYPE: " + astrotype
        elif dataset != None:
            source = "FILE: " + str(dataset)
        else:
            source = "UNKNOWN"
        #p rint "RM2088:",repr(ro.primDict)    
        #@@perform: monitory real performance loading primitives
        self.add_load_time(source, a, b)
        return ro
        
    def retrieve_primitive_set(self, dataset=None, astrotype=None):
        #p rint "RM2094:",astrotype
        val = None
        if (astrotype == None) and (dataset != None):
            val = pick_config(dataset, centralPrimitivesIndex, style="leaves")
            k = val.keys()
            #if len(k) != 1:
            #    print "RM1939:", repr(val)
                # raise RecipeExcept("CAN'T RESOLVE PRIMITIVE SET CONFLICT")
            #astrotype = k[0]
        if (astrotype != None):
            k = [astrotype]
        #p rint "RM2103:", astrotype, k, val
        primset = None
        # print "RM1475:", repr(centralPrimitivesIndex)
        primlist = []
        for astrotype in k:
            if (astrotype != None) and (astrotype in centralPrimitivesIndex):
                primdeflist = centralPrimitivesIndex[astrotype]
                #print "RM1948:", repr(primdeflist)
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
                        log = gemLog.getGeminiLog()
                        blmsg =  "##### PRIMITIVE SET IMPORT ERROR: SKIPPING %(impname)s\n" * 3
                        blmsg = blmsg % {"impname": importname}
                        msg = blmsg + traceback.format_exc() + blmsg
                        if log:
                            log.error(msg)
                        else:                    
                            print "PRINTED, not logged:\n"+msg
                    b = datetime.now()
                    try:
                        primset = eval (importname + "." + primdef[1] + "()")
                    except NameError:
                        traceback.print_exc()
                        print "NOTE "*15
                        print "NOTE: if you have had trouble with importing a Gemini primitive set,"
                        print "      you may need to create login.cl.  This can be done, if IRAF is installed,"
                        print  "      with the 'mkiraf' command"""
                        print "NOTE "*15
                        sys.exit(1)
                    
                    except:
                        print
                        print ("!@"*40)
                        print "PROBLEM CREATING PRIMITIVE SET"
                        print (("!@"*40))
                        traceback.print_exc()
                        print ("!@"*40)
                        print
                        raise
                        # set filename and directory name
                    # used by other parts of the system for naming convention based retrieval
                    # i.e. of parameters
                    primset.astrotype = astrotype
                    primset.acquire_param_dict()
                    primlist.append(primset)
        
        if len(primlist):
            return primlist
        else:
            return None
        
    def compose_recipe(self, name, recipebuffer):
        templ = """
def %(name)s(self,cfgObj):
    #print "${BOLD}RECIPE BEGINS: %(name)s${NORMAL}" #$$$$$$$$$$$$$$$$$$$$$$$$$$$
    recipeLocalParms = cfgObj.localparms
%(lines)s
    #print "${BOLD}RECIPE ENDS:   %(name)s${NORMAL}" #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
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
        dostep = (str(recipeLocalParms["%(line)s"]).lower() != "false")
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
            if (co.is_finished()):
                break
            yield co
    yield co""" % {"parms":repr(d),
                    "line":line}
            lines += newl
            
        rets = templ % {    "name" : name,
                            "lines" : lines,
                            }
        return rets
        
    def compile_recipe(self, name, recipeinpython):
        exec(recipeinpython)
        func = eval(name)
        return func
        
    def bind_recipe(self, redobj, name, recipefunc):
        rprimset = redobj.new_primitive_set(redobj.curPrimType, btype="RECIPE")
        bindstr = "rprimset.%s = new.instancemethod(recipefunc, redobj, None)" % name
        exec(bindstr)
        redobj.add_prim_set(rprimset, add_to_front = False)
        #p rint "RM2254:", redobj.primstype_order
        #p rint "RM2255:", repr(redobj.primDict)
        return redobj
    
    def check_method(self, redobj, primitivename):
        ps = redobj.get_prim_set(primitivename)
        if ps == None:
            #p rint "RM1382: %s doesn't exist" % primitivename
            # then this name doesn't exist
            return False
        else:
            #p rint "RM1382: %s does exist" % primitivename
            return True
        
    def check_and_bind(self, redobj, name, context=None):
        # print "RM1389:", name
        if self.check_method(redobj, name):
            #p rint "RM2266: checkmethod fires"
            return False
        else:
            # print "RM1078:", str(dir(context.inputs[0]))
            #p rint "RM2268:", redobj.primstype_order
            self.load_and_bind_recipe(redobj, name, astrotype = redobj.curPrimType)
            return True

    def get_applicable_parameters(self, dataset):
        '''
        
        '''
        explicitType = None
        if  type(dataset) == str:
            if os.path.exists(dataset):
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
            types = astrod.get_types()
            
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

    def retrieve_parameters(self, dataset, contextobj, name):
        '''
        
        '''
        raise "this is old code which needs removing"
        # Load defaults
        print "RM1364: here"
        defaultParamFiles = self.get_applicable_parameters(dataset)
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

from pprint import pformat

def get_recipe_info(recname, filename):
    rd = {}
    paths = filename.split("/")
    recipefile = paths[-1]
    for d in paths:
        if RECIPEMARKER in d:
            ind = paths.index(d)
            paths = paths[ind:-1]
            break
    recinfo = { "recipe_name":recname,
                "fullpath":filename,
                "basename":recipefile,
                "category_list":repr(paths),
                "package_path":"/".join(paths)
               }
    return recinfo
    
    
if True: # was firstrun logic... python interpreter makes sure this module only runs once already

    # WALK the directory structure
    # add each directory to the sytem path (from which import can be done)
    # and exec the structureIndex.***.py files
    # These indexes are meant to append it to the centralDescriptorIndex
            
    for root, dirn, files in ConfigSpace.config_walk("recipes"):
        root = os.path.abspath(root)
        #print "RM2193:", root
        sys.path.append(root)
        curdir = root
        curpack = ConfigSpace.from_which(curdir)
        for sfilename in files:
            curpath = os.path.join(curdir, sfilename)
            
            m = re.match(recipeREMask, sfilename)
            mpI = re.match(primitivesIndexREMask, sfilename)
            mri = re.match(recipeIndexREMask, sfilename)
            mro = re.match(reductionObjREMask, sfilename) 
            #if mro:
            #    print "RM2202:", mro
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
                        
                        # @@TODO: eventually continue, don't raise!
                        # don't raise, this makes bad recipe packages halt the whole package!
                        # raise now because this should NEVER happen.
                        raise RecipeExcept("Two Recipes with the same name.")
                #print "RM2412:",fullpath
                centralRecipeIndex.update({recname: fullpath})
                recinfo = get_recipe_info(recname, fullpath)
                centralRecipeInfo.update({recname: recinfo})               
                
                
                am = re.match(recipeAstroTypeREMask, m.group("recipename"))
                #print (am)
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
                            print "WARNING:\n" + rs
                for key in lpis:
                    if key not in cpis:
                        centralPrimitivesIndex.update({key:[]})
                    
                    plist = centralPrimitivesIndex[key]
                    val = lpi[key]
                    if type(val) == tuple:
                        plist.append(localPrimitiveIndex[key])
                        centralPrimitivesCatalog.add_primitive_set(
                            package = curpack, 
                            primsetEntry = val,
                            primsetPath = curpath)
                    else:
                        plist.extend(val)
                           
            elif mro: # reduction object file... contains  primitives as members
                # print "RM2271:", sfilename, fullpath
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


