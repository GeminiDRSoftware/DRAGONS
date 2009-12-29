from StandardDescriptorKeyDict import globalStdkeyDict

class CalculatorExcept:
    """This class is an exception class for the Calculator module"""
    
    def __init__(self, msg="Exception Raised in Descriptor system"):
        """This constructor accepts a string C{msg} argument
        which will be printed out by the default exception 
        handling system, or which is otherwise available to whatever code
        does catch the exception raised.
        @param msg: a string description about why this exception was thrown
        @type msg: string
        """
        self.message = msg
    def __str__(self):
        """This string operator allows the default exception handling to
        print the message associated with this exception.
        @returns: string representation of this exception, the self.message member
        @rtype: string"""
        return self.message

class Calculator(object):
    """
    The Descriptor Calculator in an object with one member fucntion for 
    each descriptor (where descriptors are conceptually types of statistical
    or other values which can be thought of as applying to all data. In practice
    a descriptor might not be 100% general, it may apply only to a vast majority
    of data types, or require some generic handling.
    Though
    in practice some descriptors may still not really apply to some data types
    they will return a valid descriptor value (e.g. if there was an instrumen without
    a filter wheel, data from that instrument would still return a sensible value
    for the filter descriptor ("none" or "blank").
    
    A Calculator is associated with particular classifications of data, such that
    it can be assigned to AstroData instances cooresponding to that classification.
    It is important that configurations not lead to multiple Calculators associated
    with one DataSet (this can happen since AstroData instances have more than one
    classification can have a Calculator associated with it.  The system handles one
    case of this where one of the two types contributing a Calculator is a subtype
    of the other, in which case the system uses the subtypes descriptor, assuming it
    "overrides" the other.  In any other case the system will throw and exception
    when trying to assign the calculator to the AstroData instance. 
    
    @note: the base class, besides being a parent class for defining new Calculators
    is also the default Calculator for when none is specifically assigned. 
    It uses "StandardDescriptorKeyDict.py"
    to map the standard key names for descriptors to specific header keys, 
    then does the retrieval from the headers in the dataset, as appropriate.
    Ideally this method should work for all prepared data, at which point we would like
    to have stored the standard values in the data header where it is directly retrieved
    rather than calculated.
    
    @ivar usage: Used to document this Descriptor.    
    """
    usage = ""
        
    stdkeyDict = globalStdkeyDict
                       
    #bound instance methods
    def gain(self, dataset):
        """ 
        Return the gain for generic AstroData instances, by reading the standard
        'gain' header from each science extension (EXTNAM="SCI"), in the case
        of the default Calculator. Subclasses of course will make data classification
        (aka "dataset types" or "AstroData types") specific calculations to obtain
        the values.
        
        Note: we never check for 'gain' in the PHU, it is supposed to accompany
        the Science data directly in that extensions own header. Since AstroData is
        assumed to be a set of data, this means the gain values are returned as a
        list of floats.
        @param dataset: the data set for which to calculate C{gain}
        @type dataset: AstroData
        @rtype: list
        @returns: array of gain values, index 0 will be the gain of science
        extension #1, i.e. (EXTNAME,EXTVER)==("SCI",1) and so on.
        """
        return None
        # BELOW LIES AN ATTEMPT AT A GENERAL GAIN DESCRIPTOR
        #retval = []
        #
        #i = 0
        #for gd in dataset["SCI"]:
        #    i += 1
        #    gain = gd.getHeader(gd.extensions[0], self.stdkeyDict["key_gain"])
        #    
        #    retval.append(gain)
        #    
        #return retval
    
    
    def filtername(self, dataset):
        """This function returns the filter descriptor, which cannot be "filter"
        in python since that is a reserved word. As with all the descriptors 
        in the default, base class, Calculator class methods, the value is
        read from the approapriate standard header. Calculators subclassed from
        Calculator can override this behavior to calculate the filtername 
        as appropriate to a particular data classification.
        """
        return self.fetchPHUValue("key_filter", dataset)
        
    def fetchPHUValue(self, keyname, dataset):
        """This utility functioin fetches a header value from the PHU of the
        given dataset. The C{keyname} given is from the standardized key names
        and are mapped into actual standard PHU header name.
        @param keyname: the standard keyname used to look up the actual header
        key name from the standard key dictionary
        @type keyname: string
        @param dataset: an AstroData set instance
        @type dataset: AstroData
        @returns: the specified value as present in the PHU
        @rtype: depends on type of header value"""
        
        try:
            keynm = self.stdkeyDict[keyname]
            retval = dataset.phuValue(keynm)
        except KeyError:
            # LOGMESSAGE OR raise CalculatorExcept("          Standard Descriptor Key \"%s\" not in PHU (generic descriptor fails)" % keyname)
            # returning None indicates imcomplete descriptor calculators
            retval = None
        return retval
    
    
    def fetchValue( self, keyname, dataset ):
        """
        A test utility function that, given a keyname, whether it be something in the header like 'INSTRUME'
        or something not in the header like 'FILTERNAME', it will either return the value from the associated
        function, or grab the value from the header.
        
        @param keyname: Name of key to get value for.
        @type keyname: string
        
        @param dataset: an AstroData set instance.
        @type dataset: AstroData
         
        @param extension: Look in a specific extension. By default, it is PHU. (No other extensions implemented at
        this time, so DO NOT USE.
        @type extension: int
        
        @return: The value in a list form.
        @rtype: list 
        """
        retval = None
        if hasattr( self, str(keyname).lower() ):
            # print "C146: calling ", str(keyname).lower()
            keyfunc = getattr( self, str(keyname).lower() )
            retval = keyfunc( dataset ) 
        else:
            # print "C148: Gathering ", str(keyname)
            for ext in dataset.getHDUList():
                #print "KAPLAH"
                try:
                    retval = ext.header[str(keyname)]
                    break
                except:
                    continue
                
            if retval is None:
                print CalculatorExcept("Standard Descriptor Key \"%s\" not in PHU (generic descriptor fails)" % keyname)
            
        #if type( retval ) != list:
        #    retval = [retval]
        
        return retval
    
    def obsepoch( self, dataset ):
        '''
        Grabs observed time.
        
        @param dataset: an AstroData set instance.
        @type dataset: AstroData
        
        @return: Time, in the format: 2009-10-02T02:32:43.44 (%Y-%m-%dT%H:%M:%S.%f)
        @rtype: string
        '''    
        value = dataset.phuHeader("DATE-OBS")
        value = value + "T" + dataset.phuHeader('TIME-OBS')
        return value
        
    def maskname( self, dataset ):
        if dataset[0].header.has_key( 'MASKNAME' ):
            return dataset.phuHeader('MASKNAME')
        else:
            return 'None'
        
    def display(self, dataset):
        return None

#@@DOCPROJECT@@ done with pass 1
