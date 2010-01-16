import sys,os
import re

"""This module contains a factory, L{getCalculator}, which will return 
the appropriate Calculator instance for the given dataset,
according to the Descriptor configuration files. There is no Descriptor
class, the classes in question are the Calculators. This function is used
by AstroData instances to get the appropriate Calculator, which is then
called by the base Descriptor related members defined in the AstroData
class.

@note: the calculator could be mixed into the AstroData instance, but this
would mean that before mixing those functions are not available. We have opted
the slightly more typing intensive (for ourselves the developers, not for users)
solution of proxying descriptor access in AstroData member functions which then
call the appropriate member function of the calculator associated with
their dataset.
"""

from ConfigSpace import configWalk
DESCRIPTORSPACE = "descriptors"

class DescriptorExcept:
    """This class is an exception class for the Descriptor module"""
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
dExcept = DescriptorExcept
# NOTE: to address the issue of Descriptors module being a singleton, instead of the 
# approach used for the ClassificationLibrary, we use the descriptors module
# itself as the singleton and thus these module level "globals" which serve
# the purpose of acting as a central location for Descriptor behavior.
firstrun = True

# calculatorIndexREMask used to identify descriptorIndex files
# these files need to set descriptorIndex to a dictionary value
# relating AstroType names to descriptor calculator names, with the
# latter being of proper form to "exec" in python. 

calculatorIndexREMask = r"calculatorIndex\.(?P<modname>.*?)\.py$"

if (True):
    #note, the firstrun logic runs but is not needed, python only imports once
    # this module operates like a singleton
    
    centralCalculatorIndex = {}
    loadedCalculatorIndex = {}
    # WALK the config space as a directory structure
    for root, dirn, files in configWalk(DESCRIPTORSPACE):
        if root not in sys.path:
	        sys.path.append(root)
        if True:
            for dfile in files:
                if (re.match(calculatorIndexREMask, dfile)):
                    fullpath = os.path.join(root, dfile)
                    diFile = open(fullpath)
                    exec diFile
                    diFile.close()
                    # file must declare calculatorIndex = {}
                
                    # note, it might be confusing to find out if
                    # one index entry stomps another... so I'm going to 
                    # check that this dict doesn't have keys already
                    # in the central dict
                
                    for key in calculatorIndex.keys():
                        if centralCalculatorIndex.has_key(key):
                            # @@log
                            msg = "Descriptor Index CONFLICT\n"
                            msg += "... type %s redefined in\n" % key
                            msg += "... %s\n" % fullpath
                            msg += "... was already set to %s\n" %centralCalculatorIndex[key]
                            msg += "... this is a fatal error"
                            raise DescriptorExcept(msg)
                        
                    centralCalculatorIndex.update(calculatorIndex)

firstrun = False

# this is down here for a good reason... it imports the
# globalStdkeyDict from StandardDescriptorKeyDict.py, which was moved to the
# descriptors subdirectory on the command of the Feb 2008 Descriptors Code Review.
from Calculator import Calculator

# Module Level Function(s)
def getCalculator(dataset):
    """ This function gets the Calculator instance appropriate for 
    the specified dataset.
    Conflicts, arising from Calculators being associated with more than one
    AstroData type classification, are resolved by traversing the type tree to see if one
    type is a subtype of the other so the more specific type can be
    used.
    @param dataset: the dataset to load a calculator for
    @type dataset: AstroData
    @returns: the appropriate Calculator instance for this type of dataset
    @rtype: Calculator
    
    @note: OPEN ISSUE: how to deal with conflicts not resolved this way... i.e.
        if there are two assignments related to types which do not appear
        in the same type trees.
        
    """
    #NOTE: Handle hdulist as well as AstroData instance as 'dataset'?
    types = dataset.discoverTypes()
    calcs = []
    
    # use classification library from dataset's context
    cl = dataset.getClassificationLibrary()
    
    calc = None
    for typ in types:
        try:
            newcalc = centralCalculatorIndex[typ]
            newcalctype = typ
            calcs.append(newcalc)
            if (calc == None):
                calc = newcalc
                calctype = newcalctype
            else:
                # if the new calc is related to a type that the old
                nt = cl.getTypeObj(newcalctype)
                if nt.isSubtypeOf(calctype):
                # subtypes "win" calculator type assignment "conflicts"
                    calc = newcalc
                    calctype = newcalctype
                else:
                    ot = cl.getTypeObj(calctype)
                    if not ot.isSubtypeOf(nt):
                        # if more than one type applies, they must have a subtype
                        raise dExcept()
        except KeyError:
            pass  # just wasn't in dictionary, no problem, most types don't have
                  #  calculators

    # by now we should have calc, and calctype, both strings... get the object
    # first, if none were found, create the default, base Calculator
    # NOTE: the base calculator looks up the descriptor in the PHU or EXTENSIONS 
    # as appropriate for that descriptor
    
    # to the descriptor type.
    if (len(calcs) == 0):
        #then none were found, use default calculator
        return Calculator()
    # first check loadedDescriptorIndex
    elif loadedCalculatorIndex.has_key(calctype):
        return loadedCalculatorIndex[calctype]
    else:
        # if here we need to import and instantiate the basic calculator
        # note: module name is first part of calc string
        modname = calc.split(".")[0]
        exec "import " + modname
        calcObj = eval (calc)
        # add this calculator to the loadedCalculatorIndex (aka "calculator cache")
        loadedCalculatorIndex.update({calctype: calcObj})
        return calcObj
    
#@@DOCPROJECT@@ done pass 1
