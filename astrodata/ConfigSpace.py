import sys
import os

CONFIGMARKER = "ADCONFIG"
spaces = {  "descriptors":"descriptors",
            "structures":"structures",
            "types":"classifications/types",
            "status": "classifications/status"
            }

class ConfigSpaceExcept:
    """This class is an exception class for the ConfigSpace module"""
    def __init__(self, msg="Exception Raised in ConfigSpace system"):
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
        
class ConfigSpace(object):
    """This class exists to connect to a configuration space, such as 
    AstroDataType libraries or Descriptors. It provides tools to simplify 
    accessing configuration information and also providing indirection to
    allow configurations to be stored with alternate storage methodologies,
    such as in relational databases.  This flexibility is useful due to the
    many deployment contexts of the Gemini Reduction Package."""
    
    
    def configWalk(self, spacename):
        """This function can be iterated over in the style of os.walk()
        @param spacename: name of the space, "types", "statustypes",
        "descriptors", or "structures".
        @param spacename: string
        @returns: via yeild, a (root, dirn, files) tuple"""
        
        dirs = self.getDirs(spacename)
        print "C46: dirs: ", dirs
        for directory in dirs:
            for elem in os.walk(directory):
                yield elem
                
    def getDirs(self, spacename):
        """This function returns a list of directories to walk for a given 
        configuration space.
        @param spacename: name of the config space to collect directories for
        @type spacename: string
        @returns: list of directories
        @rtype: list"""
        
        # get the config space standard postfix for directories
        try:
            postfix = spaces[spacename]
            print "CS61:", postfix
        except KeyError:
            raise ConfigSpaceExcept("Given ConfigSpace name not recognized (%s)" % spacename)
        
        retdirs = []
        
        # get the ADCONFIG package dirs
        adconfdirs = []
        i = 1
        for path in sys.path:
            # print "@@@@@@@@:", path,
            if os.path.isdir(path):
                # print "ISADIR"
                subdirs = os.listdir(path)
                for subpath in subdirs:
                    print "CS77:", CONFIGMARKER, subpath, 
                    if CONFIGMARKER in subpath:
                        fullpath = os.path.join(path, subpath, postfix)
                        print "full", fullpath
                        if os.path.isdir(fullpath):
                            # then this is one of the config space directories
                            adconfdirs.append(fullpath)
            else:
                pass # print ""
                
        return adconfdirs

def configWalk( spacename = None):
    cs = ConfigSpace()
    for trip in cs.configWalk(spacename):
        yield trip
