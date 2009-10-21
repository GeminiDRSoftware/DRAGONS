import sys
import os

# OPTIMISATION IDEAS #
#
# () use configdirs cache for subsequent config space calls
# () load all configdirs at startup, say iterating over "spaces"
#
######################


CONFIGMARKER = "ADCONFIG_"
spaces = {  "descriptors":"descriptors",
            "structures":"structures",
            "types":"classifications/types",
            "status": "classifications/status",
            "xmlcalibrations":"xmlcalibrations",
            }
RECIPEMARKER = "RECIPES_"
LOOKUPDIRNAME = "lookups"
cs = None
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
    
    configdirs = None
    recipedirs = None
    configpacks = None
    
    def __init__(self):
        self.configdirs = {}
        self.configpacks = []
    
    def configWalk(self, spacename):
        """This function can be iterated over in the style of os.walk()
        @param spacename: name of the space, "types", "statustypes",
        "descriptors", or "structures".
        @param spacename: string
        @returns: via yeild, a (root, dirn, files) tuple"""
        
        if spacename == "recipes":
            dirs = self.getRecipeDirs()
        else:
            dirs = self.getConfigDirs(spacename)
        # print "C46: dirs: ", dirs
        for directory in dirs:
            for elem in os.walk(directory):
                path = elem[0]
                goodpath = (".svn" not in path) and ("CVS" not in path)
                if goodpath:
                    yield elem

	# @@@WARN if there are NO good paths, this will have a problem, there is no yeild
                
    def getConfigDirs(self, spacename):
        """This function returns a list of directories to walk for a given 
        configuration space.
        @param spacename: name of the config space to collect directories for
        @type spacename: string
        @returns: list of directories
        @rtype: list"""
        if (self.configdirs != None):
            if spacename in self.configdirs:
                return self.configdirs[spacename]
            
        # get the config space standard postfix for directories
        try:
            postfix = spaces[spacename]
        except KeyError:
            raise ConfigSpaceExcept("Given ConfigSpace name not recognized (%s)" % spacename)
        
        retdirs = []
        
        # get the ADCONFIG package dirs
        adconfdirs = []
        i = 1
        for path in sys.path:
            if os.path.isdir(path):
                        # print "ISADIR"
                        subdirs = os.listdir(path)
                        for subpath in subdirs:
                            if CONFIGMARKER in subpath:
                                packdir = os.path.join(path,subpath)
                                if packdir not in self.configpacks:
                                    self.configpacks.append(packdir)
                                fullpath = os.path.join(path, subpath, postfix)
                                # print "full", fullpath
                                if os.path.isdir(fullpath):
                                    # then this is one of the config space directories
                                    adconfdirs.append(fullpath)
                        else:
                            pass # print ""
                            
        self.configdirs.update({spacename: adconfdirs})
        
        return adconfdirs

    def getRecipeDirs(self):
        """This function returns a list of directories to walk for a given 
        configuration space.
        @param spacename: name of the config space to collect directories for
        @type spacename: string
        @returns: list of directories
        @rtype: list"""
        
        if (self.recipedirs != None):
            return self.recipedirs
        
        retdirs = []
        
        # get the ADCONFIG package dirs
        adconfdirs = []
        i = 1
        for path in sys.path:
            # print "@@@@@@@@:",".svn" in path,":::",  path
            if os.path.isdir(path):
                        # print "ISADIR"
                        subdirs = os.listdir(path)
                        for subpath in subdirs:
                            if RECIPEMARKER in subpath:
                                fullpath = os.path.join(path, subpath)
                                # print "full", fullpath
                                if os.path.isdir(fullpath):
                                    # then this is one of the config space directories
                                    adconfdirs.append(fullpath)
                        else:
                            pass # print ""
        self.recipedirs = adconfdirs
        return adconfdirs

    def generalWalk( self, dir ):
        '''
        A generalized walk, that ignores all the .svn / .cvs folders. I found this can be a little useful, 
        although it will probably be thrown out.
        
        @param path: Path to throw in os.walk
        @type path: str
        
        @return: Basically, takes the output of os.walk, but without the .svn stuff.
        @rtype: yields a 3-tuple (dirpath, dirnames, filenames).
        '''
        # @@TODO: This entire method should probably be removed at some point.
        filelist = []
        for (path, directories, files) in os.walk(dir):
            goodpath = (".svn" not in path) and ("CVS" not in path)
            if goodpath:
                path = os.path.abspath(path)
                for fname in files:
                    filelist.append( os.path.join(path, fname) )
                    
        return filelist
        
def configWalk( spacename = None):
    global cs
    if (cs == None):
        cs = ConfigSpace()
        
    for trip in cs.configWalk(spacename):
        yield trip\

# @@TODO: This entire method should probably be removed at some point.
def generalWalk( spacename = None ):
    global cs
    if (cs == None):
        cs = ConfigSpace()
        
    return cs.generalWalk( spacename )

def lookupPath(name):
    """This module level function takes a lookup name and returns a path to the file."""
    global cs
    if (cs == None):
        cs = ConfigSpace()
        
    a = name.split("/")
    #print "CS198LookupPath:", a
    domain = a[0]
    pack = CONFIGMARKER + domain
    tpath = None
    #print "CS166", cs.configpacks, cs.configdirs
    for path in cs.configpacks:
        
        if path[-(len(pack)):] == pack:
            # got the right package
            tpath = path
            break
            
    if tpath == None:
        raise ConfigSpaceExcept("No Configuration Package Associated with %s" % domain)
    fpath = os.path.join(tpath, LOOKUPDIRNAME, *a[1:])
    
    if (fpath[-3:] != ".py"):
        fpath += ".py"

        
    return fpath
