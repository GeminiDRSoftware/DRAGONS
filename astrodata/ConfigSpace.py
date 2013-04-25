import sys
import os
import re
from copy import copy
# OPTIMISATION IDEAS #
#
# () use configdirs cache for subsequent config space calls
# () load all configdirs at startup, say iterating over "spaces"
#
######################

PACKAGEMARKER = "astrodata_"
CONFIGMARKER = "ADCONFIG_"
spaces = {  "descriptors":"descriptors",
            "structures":"structures",
            "types":"classifications/types",
            "status": "classifications/status",
            "xmlcalibrations":"xmlcalibrations",
            }
RECIPEMARKER = "RECIPES_"
LOOKUPDIRNAME = "lookups"
PIFMARKER = "PIF_"
CALCIFACEMARKER = "CalculatorInterface_(.*).py$"
DDLISTMARKER    = "DescriptorsList_(.*).py$"
CALCIFACEFORMAT = "CalculatorInterface_%s.py"
DDLISTFORMAT    = "DescriptorsList_%s.py"
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
    recipepath = None
    adconfigpath = None
    wholepath = None
    
    package_paths = None
    
    calc_iface_list=[]
    
    def __init__(self):
        self.configdirs = {}
        self.configpacks = []
        self.package_paths = []
 
        # support for ADCONFIGPATH and RECIPEPATH
        # NOTE: due to the way the list is extended, ADCONFIGPATH appearing second
        #       means it has precedence over RECIPEPATH, that is, ADCONFIGPATH
        #       is searched prior to RECIPEPATH
        self.recipepath = []
        if "RECIPEPATH" in os.environ:
            rpath = os.environ["RECIPEPATH"].split(":")
            # we want this path in front...
            self.recipepath = rpath

        self.adconfigpath = []
        if "ADCONFIGPATH" in os.environ:
            rpath = os.environ["ADCONFIGPATH"].split(":")
            # we want this path in front...
            self.adconfigpath = rpath
            
        self.wholepath = copy(self.recipepath)
        self.wholepath.extend(self.adconfigpath)
        self.wholepath.extend(sys.path)
        # print "CS76recipepath:", repr(self.recipepath)
        # print "CS76adconfigpath:", repr(self.adconfigpath)
        # print "CS76sys.path:", repr(sys.path)
        # print "CS76:", repr(self.wholepath)
        
    
    def config_walk(self, spacename):
        """This function can be iterated over in the style of os.walk()
        @param spacename: name of the space, "types", "statustypes",
        "descriptors", or "structures".
        @param spacename: string
        @returns: via yeild, a (root, dirn, files) tuple"""
        
        if spacename == "recipes":
            dirs = self.get_recipe_dirs()
        else:
            dirs = self.get_config_dirs(spacename)
        # print "C93: dirs: ", dirs
        for directory in dirs:
            for elem in os.walk(directory):
                path = elem[0]
                # print "CS97:", path
                goodpath = (".svn" not in path) and ("CVS" not in path)
                if goodpath:
                    if "edge" in elem: print "CS72:", elem
                    thefile = None
                    for fil in elem[2]:
                        if re.match(CALCIFACEMARKER, fil):
                            # print "CS117:", fil
                            self.calc_iface_list.append(
                                    (   "CALCIFACE",
                                        os.path.join(
                                            elem[0], fil
                                            )
                                    )    
                                )
                        ddresult = re.match(DDLISTMARKER, fil)
                        if ddresult:
                            calcname = CALCIFACEFORMAT % ddresult.group(1)
                            fcalcname = os.path.join(elem[0], calcname)
                            if not os.path.exists(fcalcname):
                                # if there is a CALCIFACEMARKER file, it supercedes this ddlist
                                # expecting to be a cache for performance purpsoes, a modified instance or otherwise
                                # the actual calculator interface
                                # print "CS133:", fil
                                
                                self.calc_iface_list.append(
                                        ( "DDLIST",
                                            os.path.join(
                                                elem[0], fil
                                                )
                                        )
                                    )
                    self.curpack = from_which(elem)        
                    yield elem
            
    def get_config_dirs(self, spacename):
        """This function returns a list of directories to walk for a given 
        configuration space.
        @param spacename: name of the config space to collect directories for
        @type spacename: string
        @returns: list of directories
        @rtype: list"""
        # print "CS152:", spacename
        if (self.configdirs != None):
            if spacename in self.configdirs:
                print "CS155:", spacename, pformat(self.configdirs)
                return self.configdirs[spacename]
        else:
            print "CS88:", "inefficency"
            
        # get the config space standard postfix for directories
        try:
            postfix = spaces[spacename]
        except KeyError:
            raise ConfigSpaceExcept("Given ConfigSpace name not recognized (%s)" % spacename)
        
        retdirs = []
        
        # get the ADCONFIG package dirs
        adconfdirs = []
        i = 1
        
        packagemask = {}
        for path in self.wholepath:
            #p rint "CS128:", path, str(os.path.abspath(path) !=  os.path.abspath(os.getcwd()))
            if  os.path.abspath(path) !=  os.path.abspath(os.getcwd()):
                if os.path.isdir(path):
                            # print "ISADIR"
                            subdirs = os.listdir(path)
                            for subpath in subdirs:
                                if not os.path.isdir(os.path.join(path,subpath)):
                                    continue
                                #print "CS134:", subpath
                                if PACKAGEMARKER in subpath:
                                    ppath = os.path.join(path, subpath)
                                    # this code exists to allow packages early in the path
                                    # to hide one's later in the path
                                    pknam = os.path.basename(ppath)
                                    if pknam in packagemask:
                                        continue
                                    else:
                                        packagemask[pknam]=ppath
                                    
                                    if ppath not in self.package_paths:
                                        self.package_paths.append(ppath)
                                    
                                    #print "CS136: package marker found"
                                    subsubpaths = os.listdir(os.path.join(path,subpath))
                                    for subsubpath in subsubpaths:
                                        if not os.path.isdir(
                                                        os.path.join(
                                                            path, subpath,subsubpath)):
                                            continue
                                        # print "CS139:",subsubpath
                                        if CONFIGMARKER in subsubpath:
                                            #print "CS141: CONFIGARKER found"
                                            packdir = os.path.join(path,subpath, subsubpath)
                                            if packdir not in self.configpacks:
                                                self.configpacks.append(packdir)
                                            fullpath = os.path.join(path, subpath, subsubpath, postfix)
                                            # print "full", fullpath
                                            if os.path.isdir(fullpath):
                                                # then this is one of the config space directories
                                                adconfdirs.append(fullpath)
                                        elif subsubpath.startswith(PIFMARKER):
                                            pifdir = os.path.join(path, subpath, subsubpath)
                                            # @@note: THIS IS ENTERED TWICE... DESCRIP AND TYPES?  SHOULDN'T BE                                            
                                            # print "cs185:", pifdir
                                            if pifdir not in sys.path:
                                                sys.path.append(pifdir)
                                            
                            else:
                                pass # print ""
                            
        self.configdirs.update({spacename: adconfdirs})
        return adconfdirs

    def get_recipe_dirs(self):
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
        pathlist = sys.path
            
        # support for ADCONFIGPATH and RECIPEPATH
        # NOTE: due to the way the list is extended, ADCONFIGPATH appearing second
        #       means it has precedence over RECIPEPATH, that is, ADCONFIGPATH
        #       is searched prior to RECIPEPATH
        if "RECIPEPATH" in os.environ:
            rpath = os.environ["RECIPEPATH"].split(":")
            # we want this path in front...
            rpath.extend(pathlist)
            pathlist = rpath

        if "ADCONFIGPATH" in os.environ:
            rpath = os.environ["ADCONFIGPATH"].split(":")
            # we want this path in front...
            rpath.extend(pathlist)
            pathlist = rpath
        # print "CS197:", repr(pathlist)
        
        pkmask = {}
        for path in pathlist:
            # print "@@@@@@@@:",".svn" in path,":::",  path
            if os.path.isdir(path):
                        # print "ISADIR"
                        subdirs = os.listdir(path)
                        for subpath in subdirs:
                            if not os.path.isdir(os.path.join(path,subpath)):
                                continue
                            if PACKAGEMARKER in subpath:
                                # this logic allows packages early in the path
                                # to hide packages late in the path
                                                                
                                pknam = os.path.basename(subpath)
                                if pknam in pkmask:
                                    continue
                                else:
                                    pkmask[pknam] = subpath

                                subsubpaths = os.listdir(os.path.join(path,subpath))
                                for subsubpath in subsubpaths:
                                    if RECIPEMARKER in subsubpath:
                                        fullpath = os.path.join(path, subpath, subsubpath)
                                        #print "RECIPEMARKER full", fullpath
                                        if os.path.isdir(fullpath):
                                            # then this is one of the config space directories
                                            adconfdirs.append(fullpath)
                            elif RECIPEMARKER in subpath:
                                fullpath = os.path.join(path, subpath)
                                if os.path.isdir(fullpath):
                                    adconfdirs.append(fullpath)
                        else:
                            pass # print ""
        self.recipedirs = adconfdirs
        # print "CS183:",repr(adconfdirs)
        return adconfdirs

    def general_walk( self, dir, exts=[] ):
        '''
        A generalized walk, that ignores all the .svn / .cvs folders. I found this can be a little useful, 
        although it will probably be thrown out at some point.
        
        @param path: Root path to throw in os.walk.
        @type path: str
        
        @param exts: list of valid type extensions to process. If exts is left as [], then everything is valid.
        the exts should be supplied in the form [".fits",".log",".jpg"].
        @type exts: list of str  
        
        @return: Basically, takes the output of os.walk, but without the .svn stuff.
        @rtype: yields a 3-tuple (dirpath, dirnames, filenames).
        '''
        filelist = []
        for (path, directories, files) in os.walk(dir):
            goodpath = ("svn" not in path) and ("CVS" not in path)
            if goodpath:
                sys.stdout.write(".")
                path = os.path.abspath(path)
                for fname in files:
                    if exts != []:
                        for ext in exts:
                            if ext == os.path.splitext(fname)[1]:
                                yield os.path.join(path, fname)
                                #filelist.append( os.path.join(path, fname) )
                                break
                    else:
                        yield os.path.join(path, fname)
                        #filelist.append( os.path.join(path, fname) )
                    
        print
        
        
def config_walk( spacename = None):
    global cs
    if (cs == None):
        cs = ConfigSpace()
        
    for trip in cs.config_walk(spacename):
        yield trip
        
def config_packs( ):
    global cs
    if (cs == None):
        cs = ConfigSpace()
        
    return cs.configpacks

def from_which(path):
    global cs
    if (cs == None):
        cs = ConfigSpace()
    # note this uses the "package name" is in the path only if it's in that package
    # trick which could be defeated by certain package structures that the 
    # system otherwise would tolerate
    pps = cs.package_paths
    #print "CS324:", repr(pps)
    
    for pack in pps:
        basepack = os.path.basename(pack)
        if pack in path:
            return basepack
    return None
            
    
def general_walk( spacename="", exts=[]):
    global cs
    if (cs == None):
        cs = ConfigSpace()
        
    return cs.general_walk( spacename, exts )

def lookup_path(name):
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
    
    if (fpath[-3:] != ".py") and (fpath[-5:]) != ".fits":
        fpath += ".py"

        
    return fpath
