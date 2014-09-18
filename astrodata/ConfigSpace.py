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
CONFIGMARKER  = "ADCONFIG_"
RECIPEMARKER  = "RECIPES_"
LOOKUPDIRNAME = "lookups"
PIFMARKER     = "PIF_"
DDLISTMARKER  = "DescriptorsList_(.*).py$"
DDLISTFORMAT  = "DescriptorsList_%s.py"
CALCIFACEMARKER = "CalculatorInterface_(.*).py$"
CALCIFACEFORMAT = "CalculatorInterface_%s.py"
spaces = {"descriptors":"descriptors",
          "structures":"structures",
          "types":"classifications/types",
          "status": "classifications/status",
          "xmlcalibrations":"xmlcalibrations",
          }
cs = None

class ConfigSpaceError(Exception):
    pass

class ConfigSpace(object):
    """
    This class exists to connect to a configuration space, such as 
    AstroDataType libraries or Descriptors. It provides tools to simplify 
    accessing configuration information and also providing indirection to
    allow configurations to be stored with alternate storage methodologies,
    such as in relational databases.  This flexibility is useful due to the
    many deployment contexts of the Gemini Reduction Package.
    """
    
    configdirs = None
    recipedirs = None
    configpacks = None
    recipepath = None
    adconfigpath = None
    wholepath = None
    package_paths = None
    calc_iface_list=[]
    
    def __init__(self):
        self.configdirs    = {}
        self.configpacks   = []
        self.package_paths = []
        self.recipepath    = []

        if "RECIPEPATH" in os.environ:
            rpath = os.environ["RECIPEPATH"].split(":")
            self.recipepath = rpath

        self.adconfigpath = []
        if "ADCONFIGPATH" in os.environ:
            rpath = os.environ["ADCONFIGPATH"].split(":")
            self.adconfigpath = rpath
            
        self.wholepath = copy(self.recipepath)
        self.wholepath.extend(self.adconfigpath)
        self.wholepath.extend(sys.path)
    
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
                                self.calc_iface_list.append(
                                        ("DDLIST", os.path.join(elem[0], fil)))
                    self.curpack = from_which(elem)        
                    yield elem
            
    def get_config_dirs(self, spacename):
        """
        This function returns a list of directories to walk for a given 
        configuration space.
        @param spacename: name of the config space to collect directories for
        @type spacename: string
        @returns: list of directories
        @rtype: list
        """
        if (self.configdirs != None):
            if spacename in self.configdirs:
                return self.configdirs[spacename]
        else:
            print "CS88:", "inefficency"
            
        # get the config space standard postfix for directories
        try:
            postfix = spaces[spacename]
        except KeyError:
            raise ConfigSpaceError("ConfigSpace name not recognized (%s)" \
                                    % spacename)
        
        retdirs = []
        
        adconfdirs = []
        packagemask = {}
        i = 1
        for path in self.wholepath:
            if  os.path.abspath(path) !=  os.path.abspath(os.getcwd()):
                if os.path.isdir(path):
                            subdirs = os.listdir(path)
                            for subpath in subdirs:
                                if not os.path.isdir(os.path.join(path,subpath)):
                                    continue
                                if PACKAGEMARKER in subpath:
                                    ppath = os.path.join(path, subpath)
                                    pknam = os.path.basename(ppath)
                                    if pknam in packagemask:
                                        continue
                                    else:
                                        packagemask[pknam]=ppath
                                    
                                    if ppath not in self.package_paths:
                                        self.package_paths.append(ppath)
                                    
                                    subsubpaths = os.listdir(os.path.join(path,subpath))
                                    for subsubpath in subsubpaths:
                                        if not os.path.isdir(
                                                        os.path.join(
                                                            path, subpath,subsubpath)):
                                            continue
                                        if CONFIGMARKER in subsubpath:
                                            packdir = os.path.join(path,subpath, subsubpath)
                                            if packdir not in self.configpacks:
                                                self.configpacks.append(packdir)
                                            fullpath = os.path.join(path, subpath, subsubpath, postfix)
                                            if os.path.isdir(fullpath):
                                                adconfdirs.append(fullpath)
                                        elif subsubpath.startswith(PIFMARKER):
                                            pifdir = os.path.join(path, subpath, subsubpath)
                                            if pifdir not in sys.path:
                                                sys.path.append(pifdir)
                                            
                            else:
                                pass
                            
        self.configdirs.update({spacename: adconfdirs})
        return adconfdirs

    def get_recipe_dirs(self):
        """
        This function returns a list of directories to walk for a given 
        configuration space.

        @param spacename: name of the config space to collect directories for
        @type  spacename: string

        @returns: list of directories
        @rtype:   list
        """
        
        if (self.recipedirs != None):
            return self.recipedirs
        
        retdirs = []
        adconfdirs = []
        pathlist = sys.path
        i = 1
        if "RECIPEPATH" in os.environ:
            rpath = os.environ["RECIPEPATH"].split(":")
            rpath.extend(pathlist)
            pathlist = rpath

        if "ADCONFIGPATH" in os.environ:
            rpath = os.environ["ADCONFIGPATH"].split(":")
            rpath.extend(pathlist)
            pathlist = rpath
        
        pkmask = {}
        for path in pathlist:
            if os.path.isdir(path):
                        subdirs = os.listdir(path)
                        for subpath in subdirs:
                            if not os.path.isdir(os.path.join(path,subpath)):
                                continue
                            if PACKAGEMARKER in subpath:
                                pknam = os.path.basename(subpath)
                                if pknam in pkmask:
                                    continue
                                else:
                                    pkmask[pknam] = subpath

                                subsubpaths = os.listdir(os.path.join(path,subpath))
                                for subsubpath in subsubpaths:
                                    if RECIPEMARKER in subsubpath:
                                        fullpath = os.path.join(path, subpath, subsubpath)
                                        if os.path.isdir(fullpath):
                                            adconfdirs.append(fullpath)
                            elif RECIPEMARKER in subpath:
                                fullpath = os.path.join(path, subpath)
                                if os.path.isdir(fullpath):
                                    adconfdirs.append(fullpath)
                        else:
                            pass
        self.recipedirs = adconfdirs

        return adconfdirs

    def general_walk( self, dir, exts=[] ):
        """
        A generalized walk, that ignores all the .svn / .cvs folders. 
        
        @param path: Root path to throw in os.walk.
        @type path: str
        
        @param exts: list of valid type extensions to process. 
                     If exts is left as [], then everything is valid.
                     exts should be in the form [".fits",".log",".jpg"].
        @type exts: list of str  
        
        @return: os.walk without the .svn
        @rtype:  yields a 3-tuple (dirpath, dirnames, filenames).
        """
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
                                break
                    else:
                        yield os.path.join(path, fname)
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

    pps = cs.package_paths
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
    """
    This module level function takes a lookup name and returns a path to the file.
    """
    global cs
    if (cs == None):
        cs = ConfigSpace()
        
    a = name.split("/")
    domain = a[0]
    pack = CONFIGMARKER + domain
    tpath = None
    for path in cs.configpacks:
        if path[-(len(pack)):] == pack:
            tpath = path
            break
            
    if tpath == None:
        raise ConfigSpaceError("No Configuration Package Associated with %s" \
                                % domain)
    fpath = os.path.join(tpath, LOOKUPDIRNAME, *a[1:])
    if (fpath[-3:] != ".py") and (fpath[-5:]) != ".fits":
        fpath += ".py"
        
    return fpath
