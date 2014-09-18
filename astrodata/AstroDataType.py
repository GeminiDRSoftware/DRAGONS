#
#                                                                  gemini_python
#
#                                                         astrodata_X1/astrodata
#                                                               AstroDataType.py
# ------------------------------------------------------------------------------
# $Id$
# ------------------------------------------------------------------------------
__version__      = '$Revision$'[11:-2]
__version_date__ = '$Date$'[7:-2]
# ------------------------------------------------------------------------------
"""
This module provides the following class definitions:

  -- LibraryNotLoaded
  -- BadArgument
  -- ClassReq             aliased as ISCLASS
  -- PHUReq               aliased as PHU
  -- DataClassification

These provide functionality to the AstroData typing system.
"""
# ------------------------------------------------------------------------------
import os
import re
import pyfits

import AstroData

from ConfigSpace import config_walk
from astrodata import new_pyfits_version
from astrodata.Requirements import Requirement
from astrodata.Requirements import AND, OR, NOT
# ------------------------------------------------------------------------------
verbose = False
verbt   = False
ldebug  = False
verboseLoadTypes = True
# ------------------------------------------------------------------------------
class LibraryNotLoaded(Exception):
    """Class for raising a particular exceptions"""
    pass
    
class BadArgument(Exception):
    """Class for raising a particular exception"""
    pass

# ------------------------------------------------------------------------------
def get_classification_library():
    try:
        classificationLibrary = ClassificationLibrary()
        return classificationLibrary
    except CLAlreadyExists, s:
        return s.clInstance
    return None

# ------------------------------------------------------------------------------
class ClassReq(Requirement):
    typeReqs = None
    def __init__(self, *args):
        lst = []
        for arg in args:
            if hasattr(arg, "__iter__"):
                lst.extend(arg)
            else:
                lst.append(arg)
        self.typeReqs = lst

    def satisfied_by(self, hdulist):
        numsatisfied = 0
        library = get_classification_library()
        for typ in self.typeReqs:
            if (verbt): 
                print "ADT127 TYPES:", self.typeReqs
            if (library.check_type(typ, hdulist)):
                if(verbt): 
                    print "satisfied"
                numsatisfied = numsatisfied + 1
            else:
                if (verbt): 
                    print "unsatisfied"
                return False
        return True
ISCLASS = ClassReq


class PHUReq(Requirement):
    def __init__(self, phureqs=None, **argd):
        if phureqs == None:
            phureqs = {}
        phureqs.update(argd)
        self.phuReqs = phureqs
        
    def satisfied_by(self, hdulist):
        try:
            if new_pyfits_version:
                phuHeader = hdulist[0].header
                phuCards = phuHeader.cards
                phuCardsKeys = phuHeader.keys()
            else:
                phuCards = hdulist[0].header.ascard
                phuCardsKeys = phuCards.keys()
        except KeyError:
            return False
        
        numreqs = len(self.phuReqs) 
        numsatisfied = 0
        numviolated  = 0
        for reqkey in self.phuReqs.keys():
            # Note: the key can have special modifiers to indicate the
            # key is a regexp (otherwise it's a string literal).
            # May indicate that a flag is not required but PROHIBITED.
            # Other modifiers can be supported, and should appear in a 
            # comma separated list within "{}" before the KEY's text. 
            # E.g. "{re}.*?PREPARE"

            # assume no key mods
            mods_re = False
            mods_prohibit = False
            m = re.match(r"(\{(?P<modifiers>.*?)\}){0,1}(?P<key>.*?)$", reqkey)
            
            # cleanreqkey ... possibly a regexp
            cleanreqkeyPRE = reqkey
            if (m):
                modsstr = m.group("modifiers")
                if modsstr != None:
                    mods = modsstr.split(",")
                    if ("re" in mods):
                        mods_re = True
                    if ("prohibit" in mods):
                        mods_prohibit = True
                    cleanreqkeyPRE = m.group("key")
            
            # before checking value, check if it exists
            # get list of keys
            if (mods_re):
                cleanreqkeylist = AstroData.re_header_keys(cleanreqkeyPRE, 
                                                           hdulist[0].header)
                if cleanreqkeylist is None or len(cleanreqkeylist) == 0:
                    # prohibited flag, return true
                    if (mods_prohibit == True):
                        numsatisfied += 1
                        continue
                    else:
                        return False
            else:
                cleanreqkeylist = [cleanreqkeyPRE]

            # @note: trickery supporting "prohibited" tags and re Keys: 
            # to make this work for regexp lists of keys that matched, 
            # and treat the original non re-case as a special case involves 
            # changing the 'return FALSE' logic to return false ALL the keys 
            # must fail to match; start with a count equal to the number of 
            # keys in the list, and only return false if all of them result 
            # in False
            numviablekeys = len(cleanreqkeylist)
            no_match_msg  = "%s header DID NOT match %s with value of %s"
            for cleanreqkey in cleanreqkeylist:
                if cleanreqkey in phuCardsKeys:
                    try:
                        match = None
                        try:
                            match = re.match(str(self.phuReqs[reqkey]), 
                                             str(phuCards[cleanreqkey].value))
                        except re.error:
                            print """
                            BAD PHU Requirement in Classification '%s'
                            phuKey: '%s'
                            regex : '%s' """ % (repr(self.__class__),
                                                reqkey, str(self.phuReqs[reqkey])
                                               )
                            raise
                            
                        if (match):
                            if (verbose) : 
                                print "%s header matched %s with value: %s" % \
                                    (fname, reqkey, phuCards[reqkey])
                            if (mods_prohibit == False): 
                                numsatisfied = numsatisfied + 1
                            else:
                                numviablekeys -= 1
                                if numviablekeys == 0:
                                    return False
                        else :
                            if (verbose) : 
                                print no_match_msg % \
                                    (fname, reqkey, phuCards[reqkey])
                            if (mods_prohibit == True): 
                                numsatisfied = numsatisfied + 1
                            else:
                                numviablekeys -= 1
                                if numviablekeys == 0:                     
                                    return False
                    except KeyError:
                        if (mods_prohibit == True):
                            numsatisfied = numsatisfied + 1
                        else:
                            if (verbose) : 
                                print no_match_msg % \
                                (fname, reqkey, phuCards[reqkey])
                            numviablekeys -= 1
                            if numviablekeys == 0:
                                return False
                else:
                    # not there, fail unless prohibited
                    if (mods_prohibit == True):
                        numsatisfied += 1
                    else:
                        numviablekeys -= 1
                        if numviablekeys == 0:
                            return False
        if (verbt):
            print ("numreqs=%d\nnumsatisfied=%d\nnumviolated=%d" % \
                   (numreqs, numsatisfied, numviolated))
            
        if (verbose) : print "match"
        return True
PHU = PHUReq

# ------------------------------------------------------------------------------
class DataClassification(object):
    """
    The DataClassification Class encapsulates a single classification type, and
    knows how to recognize that type when given a pyfits.HDUList instance.
    Matching is currently done against PHU header keys, though the object
    is designed to be able to look elsewhere for classifying information.
    Classification configurations are classes subclassed from DataClassification
    with the class variables set appropriately to indicate the PHU requirements.

    The DataClassification class also allows specifying one type as dependant on
    another type, in which case the other type will try to match the PHU headers
    defined for it. When used through AstroData applicable classification names 
    are cached so the PHU is not checked repeatedly.
    
    This object is not intended for general us, and is a worker class for
    the L{ClassificationLibrary}, from the users point of view data 
    classifications are handled as strings, i.e. classification names.
    L{ClassificationLibrary} is therefore the proper interface to use
    for retrieving type information. However, most users will use the 
    L{AstroData} classification interface which in turn rely on 
    L{ClassificationLibrary}.
    
    NOTE: The configuration system and public interface makes a distinction 
    between "typology" classifications and "processing status" classifications.
    Technically there is no real difference between these two types of 
    classification, the difference occurs in the purpose of the two, and the 
    interfaces allow getting one or the other type, or both. In principle 
    however, typology classifications relate to instrument-modes or other 
    classifications that more or less still apply to the data after it has 
    been transformed by processing (e.g. GMOS_IMAGE data is still GMOS_IMAGE 
    data after flat fielding), and processing status classifications will fail 
    to apply after processing.
    E.g. After running 'prepare', GMOS_UNPREPARED data changes to GMOS_PREPARED.
    """
    # RAW TYPE: this is the raw type of the data, None, means search parent 
    # types and use their setting if need be. This support calling descriptors
    # in raw mode.
    rawType = None

    # classification library
    library = None

    # new type requirements
    requirement = None
    
    # Parameter Requirement Dictionary
    usage = ""    
    phuReqs = {}

    # type name
    name = "Unclassified"
    parent    = None
    children  = None
    parentDCO = None           # parent DataClassification Object
    childDCOs = None           # list of child DataClassification Object
    
    # So the type knows its source file... if we don't store them on disk
    # then this would become some other locator.  Valuable if there is a
    # reason to update type definitions dynamically (i.e. to support a 
    # type definition editor)
    fullpath   = ""
    phuReqDocs = {
        "FILTER3" : "NIRI Instrument Mode: Imaging or Spectroscopy",
        "INSTRUME": "Instrument Used",
        "MASKNAME": "Mask for GMOS MOS Spectroscopy",
        "OBSMODE" : "GMOS Mode: IMAGE, IFU, MOS, or LONGSLIT.",
        "OBSERVAT": "Observatory, Gemini-South or Gemini-North",
        "TELESCOP": "Observatory, Gemini-South or Gemini-North"
    }


    def add_child(self, child):
        if self.children == None:
            self.children = []
        if self.childDCOs == None:
            self.childDCOs = []
        if child.name not in self.children:
            self.children.append(child.name)
        if child not in self.childDCOs:
            self.childDCOs.append(child)
        
    def assert_type(self, hdulist):
        """
        This function will check to see if the given HDUList instance is
        of its classification. Currently this function checks PHU keys
        in the C{hdulist} argument as well as check to see if any 
        classifications upon which this classification is dependent apply.
        To extend what is checked to other details, such as headers in 
        data extensions, this function must change or be overridden by a 
        child class.

        @param hdulist: an HDUList as returned by pyfits.open()
        @type  hdulist: <pyfits.HDUList>

        @return: C{True} if class applies to C{hdulist}, OR C{False}.
        @rtype:  <bool>
        """
        if (ldebug):
            print "asserting %s"  % self.name
            
        if (self.library == None):
            raise LibraryNotLoaded
            
        # New Requirement style
        if (self.requirement):
            return self.requirement.satisfied_by(hdulist)
        else:
            return False
            
    def is_subtype_of(self, supertype):
        """This function is used to check type relationships. For this type to 
        bea "subtype" of the given type both must occur in a linked tree.
        A node which is a relative leaf is a considered a subtype of its 
        relative root.

        This is used to resolve conflicts that can occur when objects, features
        or subsystems are associated with a given classification. Since 
        AstroData instances generally have more than one classification 
        and some associated objects, features, or subsystems require that
        they are the only one ultimately associated with a particular AstroData
        instance,  as with Descriptors, this function is used to correct the 
        most common case of this, in which one of the types is a subtype of 
        the other, and therefore can be taken to override the parent level 
        association.

        @param supertype: string name for "supertype"
        @type  supertype: <str>

        @returns: True if the classification detected by this DataClassification 
                  is subtype of the named C{supertype}.
        @rtype:   <bool>
        """
        # seek supertype in my parent (typeReqs).
        if supertype == self.parent:
            return True
        else:
            if self.parentDCO and self.parentDCO.is_subtype_of(supertype):
                return True
        return False
        
    def get_super_types( self, append_to=None ):
        """
        Returns a list of all immediate parents.
        
        @return: A List of parents of DataClassificationType. To get the name 
                 of the type, simply take an element from the list use the 
                 '.name'
        @rtype: <list>
        """
        if append_to == None:
            superTypes = []
        else:
            superTypes = append_to
        
        if self.parentDCO:
            superTypes.append(self.parentDCO)
            superTypes = self.parentDCO.get_super_types(append_to=superTypes)
        return superTypes
        
    def get_all_super_types( self, height=0 ):
        """
        Returns a list of all parents and grandparents of self in level order of 
        self. Within each level, sorting is based on how typereqs was input. 
        For example, if typereqs = [a,b,d,c], then that level will be [a,b,d,c] 
        and the first parents in the level above will be from a, then b, then d 
        and then c.
        
        @param height: The height from which to get achieve parents up to. 
        A height <= 0, will return all parents/grandparents.
        @type height: int
        
        @return: A List of parents and grandparents of DataClassification type.
                 To get the name of the type, simply take an element from the 
                 list use the '.name'
        @rtype: <list>
        """

        immediate_superTypes = self.get_super_types()
        all_superTypes = immediate_superTypes
        
        if height <= 0 or height > 1000:
            height = 1000
        
        counter = 0  
        while counter < height:
            next_superTypes = []
            for superType in immediate_superTypes:
                temp = superType.get_super_types()
                next_superTypes += temp
                all_superTypes += temp
            immediate_superTypes = list(set(next_superTypes))
            if immediate_superTypes == []:
                # There are no more parents
                break
            counter += 1
        
        if counter >= 1000:
            err = "Cannot resolve '%(name)s's type requirements."
            raise RuntimeError(err % {"name": str(self)})
        return list(set(all_superTypes))          # Removes duplicates


    def walk(self, style = "both"):
        yield self
        if style == "children":
            if self.childDCOs:
                for childDCO in self.childDCOs:
                    for typ in childDCO.walk(style="children"):
                        yield typ
        elif style == "parent":
            if self.parentDCO:
                for typ in self.parentDCO.walk(style="parent"):
                    yield typ
        else:
            for typ in self.walk(style="parent"):
                yield typ
            for typ in self.walk(style="children"):
                yield typ
    
    def python_class(self):
        """
        This function generates a DataClassification Class based on self.
        The purpose of this is to support classification editors.
        @returns: a string containing python source code for this instance. 
        Note, of course, if you add functions or members to a child class 
        derived from DataClassification, they will not be recognized by this 
        function and will not be represented in the outputed code.

        @rtype: <str>
        """
        class_templ = """
        class %(typename)s(DataClassification):
        name="%(typename)s"
        usage = "%(usage)s"
        typeReqs= %(typeReqs)s
        phuReqs= %(phuReqs)s
        newtypes.append(%(typename)s())
        """

        code = class_templ % { "typename" : self.name,
                                "phuReqs" : str(self.phuReqs),
                                "typeReqs": str(self.typeReqs),
                                "usage"   : self.usage
                               }
        return code

    def json_typetree(self):
        sdict = {"name":self.name,}
        if self.childDCOs:
            cdict = []
            sdict["children"]=cdict
            for childdco in self.childDCOs:
                cdict.append(childdco.json_typetree())
        return sdict

# ------------------------------------------------------------------------------
class CLAlreadyExists():
    """
    This class exists to return a singleton of the ClassificationLibrary 
    instance. See L{ClassificationLibrary} for more information.
    
    NOTE: This should be refactored into a non-exception calling version 
    that uses the __new__ operator instead of __init__. This method is required 
    because __init__ cannot return a value, so instead throws an exception if 
    the ClassficationLibrary has already been created. Please use the AstroData 
    interface for the classification to access data classification information 
    so that your code does not break when this is refactored. That interface is 
    far more convienent and in most cases you only need to use the 
    ClassificationLibrary if you are working with data and therefore have access 
    to an AstroData instance.
    """
    clInstance = None           
              
# ------------------------------------------------------------------------------
class ClassificationLibrary (object):
    """
    This class exists as the proper full interface to the classification 
    features, though most users should merely use the classification interface 
    provided through AstroData. Single DataClassification class instances can 
    report if their own classifications apply, but only a complete library 
    encapsulates the whole classification system. To find if a single 
    classification applies, the coder asks the library by providing the 
    classification by name.

    DataClassification objects are not passed around, but used only to detect 
    the classfication. Script authors therefore should not converse directly 
    with the data classification classes, and instead allow them to be managed 
    by the Library. 
    
    This Library also knows how to produce HTML documentation of itself, so 
    that the classification definitions can be the sole source of such 
    information and thus keep the documentation as up to date as possible.
    
    @note: Classification Names, aka Gemini Type Names, are strings, the python 
    objects are not used to report types, only detect them.  When passed in and 
    out of functions classifications are always represented by their string 
    names.
    
    @note: This class is a singleton, which means on the second attempt to 
    create it, instead of a new instance one will recieve a pointer to the 
    instance already created. This ensures that only one library will be loaded 
    in a given program, which desireable for efficiency and coherence to a 
    single type family for a given processing session.  
    
    This is accomplished as the constructor for ClassificationLibrary
    keeps a class static variable with it's own instance pointer, if this 
    pointer is already set, the constructor (aka __init__()) throws an 
    exception, an instance of CLAlreadyExists which will contain the reference 
    to the ClassificationLibrary instance.
    
    It is not advised to instantiate the ClassificationLibrary in a regular 
    call like,

    C{cl = ClassificationLibrary()}, 

    instead, use code such as the following:
    
      if (self.classification_library == None):
            try:
                self.classification_library = ClassificationLibrary()
            except CLAlreadyExists, s:
                self.classification_library = s.clInstance
                
        return self.classification_library
    
    The L{AstroData.get_classification_library} function retrieves the instance
    handle this way.
    
    This method is slated to be replaced, to avoid being affected by the change
    use the AstroData class' interface to classification features.
    """
    definitionsPath = None          # set in __init__(..)
    definitionsStorageREMask = None # set in __init(..)
    
    # This class acts as a singleton.
    __single = None

    # type dictionaries
    typesDict    = None
    statusDict   = None
    typologyDict = None
    
    @classmethod
    def get_classification_library(cls):
        if ClassificationLibrary.__single:
            return ClassificationLibrary.__single
        try:
            classification_library = ClassificationLibrary()
        except CLAlreadyExists, s:
            classification_library = s.clInstance
        return classification_library
                
    # This is the directory to look for parameter requirements 
    #(which serve as type definitions)
    def __init__(self, context="default"):
        """
        parameters: context -- Name of the context from which to retrieve
                               the library.
        @type       context: <str>
        """
        
        if (ClassificationLibrary.__single):
            cli = CLAlreadyExists()
            cli.clInstance = ClassificationLibrary.__single
            raise cli
        else:
            # NOTE: Use this file's name to get path to types
            rootpath = os.path.dirname(os.path.abspath(__file__))
            self.definitionsPath = os.path.join(rootpath ,"types")
            self.definitionsStorageREMask = r"(gemdtype|adtype)\.(?P<modname>.*?)\.py$"
            self.typesDict = {}
            self.typologyDict = {}
            self.statusDict = {}
            self.load()
            ClassificationLibrary.__single = self

    def load(self):
        """
        This function loads the classification library, in general 
        by obtaining python modules containing classes 
        which descend from DataClassification (or share its interface)
        and evaluating them.
        @returns: Nothing
        """
        if (verbose):
            print __file__
            print "definitionsPath=", self.definitionsPath
            print "definitionsStorageREMask=", self.definitionsStorageREMask
 
        self.load_types("types", self.typesDict, self.typologyDict)
        self.load_types("status", self.typesDict, self.statusDict)
        self.trace_parents()
        return
        
    def load_types(self, spacename, globaldict, tdict):
        """
        This function loads all the modules matching a given naming convention
        (regular expression "B{gemdtype\.(?P<modname>.*?)\.py$}")
        recursively within the given path. Loaded files are assumed to be
        either DataClassifications or share DataClassification's interface.
        Note, multiple DataClassifications can be put in the same file, 
        and it is important to add the new class to the C{newtypes} variable
        in the definition file, or else the load function will not see it.
        """
        for root, dirn, files in config_walk(spacename):
            for dfile in files:
                if (re.match(self.definitionsStorageREMask, dfile)):
                    fullpath = os.path.join(root, dfile)
                    import py_compile
                    defsFile = open(fullpath)
                    newtypes = []
                    exec (defsFile)
                    defsFile.close()
                    # newtype is declared here; used in the definition file to 
                    # pack in new types and return them to this scope.
                    for newtype in newtypes:
                        newtype.fullpath = fullpath
                        newtype.library = self
                        globaldict[newtype.name] = newtype
                        tdict[newtype.name] = newtype
                else :
                    if (verbose) : print "ignoring %s" % dfile

        for typ in globaldict.keys():
            typo = globaldict[typ]
            if typo.parent:
                if typo.parent in globaldict:
                    globaldict[typo.parent].add_child(typo)
        return


    def check_type(self, typename, dataset):
        """
        This function will check to see if a given type applies to a given 
        dataset.
        
        @param typename: then name of the type to check the C{dataset} against.
        @type  typename: <str>
        
        @param dataset: the data set in question
        @type  dataset: L{AstroData} instance or B{string}

        @returns: True if type applies to the dataset, False otherwise
        @rtype:   <bool>
        """
        
        if isinstance(dataset, AstroData.AstroData):
            hdulist = dataset.hdulist
        else:
            if not isinstance(dataset, pyfits.HDUList):
                raise BadArgument
            hdulist = dataset
        
        if typename in self.typesDict:
            retval = self.typesDict[typename].assert_type(hdulist)
        else:
            return False         
        return retval

    def is_name_of_type(self, typename):
        if typename in self.typesDict:
            return True
        else:
            return False
            
    def get_available_types(self):
        return self.typesDict.keys()
        
    def get_type_obj(self, typename):
        """
        Generally users do not need DataClassification instances, however
        if you really do need that object, say to write an editor... this 
        function will retrieve it.

        @param typename: name of the classification for which you want the
                         associated DataClassification instance
        @type typename: string

        @returns: the correct DataClassification instance
        @rtype: DataClassification
        """
        try:
            rettype = self.typesDict[typename]
            return rettype
        except KeyError:
            return None
            
    def type_is_child_of(self, typename, parenttyp):
        child = self.get_type_obj(typename)
        parent = self.get_type_obj(parenttyp)
        return child.is_subtype_of(parenttyp)

    def discover_types(self, dataset, all=False):
        """
        This function returns a list of string names for the classifications
        which apply to this dataset.
        @param dataset: the data set in question
        @type dataset: either L{AstroData} instance or L{HDUList}
        @param all: flag to drive if a dictionary of three different lists
        is returned, if C{all} is True.  If False, a list of all status and 
        processing types is returned as a list of strong names.  If True, a 
        dictionary is returned.  Element with the key "typology" will contain 
        the typology related classifications, key "status" will contain the 
        processing status related classifications, and key "all" will contain 
        the union of both sets and is the list returned when "all" is False.
        @type all: Bool
        @returns: the data type classifications which apply to the given dataset
        @rtype: list or dict of lists
        """
        retarya = self.discover_classifications(dataset, self.typologyDict)
        retaryb = self.discover_classifications(dataset, self.statusDict)
        retary = []
        retary.extend(retarya)
        retary.extend(retaryb)

        if (all == True):
            retdict = {}
            retdict["all"] = retary
            retdict["typology"] = retarya
            retdict["status"]   = retaryb
            return retdict
        else:
            return retary
    
    def discover_status(self, dataset):
        """
        Returns a list of string names for the processing status related 
        classifications which apply to this dataset.

        @param dataset: the data set in question
        @type dataset: either L{AstroData} instance or L{HDUList}

        @returns: type classifications which apply to the given dataset
        @rtype:   <list> of strings
        """
        return self.discover_classifications(dataset, self.statusDict)
        
    def discover_typology(self, dataset):
        """Returns a list of string names for the typological
        classifications which apply to this dataset.

        @param dataset: the data set in question
        @type  dataset: L{AstroData} instance or L{HDUList}

        @returns: the data type classifications which apply to the given dataset
        @rtype:   <list> of strings
        """
        return self.discover_classifications(dataset, self.typologyDict)
        
    def discover_classifications(self, dataset, classification_dict):
        """
        Returns a list of classifications ("data types").
        
        @param dataset: the data set in question
        @type  dataset: L{AstroData} instance or B{string}

        @return: Returns list of DataClassification names
        @rtype:  <list> of strings
        """
        typeList = []
        closeHdulist = False

        if isinstance(dataset, basestring):
            try:
                hdulist = pyfits.open(dataset)
                closeHdulist = True
            except:
                print ("we have a problem with ", dataset)
                raise
        elif isinstance(dataset, AstroData.AstroData):
            hdulist = dataset.hdulist
            closeHdulist = False
        else:
            hdulist = dataset
            
        for tkey, tobj in classification_dict.items():
            if (tobj.assert_type(hdulist)):
                typeList.append(tobj.name)
                
        if (closeHdulist):
            hdulist.close()
        return typeList

    def trace_parents(self):
        """
        DataClassifications are generally specified by name for the user
        and only the AstroDataType module cares about the actual 
        DataClassification object.  However, it's needed in order to trace 
        precedence when assigning features, like descriptor calculator or 
        primitive sets.  This function traces through the string names of 
        parents, and create members which will make it easy to walk the tree.
        
        The short explanation of what this function does is set the DCO members,
        parentDCO and childDCOs, as well as the children member which contains 
        the string names of the child types.
        """
        
        for typ in self.typesDict.keys():
            dco = self.typesDict[typ]
            if dco.parent:
                dco.parentDCO = self.get_type_obj(dco.parent)
                dco.parentDCO.add_child(dco)
        return
