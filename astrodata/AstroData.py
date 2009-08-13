#!/bin/env pyth
import sys
import pyfits

from AstroDataType import *

import Descriptors


from CalculatorInterface import CalculatorInterface
import Calculator
import re

verbose = False
verboseLoadTypes = True
verbt = False

#FUNCTIONS
def reHeaderKeys(rekey, header):
    """This utility function returns a list of keys from 
    the header passed in which match the given regular expression.
    @param rekey: a regular expresion to match to keys in header
    @type rekey: string
    @param header: a Header object as returned
    @type header: pyfits.Header"""
    retset = []
    for k in header.ascardlist().keys():
        # print "gd278: key=%s" % k
        if re.match(rekey, k):
            retset.append(k)

    if len(retset) == 0:
       retset = None
       
    return retset

class gdExcept:
    pass

class AstroData(object, CalculatorInterface):
    """
    The AstroData Class is a class designed to handle sets of astronomic data as 
    a single unit. Currently the 
    data must be stored in Multi-extension FITS files (aka "MEF files"). Other 
    file types could be supported, but
    at this time only MEF files are. AstroData adds some semantic knowledge
    over what one has when accessing such files directly with pyfits (etc),
    information about the relationships between the data and allows you to look at a
    complete bundle of data which includes multiple sets of data and a variety
    of meta information about the data. 
    
    The class allows ways to define, in 
    configurations, how the separate extensions in a MEF file are related, e.g.
    allowing the class to recognize the relationship between three extensions when
    one is the science, one is a varience plane, and the other a data mask, or to
    understand that the a mask definition is associated with a series of spectra,
    so that the definition can be propagated even by scripts which do not know about,
    or care about that particular association, at that particular point in processing.
    All type and related definitions are loaded as configurations, so this semantic
    knowledge is not encoded directly in this class, but sits in configuration files
    used
    by subordinate classes such as the L{AstroDataType Class<AstroDataType>}.
    
    In general one can consider the functionality to consist of
    file handling, data handling, type checking, and managing
    meta-information in the case of MEF file. However, the class proper really
    just coordinates other classes to accomplish it's goals, e.g. for file handling 
    and data handling the
    class uses python standard Pyfits and NumPy. The type
    services, projection of structures, normalization of standard meta data about
    and observation, are handled by custom classes. See also L{AstroDataType},
    L{Structure}, and L{Descriptors}.
    """
    
    types = None
    typesStatus = None
    typesTypology = None
    filename = None
    hdulist = None
    hdurefcount = 0
    mode = "readonly"
    descriptorCalculator = None
    # index is for iterator behavior
    index = 0
    # for subdata
    borrowedHDUList = False # if we took the hdul from another GD instance
    container = None # AstroData instance we took hdul from
    
    # None means "all", otherwise, an array of extensions
    extensions = None
    
    # ClassificationLibrary Singleton, must be retrieved through
    #   getClassificationLibrary()
    classificationLibrary = None
    
    def __del__(self):
        """ This is the destructor for AstroData. It performs reference 
        counting and behaves differently when this instance is subdata, since
        in that case some other instance "owns" the pyfits HDUs instance.
        """
        
        if self.borrowedHDUList:
            self.container.relhdul()
            self.hdulist = None
        else:
            if (self.hdulist != None):
                self.hdulist.close()
                
    def __getitem__(self,ext):
        """This function support the "[]" syntax.  We use it to create
        AstroData objects associated with "subdata"... that is, a limited
        subset of extensions in the given MEF. e.g.::
        
            datasetA = AstroData.AstroData("file.fits")
            datasetB = datasetA[SCI]
            
        In this case datasetB would be a AstroData object associated with the
        same mef, but which will behave as if the SCI extensions are the only
        extensions in the file.  Note, datasetA and datasetB share the PHU and
        also the datastructures of the HDUs they have in common, but different 
        HDUList instances.
        
        @param ext: EXTNAME name for this subdata instance.
        @type ext: string
        @returns: AstroData instance associated with the subset of data
        @rtype: AstroData
        """
        hdul = self.gethdul()
        # ext can be tuple, an int, or string ("EXTNAME")
        
        exs = []
        if (type(ext) == str):
            # str needs to be EXTNAME, so we go over the extensions
            # to collect those with the correct extname
            hdul = self.gethdul()
            maxl = len(hdul)
            count = 0
            extname = ext
            #print "gd80: %s" % ext
            for i in range(0,maxl):
                try:
                    # note, only count extension in our extension
                    
                    if (hdul[i].header["EXTNAME"] == extname):
                        try:
                            extver = hdul[i].header["EXTVER"]
                        except KeyError:
                            extver = 1
                            
                        exs.append((extname, extver))
             #           print "gd87:hdul[i] == hdul[%s]" % exs
                except KeyError:
                    #print " gd84: keyerror:[%s]" % extname
                    pass
            self.relhdul()

            return AstroData(self, exts=exs)
            
        elif (type (ext) == tuple) or (type(ext) == int):
            # print "gd121: TUPLE or INT!"
            try:
                #make sure it exists
                if type(ext) == int:
                    ext = ext+1 # so 0 does not mean HDU, but 0'th content-extension
                exttmp = hdul[ext]
            except KeyError:
                # print "gd105: keyerror:[%s]" % ext
                # selector not valid
                self.relhdul()
                return None

            gdpart = AstroData(self, exts=[ext])
            self.relhdul()
            #print "gd132: %s" % str(gdpart)
            return gdpart
        else:
            raise KeyError()
            
    def __len__(self):
        """This is the length operator for AstroData.
        @returns: number of extensions minus the PHU
        @rtype: int"""
        return len(self.hdulist)-1
    
    def __init__(self, fname, mode="readonly", exts = None, extInsts = None):
        """
        Constructor for AstroData. Note, the file will be opened.
        @param fname: filename of MEF to load
        @type fname: string
        @param mode: an [optional] IO access mode, same as pyfits mode, see
        L{open()} for a list of supported modes.
        @type mode: string
        @param exts: a list of extensions this instance should refer to, given 
        integer or tuples specifying each extention. I.e. (EXTNAME,EXTVER) tuples or 
        and integer index which specifies the ordinal position of the extension in the MEF 
        file, begining with index 0 for the PHU. NOTE: if present this option will
        override and obscure the extInsts argument which will be ignored.
        @type exts: list
        @param extInsts: a list of extensions this instance should refer to, given as
        actual pyfits.HDU instances. NOTE: if the "exts" argument is also set,
        this argument is ignored.
        @type extInsts: list
        """
        
        # not actually sure we should open right away, but 
        #  I do suppose the user expects to give the file name
        #  on construction.  To not open now requires opening
        #  in a lazy manner, which means setting up a system for that.
        #  Therefore, I'll adopt the assumption that the file is 
        #  open on construction, and later, add lazy-open to all
        #  functions that would require it, should we optimize that way.

        # !!NOTE!! CODE FOLLOWING THIS COMMENT IS REQUIRED BY DESIGN
        # "extensions" first so other
        # initialization code knows this is subdata
        # set the parts
        # print exts
        self.extensions = exts
        self.extInsts = extInsts
        
        self.open(fname, mode)
        
    #ITERATOR FUNCTIONS
    def __iter__(self):
        """This function exists so that AstroData can be used as an iterator.
        It initializes the iteration process, resetting the index of the 
        current extension.
        @returns: self
        @rtype: AstroData"""
        self.index = 0
        return self
        
    def next(self):
        """This function exists so that AstroData can be used as an iterator.
        This function returns the objects "ext" in the following line:
        
        for ext in gemdatainstance:
        
        If this AstroData instance is associated with a subset of the data in
        the MEF to which it refers, then this iterator goes through that subset
        order (as given by the 
        
        @returns: a single extension AstroData instance representing the current
        extension in the data.
        @rtype: AstroData
        """
        try:
            if self.extensions == None:
                # note, I do not iterate over the PHU at index = 0
                # we consider this a MEF header, not extension header
                # though strictly speaking it is an "extension" from a 
                # FITS standard point of view.
                ext = self.index
            else:
                ext = self.extensions[self.index]
        except IndexError:
            raise StopIteration
        
        self.index += 1
        try:
            # don't be fooled by self[ext]... this creates an AstroData instance
            retobj = self[ext]
        except IndexError:
            raise StopIteration
        
        return retobj
    
    def append(self, moredata=None, data=None, header=None):
        """This function appends more HDUs to this AstroData
        instance.
        @param moredata: either an AstroData instance, an HDUList instance, 
        or an HDU instance. When present, data and header will be ignored.
        @type moredata: pyfits.HDU, pyfits.HDUList, or AstroData
        @param data: if moredata not specified, data and header are used to make 
        and HDU which is then added to the HDUList associated with this
        AstroData instance.
        @type data: numarray.numaraycore.NumArray
        @param header: if moredata not specified, data and header are used to make 
        and HDU which is then added to the HDUList associated with this
        AstroData instance.
        @type header: pyfits.Header
        """
        if (moredata == None):
            self.hdulist.append(pyfits.ImageHDU(data = data, header=header))
        elif isinstance(moredata, AstroData):
            self.hdulist.append(moredata.hdulist[1:])
        elif isinstance(moredata, pyfits.HDU):
            self.hdulist.append(moredata)
        elif type(moredata) is pyfits.HDUList:
            for hdu in moredata[1:]:
                self.hdulist.append(hdu)
    
    def close(self):
        """This function will close the attachment to the file on disk
        if this instance opened that file.  If this is subdata, e.g.
        sd = gd[SCI] where gd is another AstroData instance, sd.close()
        will not close the hdulist because gd will actually own the
        hold on that file."""
        
        if self.borrowedHDUList:
            self.container.relhdul()
            self.hdulist = None
        else:
            if self.hdulist != None:
                self.hdulist.close()
                self.hdulist = None

    def getData(self):
        """Function returns data member(s), specifically for the case in which
        the AstroData instance has ONE extension (in addition to PHU). This
        allows a single extension AstroData instance to be used as though
        it is simply one extension, e.g. allowing gd.data to be used in
        place of the more esoteric and ultimately more dangerous gd[1].data.
        One can assure one is dealing with single extension AstroData instances
        when iterating over the AstroData extensions, e.g.::
        
            for gd in dataset[SCI]:
                ...
                
        @raise: gdExcept if AstroData instance has more than one extension 
        (not including PHU).
        @return: data array associated with the single extension
        @rtype: NumArray
        """
        hdl = self.gethdul()
        if len(hdl) == 2:
            retv = hdl[1].data
        else:
            # print "gd207: %d" % len(hdl)
            raise gdExcept()
            
        self.relhdul()
        return retv

    def setData(self, newdata):
        """This function sets the data member(s) of a data section of an HDU, specifically for the case in which
        the AstroData instance has ONE extension (in addition to PHU).  This cases
        should be assured when iterating over the AstroData extensions, e.g.::
        
            for gd in dataset[SCI]:
                ...
                
        @raise gdExcept: if AstroData instance has more than one extension 
        (not including PHU).
        @param newdata: new data objects
        @type newdata: numarray.numarraycore.NumArray
        """
        hdl = self.gethdul()
        if len(hdl) == 2:
            retv = hdl[1].data
        else:
            raise gdError()
            
        self.relhdul()
        return retv
        
    data = property(getData, setData)
    
    def getHeader(self, extension = None):
        """
        Function returns header member for SINGLE EXTENSION MEFs (which are those that
        have only one extension plus PHU). This case 
        is assured when iterating over extensions using AstroData, e.g.:
        
        for gd in dataset[SCI]: ...
        
        @raise gdExcept: Will raise a gdExcept exception if more than one extension exists. 
            (note: The PHU is not considered an extension in this case)
        @return: header
        @rtype: pyfits.Header
        """
        if extension == None:
            hdl = self.gethdul()
            if len(hdl) == 2:
                retv = hdl[1].header
            else:
                print "numexts = %d" % len(hdl)
                raise gdExcept()

            self.relhdul()
            return retv
        else: 
            hdl = self.gethdul()
            retv = hdl[extension].header
            self.relhdul()
            return retv
            
    def setHeader(self, header, extension=None):
        """
        Function sets the extension header member for SINGLE EXTENSION MEFs 
        (which are those that have only one extension plus PHU). This case 
        is assured when iterating over extensions using AstroData, e.g.:
        
        for gd in dataset[SCI]: ...
        
        @param header: header to set for given extension
        @type header: pyfits.Header
        
        @param extension: Extension index from which to retrieve header, if None or not present then this must be
        a single extension AstroData instance, which contains just the PHU and a single data extension, and the data
        extension's header is returned.
        @type extension: int or tuple, pyfits compatible extension index
        
        @raise gdExcept: Will raise a gdExcept exception if more than one extension exists. 
            (note: The PHU is not considered an extension in this case)
        """
        if extension == None:
            hdl = self.gethdul()
            if len(hdl) == 2:
                hdl[1].header = header
            else:
                raise gdExcept("Not single extension AstroData instance, cannot call without explicit extension index.")

            self.relhdul()
        else:
            self.hdulist[extension].header = header
                    
    header = property(getHeader,setHeader)

    def getHeaders(self):
        """
        Function returns header member(s) for all extension (except PHU).
        @return: list of pyfits.Header instances
        @rtype: pyfits.Header
        """
        hdl = self.gethdul()
        
        retary = []
        
        for hdu in hdl:
            retary.append(hdu.header)

        self.relhdul()
        return retary
        
    
    def open(self, source, mode = "readonly"):
        '''
        This function initiates interaction with a given set of
        AstroData. Note, this is not the way one generally opens a 
        MEF with AstroData, instead, pass the filename into the
        constructor. This function can still be of use if
        the AstroData object has been closed.
        @param source: source for data to be associated with this instance, can be 
        an AstroData instance, a pyfits.HDUList instance, or a string filename.
        @type source: string | AstroData | pyfits.HDUList
        @param mode: IO access mode, same as the pyfits open mode, C{readonly},
        C{update}, or C{append}.  The mode is passed to pyfits so if it is an
        illegal mode name, pyfits will be the subsystem reporting the error. 
        @type mode: string
        @return: nothing
        '''
                
        # might not be a filename, if AstroData instance is passed in
        #  then it has opened or gets to open the data...
        if isinstance(source, AstroData):
            self.filename = source.filename
            self.borrowedHDUList = True
            self.container = source
            # @@REVISIT: should this cache copy of types be here?
            # probably not... works now where type is PHU dependent, but
            # this may not remain the case... left for now.
            if (source.types != None) and (len(source.types) != 0):
                self.types = source.types
                        
            chdu = source.gethdul()
            sublist = [chdu[0]]
            if self.extensions != None:
                # then some extensions have been identified to refer to
                for extn in self.extensions:
                    sublist.append(chdu[extn])
            elif (self.extInsts != None):
                # then some extension (HDU objects) have been given in a list
                sublist += self.extInsts
            self.hdulist = pyfits.HDUList(sublist)
        elif type(source) == pyfits.HDUList:
            self.hdulist = source
        else:
            self.filename = source
            try:
                self.hdulist = pyfits.open(self.filename, mode = mode)
            except IOError:
                print "can't open %s, mode=%s" % (self.filename, mode)
                raise
        
        self.discoverTypes()
            
    def close(self):
        """
        This function closes the pyfits.HDUList object if this instance
        is the owner (the instance originally calling pyfits.open(..) on this
        MEF).
        @return: nothing
        """
        if (self.borrowedHDUList == False):
            self.hdulist.close()
        else:
            self.container.relhdul()
            self.container = None
            
        self.hdulist = None
            
    
    def getHDUList(self):
        """
        This function retrieves the HDUList. NOTE: The HDUList should also be "released"
        by calling L{releaseHDUList}, as access is reference-counted. This function is
        also aliased to L{gethdul(..)<gethdul>}.
        @return: The AstroData's HDUList as returned by pyfits.open()
        @rtype: pyfits.HDUList
        """
        self.hdurefcount = self.hdurefcount + 1
        return self.hdulist
                
    gethdul = getHDUList # function alias
    
    def releaseHDUList(self):
        """
        This function will release a reference to the HDUList... don't call unless you have called
        L{getHDUList} at some prior point. Note, this function is aliased to L{relhdul(..)<relhdul>}.
        """
        self.hdurefcount = self.hdurefcount - 1
        return
    relhdul = releaseHDUList # function alias
            
    def getClassificationLibrary(self):
        """
        This function will return a handle to the ClassificationLibrary.  NOTE: the ClassificationLibrary
        is a singleton, this call will either return the currently extant instance or, if not extant,
        will create the classification library (using the default context).
        @return: A reference to the system classification library
        @rtype: L{ClassificationLibrary}
        """
        if (self.classificationLibrary == None):
            try:
                self.classificationLibrary = ClassificationLibrary()
            except CLAlreadyExists, s:
                self.classificationLibrary = s.clInstance
	                
        return self.classificationLibrary
    
    def getTypes(self, prune = False):
        """This function returns an array of string type names, just as discoverTypes
        but also takes arguments to modify the list. 
        @param prune: flag which controls 'pruning' the returned type list so that only the
        leaf node type for a given set of related types is returned.
        @type prune: Bool
        @returns: a list of classification names that apply to this data
        @rtype: list
        """
        
        retary = self.discoverTypes()
        if prune :
            # since there is no particular order to identifying types, I've deced to do this
            # here rather than try to build the list with this in mind (i.e. passing prune to
            # ClassificationLibrary.discoverTypes()
            #  basic algo: run through types, if one is a supertype of another, 
            #  remove the supertype
            
            cl = self.getClassificationLibrary()
            pary = list(retary)
            for typ in retary:
                to = cl.getTypeObj(typ) 
                for supertyp in retary:
                    if typ != supertyp:
                        if to.isSubtypeOf(supertyp):
                            # then remove supertype, only subtypes allowed in pruned list
                            if supertyp in pary:
                                pary.remove(supertyp)
            retary = pary
                            

        return retary
        
    def discoverTypes(self, all = False):
        """
        This function provides a list of classifications of both processing status
        and typology which apply to the data encapsulated by this instance, 
        identified by their string names.
        @param all: a flag which  controls how the classes are returned... if True,
        then the function will return a dictionary of three lists, "all", "status",
        and "typology".  If Fa
lse, the return value is a list which is in fact
        the "all" list, containing all the status and typology related types together.
        @return: a list of DataClassification objects, or a dictionary of lists
        if the C{all} flag is set.
        @rtype: list | dict
        """
        if (self.types == None):
            cl = self.getClassificationLibrary()

            alltypes = cl.discoverTypes(self, all=True)
            self.types = alltypes["all"]
            self.typesStatus = alltypes["status"]
            self.typesTypology = alltypes["typology"]
        
        if (all == True):
            retdict = {}
            retdict["all"] = self.types
            retdict["status"] = self.typesStatus
            retdict["typology"] = self.typesTypology
            return retdict
        else:
            return self.types
        
    def getStatus(self):
        """ This function returns specifically "status" related classifications
        about the encapsulated data.
        @returns: a list of string classification names
        @rtype: list
        """
        
        retary = self.discoverStatus()
        return retary
    
    def discoverStatus(self):
        """
        This function returns the set of processing types applicable to 
        this dataset.
        @returns: a list of classification name strings
        @rtype: list
        """
        if (self.typesStatus == None):
            self.typesStatus = cl.discoverStatus(self)
            
        return self.typesStatus

    def getTypology(self):
        """
        This function returns the set of typology types applicable to
        this dataset.  "Typology" classifications are those which tend
        to remain with the data for it's lifetime, e.g. those related
        to the instrument mode of the data.
        @returns: a list of classification name strings
        @rtype: list"""
        
        self.discoverTypology()
        
        retary = [i.name for i in self.typesTypology]
        return retary

    def discoverTypology(self):
        """
        This function returns a list of classification names
        for typology related classifications, as apply to this
        dataset.
        @return: DataClassification objects in a list
        @rtype: list
        """
        if (self.typesStatus == None):
            self.typesStatus = cl.discoverStatus(self)
            
        return self.typesTypology

        
    def checkType(self, *typenames):
        """
        This function checks the type of this data to see if it can be characterized 
        as the type specified 
        by C{typename}.
        @param typename: Specifies the type name to check.
        @type typename: string
        @returns: True if the given type applies to this dataset, False otherwise.
        @rtype: Bool
        """
        if (self.types == None):
            cl = self.getClassificationLibrary()
            self.types = cl.discoverTypes(self)
            typestrs = self.getTypes()
        for typen in typenames:
            if typen in self.types:
                pass
            else:
                return False
                
        return True
    isType = checkType

    def rePHUKeys(self, rekey):
        """ reKeys returns all keys in this dataset's PHU which match the given 
        regular expression.
        @param rekey: A regular expression
        @type rekey: string
        @returns: a list of keys from the PHU that matched C{rekey}
        @rtype: list"""
        phuh = self.hdulist[0].header
        
        retset = reHeaderKeys(rekey, phuh)
            
        return retset
            
    # PHU manipulations
    def phuValue(self, key):
        """
        This function returns a header from the primary header unit 
        (extension 0 in a MEF).
        @param key: name of header entry to retrieve
        @type key: string
        @rtype: depends on datatype of the key's value
        @returns: the key's value or None if not present
        """
        try:
            hdus = self.getHDUList()
            retval = hdus[0].header[key]
            self.relhdul()
            return retval
        except KeyError:
            self.relhdul()
            return None
    phuHeader = phuValue
            
    def translateIntExt(self, integer):
        """This function is used internally to support AstroData
        instances associated with a subset of the full MEF file
        associated with this instance. This function, if this instance
        is not associated with the entire MEF, will return whatever
        is in the extensions list at that index, otherwise, it returns
        the integer passed in. In the first case it can be a tuple which
        is returned if tuples were used to specify the subset of 
        extensions from the MEF which are associated with this instance.
        
        @rtype: int | tuple
        @returns: the actual extension relative to the containing MEF
        """
        if (self.extensions == None):
            return integer
        else:
            return self.extensions[integer-1]
    
    def getHeaderValue(self, extension, key):
        """This function returns the value from the given extension's
        header.
        @param extension: identifies which extension
        @type extension: int or (EXTNAME, EXTVER) tuple
        @param key: name of header entry to retrieve
        @type key: string
        @rtype: depends on datatype of the key's value
        @returns: the key's value or None if not present
        """
        
        if type(extension) == int:
            extension = self.translateIntExt(extension)
            
        #make sure extension is in the extensions list
        if (self.extensions != None) and (not extension in self.extensions):
            return None
            
        hdul = self.gethdul()
        try:
            retval = hdul[extension].header[key]
        except KeyError:
            return None
            
        self.relhdul()
        return retval
   
    def info(self):
        """This function calls the pyfits.HDUList C{info(..)} function
        on this instances C{hdulist} member.  The output goes whereever
        C{HDUList.info(..)} goes, namely, standard out."""
        self.hdulist.info()
        
 
    def write(self, fname = None):
        """
        This function acts similarly to C{HDUList.writeto} if name is given, 
        or C{HDUList.update} if none is given,
        that is it will write a new file if fname is given, otherwise it will
        overwrite the source file originally loaded.
        @param fname: file name, optional if instance already has name, which
        might not be the case for new AstroData instances created in memory.
        @type fname: string
        """
        hdul = self.gethdul()
        if fname != None:
            self.filename = fname
            
        if (self.filename == None):
            # @@FUTURE:
            # perhaps create tempfile name and use it?
            raise gdExcept()
            
        hdul.writeto(self.filename)
        
        
    # MID LEVEL MEF INFORMATION
    #
    def countExts(self, extname):
        """This function will counts the extensions of a given name
        which are associated with this AstroData instance. Note, if 
        this instance is associated with a subset of the extensions
        in the source MEF, only those in the subset will be counted.
        @param extname: the name of the extension, equivalent to the
        value associated with the "EXTNAM" key in the extension header.
        @param extname: The name of EXTNAM, or "None" if there should be
        no extname key at all.
        @type extname: string
        @returns: number of extensions of that name
        """
        hdul = self.gethdul()
        maxl = len(hdul)
        count = 0
        for i in range(0,maxl):
            try:
                # note, only count extension in our subdata extension list
                if (self.extensions == None) or ((extname,i) in self.extensions):
                    if (hdul[i].header["EXTNAME"] == extname):
                        count += 1
            except KeyError:
                #no biggie if some extention has no EXTNAME
                if extname == None:
                    count += 1  # in this case we are counting when there is no
                                # EXTNAME in the header
        self.relhdul()
        
        return count
        
    def getHDU(self, extid):
        """This function returns the HDU identified by the C{extid} argument. This
        argument can be an integer or (EXTNAME, EXTVER) tuple.
        @param extid: specifies the extention (pyfits.HDU) to return.
        @type extid: int | tuple
        @returns:the extension specified
        @rtype:pyfits.HDU
        """
        return self.hdulist[extid]
        
    def getPHUHeader(self):
        return self.getHDU(0).header
            
        
# SERVICE FUNCTIONS and FACTORIES
def correlate( *iary):
    """
    This AstroData helper function returns a list of tuples of Single Extension 
    AstroData instances. It accepts a variable number of arguments, all of which
    should be AstroData instances. The returned list contains tuples of associated 
    extensions... that is, given three inputs, C{a}, C{b} and C{c},
    assuming all have (SCI,1) extensions, then in there will be a tuple in the list
    (a[(SCI,1)], b[(SCI,1)], c[(SCI,1)]).  This is useful for processes which
    combine extensions this way, i.e. with the Gemini system to "add MEFa to MEFb" 
    means precicely this, to add MEFa[(SCI,1)] to MEFb[(SCI,1)], 
    MEFa[(SCI,2)] to MEFb[(SCI,2)] and so on. Similarly when handling variance or
    data quality planes, such correlations by (EXTNAME,EXTVER) are used to know
    what data in separate MEFs should be combined in the specified operation.
    @parm iary: any number of AstroData instances
    @type iary: arglist
    @returns: a list of tuples containing correlated extensions from the arguments. 
    Note: to appear in the list, all the given arguments must have an extension
    with the given (EXTNAME,EXTVER) for that tuple.
    @returns: list of tuples
    """
    numinputs = len(iary)
    
    if numinputs < 1:
        raise gdExcept()
    
    outlist = []
    
    outrow = []
    
    baseGD = iary[0]
    
    for extinbase in baseGD:
        extname = extinbase.header["EXTNAME"]
        extver  = extinbase.header["EXTVER"]
        outrow = [ extinbase ]
        #print "gd610: (%s,%d)" % (extname,extver)
        for gd in iary[1:]:
            correlateExt = gd[(extname, extver)]
            # print "gd622: " + str(correlateExt.info())
            #print "gd614: %s" % str(correlateExt)
            if correlateExt == None:
                #print "gd615: no extension matching that"
                break
            else:
                outrow.append(correlateExt)
        
        #print "gd621: %d %d" %(len(outrow), numinputs)
        if len(outrow) == numinputs:
            # if the outrow is short then some input didn't correlate with the
            # cooresponding extension, otherwise, add it to the table (list of lists)
            outlist.append(outrow)
    return outlist    

def prepOutput(inputAry = None, name = None):
    """
    This function creates an output AstroData with it's own PHU,
    with associated data propagated, but not written to disk.
    @param inputAry : (1) single argument or list, use propagation and other
    standards to create output and (2) None means create empty AstroData.
    Note: the first element in the list is used as the I{reference} AstroData 
    from which the PHU is
    propagated to the prepared output file.
    @param name: File name to use for return AstroData, optional.
    @returns: an AstroData instance with associated data from the inputAry 
    properly forwarded to the returned instance, which is likely going to
    recieve output from further processing.
    File will not exist on disk yet.
    @rtype: AstroData
    """
    if inputAry == None:
        raise gdExcept()
        return None
    
    if type(inputAry) != list:
        iary = [inputAry]
    else:
        iary = inputAry
    
    #get PHU from inputAry[0].hdulist
    hdl = iary[0].gethdul()
    outphu = hdl[0]
        
    # make outlist the complete hdulist
    outlist = [outphu]

    #perform extension propagation
     
    newhdulist = pyfits.HDUList(outlist)
    
    retgd = AstroData(newhdulist)
    
    return retgd

# @@DOCPROJECT@@ done pass 1
