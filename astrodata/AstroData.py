#!/bin/env pyth
import sys
from copy import copy, deepcopy
import pyfits
__docformat__ = "restructuredtext" #for epydoc

from AstroDataType import *

import Descriptors
# this gets SCI (== "SCI") etc
from gemconstants import *
import Calculator

try:
    from CalculatorInterface import CalculatorInterface
except ImportError:
    class CalculatorInterface:
        pass

import re

verbose = False
verboseLoadTypes = True
verbt = False


class ADExcept:
    """This class is an exception class for the Calculator module"""
    
    def __init__(self, msg="Exception Raised in AstroData system"):
        """This constructor accepts a string C{msg} argument
        which will be printed out by the default exception 
        handling system, or which is otherwise available to whatever code
        does catch the exception raised.
        
        :param: msg: a string description about why this exception was thrown
        
        :type: msg: string
        """
        self.message = msg
    def __str__(self):
        """This string operator allows the default exception handling to
        print the message associated with this exception.
        :returns: string representation of this exception, the self.message member
        :rtype: string"""
        return self.message
class SingleHDUMemberExcept(ADExcept):
    def __init__(self, msg=None):
        if msg == None:
            self.message = "This member can only be called for Single HDU AstroData instances"
        else:
            self.message = "SingleHDUMember: "+msg     

class OutputExists(ADExcept):
    def __init__(self, msg=None):
        if msg == None:
            self.message = "Output Exists"
        else:
            self.message = "Output Exists: "+ msg
            

#FUNCTIONS
def reHeaderKeys(rekey, header):
    """This utility function returns a list of keys from 
    the header passed in which match the given regular expression.
    :param rekey: a regular expresion to match to keys in header
    :type rekey: string
    :param header: a Header object as returned
    :type header: pyfits.Header"""
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
The AstroData Class is  designed to handle sets of astronomy data as 
a single unit. I uses the MEF file format for data storage, and keeps
pyfits related data structures in memory. In principle other data storage types
could be supported and also datasets distributed across multiple files.
However, currently, AstroData presumes that a MEF is a single dataset, and thus
applies associations only internally to single MEFs. Thus a MEF is a dataset,
and an AstroData instance encapsulates a dataset, and reads and writes itself to
disk as a MEF. AstroData thus maintains the individual header-data units within
the MEF as pyfits.HDUs. To the programmer is  itself as a collection of 
AstroData instances, each containing one of the HDUs in the whole set found.

The class loads configurations found on ADCONFIGPATH, RECIPEPATH, and the PYTHONPATH,
in that order, with the naming convention "ADCONFIG_xyz" (the recipe system packages 
are named "RECIPES_xyz"). The configurations allow defining
how the separate extensions in a MEF file of a known dataset type are related. This
allows subsystems to in turn recognize the relationship between three extensions.

For example:

+ to recognize of three separate HDUs
  one HDU is the science, one is a variance plane, and the other a data mask, and that
  they relate to the same exposure so the variance and data quality mask can be
  automatically transformed when the science
  data is transformed.
+ to understand that the a mask definition HDU is associated with a series of spectra,
  allowing the system to help ensure this HDU is propagated to output, even when 
  transformed by generic transformations which do not "know" the transformation is
  for spectroscopy (e.g. a simple image subtraction)

All type and related definitions are loaded as configurations, so this semantic
knowledge is not encoded directly in this class, but sits in configuration files
used
by subordinate classes such as the :class:`Classification Library<astrodata.datatypes.ClassificationLibrary>`.
All access to configurations goes through a :class:`ConfigSpace object<astrodata.ConfigSpace.ConfigSpace`.

In general one can consider the functionality to consist of
file handling, data handling, type checking, and managing
meta-information in the case of the MEF file. AstroData uses subsidiary classes to provide
most functionality, e.g. for file handling 
and data handling the
class uses python standard Pyfits and NumPy. The type
services, projection of structures, normalization of standard high-level meta data about
and observation, are handled by custom package classes. See also the 
:class:`Classification Library<astrodata.datatypes.ClassificationLibrary>
AstroDataType}, the
:class:`Structure Class<astrodata.Structures.Structure>`, and the 
:class:`Descriptor Class<astrodata.descriptors.Descriptor`
.
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
    def __init__(self, fname=None, mode="readonly", exts = None, extInsts = None):
        """
        Constructor for AstroData. Note, the file will be opened.
        
        :param fname: filename of MEF to load
        :type fname: string
        :param mode: IO access mode, same as pyfits mode, see
                     :meth:`open(..)<open>` for a list of supported modes.
        :type mode: string
        :param exts: a list of extensions this instance should refer to, given 
                     integer or tuples specifying each extention. I.e. (EXTNAME,EXTVER) tuples or 
                     and integer index which specifies the ordinal position of the extension in the MEF 
                     file, begining with index 0 for the PHU. NOTE: if present this option will
                     override and obscure the extInsts argument which will be ignored.
        :type exts: list
        :param extInsts: a list of extensions this instance should refer to, given as
                         actual pyfits.HDU instances. NOTE: if the "exts" argument is also set,
                         this argument is ignored.
        :type extInsts: list
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
        also the data structures of the HDUs they have in common. So a change
        to datasetA[("SCI",1)].data will change datasetB[("SCI",1)].data member
        because they are in fact the same numpy array in memory. The HDUList is
        a different list, however, that references common HDUs.

        NOTE: Integer extensions start at 0 for the data-containing extensions.
        ad[0] is the first extension AFTER the PHU, it is not the PHU!  In
        AstroData instances, the PHU is purely a header.
        
        :param ext: EXTNAME name for this subdata instance.
        :type ext: string
        :returns: AstroData instance associated with the subset of data
        :rtype: AstroData
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
                            extver = int(hdul[i].header["EXTVER"])
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
                    ext = ext+1 # so 0 does not mean PHU, but 0'th content-extension
                # print "AD262:", repr(ext)
                # hdul.info()
                # self.hdulist.info()
                exttmp = self.hdulist[ext]
                # exttmp = hdul[ext]
            except KeyError:
                print "gd105: keyerror:[%s]" % str(ext)
                # selector not valid
                self.relhdul()
                return None

            gdpart = AstroData(self, exts=[ext])
            self.relhdul()
            # print "gd132: %s" % str(gdpart)
            return gdpart
        else:
            raise KeyError()
            
    def __len__(self):
        """This is the length operator for AstroData.
        :returns: number of extensions minus the PHU
        :rtype: int"""
        return len(self.hdulist)-1
    
        
    #ITERATOR FUNCTIONS
    def __iter__(self):
        """This function exists so that AstroData can be used as an iterator.
        It initializes the iteration process, resetting the index of the 
        current extension.
        :returns: self
        :rtype: AstroData"""
        self.index = 0
        return self
        
    def next(self):
        """This function exists so that AstroData can be used as an iterator.
        This function returns the objects "ext" in the following line:
        
        for ext in gemdatainstance:
        
        If this AstroData instance is associated with a subset of the data in
        the MEF to which it refers, then this iterator goes through that subset
        order (as given by the 
        
        :returns: a single extension AstroData instance representing the current
        extension in the data.
        :rtype: AstroData
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
        
    def __deepcopy__(self, memo):
        # pyfits throws exception on deepcopy
        #self.hdulist = deepcopy(self.hdulist, memo)
        lohdus = []
        for hdu in self.hdulist:
            nhdu = copy(hdu)
            nhdu.header = nhdu.header.copy()
            lohdus.append(nhdu)
        
        hdulist = pyfits.HDUList(lohdus)
        
        return AstroData(hdulist)
            
        
        print "AD298: copy?"
    
    def append(self, moredata=None, data=None, header=None):
        """
This function appends more data units (aka an "HDU") to the AstroData
instance.

:param moredata: Either an AstroData instance, an HDUList instance, 
    or an HDU instance. When present, data and header will be ignored.

:type moredata: pyfits.HDU, pyfits.HDUList, or AstroData
)
:param data: if moredata *is not* specified, data and header should 
    both be set and areare used to instantiate
    a new HDU which is then added to the 
    AstroData instance.

:type data: numarray.numaraycore.NumArray

:param header: if moredata *is not* specified, data and header are used to make 
    an HDU which is then added to the HDUList associated with this
    AstroData instance.

:type header: pyfits.Header
        """
        if (moredata == None):
            if len(self.hdulist) == 0:
                self.hdulist.append(pyfits.PrimaryHDU(data = data, header=header))
            else:
                self.hdulist.append(pyfits.ImageHDU(data = data, header=header))
        elif isinstance(moredata, AstroData):
            for hdu in moredata.hdulist[1:]:
                self.hdulist.append(hdu)
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
        """
The 'data' property is returned by the getData(..) member, and returns data member(s)
specifically for the case in which
the AstroData instance has ONE HDU (in addition to PHU). This
allows a single extension AstroData instance to be used as though
it is simply one extension, e.g. allowing gd.data to be used in
place of the more esoteric and ultimately more dangerous gd[1].data.
One can assure one is dealing with single extension AstroData instances
when iterating over the AstroData extensions, e.g.:

.. code-block: python

    for gd in dataset[SCI]:
        pass

:raise: gdExcept if AstroData instance has more than one extension 
    (not including PHU).
:return: data array associated with the single extension
:rtype: NumArray
        """
        hdl = self.gethdul()
        if len(hdl) == 2:
            retv = hdl[1].data
        else:
            # print "gd207: %d" % len(hdl)
            raise ADExcept("getData must be called on single extension instances")
            
        self.relhdul()
        return retv

    def setData(self, newdata):
        """This function sets the data member(s) of a data section of an HDU, specifically for the case in which
        the AstroData instance has ONE extension (in addition to PHU).  This cases
        should be assured when iterating over the AstroData extensions, e.g.::
        
            for gd in dataset[SCI]:
                ...
                
        :raise gdExcept: if AstroData instance has more than one extension 
        (not including PHU).
        :param newdata: new data objects
        :type newdata: numarray.numarraycore.NumArray
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
        
        :raise gdExcept: Will raise a gdExcept exception if more than one extension exists. 
            (note: The PHU is not considered an extension in this case)
        :return: header
        :rtype: pyfits.Header
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
        
        :param header: header )to set for given extension
        :type header: pyfits.Header
        
        :param extension: Extension index from which to retrieve header, if None or not present then this must be
        a single extension AstroData instance, which contains just the PHU and a single data extension, and the data
        extension's header is returned.
        :type extension: int or tuple, pyfits compatible extension index
        
        :raise gdExcept: Will raise a gdExcept exception if more than one extension exists. 
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
        :return: list of pyfits.Header instances
        :rtype: pyfits.Header
        """
        hdl = self.gethdul()
        
        retary = []
        
        for hdu in hdl:
            retary.append(hdu.header)

        self.relhdul()
        return retary

    def hasSingleHDU(self):
        return len(self.hdulist) == 2
    
    def setExtname(self, name, ver):
        """WARNING: this function recreates HDUs and is not to be used on subdata"""
        if self.borrowedHDUList:
            raise ADExcept("cannot setExtname on subdata")
        
        print "AD545:", name, ver
        if not self.hasSingleHDU():
            raise SingleHDUMemberExcept("ad.setExtname(%s,%s)"%(str(name), str(ver)))
        
        if True:
            hdu = self.hdulist[1]
            nheader = hdu.header
            nheader.update("extname", name, "added by AstroData")
            nheader.update("extver", ver, "added by AstroData")
            hdu.name = name
            hdu._extver = ver
            # print "AD553:", repr(hdu.__class__)
            

    def open(self, source, mode = "readonly"):
        '''
        This function initiates interaction with a given set of
        AstroData. Note, this is not the way one generally opens a 
        MEF with AstroData, instead, pass the filename into the
        constructor. This function can still be of use if
        the AstroData object has been closed.
        :param source: source for data to be associated with this instance, can be 
        an AstroData instance, a pyfits.HDUList instance, or a string filename.
        :type source: string | AstroData | pyfits.HDUList
        :param mode: IO access mode, same as the pyfits open mode, C{readonly},
        C{update}, or C{append}.  The mode is passed to pyfits so if it is an
        illegal mode name, pyfits will be the subsystem reporting the error. 
        :type mode: string
        :return: nothing
        '''
                
        inferRAW = True
        # might not be a filename, if AstroData instance is passed in
        #  then it has opened or gets to open the data...
        if isinstance(source, AstroData):
            inferRAW = False
            self.filename = source.filename
            self.borrowedHDUList = True
            self.container = source
            # @@REVISIT: should this cache copy of types be here?
            # probably not... works now where type is PHU dependent, but
            # this may not remain the case... left for now.
            if (source.types != None) and (len(source.types) != 0):
                self.types = source.types
                        
            chdu = source.gethdul()
            # include the phu no matter what
            sublist = [chdu[0]]
            if self.extensions != None:
                # then some extensions have been identified to refer to
                for extn in self.extensions:
                    # print "AD553:", extn, chdu[extn].header["EXTVER"]
                    sublist.append(chdu[extn])
            elif (self.extInsts != None):
                # then some extension (HDU objects) have been given in a list
                sublist += self.extInsts
            self.hdulist = pyfits.HDUList(sublist)
            # print "AD559:", self.hdulist[1].header["EXTVER"]
        elif type(source) == pyfits.HDUList:
            self.hdulist = source
        else:
            if source == None:
                phu = pyfits.PrimaryHDU()
                self.hdulist = pyfits.HDUList([phu])
            else:
                if not os.path.exists(source):
                    raise ADExcept("Cannot open "+source)
                self.filename = source
                try:
                    if mode == 'new':
                        if os.access(self.filename, os.F_OK):
                            os.remove(self.filename)
                        mode = 'append'
                    self.hdulist = pyfits.open(self.filename, mode = mode)
		    #print "AD591:", self.hdulist[1].header["EXTNAME"]
                    #print "AD543: opened with pyfits", len(self.hdulist)
                except IOError:
                    print "CAN'T OPEN %s, mode=%s" % (self.filename, mode)
                    raise

        #if mode != 'append':
        if len(self.hdulist):
            try:
                self.discoverTypes()
            except:
                raise ADExcept("discover types failed")

        # do inferences
        if inferRAW and self.isType("RAW"):
            
            # for raw, if no extensions are named
            # infer the name as "SCI"
            hdul = self.hdulist
            namedext = False
	    
            for hdu in hdul[1:]:
    	        # print "AD642:",hdu.name

                if hdu.name or "extname" in hdu.header: 
                    namedext = True
                    #print "AD615: Named", hdu.header["extname"]
                else:
                    # print "AD617: Not Named"
                    pass
                    
            if namedext == False:
                # print "AD567: No named extension"
                l = len(hdul) # len w/phu
                # print "AD567: len of hdulist ",l

                
                # nhdul = [hdul[0]]
                # nhdulist = pyfits.HDUList(nhdul)

                for i in range(1, l):
                    hdu = hdul[i]
                    #print "AD667:",hdu.name
                    hdu.header.update("EXTNAME", "SCI", "added by AstroData", after='GCOUNT')
                    hdu.header.update("EXTVER", i, "added by AstroData", after='EXTNAME')
                    hdu.name = SCI
                    hdu._extver = i
                    #print "AD672:", hdu._extver, i, id(hdu)
                    #nhdu = hdu.__class__( data= hdu.data, header = hdu.header, name = "SCI")
                    #nhdu._extver = i;
                    #nhdu.header.update("EXTNAME", "SCI", "added by AstroData", after='GCOUNT')
                    # print "AD631:extname = ", nhdu.header["EXTNAME"]
                    # nhdu.header.__delitem__("EXTVER")
                    #nhdu.header.update("EXTVER", str(i), "added by AstroData", after='EXTNAME')
                    #print "AD681:",nhdu.name
                    #nhdul.append(nhdu)
                    #print "AD570:", repr(self.extGetKeyValue(i,"EXTNAME"))
                #for hdu in hdul[1:]:
                #    nhdu = hdu.__class__(hdu.data, hdu.header, ext=(str(hdu.header["EXTNAME"]), int( hdu.header["EXTVER"])))
                #    nhdul.append(nhdu)
                #    nhdulist[(str(hdu.header["EXTNAME"]), int( hdu.header["EXTVER"]))] = nhdu
                #del(hdul)
                #self.hdulist = pyfits.HDUList(nhdul)
                
                #print "AD646: nhdul.info()"
                #self.hdulist.info()
                # @@NOTE: should we do something make sure the
                # dropped hdulist goes away?
    def close(self):
        """
        This function closes the pyfits.HDUList object if this instance
        is the owner (the instance originally calling pyfits.open(..) on this
        MEF).
        :return: nothing
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
        :return: The AstroData's HDUList as returned by pyfits.open()
        :rtype: pyfits.HDUList
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
        :return: A reference to the system classification library
        :rtype: L{ClassificationLibrary}
        """
        if (self.classificationLibrary == None):
            try:
                self.classificationLibrary = ClassificationLibrary()
            except CLAlreadyExists, s:
                self.classificationLibrary = s.clInstance
	                
        return self.classificationLibrary
    
    def pruneTypelist(self, typelist):
        cl = self.getClassificationLibrary()
        retary = typelist;
        pary = []
        for typ in retary:
            notSuper = True
            for supertype in retary:
                sto = cl.getTypeObj(supertype)
                if sto.isSubtypeOf(typ):
                    notSuper = False
            if notSuper:
                pary.append(typ)
        return pary

    
    def getTypes(self, prune = False):
        """This function returns an array of string type names, just as discoverTypes
        but also takes arguments to modify the list. 
        :param prune: flag which controls 'pruning' the returned type list so that only the
        leaf node type for a given set of related types is returned.
        :type prune: Bool
        :returns: a list of classification names that apply to this data
        :rtype: list
        """
        
        retary = self.discoverTypes()
        if prune :
            # since there is no particular order to identifying types, I've deced to do this
            # here rather than try to build the list with this in mind (i.e. passing prune to
            # ClassificationLibrary.discoverTypes()
            #  basic algo: run through types, if one is a supertype of another, 
            #  remove the supertype
            retary = self.pruneTypelist(retary)
            
        
        return retary
        
    def discoverTypes(self, all  = False):
        """
        This function provides a list of classifications of both processing status
        and typology which apply to the data encapsulated by this instance, 
        identified by their string names.
        :param all: a flag which  controls how the classes are returned... if True,
        then the function will return a dictionary of three lists, "all", "status",
        and "typology".  If Fa
lse, the return value is a list which is in fact
        the "all" list, containing all the status and typology related types together.
        :return: a list of DataClassification objects, or a dictionary of lists
        if the C{all} flag is set.
        :rtype: list | dict
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
        
    def getStatus(self, prune=False):
        """ This function returns specifically "status" related classifications
        about the encapsulated data.
        :returns: a list of string classification names
        :rtype: list
        """
        retary = self.discoverStatus()
        if prune:
            retary = self.pruneTypelist(retary)

        return retary
    
    def discoverStatus(self):
        """
        This function returns the set of processing types applicable to 
        this dataset.
        :returns: a list of classification name strings
        :rtype: list
        """

        if (self.typesStatus == None):
            cl = self.getClassificationLibrary()
            self.typesStatus = cl.discoverStatus(self)
            
        return self.typesStatus

    def getTypology(self):
        """
        This function returns the set of typology types applicable to
        this dataset.  "Typology" classifications are those which tend
        to remain with the data for it's lifetime, e.g. those related
        to the instrument mode of the data.
        :returns: a list of classification name strings
        :rtype: list"""
        
        retary = self.discoverTypology()
        if prune:
            retary = self.pruneTypelist(retary)
        return retary

    def discoverTypology(self):
        """
        This function returns a list of classification names
        for typology related classifications, as apply to this
        dataset.
        :return: DataClassification objects in a list
        :rtype: list
        """
        if (self.typesTypology == None):
            cl = self.getClassificationLibrary()
            self.typesTypology = cl.discoverTypology(self)
            
        return self.typesTypology

        
    def checkType(self, *typenames):
        """
        This function checks the type of this data to see if it can be characterized 
        as the type specified 
        by C{typename}.
        :param typename: Specifies the type name to check.
        :type typename: string
        :returns: True if the given type applies to this dataset, False otherwise.
        :rtype: Bool
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
        :param rekey: A regular expression
        :type rekey: string
        :returns: a list of keys from the PHU that matched C{rekey}
        :rtype: list"""
        phuh = self.hdulist[0].header
        
        retset = reHeaderKeys(rekey, phuh)
            
        return retset
            
    # PHU manipulations
    def phuValue(self, key):
        """
        This function returns a header from the primary header unit 
        (extension 0 in a MEF).
        :param key: name of header entry to retrieve
        :type key: string
        :rtype: depends on datatype of the key's value
        :returns: the key's value or None if not present
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
    phuGetKeyValue = phuValue
    
    def phuSetKeyValue(self, key, value, comment = None):
        hdus = self.hdulist
        hdus[0].header.update(key, value, comment)
        return
        
    def getPHU(self):
        return self.hdulist[0]
    
    def setPHU(self, phu):
        self.hdulist[0] = phu
        return
        
    phu = property(getPHU, setPHU)

            
    def translateIntExt(self, integer):
        """This function is used internally to support AstroData
        instances associated with a subset of the full MEF file
        associated with this instance. This function, if this instance
        is not associated with the entire MEF, will return whatever
        is in the extensions list at that index, otherwise, it returns
        the integer passed in. In the first case it can be a tuple which
        is returned if tuples were used to specify the subset of 
        extensions from the MEF which are associated with this instance.
        
        :rtype: int | tuple
        :returns: the actual extension relative to the containing MEF
        """

        return integer+1
# for the life of me I can't remember why I'm using self.extensions...
# @@TODO: remove self.extensions completely?  might be useful to know
# the hdu's extension in the original file... ?
#        if (self.extensions == None):
#            return integer+1
#        else:
#            print "AD874:", repr(self.extensions)
#            return self.extensions[integer]
    
    def getHeaderValue(self, key):
        if len(self.hdulist) == 2:
            return self.extGetKeyValue(0,key)
        else:
            raise ADExcept("getHeaderValue must be called on single extension instance")
    getKeyValue = getHeaderValue
           
    def extGetKeyValue(self, extension, key):
        """This function returns the value from the given extension's
        header.
        :param extension: identifies which extension
        :type extension: int or (EXTNAME, EXTVER) tuple
        :param key: name of header entry to retrieve
        :type key: string
        :rtype: depends on datatype of the key's value
        :returns: the key's value or None if not present
        """
        
        if type(extension) == int:
            extension = self.translateIntExt(extension)
        #make sure extension is in the extensions list
        #@@TODO: remove these self.extensions lists
        
        # if (self.extensions != None) and (not extension in self.extensions):
        #    return None
            
        hdul = self.gethdul()
        try:
            retval = hdul[extension].header[key]
        except KeyError:
            raise "AD912"
            return None
        # print "AD914:", key, "=",retval    
        return retval
    
    def setKeyValue(self, key):
        if len(self.hdulist) == 2:
            self.extSetKeyValue(0,key)
        else:
            raise ADExcept("setKeyValue must be called on single extension instance")
    
    def extSetKeyValue(self, extension, key, value, comment = None):

        origextension = extension
        if type(extension) == int:
            # this translates ints from our 0-relative base of AstroData to the 
            #  1-relative base of the hdulist, but leaves tuple extensions
            #  as is.
            #print "AD892: pre-ext", extension
            extension = self.translateIntExt(extension)
            #print "AD892: ext", extension
            
        #make sure extension is in the extensions list if present
        if (self.extensions != None) and (not extension in self.extensions):
            raise ADExcept("Extention %s not present in AstroData instance" % str(origextension))
            
        hdul = self.gethdul()
        hdul[extension].header.update(key, value, comment)
            
        self.relhdul()
        return 
   
    def info(self):
        """This function calls the pyfits.HDUList C{info(..)} function
        on this instances C{hdulist} member.  The output goes whereever
        C{HDUList.info(..)} goes, namely, standard out."""
        self.hdulist.info()
        
 
    def write(self, fname = None, clobber = False):
        """
        This function acts similarly to C{HDUList.writeto} if name is given, 
        or C{HDUList.update} if none is given,
        that is it will write a new file if fname is given, otherwise it will
        overwrite the source file originally loaded.
        :param fname: file name, optional if instance already has name, which
        might not be the case for new AstroData instances created in memory.
        :type fname: string
        """
        hdul = self.gethdul()
        if fname != None:
            self.filename = fname
            
        if (self.filename == None):
            # @@FUTURE:
            # perhaps create tempfile name and use it?
            raise gdExcept()
           
        if os.path.exists(self.filename):
            if clobber:
                os.remove(self.filename)
            else:
                raise OutputExists(self.filename)
        hdul.writeto(self.filename)
        
        
    # MID LEVEL MEF INFORMATION
    #
    def countExts(self, extname):
        """This function will counts the extensions of a given name
        which are associated with this AstroData instance. Note, if 
        this instance is associated with a subset of the extensions
        in the source MEF, only those in the subset will be counted.
        :param extname: the name of the extension, equivalent to the
        value associated with the "EXTNAM" key in the extension header.
        :param extname: The name of EXTNAM, or "None" if there should be
        no extname key at all.
        :type extname: string
        :returns: number of extensions of that name
        """
        hdul = self.gethdul()
        maxl = len(hdul)
        count = 0
        for i in range(1,maxl):
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
        :param extid: specifies the extention (pyfits.HDU) to return.
        :type extid: int | tuple
        :returns:the extension specified
        :rtype:pyfits.HDU
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
    :parm iary: any number of AstroData instances
    :type iary: arglist
    :returns: a list of tuples containing correlated extensions from the arguments. 
    Note: to appear in the list, all the given arguments must have an extension
    with the given (EXTNAME,EXTVER) for that tuple.
    :returns: list of tuples
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

def prepOutput(inputAry = None, name = None, clobber = False):
    """
    This function creates an output AstroData with it's own PHU,
    with associated data propagated, but not written to disk.
    :param inputAry : (1) single argument or list, use propagation and other
    standards to create output and (2) None means create empty AstroData.
    Note: the first element in the list is used as the I{reference} AstroData 
    from which the PHU is
    propagated to the prepared output file.
    :param name: File name to use for return AstroData, optional.
    :returns: an AstroData instance with associated data from the inputAry 
    properly forwarded to the returned instance, which is likely going to
    recieve output from further processing.
    File will not exist on disk yet.
    :rtype: AstroData
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

    if name != None:
        if os.path.exists(name):
            if clobber == False:
                raise OutputExists(name)
            else:
                os.remove(name)
        retgd.filename = name
    
    return retgd
