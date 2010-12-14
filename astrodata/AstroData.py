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

from astrodata.adutils import arith

try:
    from CalculatorInterface import CalculatorInterface
except ImportError:
    class CalculatorInterface:
        pass

import re
from datetime import datetime

from adutils.netutil import urlfetch

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
The AstroData Class which represents a dataset in memory much like any I/O 
interface, e.g. you can provide a filename and it will construct and object
representing that file that can be used to access information therein. It uses
pyfits for actual data manipulation and focusses instead of representing the
dataset as a whole and negotiating the incidental difference between datasets 
from different instruments and modes.

AstroData relies on the MEF file format for data storage, and pyfits structures
for the dataset when in memory. AstroData presumes that a MEF is a single dataset
and acts as a collection of AstroData instances associated with each header-data
the file (note the PHU is handles specially, and is not represented as a part of 
the collection. Note, these sub-instances are not maintained in memory, but are
created by the __getitem__ member function, i.e. the "[]" interface.

The class loads configurations found on ADCONFIGPATH, RECIPEPATH, and the PYTHONPATH,
in that order, in directories named "astrodata_<ANYTHING>" containing subdirectories
with the naming convention "ADCONFIG_<ANYTHING>". Configurations define AstroData Types,
descriptors which produce high-level metadata, primitives, which are units of 
transformation which may be implemented differently based on AstroData Type, and 
relationships between extensions (the prototype Structures feature).

This allows AstroData, for example...:

+ to regularize the reduction script determines the instrument-mode the dataset was taken in

+ to normalize differences in how meta-data is presented

+ to discriminate types of data arrays, e.g. science data from variance and data quality 
  mask data, and to know with which of the former the latter are associated.
  
+ to propagate header-data units important to the given instrument mode, but which are
  not directly part of the current transformation, e.g. propagating Mask Definition 
  extensions when running a simple general-purpose reduction step that is no GMOS or MOS specific,
  such as an image subtraction.

All type and high level meta-data definitions, and supporting information, is loaded 
from configuration directories. 
Semantic knowledge about Gemini data, the dataset types, particulare
name of high-level metadata and so on,
is not encoded directly in the AstroData class, but sits in configuration files
which are used to detect and provide the information desired.
All access to configurations goes 
through a :class:`ConfigSpace<astrodata.ConfigSpace.ConfigSpace>` class,
which by default looks on the paths mentioned above for packages named "astrodata_<ANYTHING>".

In general one can consider the functionality to consist of
file handling, data handling, type checking, and managing
meta-information for complex datasets. AstroData uses subsidiary classes to provide
most functionality, e.g. for file handling 
and data handling the
class uses python standards Pyfits and NumPy. The type
services, projection of structures, normalization of standard high-level meta data 
are provided by custom classes in the astrodata package, generally through the AstroData class or RecipeSystem.
For example, the regular user uses the AstroData type interface to access type information, while
internally the AstroDataType module relies on the 
:class:`ClassificationLibrary<astrodata.datatypes.ClassificationLibrary>` and related 
:class:`DataClassification<astrodata.datatypes.DataClassficiation>` to
execute the type detection code.

Note, the variable "ad" is generally used to represent an already constructed AstroData instance.
"""
    
    types = None
    typesStatus = None
    typesTypology = None
    filename = None
    url = None # if retrieved
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
    tlm=None
    # ClassificationLibrary Singleton, must be retrieved through
    #   getClassificationLibrary()
    classificationLibrary = None

    def __init__(self, dataset=None, mode="readonly", exts = None, extInsts = None,
                    header = None, data = None, store = None, storeClobber = False):
        """
        The AstroData constructor constructs an in memory representation of a dataset.
        If given a filename it uses pyfits to open the dataset, reads the header
        and detects applicable types. Binary data, such as pixel data, is left on disk until referenced.
        
        :param dataset: the dataset to load, either a filename (string), an AstroData instance, or a pyfits.HDUList
        :type dataset:  string, AstroData, HDUList
        :param mode: IO access mode, same as pyfits mode ("readonly", "update", "or append") with one additional 
                    supported mode, "new". If the mode is "new", then the constructor checks the named file does not
                    exist, and if it does not already exist, it creates an empty AstroData of that name, but does not
                    write it to disk.  Such an AstroData instance is ready to have HDUs appended, and to be written to
                    disk as the user's command with "ad.write()".
        :type mode: string
        :param exts: (advanced) a list of extension indexes in the parent HDUList that this instance should refer to, given 
                     integer or tuples specifying each extention in the "pyfits" index space where
                     the PHU is 0, and first data extension is 1, and so on.
                     I.e. (EXTNAME,EXTVER) tuples or 
                     and integer index which specifies the ordinal position of the extension in the MEF 
                     file, begining with index 0 for the PHU. NOTE: if present this option will
                     Foverride and obscure the extInsts argument which will be ignored.
                     This is generally used internally to support subdata, i.e. sci_subdata = ad["SCI"]
                     is constructed passing the "SCI" extensions to this constructor.
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
        fname = None
        headers = None
        if type(dataset) == str:
            parts = dataset.split(":")
            if len(parts)>1:
                # then the string is an URL, retrieve it
                import urllib
                from urllib import urlretrieve
                savename = os.path.basename(dataset)
                print "${BOLD}AstroData${NORMAL} retrieving remote file: "
                print "     from %s" % dataset
                print "     to   %s" % os.path.join(store,savename)
                if store:
                    # print "AD230: Storing in,", store
                    fname = urlfetch(dataset, store = store, clobber = storeClobber)
                    #fname,headers = urlretrieve(dataset, os.path.join(store, savename), None, 
                    #    urllib.urlencode({"gemini_fits_authorization":"good_to_go"}))
                else:
                    # print "AD235: Retrieved to temp file"
                    fname = urlfetch(dataset)
                    #fname, headers = urlretrieve(dataset)
                dataset = savename
            elif store:
                import shutil
                shutil.copy(dataset, store)
                dataset = os.path.join(store,dataset)
                # print "AD235:", dataset
            
        if (dataset == None) and (header != None) and (data != None):
            dataset = pyfits.ImageHDU(data = data, header=header)
            
        if fname == None:
            self.open(dataset, mode)
        else:
            # fname is set when retrieving an url, it will be the temporary 
            #   filename that urlretrieve returns.  Nice to use since it's 
            #   guarenteed to be unique.  But it's only on disk to load (we
            #   could build the stream up from the open url but that might be
            #   a pain (depending on how well pyfits cooperates)... and this means
            #   we don't need to load the whole file into memory.
            self.open(fname, mode)
            if store == None:
                # because if store == None then file was retrieved to a persistent location
                os.remove(fname)
                

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
    
    def __contains__(self, ext):
        try:
            val = self[ext]
            if val == None:
                return False
        except:
            return False
        return True
                    
    def __getitem__(self,ext):
        """This function support the "[]" syntax on AstroData instances,
        e.g. ad[("SCI",1)].  We use it to create
        AstroData objects associated with "subdata"... that is, a limited
        subset of extensions in the given MEF. e.g.::
        
            datasetA = AstroData.AstroData("datasetMEF.fits")
            datasetB = datasetA[SCI]
            
        In this case datasetB would be a AstroData object associated with the
        same MEF, sharing some of the the same actual HDUs in memory as 
        datasetA.  The "datasetB" will behave as if the SCI extensions are the only
        extensions in the HDUList, and indeed will have it's own HDUList.
        Note, datasetA and datasetB share the PHU and
        also the data structures of the HDUs they have in common. So a change
        to "datasetA[('SCI',1)].data" will change "datasetB[('SCI',1)].data" member
        because they are in fact the same numpy array in memory. The HDUList is
        a different list, however, that references common HDUs.  Thus if written to disk,
        the resulting MEF will contain only the extensions in datasetB's HDUList.

        NOTE: Integer extensions start at 0 for the data-containing extensions,
        not at the PHU as with pyfits.  This is important:
        ad[0] is the first extension AFTER the PHU; it is not the PHU!  In
        AstroData instances, the PHU is purely a header, which can be accessed 
        with the code, "ad.phu".
        
        :param ext: EXTNAME name for this subdata's HDU. If an int or tuple, a 
            single extension is wrapped with an AstroData instance, if a string
            then all extensions with the given EXTNAME will be wrapped by the 
            new AstroData instance.
        :type ext: string, int, or tuple
        :returns: an AstroData instance associated with the subset of data
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
                # print 'Extension "%s" does not exist' % str(ext)
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
        'current' extension to the first data extension.
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
        in order.
        
        :returns: a single extension AstroData instance representing the 'current'
        extension in the AstroData iteration loop.
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
        This function appends more data units (aka "HDUs") to the AstroData
        instance.

        :param moredata: either an AstroData instance, an HDUList instance, 
            or an HDU instance. When present, data and header will be ignored.

        :type moredata: pyfits.HDU, pyfits.HDUList, or AstroData
        :param data: if moredata *is not* specified, data and header should 
            both be set and are used to instantiate
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
        elif type(moredata) is pyfits.ImageHDU:
            self.hdulist.append(moredata)

    def insert(self, index, moredata=None, data=None, header=None):
        """
        This function inserts more data units (aka an "HDU") to the AstroData
        instance.

        :param index: the extension index, either an int or (EXTNAME, EXTVER) pair
                      before which the extension is to be inserted. Note, the first data extension
                      is [0], you cannot insert before the PHU.
        :param moredata: Either an AstroData instance, an HDUList instance, 
                      or an HDU instance. When present, data and header will be ignored.

        :type moredata: pyfits.HDU, pyfits.HDUList, or AstroData
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
        # print "AD416", type(index), index
        if type(index) == tuple:
            index = self.getIntExt(index, hduref=True)
        # print "AD416", type(index), index
            
        
        if (moredata == None):
            if len(self.hdulist) == 0:
                self.hdulist.insert(index, pyfits.PrimaryHDU(data = data, header=header))
            else:
                self.hdulist.insert(index,pyfits.ImageHDU(data = data, header=header))
        elif isinstance(moredata, AstroData):
            for hdu in moredata.hdulist[1:]:
                self.hdulist.insert(index, hdu)
                index+=1
        elif type(moredata) is pyfits.HDUList:
            for hdu in moredata[1:]:
                self.hdulist.insert(index, hdu)
                index += 1
        elif type(moredata) is pyfits.ImageHDU:
            self.hdulist.insert(index, moredata)
    
    def close(self):
        """The close(..) function will close the associated HDUList
        If this is subdata, e.g.
        (sd = gd[SCI] where gd is another AstroData instance, sd is "sub-data") 
        then sd.close()
        will not close the hdulist because gd will actually own the
        hold on that file."""
        
        if self.borrowedHDUList:
            self.container.relhdul()
            self.hdulist = None
        else:
            if self.hdulist != None:
                self.hdulist.close()
                self.hdulist = None
                
    def infostr(self, asHTML = False):
        if not asHTML:
            rets = ""
            for ext in self:
                rets += "id =",id(ext.hdulist[1]), "\n"
                rets += "    ", ext.extname(), ext.extver(), "\n"
        else:
            rets="<b>Extension List</b>: %d in file" % len(self)
            rets+="<ul>"
            for ext in self:
                rets += "<li>(%s, %s)</li>" % (ext.extname(), str(ext.extver()))
            rets += "</ul>"
                     
        return rets
        
    def exceptIfSingle(self):
        if len(self.hdulist) != 2:
            raise SingleHDUMemberExcept()
            
    def extname(self):
        self.exceptIfSingle()
        return self.hdulist[1].header.get("EXTNAME", None)
        
    def extver(self):
        self.exceptIfSingle()
        retv = self.hdulist[1].header.get("EXTVER", None)
        if retv:
            retv = int(retv)
        return retv
        

    def getData(self):
        """
        The 'data' property is returned by the getData(..) member, and returns HDU data member(s)
        specifically for the case in which
        the AstroData instance has ONE HDU (in addition to PHU). This
        allows a single-extension AstroData, such as AstroData generates through iteration, 
        to be used as though
        it simply is just the one extension, e.g. allowing gd.data to be used in
        place of the more esoteric and ultimately more dangerous gd[0].data.
        One can assure one is dealing with single extension AstroData instances
        when iterating over the AstroData extensions, and when picking out an extension 
        by integer or tuple indexing, e.g.:

        .. code-block: python

            for gd in dataset[SCI]:
                # gd is a single-HDU index
                gd.data = newdata

            # assuming the named extension exists, sd will be a single-HDU AstroData
            sd = dataset[("SCI",1)]


        :raise: gdExcept if AstroData instance has more than one extension 
            (not including PHU).
        :return: data array associated with the single extension
        :rtype: pyfits.ndarray
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
        """
        This function sets the data member(s) of a data section of an HDU, 
        specifically for the case in which
        the AstroData instance has ONE extension (in addition to PHU).  This case
        is assured when iterating over the AstroData extensions, e.g.::
        
            for gd in dataset[SCI]:
                ...
                
        :raise gdExcept: if AstroData instance has more than one extension 
        (not including PHU).
        :param newdata: new data objects
        :type newdata: numarray.numarraycore.NumArray
        """
        hdl = self.gethdul()
        if len(hdl) == 2:
            # note: should we check type of newdata?
            hdl[1].data = newdata
        else:
            raise gdError()
            
        self.relhdul()
        return
    
    data = property(getData, setData, None, """
            The data property can only be used for single-HDU AstroData instances, such as those returned
            during iteration. It is a property attribute which uses getData(..) and setData(..)
            to access the data members with "=" syntax.
            To set the data member, use "ad.data = newdata", where "newdata" must be a numpy array
            To get the data member, use "npdata = ad.data".
            """)
    
    def getHeader(self, extension = None):
        """
        The getHeader(..) function returns header member for Single-HDU AstroData instances
        (which are those that
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
        The setHeader(..) function sets the extension header member for SINGLE EXTENSION MEFs 
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
                    
    header = property(getHeader,setHeader, None, """
                The header property can only be used for single-HDU AstroData instances, such as those returned
                during iteration. It is a property attribute which uses getHeader(..) and setHeader(..)
                to access the data members with "=" syntax.
                To set the header member, use "ad.header = newheader", where "newheader" must be a pyfits.Header object.
                To get the header member, use "hduheader = ad.header".
                """
                )

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
    
    def allDescriptorNames(self):
        funs = dir(CalculatorInterface)
        descs = []
        for fun in funs:
            if "_" != fun[0] and (fun.lower() == fun):
                descs.append(fun)
        return descs
        
    def allDescriptors(self):
        funs = self.allDescriptorNames()
        rdict = {}
        for fun in funs:
            # print "AD727:", repr(fun)
            try:
                val = eval("self.%s(asList=True)" % fun)
            except:
                val = "ERROR Getting Value"
            rdict.update({fun:val})
        return rdict
        
    
    def getIntExt(self, extension, hduref=False):
        """getInxExt takes an extension index, either an integer
        or (EXTNAME, EXTVER) tuple, and returns the index location
        of the extension.  If hduref is set to True, then the index
        returns is relative to the HDUList (0=PHU, 1=First non-PHU extension).
        If hduref == False (the default) then the index returned is relative to 
        the AstroData numbering convention, where index=0 is the first non-PHU
        extension in the MEF file.
        """
        if type(extension) == int:
            return extension + 1
        if type(extension) == tuple:
            
            for i in range(1, len( self.hdulist)):
                hdu = self.hdulist[i]
                nam = hdu.header.get("extname", None)
                ver = int(hdu.header.get("extver", None))
                if nam and ver:
                    if nam == extension[0] and ver == extension[1]:
                        if not hduref:
                            return i -1
                        else:
                            return i
        return None
    
    def renameExt(self, name, ver = None, force = False):
        """The renameExt() function is needed in order to give an HDU an new EXTNAME and EXTVER based identifier.  Merely changing
        the values in the extensions header are not sufficient, as there are in-memory properties of the HDU that are not changes when
        the header changes.
        WARNING: this function maniplates private (or somewhat private) 
            HDU members, name and _extver. STSCI has been informed of the issue and has made us a special HDU function for 
            performing the renaming.  When generally available, this new function will be used instead of manipulating the 
            HDU's properties directly."""
        if force != True and self.borrowedHDUList:
            raise ADExcept("cannot setExtname on subdata")
        
        if type(name) == tuple:
            name = name[0]
            ver = name[1]
        
        if ver == None:
            ver = 1
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
    setExtname = renameExt
            
            

    def open(self, source, mode = "readonly"):
        '''
        This function wraps a source dataset, which can be in memory as another 
        AstroData or pyfits HDUList, or on disk, given as the string filename.
        
        This is not the way one generally creates an AstroData instance from a 
        filename or other object. Instead, we generally pass the filename or HDUList instance
        into the
        constructor, which calls open. This function can still be of use if
        the AstroData object has been closed, but should probably be left to advanced
        users. Most people should use the constructor.

        :param source: source for data to be associated with this instance, can be 
                       an AstroData instance, a pyfits.HDUList instance, or a string filename.
        
        :type source: string | AstroData | pyfits.HDUList
        
        :param mode: IO access mode, same as the pyfits open mode, C{readonly},
                     C{update}, or C{append}.  The mode is passed to pyfits so if it is an
                     illegal mode name, pyfits will be the subsystem reporting the error. 
        
        :type mode: string
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
        elif type(source) == pyfits.ImageHDU:
            phu = pyfits.PrimaryHDU()
            self.hdulist= pyfits.HDUList([phu, source])
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

                if hdu.name or ("extname" in hdu.header): 
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
                    hdu.header.update("EXTNAME", "SCI", "added by AstroData", after='GCOUNT')
                    hdu.header.update("EXTVER", i, "added by AstroData", after='EXTNAME')
                    hdu.name = SCI
                    hdu._extver = i
    def close(self):
        """
        This function closes the pyfits.HDUList object if this instance
        is the owner (the instance originally calling pyfits.open(..) on this
        MEF).  Note... it may be bad to close files which have slices still
        in memory, the system currently does not ensure the HDUList does not contain
        HDUs still in use.
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
        """This function returns an array of type names as a list of strings.  It is possible to
        "prune" the list so that only leaf nodes appear, since these are the most specific types available.
        
        Note: types are divided into two categories, one intended for types which detect processing
        status (i.e. RAW vs PREPARED), and the other which contains a more traditional "typology" consisting of heirarchical
        trees of related datastypes.  To retrieve just status types, use getStatus(..), and to retreive just typological
        types use getTypology(..).  Note, the system does not enforce what checks are actually performed by types in each
        category, and both use the same DataClassification objects to define and discover type.  The distinction is
        organizational, and to allow assignment of features in status-related types to not conflict features assigned in
        typological-related types.
        
        :param prune: flag which controls 'pruning' the returned type list so that only the
                      leaf node type for a given set of related types is returned.
        :type prune: Bool
        :returns: a list of classification names that apply to this data
        :rtype: list of strings
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
        """ This function returns the set of type names (strings) which apply to
        this dataset and which
        come from the status section of the AstroData Type
        library. "Status" classifications are those which tend
        to change during the reduction of a dataset, 
        e.g. those related
        to the processing status of the data, e.g. RAW vs PREPARED. 
        Strictly a "status" type is any
        type defined in or below the "astrodata_Gemini/ADCONFIG/classification/status" 
        directory.

        :returns: a list of string classification names
        :rtype: list of strings
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
        :rtype: list of strings
        """

        if (self.typesStatus == None):
            cl = self.getClassificationLibrary()
            self.typesStatus = cl.discoverStatus(self)
            
        return self.typesStatus

    def getTypology(self):
        """
        This function returns the set of type names (strings) which apply to
        this dataset and which
        come from the typology section of the AstroData Type
        library. "Typology" classifications are those which tend
        to remain with the data in spite of reduction, e.g. those related
        to the instrument mode of the data. Strictly it is any
        type defined in or below the "astrodata_Gemini/ADCONFIG/classification/types" 
        directory.
        :returns: a list of classification name strings
        :rtype: list of strings"""
        
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

        
    def isType(self, *typenames):
        """
        This function checks the type of this data to see if it can be characterized 
        as the type specified 
        by C{typename}.
        
        :Note: "AstroData.checkType" is an alias for "AstroData.isType"
        
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
    checkType = isType

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
    def phuGetKeyValue(self, key):
        """
        The phuGetKeyValue(..) function returns the value associated with the
        given key within the primary header unit
        of the dataset.
        
        :param key: name of header value to retrieve
        :type key: string
        :rtype: string
        :returns: the key's value or None if not present, user must convert
        from string type.
        """
        try:
            hdus = self.getHDUList()
            retval = hdus[0].header[key]
            self.relhdul()
            return retval
        except KeyError:
            self.relhdul()
            return None
    phuValue = phuGetKeyValue
    phuHeader = phuValue
    
    def phuSetKeyValue(self, key, value, comment = None):
        """
        The phuSetKeyValue(..) function is used to set the value 
        (and optionally
        the comment) associated
        with a given key in the primary header unit of the dataset.
        
        :param key: name of PHU header value to set
        :type key: string
        :param value: value to apply to PHU header
        :type value: string (or can be converted to string)
        """
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
        :returns: the pyfits-index, relative to the containing HDUList
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
    
    def getKeyValue(self, key):
        """
        The getKeyValue(..) function is used to get the value associated
        with a given key in the data-header unit of a single-HDU
        AstroData instance (such as returned by iteration).
        
        :param key: name of header value to set
        :type key: string
        :returns: the specified value
        :rtype: string
        """
        if len(self.hdulist) == 2:
            return self.extGetKeyValue(0,key)
        else:
            raise ADExcept("getHeaderValue must be called on single extension instance")
    getHeaderValue = getKeyValue
           
    def extGetKeyValue(self, extension, key):
        """This function returns the value from the given extension's
        header, with "0" being the first data extension.  To get
        values from the PHU use phuGetKeyValue(..).
        
        :param extension: identifies which extension, either an integer index 
                          or (EXTNAME, EXTVER) tuple
        :type extension: int or (EXTNAME, EXTVER) tuple
        :param key: name of header entry to retrieve
        :type key: string
        :rtype: string
        :returns: the value associated with the key, or None if not present
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
        """
        The setKeyValue(..) function is used to set the value (and optionally
        the comment) associated
        with a given key in the data-header of a single-HDU AstroData instance.
        
        :param key: name of data header value to set
        :type key: string
        :param value: value to apply to header
        :type value: string (or can be converted to string)
        """
        if len(self.hdulist) == 2:
            self.extSetKeyValue(0,key)
        else:
            raise ADExcept("setKeyValue must be called on single extension instance")
    
    def extSetKeyValue(self, extension, key, value, comment = None):
        """
        The extSetKeyValue(..) function is used to set the value (and optionally
        the comment) associated
        with a given key in the header unit of the given extension within
        the dataset.
       
        :param extension: identifies which extension, either an integer index 
                          or (EXTNAME, EXTVER) tuple
        :type extension: int or (EXTNAME, EXTVER) tuple
        :param key: name of PHU header value to set
        :type key: string
        :param value: value to apply to PHU header
        :type value: string (or can be converted to string)
        """
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
        """The info(..) function calls the pyfits.HDUList.info(..) function
        on this instances "hdulist" member.  This function outputs information
        about the datasets HDUList to standard out. There are plans to
        replace this with a more AstroData centric display, e.g. the integer
        indexes given are relative to the HDUList and not the AstroData
        instance, and to return the report as a string.  Currently this is
        instead a convienience for viewing the state of an AstroData's
        HDUList."""
        self.hdulist.info()
        
    def displayID(self):
        import IDFactory
        return IDFactory.generateStackableID(self)
 
    def write(self, filename = None, clobber = False, rename=True):
        """
        The write function acts similarly to the HDUList.writeto(..) function
        if a filename is given, or like HDUList.update(..) if no name is given.
        When a name is given, this becomes the new on-disk name of the AstroData
        and will be used on subsequent calls to write for which the filename is
        not provided. If the clobber flag is False (the default) then
        write(..) throws an exception if the file already exists.
        
        :param fname: file name to write to, optional if instance already has name, which
                      might not be the case for new AstroData instances created in memory.
        :type fname: string
        :param clobber: This flag drives if AstroData will overwrite an existing
                    file.
        :type clobber: bool
        :param rename: This flag allows you to write the AstroData instance to a new
        filename, but leave the "current" name in tact.
        :type rename: bool
        """
        fname = filename
        hdul = self.gethdul()
        if fname == None:
            rename = False
            fname = self.filename
        else:
            if rename == True:
                self.filename = fname
           
        # by here fname is either the name passed in, or if None, it is self.filename
        if (fname == None):
            # @@FUTURE:
            # perhaps create tempfile name and use it?
            raise gdExcept()
           
        if os.path.exists(fname):
            if clobber:
                os.remove(fname)
            else:
                raise OutputExists(fname)
        hdul.writeto(fname)
        
        
    # MID LEVEL MEF INFORMATION
    #
    def countExts(self, extname):
        """The countExts(..) function counts the extensions of a given name
        (as stored in the HDUs "EXTVER" header). 

        :param extname: the name of the extension, equivalent to the
                        value associated with the "EXTNAM" key in the extension header.
        :type extname: string
        :returns: number of extensions of that name
        :rtype: int
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
            
    def historyMark(self, key=None, comment=None, stomp=True):
        """
        This function will add the timestamp type keys to the astrodata instance's PHU.
        The default will be to update the GEM-TLM key by just calling ad.historyMark() 
        without any input vals. Value stored is the UT time in the same format as the CL scripts.
        The GEM-TLM key will be updated along with the specified key automatically.
        
        param key: header keyword to be changed/added
        type key: string
        param comment: comment for the keyword in the PHU, keep it short
                       default if key is provided is 'UT Time stamp for '+key 
        type comment: string
        param stomp: if True, use the current time; if False, use the latest saved time
        type stomp: boolean (True/False)
        """
        if stomp:
            self.tlm = datetime.now().isoformat()[0:-7]
        elif (stomp==False) and (self.tlm==None):
            self.tlm = datetime.now().isoformat()[0:-7]
        
        if comment==None and key!=None:
            comment='UT Time stamp for '+key
        
        # Updating PHU with specified key and GEM-TLM    
        if key !=None:
            self.phuSetKeyValue(key,self.tlm,comment)
            self.phuSetKeyValue('GEM-TLM',self.tlm,'UT Last modification with GEMINI')
        # Only updating the GEM-TLM PHU key
        else:
             self.phuSetKeyValue('GEM-TLM',self.tlm,'UT Last modification with GEMINI')     
        
        return self.tlm
    #-------------------------------------------------------
    
    def div(self,denominator):
        
        adOut=arith.div(self,denominator)
        return adOut
    
    def mult(self,inputB):
        adOut=arith.mult(self,inputB)
        return adOut
    
    def add(self,inputB):
        adOut=arith.add(self,inputB)
        return adOut
    
    def sub(self,inputB):
        adOut=arith.sub(self,inputB)
        return adOut
    
# SERVICE FUNCTIONS and FACTORIES
def correlate( *iary):
    """
    The AstroData helper function, correlate(..), returns a list of tuples of 
    Single Extension AstroData instances which associate members extension
    among the input array.
    It accepts a variable number of arguments, all of which
    should be AstroData instances. The returned list contains tuples of associated 
    extensions... that is, given three inputs, 'a', 'b' and 'c'. 
    Assuming all have (SCI,1) extensions, then within the returned lists will
    be the equivalent of the following tuple
    (a[(SCI,1)], b[(SCI,1)], c[(SCI,1)]).  This is useful for processes which
    process multiple input images, i.e. within the Gemini system, to "add MEFa to MEFb" 
    means precicely this, to add MEFa[(SCI,1)] to MEFb[(SCI,1)], 
    MEFa[(SCI,2)] to MEFb[(SCI,2)] and so on. Similarly, such an operation of "SCI" frames
    will imply complementary changes on "VAR" and "DQ" frames which also will use
    the correlate function.
    
    :parm iary: any number of AstroData instances
    :type iary: arglist
    :returns: a list of tuples containing correlated extensions from the arguments. 
    :returns: list of tuples
    
    :Note: to appear in the list, all the given arguments must have an extension
    with the given (EXTNAME,EXTVER) for that tuple.
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
    The prepOutput(..) function creates a new AstroData ready for appending
    output information.  While you can also simply create an empty AstroData
    instance by giving no arguments to the AstroData constructor 
    (i.e. "ad = AstroData()"), prepOutput(..) will take into account the input
    array provided to:
    
    + Copy the PHU of the reference image (inputAry[0])
    + Propagate associated information such as the MDF in the case of a MOS 
      observation
    
    :param inputAry : The input array from which propagated content (such as the 
        source PHU) will be taken. Note: the zero-th element in the list is 
        used as the reference dataset, for PHU or other items which require
        a particular reference.
    :param name: File name to use for returned AstroData, optional.
    :returns: an AstroData instance with associated data from the inputAry 
        properly represented to the returned instance.
        File will not have been written to disk by prepOutput(..).
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
