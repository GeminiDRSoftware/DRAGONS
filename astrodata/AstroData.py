#!/bin/env pyth
__docform__ = "restructuredtext" #for epydoc

import sys
import os
import re
from datetime import datetime
from copy import copy, deepcopy
import urllib2
import pyfits
import numpy

import astrodata
from AstroDataType import *
import Descriptors
# this gets SCI (== "SCI") etc
from gemconstants import *
import Calculator
from astrodata.adutils import arith
from astrodata import Errors
from adutils.netutil import urlfetch
from astrodata.ExtTable import ExtTable
from adutils.gemutil import rename_hdu
try:
    from CalculatorInterface import CalculatorInterface
except ImportError:
    class CalculatorInterface:
        pass

verbose = False
verboseLoadTypes = True
verbt = False

def re_header_keys(rekey, header):
    """
    :param rekey: a regular expresion to match to keys in header
    :type rekey: string
    
    :param header: a pyfits.Header object as returned by *ad[("SCI",1)].header*
    :type header: pyfits.Header
    
    :return: a list of keys that appear in the given header
    :rtype: list of strings

    This utility function returns a list of keys from 
    the header passed in which match the given regular expression.
    """
    retset = []
    for k in header.ascardlist().keys():
        # print "gd278: key=%s" % k
        if re.match(rekey, k):
            retset.append(k)

    if len(retset) == 0:
       retset = None
    return retset

class AstroData(object, CalculatorInterface):
    """
The AstroData class abstracts datasets stored in MEF files
and provides uniform interfaces for working on datasets from different
instruments and modes.  Configuration packages are used to describe
the specific data characteristics, layout, and to store type-specific
implementations.

MEFs can be generalized as lists of header-data units, with key-value 
pairs populating headers and pixel data populating data,
AstroData interprets a MEF as a single complex entity.  The individual
"extensions" with the MEF are available using python list ("[]") syntax and are 
wrapped in AstroData objects (see 
:meth:AstroData.__getitem__()<astrodata.data.AstroData>). AstroData uses 
pyfits for MEF I/O and numpy for pixel manipulations. 

While the pyfits and numpy objects are available to the programmer, AstroData
provides analagous methods for most pyfits functionality which allows it to
maintain the dataset  as a cohesive whole. The programmer does however use the
numpy pixel arrays directly for pixel manipulation.

In order to identify types of dataset and provide type-specific behavior
AstroData relies on configuration packages either in the PYTHONPATH environment
variable or the Astrodata package environment variables, ADCONFIGPATH, or
RECIPEPATH. The configuration (i.e. astrodata_Gemini) contains definitions for
all  instrument-mode-specific behavior. The configuration contains type
definitions, meta-data functions, information lookup tables, and any other code
or information needed to handle specific types of dataset.

This allows AstroData to manage access to the dataset for convienience and
consistency.
For example AstroData is able...:

- ... to allow reduction scripts to have easy access to dataset classification 
  information in a consistent way across all instrument-modes
- ... to provide consistent interfaces for obtaining common meta-data across all
  instrument modes
- ... to relates internal extensions, e.g. discriminate between science and 
  variance arrays and associate them properly
- ... to help propagate header-data units important to the given instrument mode,
  but unknown to general purpose transformations

AstroData's purpose in general is to provide smart dataset-centered interfaces
which adapt to dataset type. The primary interfaces of note are for file
handling, dataset-type checking, and managing meta-data, but AstroData also
integrates other functionality.
"""
    types = None
    typesStatus = None
    typesTypology = None
    _filename = None
    __origFilename = None
    url = None # if retrieved
    hdulist = None
    hdurefcount = 0
    mode = "readonly"
    descriptor_calculator = None
    descriptorFormat = None
    # index is for iterator behavior
    index = 0
    # for subdata
    borrowed_hdulist = False # if we took the hdul from another GD instance
    container = None # AstroData instance we took hdul from
    
    # None means "all", otherwise, an array of extensions
    extensions = None
    tlm=None
    # ClassificationLibrary Singleton, must be retrieved through
    #   get_classification_library()
    classification_library = None

    def __init__(self, dataset=None, mode="readonly", exts=None, extInsts=None,
                    phu=None, header=None, data=None, store=None, 
                    storeClobber=False):
        """
        :param dataset: the dataset to load, either a filename (string), an
            AstroData instance, or a pyfits.HDUList 
        :type dataset:  string, AstroData, HDUList

        :param mode: IO access mode, same as pyfits mode ("readonly", "update",
            "or append") with one additional AstroData-specific mode, "new".
            If the mode is "new", and a filename is provided, the constructor
            checks the named file does not exist, and if it does not already
            exist, it creates an empty AstroData of that name (to save the
            filename), but does not write it to disk. Such an AstroData 
            instance is ready to have HDUs appended, and to be written to disk
            at the user's command with "ad.write()".
        :type mode: string
        
        :param exts: (advanced) a list of extension indexes in the parent
            HDUList that this instance should refer to, given  integer or 
            (EXTNAME, EXTVER) tuples specifying each extention in the "pyfits"
            index space where the PHU is at index 0, the first data extension
            is at index 1, and so on. I.e. This is primarilly intended for 
            internal use creating "sub-data", which are AstroData instances
            that represent a slice, or subset, of some other AstroData instance.
            
            NOTE: if present this option will override and obscure the extInsts
            argument which will be ignored. 
            
            Example of sub-data:

                sci_subdata = ad["SCI"]

            The sub-data is is created by passing "SCI" as an argument to the
            constructor. The 'sci_subdata' object would consist of its own 
            AstroData instance referring to it's own HDUList, but the HDUs in
            this list would still be shared (in memory) with the 'ad' object,
            and appear in its HDUList as well.
        :type exts: list
        
        :param extInsts: a list of extensions this instance should contain,
            given as actual pyfits.HDU instances. NOTE: if the "exts" argument
            is also set, this argument is ignored.
        :type extInsts: list of pyfits.HDU objects

        :param phu: primary header unit. This object is propagated to all 
            astrodata sub-data ImageHDUs. Special handling is made for header
            instances that are passed in as this arg., where a phu will be 
            created and the '.header' will be assigned (ex. hdulist[0], ad.phu,
            ad[0].hdulist[0], ad['SCI',1].hdulist[0], ad[0].phu, 
            ad['SCI',1].phu, and all the previous with .header appended) 
        :type phu: pyfits.core.PrimaryHDU, pyfits.core.Header 


        :param header: extension header for images (ex. hdulist[1].header,
            ad[0].hdulist[1].header, ad['SCI',1].hdulist[1].header)
        :type phu: pyfits.core.Header

        :param data: the image pixel array (ex. hdulist[1].data,
            ad[0].hdulist[1].data, ad['SCI',1].hdulist[1].data)
        :type data: numpy.ndarray
        
        :param store: directory where a copy of the original file will be 
            stored. Special handling is done for remote files.
        :type store: string
        
        :param storeClobber: remote file handling for existing files with the
            same name.  If true will save, if not, will delete.
        :type storeClobber: boolean

        The AstroData constructor constructs an in-memory representation of a
        dataset. If given a filename it uses pyfits to open the dataset, reads
        the header and detects applicable types. Binary data, such as pixel
        data, is left on disk until referenced.
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
            # print "AD257:", repr(parts)
            if len(parts) > 1:
                if parts[0] == "file":
                    remoteFile = False
                    dataset = parts[1][2:]
                else:
                    remoteFile = True
            else:
                remoteFile = False
            if remoteFile:
                # then the string is an URL, retrieve it
                import urllib
                from urllib import urlretrieve
                savename = os.path.basename(dataset)
                #print "AstroData retrieving remote file: "
                #print "     from %s" % dataset
                #print "     to   %s" % os.path.join(store,savename)
                try:
                    if store:
                        # print "AD230: Storing in,", store
                        fname = urlfetch(dataset, store=store, clobber= True) #storeClobber)
                        #fname,headers = urlretrieve(dataset, os.path.join(store, savename), None, 
                        #    urllib.urlencode({"gemini_fits_authorization":"good_to_go"}))
                    else:
                        # print "AD235: Retrieved to temp file"
                        fname = urlfetch(dataset, clobber=True)
                        #fname, headers = urlretrieve(dataset)
                    dataset = savename
                except urllib2.HTTPError, error:
                    raise Errors.AstroDataError("AstroData could not load via http: %s" % dataset)
            elif store:
                import shutil
                shutil.copy(dataset, store)
                dataset = os.path.join(store,dataset)
                # print "AD235:", dataset
            
        if dataset is None:
            if (type(data) is list):
                raise TypeError("cannot accept data as a list")
            if phu is None:
                hdu = pyfits.PrimaryHDU()
                # create null phu 
                dataset = pyfits.HDUList(hdu)
                # if data and/or header is None, pyfits will allow it 
                if data is not None:
                    dataset.append(pyfits.ImageHDU(data=data, header=header))
            else: 
                hdu = pyfits.PrimaryHDU()
                dataset = pyfits.HDUList(hdu)
                if type(phu) is pyfits.core.PrimaryHDU:
                    dataset[0] = phu
                # if phu is a header, then it will be assigned to a new phu
                elif phu.__class__ == pyfits.core.Header:
                    dataset[0].header = phu
                else:
                    raise TypeError("phu is of an unsupported type")
                if data is not None:
                    dataset.append(pyfits.ImageHDU(data=data, header=header))
        if fname == None:
            self.open(dataset, mode)
        else:
            # fname is set when retrieving an url, it will be the temporary 
            #   filename that urlretrieve returns.  Nice to use since it's 
            #   guarenteed to be unique.  But it's only on disk to load (we
            #   could build the stream up from the open url but that might be
            #   a pain (depending on how well pyfits cooperates)... and this 
            #   means we don't need to load the whole file into memory.
            self.open(fname, mode)
            if store == None:
                # because if store == None, file was retrieved to a persistent
                # location
                os.remove(fname)
    
    def __del__(self):
        """ This is the destructor for AstroData. It performs reference 
        counting and behaves differently when this instance is subdata, since
        in that case some other instance "owns" the pyfits HDUs instance.
        """
        if self.borrowed_hdulist:
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
        """
        :param ext: The integeter index, indexing(EXTNAME, EXTVER) tuple,
            EXTNAME name for the desired subdata. If an int or tuple, a  single
            extension is wrapped with an AstroData instance, and
            "single-extension" members of the AstroData object can be used. If
            a string is given then all extensions with the given EXTNAME will
            be wrapped by the  new AstroData instance.
        :type ext: string, int, or tuple
        
        :returns: an AstroData instance associated with the subset of data.
        :rtype: AstroData

        
        This function supports the "[]" syntax for AstroData instances,
        e.g. *ad[("SCI",1)]*.  We use it to create
        AstroData objects associated with "subdata" of the parent
        AstroData object, that is, consisting of an HDUList which
        consists of some subset of the parent MEF. e.g.::
        
            datasetA = AstroData.AstroData("datasetMEF.fits")
            datasetB = datasetA[SCI]
            
        In this case, after the operations, datasetB is an AstroData object
        associated with the same MEF, sharing some of the the same actual HDUs
        in memory as  datasetA. The object in "datasetB" will behave as if the
        SCI extensions are its only members, and it does in fact have its own
        pyfits.HDUList. Note that datasetA and datasetB share the PHU and also
        the data structures of the HDUs they have in common, so that a change
        to "datasetA[('SCI',1)].data" will change the 
        "datasetB[('SCI',1)].data" member and vice versa. They are in fact both
        references to the same numpy array in memory. The HDUList is a 
        different list, however, that references common HDUs. If a subdata 
        related AstroData object is written to disk, the resulting MEF will
        contain only the extensions in datasetB's HDUList.

        :note: Integer extensions start at 0 for the data-containing 
            extensions, not at the PHU as with pyfits.  This is important:
            ad[0] is the first content extension, in traditional MEF 
            percpective, the extension AFTER the PHU; it is not the PHU!  In
            AstroData instances, the PHU is purely a header, and not counted
            as an extension in the way that headers generally are not counted
            as their own elements in the array they contain meta-data for.
            The PHU can be accessed  with the code, ''ad.phu'', or via the
            AstroData PHU-related member functions.
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
            
            if len(exs):
                return AstroData(self, exts=exs)
            else:
                return None
            
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
                #print "AD426: keyerror"
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
    
        
    # ITERATOR PROTOCOL FUNCTIONS
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
        
        :returns: a single extension AstroData instance representing the
            'current' extension in the AstroData iteration loop.
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
                #ext = self.extensions[self.index]
                ext = self.index
        except IndexError:
            raise StopIteration
        #print "AD478:", self.index
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
        # loop through headers and copy to a new list
        for hdu in self.hdulist:
            nhdu = copy(hdu)
            #print "AD496:", repr(hdu.header), id(hdu.data)
            nhdu.header = nhdu.header.copy()
            lohdus.append(nhdu)
        # Load copied list of headers into pyfits
        hdulist = pyfits.HDUList(lohdus)
        # loading the pyfits hdulist into AstroData
        adReturn = AstroData(hdulist)
        # Copying over private variables over to copied object
        adReturn.__origFilename = self.__origFilename
        adReturn._filename = self._filename
        # Return fully copied AD instance
        return adReturn
    
    # -------- private variable: filename ---------------------------------
    def get_filename(self):
        return self._filename
    def set_filename(self, newfn):
        if self.mode == "readonly":
            self.mode = "update"
        self._filename = newfn

    filename = property(get_filename, set_filename, None, "The filename"
                " member is monitored so that the mode can be changed from"
                " readonly when the filename is changed.")
    
    def moredata_check(self, md=None, append=False, insert=False, \
                       replace=False, index=None):
        if isinstance(md, AstroData):
            return md.hdulist
        elif type(md) is pyfits.HDUList:
            return md
        elif isinstance(md, pyfits.core._AllHDU):
            try:
                if append:
                    self.hdulist.append(md)
                    print "WARNING: Appending unknown HDU type"
                    return False
                elif insert or replace:
                    if not index:
                        raise Errors.AstroDataError(\
                            "index required to insert")
                    else:
                        if replace:
                            self.remove(index)
                        self.hdulist.insert(i + 1, md)
                        print "WARNING: Inserting unknown HDU type"
                        return False
            except:
                raise Errors.AstroDataError(\
                   "cannot operate on pyfits.core._AllHDU instance")
        else:
            raise Errors.AstroDataError(\
                "The 'moredata' argument is of an unsupported type")


    def moredata_work(self, append=False, insert=False, autonum=False, \
                      et_host=None, et_guest=None, hduindx=None, hdul=None): 
        """
        create a master table out of the host and update the EXTVER 
        for the guest as it is being updated in the table
        """
        if autonum:
            guest_bigver = et_guest.largest_extver()
            count = 0
            ext = 0
            for row in et_guest.rows():
                host_bigver = et_host.largest_extver()
                for tup in row:
                    if None in et_guest.xdict[tup[0]].keys():
                        et_host.putAD(extname=tup[0], extver=None,\
                            ad=tup[1])
                        rename_hdu(name=tup[0], ver=None, \
                            hdu=hdul[tup[1][1]])
                        ext += 1
                    else:
                        et_host.putAD(extname=tup[0], \
                            extver=host_bigver + 1, ad=tup[1])
                        rename_hdu(name=tup[0], ver=host_bigver + 1, \
                            hdu=hdul[tup[1][1]])
                        ext += 1
                count +=1
            for i in range(1,len(hdul)):
                if append:
                    self.hdulist.append(hdul[i])
                elif insert:
                    if len(self.hdulist) == 1:
                        self.hdulist.append(hdul[i])
                    else:
                        self.hdulist.insert(hduindx, hdul[i])
        else:
            for ext in et_guest.xdict.keys():
                if ext in et_host.xdict.keys():
                    for ver in et_guest.xdict[ext].keys():
                        if ver in et_host.xdict[ext].keys():
                            raise Errors.AstroDataError(\
                        "EXTNAME, EXTVER conflict, use auto_number")
            for hdu in hdul[1:]:
                if append:
                    self.hdulist.append(hdu)
                elif insert:
                    self.hdulist.insert(hduindx, hdu)
   
    def append(self, moredata=None, data=None, header=None, auto_number=False,\
               extname=None, extver=None):
        """
        :param moredata: either an AstroData instance, an HDUList instance, 
            or an HDU instance to add to this AstroData object.
            When present, data and header arguments will be ignored.
        :type moredata: pyfits.HDU, pyfits.HDUList, or AstroData
        
        :param data: if moredata *is not* specified, data and header should 
            both be set and are used to construct a new HDU which is then 
            added to the  AstroData object. The 'data' argument should be set
            to a valid numpy array.
        :type data: numarray.numaraycore.NumArray
        
        :param header: if moredata *is not* specified, data and header are used
            to make  an HDU which is then added to the HDUList associated with
            this AstroData instance. The 'header' argument should be set to a
            valid pyfits.Header object.
        :type header: pyfits.Header

        :param auto_number: auto-increment extver to fit file convention
        :type auto_number: boolean

        :param extname: extension name (ex, 'SCI', 'VAR', 'DQ')
        :type extname: string

        This function appends more data units (aka "HDUs") to the AstroData
        instance.
        """
        et_host = None
        et_guest = None
        hdulist = None
        override_extver=False
        if extver:
            override_extver=True
        
        if moredata:
            hdulist = self.moredata_check(md=moredata, append=True)
            if not hdulist:
                return
            ethost = ExtTable(hdul=self.hdulist)
            etguest = ExtTable(ad=moredata)
            self.moredata_work(append=True, autonum=auto_number, \
                et_host=ethost, et_guest=etguest, hdul=hdulist)
        else:
            if header is None or data is None: 
                raise Errors.AstroDataError(\
                    "both header and data is required")
             
            #  override header extname, extver if given in args, then
            #  if host ad has 1 or 0 exts, will just append and exit
            header = self.verify_header(extname=extname, extver=extver, \
                header=header)
            # check for conflict if not auto_number then increment
            # off the largest host extver unless the extname is larger
            et_host = ExtTable(self)
            if not auto_number:
                for ext in et_host.xdict.keys():
                    if header["EXTNAME"] == ext:
                        if header["EXTVER"] in et_host.xdict[ext].keys():
                            raise Errors.AstroDataError(\
                                "EXTNAME EXTVER conflict, use auto_number") 
            host_bigver = et_host.largest_extver()
            if header["EXTVER"] > host_bigver:
                extver = header["EXTVER"]
            if not override_extver:
                extver = host_bigver + 1
            header.update("EXTVER", extver, "Added by AstroData")
            self.hdulist.append(pyfits.ImageHDU(data=data,\
                header=header))
    
   
    def close(self):
        """The close(..) function will close the HDUList associated with this
        AstroData instance. If this is subdata, e.g. (sd = gd[SCI] where gd is
        another AstroData instance, sd is "sub-data")  then sd.close() will not
        close the original hdulist because gd will actually own the hold on 
        that HDUList and its related file."""
        if self.borrowed_hdulist:
            self.container.relhdul()
            self.hdulist = None
        else:
            if self.hdulist != None:
                self.hdulist.close()
                self.hdulist = None
 
    def remove(self, index):
        """
        :param index: the extension index, either an int or (EXTNAME, EXTVER)
            pair before which the extension is to be inserted. Note, the 
            first data extension is [0], you cannot insert before the PHU.
            Index always refers to Astrodata Numbering system, 0 = HDU
        :type index: integer or (EXTNAME,EXTVER) tuple
        
        """
        if type(index) == tuple:
            index = self.get_int_ext(index, hduref=True)
            self.hdulist.__delitem__(index)
        else:    
            if index > len(self) - 1:
                raise Errors.AstroDataError("Index out of range")
            self.hdulist.__delitem__(index - 1)
            
    def insert(self, index, moredata=None, data=None, header=None, \
               auto_number=False, extname=None, extver=False):
        """
        :param index: the extension index, either an int or (EXTNAME, EXTVER)
            pair before which the extension is to be inserted. Note, the 
            first data extension is [0], you cannot insert before the PHU.
            Index always refers to Astrodata Numbering system, 0 = HDU
        :type index: integer or (EXTNAME,EXTVER) tuple
        
        :param moredata: Either an AstroData instance, an HDUList instance, or
            an HDU instance. When present, data and header will be ignored.
        :type moredata: pyfits.HDU, pyfits.HDUList, or AstroData
        
        :param data: if moredata *is not* specified, data and header should 
            both be set and are used to construct a new HDU which is then 
            added to the AstroData instance.
        :type data: numarray.numaraycore.NumArray

        :param header: if moredata *is not* specified, data and header are 
            used to make an HDU which is then added to the HDUList associated
            with this AstroData instance.
        :type header: pyfits.Header
        
        :param auto_number: auto-increment appends to match existing extname - extver 
            convention.
        :type auto_number: boolean

        :param extname: extension name (ex, 'SCI', 'VAR', 'DQ')
        :type extname: string

        :param extver: extension version (ex, 1, 2, 3) 
        :type extver: integer
        
        This function inserts more data units (aka an "HDU") to the AstroData
        instance.
        """
        et_host = None
        et_guest = None
        hdulist = None
        hdu_index = None
        if type(index) == tuple:
            hdu_index = self.get_int_ext(index, hduref=True)
        else:    
            hdu_index = index + 1
        if hdu_index > len(self.hdulist):
            raise Errors.AstroDataError("Index out of range")
        if moredata:
            hdulist = self.moredata_check(md=moredata, insert=True, index=index)
            if not hdulist:
                return
            
            ethost = ExtTable(hdul=self.hdulist)
            etguest = ExtTable(ad=moredata)
            self.moredata_work(insert=True, autonum=auto_number, \
                et_host=ethost, et_guest=etguest, hdul=hdulist, \
                hduindx=hdu_index)
        else:
            if header is None or data is None: 
                raise Errors.AstroDataError(\
                    "both header and data is required")
             
            #  override header extname, extver if given in args, then
            #  if host ad has 1 or 0 exts, will just insert and exit
            header = self.verify_header(extname=extname, extver=extver, \
                header=header)
            if len(self.hdulist) == 0:
                self.hdulist.insert(hdu_index, pyfits.PrimaryHDU(data=data, \
                    header=header))
                return
            if len(self.hdulist) == 1:
                self.hdulist.insert(hdu_index, pyfits.ImageHDU(data=data, \
                    header=header))
                return 
            
            et_host = ExtTable(self)
            if not auto_number:
                for ext in et_host.xdict.keys():
                    if header["EXTNAME"] == ext:
                        if header["EXTVER"] in et_host.xdict[ext].keys():
                            raise Errors.AstroDataError(\
                                "EXTNAME EXTVER conflict, use auto_number") 
            host_bigver = et_host.largest_extver()

            if header["EXTVER"] > host_bigver:
                extver = header["EXTVER"]
            else:
                extver = host_bigver + 1
            header.update("EXTVER", extver, "Added by AstroData")
            self.hdulist.insert(hdu_index, pyfits.ImageHDU(data=data,\
                header=header))
    
               
    def infostr(self, as_html=False, verbose=False, table=False):
        """
        :param as_html: boolean that indicates if the string should be HTML
                       formatted or not
        :type as_html: bool
        
        :param verbose: boolean that will add alias and object id info
        :type verbose: bool

        The infostr(..) function is used to get a string ready for display
        either as plain text or HTML.  It provides AstroData-relative
        information, unlike the pyfits-forwarded function AstroData.info(),
        and so uses AstroData relative indexes, descriptors, and so on.  
        """
        if not as_html:
            hdulisttype = ""
            phutype = None
            #Check basic structure of ad
            if isinstance(self, astrodata.AstroData):
                selftype = "AstroData"
            if isinstance(self.hdulist, pyfits.core.HDUList):
                hdulisttype = "HDUList"
            if isinstance(self.phu, pyfits.core.PrimaryHDU):
                phutype = "PrimaryHDU"
            if isinstance(self.phu.header, pyfits.core.Header):
                phuHeaderType = "Header"
            rets = ""
            
            # Create Primary AD info
            rets += "\nFilename: %s" % str(self.filename)
            if verbose:
                rets += "\n Obj. ID: %s" % str(id(self))
            rets += "\n    Type: %s" % selftype
            rets += "\n    Mode: %s" % str(self.mode)
            if verbose:
                rets += "\n\nAD No.    Name          Type      MEF No."
                rets += "  Cards    Dimensions   Format   ObjectID   "
                rets += "\n%shdulist%s%s%s%s" % (" "*8, " "*7, \
                    hdulisttype, " "*45, str(id(self.hdulist)))
                rets += "\n%sphu%s%s    0%s%d%s%s" % (" "*8, " "*11, \
                    phutype, " "*7, len(self.phu._header.ascard),\
                    " "*27, str(id(self.phu)))
                rets += "\n%sphu.header%s%s%s%s" % (" "*8, " "*4, \
                    phuHeaderType, " "*46, str(id(self.phu.header)))
            else:
                rets += "\n\nAD No.    Name          Type      MEF No."
                rets += "  Cards    Dimensions   Format   "
                rets += "\n%shdulist%s%s" % (" "*8, " "*7, hdulisttype)
                rets += "\n%sphu%s%s    0%s%d" % (" "*8, " "*11, \
                    phutype, " "*7, len(self.phu._header.ascard))
                rets += "\n%sphu.header%s%s" % (" "*8, " "*4, phuHeaderType)
            hdu_indx = 1
            for hdu in self.hdulist[1:]:
                #if ext.extname() is None:
                #    rets += "\n\t* There are no extensions *"
                #else:
                # Check hdulist instances
                if isinstance(hdu, pyfits.core.ImageHDU):
                    extType = "ImageHDU"
                    if isinstance(hdu.header, pyfits.core.Header):
                        extHeaderType = "Header"
                    else:
                        extHeaderType = ""
                    if isinstance(hdu.data, numpy.ndarray):
                        extDataType = "ndarray"
                    elif hdu.data is None:
                        extDataType = "None"
                    else:
                        extDataType = ""
                elif isinstance(hdu, pyfits.core.BinTableHDU):
                    extType = "BinTableHDU"
                    if isinstance(hdu.header, pyfits.core.Header):
                        extHeaderType = "Header"
                    else:
                        extHeaderType = ""
                    if isinstance(hdu.data, pyfits.core.FITS_rec):
                        extDataType = "FITS_rec"
                    elif hdu.data is None:
                        extDataType = "None"
                    else:
                        extDataType = ""
                else:
                    extType = ""
                
                # Create sub-data info lines
                adno_ = "[" + str(hdu_indx - 1) + "]"
                try:
                    name_ = None
                    cards_ = None
                    if not hdu.header.has_key("EXTVER"):
                        name_ = hdu.header["EXTNAME"]
                    else:
                        name_ = "('" + hdu.header['EXTNAME'] + "', "
                        name_ += str(hdu.header['EXTVER']) + ")"
                    cards_ = len(self.hdulist[hdu_indx]._header.ascard)
                except:
                    pass
                if extType == "ImageHDU":
                    if self.hdulist[hdu_indx].data == None:
                        dimention_ = None
                        form_=None
                    else:
                        dimention_ = self.hdulist[hdu_indx].data.shape
                        form_ = self.hdulist[hdu_indx].data.dtype.name
                else:
                    dimention_ = ""
                    form_ = ""
                if verbose:
                    rets += "\n%-7s %-13s %-13s %-8s %-5s %-13s %s  %s" % \
                        (adno_, name_, extType, str(hdu_indx), str(cards_), \
                            dimention_, form_, \
                            str(id(self.hdulist[hdu_indx])))
                    if extType == "ImageHDU" or extType == "BinTableHDU":
                        rets +="\n           .header    %s%s%s" % \
                            (extHeaderType, " "*46, \
                            str(id(self.hdulist[hdu_indx].header)))
                        rets +="\n           .data      %s%s%s" % \
                            (extDataType, " "*45, \
                            str(id(self.hdulist[hdu_indx].data)))
                else:
                    rets += "\n%-7s %-13s %-13s %-8s %-5s %-13s %s" % \
                        (adno_, name_, extType, str(hdu_indx), str(cards_), \
                            dimention_, form_)
                    if extType == "ImageHDU" or extType == "BinTableHDU":
                        rets +="\n           .header    %s" % extHeaderType 
                        rets +="\n           .data      %s" % extDataType
                hdu_indx += 1
            if verbose:
                s = " "*24
                rets += """


Sub-data Information:

An AstroData instance (AD) is always associated with a second AstroData
instance, or sub-data(AD[]).  This allows users the convenience of accessing
header and image data directly (ex ad[0].data, ad('SCI', 1).data).  Both the
AD and sub-data share objects in memory, which cause many aliases (see below).
Also note that the sub-data mode='update' property cannot be changed and 
AD.filename is assigned to AD[].filename but cannot be changed by the sub-data
instance.
                
      Name Mapping for AD and sub-data(AD[]) for a 3 ext. MEF

AD.phu == AD.hdulist[0] == AD[0].hdulist[0] == AD('SCI', 1).hdulist[0] 
                        == AD[1].hdulist[0] == AD('SCI', 2).hdulist[0]
                        == AD[2].hdulist[0] == AD('SCI', 3).hdulist[0]
                        == AD[0].phu == AD('SCI', 1).phu 
                        == AD[1].phu == AD('SCI', 2).phu
                        == AD[2].phu == AD('SCI', 3).phu

AD.phu.header == all of the above with .header appended

AD[0].data == AD.hdulist[1].data == AD.('SCI', 1).data          
AD[1].data == AD.hdulist[2].data == AD.('SCI', 2).data          
AD[2].data == AD.hdulist[3].data == AD.('SCI', 3).data          
    
AD[0].header == AD.hdulist[1].header == AD.('SCI', 1).header          
AD[1].header == AD.hdulist[2].header == AD.('SCI', 2).header        
AD[2].header == AD.hdulist[3].header == AD.('SCI', 3).header

                     Relationship to pyfits

The AD creates a pyfits HDUList (if not supplied by one) and attaches it 
to itself as AD.hdulist.  The sub-data also creates its own unique HDUList as 
AD[?].hdulist or AD('?', ?).hdulist, but shares in memory the phu (including
the phu header) with the primary AD HDUList. 

The AD.hdulist may have more than one extension, however, the sub-data is 
limited to one extension. This sub-data hdulist extension shares memory with
its corresponding AD.hdulist extension (ex. AD[0].hdulist[1] == AD.hdulist[1])

One important difference to note is that astrodata begins its first element 
'0' with data (ImageHDU), where pyfits HDUList begins its first element '0'
with meta-data (PrimaryHDU). This causes a 'one off' discrepancy. 

                """
        else:
            rets="<b>Extension List</b>: %d in file" % len(self)
            rets+="<ul>"
            for ext in self:
                rets += "<li>(%s, %s)</li>" % (ext.extname(), str(ext.extver()))
            rets += "</ul>"
        if table:
            rets = ""
            count = 0
            for ext in self:
                if isinstance(ext.hdulist[1], pyfits.core.BinTableHDU):
                    count += 1
                    rets += "\n" + "="*79 + "\n" + str(count) 
                    rets += ". BinTableHDU: " + ext.extname() + "\n" + "="*79
                    rets += "\n      Name            Value" + " "*25 + "Format"
                    rets += "\n" + "-"*79
                    fitsrec = ext.hdulist[1].data
                    for i in range(len(fitsrec.names)):
                        fstr = eval(\
        "ext.hdulist[1].header.ascard['TFORM%s']._cardimage.split(':')[1]" % (i + 1))
                        rets += "\n%-15s : %-15s         %3s (%-10s)" % \
                        (fitsrec.names[i],fitsrec[0][i], fitsrec.formats[i], fstr)
                    rets += "\n" + "="*79
        return rets
        
    def except_if_single(self):
        if len(self.hdulist) != 2:
            raise Errors.SingleHDUMemberExcept()
            
    def extname(self):
        self.except_if_single()
        return self.hdulist[1].header.get("EXTNAME", None)
        
    def extver(self):
        self.except_if_single()
        retv = self.hdulist[1].header.get("EXTVER", None)
        if retv:
            retv = int(retv)
        return retv
        
    def get_data(self):
        """
        :return: data array associated with the single extension
        :rtype: pyfits.ndarray

        The *get_data(..)* member is the function behind the property-style
        "data" member and returns appropriate HDU's data member(s) specifically
        for the case in which the AstroData instance has ONE HDU (in addition to
        the PHU). This allows a single-extension AstroData, such as AstroData
        generates through iteration,  to be used as though it simply is just the
        one extension, e.g. allowing *gd.data* to be used in place of the more
        esoteric and ultimately more dangerous *gd[0].data*. One can assure one
        is dealing with single extension AstroData instances when iterating over
        the AstroData extensions and when picking out an extension  by integer
        or tuple indexing, e.g.::

            for gd in dataset[SCI]:
                # gd is a single-HDU index
                gd.data = newdata

            # assuming the named extension exists,
            # sd will be a single-HDU AstroData
            sd = dataset[("SCI",1)]
        """
        hdl = self.gethdul()
        if len(hdl) == 2:
            retv = hdl[1].data
        else:
            raise Errors.SingleHDUMemberExcept()
        self.relhdul()
        return retv

    def set_data(self, newdata):
        """
        :param newdata: new data objects
        :type newdata: numarray.numarraycore.NumArray

        :raise Errors.SingleHDUMemberExcept: if AstroData instance has more 
            than one extension (not including PHU).

        This function sets the data member of a data section of an AstroDat
        object, specifically for the case in which the AstroData instance has
        ONE header-data unit (in addition to PHU).  This case is assured when
        iterating over the AstroData extensions, e.g.::

            for gd in dataset[SCI]:
                ...
        """
        hdl = self.gethdul()
        if len(hdl) == 2:
            # note: should we check type of newdata?
            hdl[1].data = newdata
        else:
            raise Errors.SingleHDUMemberExcept()
        self.relhdul()
        return 
    
    data = property(get_data, set_data, None, """
            The data property can only be used for single-HDU AstroData
            instances, such as those returned during iteration. It is a property
            attribute which uses *get_data(..)* and *set_data(..)* to access the
            data members with "=" syntax. To set the data member, use *ad.data =
            newdata*, where *newdata* must be a numpy array. To get the data
            member, use *npdata = ad.data*.
            """)
    
    def get_header(self, extension = None):
        """
        :return: header
        :rtype: pyfits.Header

        :raise Errors.SingleHDUMemberExcept: Will raise an exception if more
            than one extension exists. 
            (note: The PHU is not considered an extension in this case)
        
        The get_header(..) function returns the header member for Single-HDU
        AstroData instances (which are those that have only one extension plus
        PHU). This case  can be assured when iterating over extensions using
        AstroData, e.g.::
        
            for gd in dataset[SCI]: 
                ...
        """
        if extension == None:
            hdl = self.gethdul()
            if len(hdl) == 2:
                retv = hdl[1].header
            else:
                #print "numexts = %d" % len(hdl)
                raise Errors.SingleHDUMemberExcept()
            self.relhdul()
            return retv
        else: 
            hdl = self.gethdul()
            retv = hdl[extension].header
            self.relhdul()
            return retv
            
    def set_header(self, header, extension=None):
        """
        :param header: pyfits Header to set for given extension
        
        :type header: pyfits.Header
        
        :param extension: Extension index to retrieve header, if None
            or not present then this must be a single extension AstroData
            instance, which contains just the PHU and a single data extension,
            and the data extension's header is returned.

        :type extension: int or tuple, pyfits compatible extension index
        
        :raise Errors.SingleHDUMemberExcept: Will raise an exception if more 
            than one extension exists. 

        The set_header(..) function sets the extension header member for single
        extension (which are those that have only one extension plus PHU). This
        case  is assured when iterating over extensions using AstroData, e.g.:

            for gd in dataset[SCI]: 
                ...
        """
        if extension == None:
            hdl = self.gethdul()
            if len(hdl) == 2:
                hdl[1].header = header
            else:
                raise Errors.SingleHDUMemberExcept()
            self.relhdul()
        else:
            self.hdulist[extension].header = header
                    
    header = property(get_header, set_header, None, """
                The header property can only be used for single-HDU AstroData
                instances, such as those returned during iteration. It is a
                property attribute which uses *get_header(..)* and
                *set_header(..)* to access the header member with the "=" syntax.
                To set the header member, use *ad.header = newheader*, where
                *newheader* must be a pyfits.Header object. To get the header
                member, use *hduheader = ad.header*.
                """
                )

    def get_headers(self):
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

    def has_single_hdu(self):
        return len(self.hdulist) == 2
    
    def all_descriptor_names(self):
        funs = dir(CalculatorInterface)
        descs = []
        for fun in funs:
            if "_" != fun[0] and (fun.lower() == fun):
                descs.append(fun)
        return descs
        
    def all_descriptors(self):
        funs = self.all_descriptor_names()
        rdict = {}
        for fun in funs:
            # print "AD727:", repr(fun)
            try:
                val = eval("self.%s(asList=True)" % fun)
            except AttributeError:
                val = 'ERROR: No Descriptor Function Named "%s"' % fun  
            except:
                val = "ERROR: %s" % repr(sys.exc_info()[1])
            rdict.update({fun:val})
        return rdict
        
    def get_int_ext(self, extension, hduref=False):
        """getInxExt takes an extension index, either an integer
        or (EXTNAME, EXTVER) tuple, and returns the index location
        of the extension.  If hduref is set to True, then the index
        returned is relative to the HDUList (0=PHU, 1=First non-PHU extension).
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
    
    def open(self, source, mode = "readonly"):
        """
        :param source: source contains some reference for the dataset to 
                       be opened and associated with this instance. Generally
                       it would be a filename, but can also be
                       an AstroData instance or a pyfits.HDUList instance.
        
        :type source: string | AstroData | pyfits.HDUList
        
        :param mode: IO access mode, same as the pyfits open mode, "readonly,
                     "update", or "append".  The mode is passed to pyfits so
                     if it is an illegal mode name, pyfits will be the
                     subsystem reporting the error. 
        
        :type mode: string

        This function wraps a source dataset, which can be in memory as another
        AstroData or pyfits HDUList, or on disk, given as the string filename.
        
        Please note that generally one does not use "open" directly, but passes
        the filename to the AstroData constructor. The constructor uses
        open(..) however.  Most users should use the constructor, which may 
        perform extra operations.
        """
        inferRAW = True
        # might not be a filename, if AstroData instance is passed in
        #  then it has opened or gets to open the data...
        if isinstance(source, AstroData):
            inferRAW = False
            self.filename = source.filename
            self.__origFilename = source.filename
            self.borrowed_hdulist = True
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
        elif isinstance(source, pyfits.core._AllHDU):
            phu = pyfits.PrimaryHDU()
            self.hdulist= pyfits.HDUList([phu, source])
        else:
            if source == None:
                phu = pyfits.PrimaryHDU()
                self.hdulist = pyfits.HDUList([phu])
            else:
                if not os.path.exists(source):
                    raise IOError("Cannot open " + source)
                self.filename = source
                self.__origFilename = source
                try:
                    if mode == "new":
                        if os.access(self.filename, os.F_OK):
                            os.remove(self.filename)
                        mode = "append"
                    self.hdulist = pyfits.open(self.filename, mode=mode)
                    self.mode = mode
                    if len(self.hdulist) == 1:   # This is a single FITS
                        hdu = self.hdulist[0]
                        nhdu = pyfits.PrimaryHDU()
                        hdulist = pyfits.HDUList([nhdu])
                        imagehdu = pyfits.ImageHDU(header=hdu.header, \
                            data=hdu.data)
                        hdulist.append(imagehdu)
                        self.hdulist=hdulist
                        kafter = "GCOUNT"
                        if not hdu.header.has_key(kafter): 
                            kafter = None
                        if hdu.header.get("TFIELDS"): 
                            kafter = "TFIELDS"
                        hdu.header.update("EXTNAME", "SCI", \
                            "added by AstroData", after=kafter)

		    #print "AD591:", self.hdulist[1].header["EXTNAME"]
                    #print "AD543: opened with pyfits", len(self.hdulist)
                except IOError:
                    print "CAN'T OPEN %s, mode=%s" % (self.filename, mode)
                    raise

        if len(self.hdulist):
            try:
                self.discover_types()
            except KeyboardInterrupt:
                raise
            except:
                raise
                raise Errors.AstroDataError("discover types failed")
        
        if inferRAW and self.is_type("RAW"):
            # for raw, if no extensions are named
            # infer the name as "SCI"
            hdul = self.hdulist
            namedext = False
            for hdu in hdul[1:]:
    	        #print "AD1036:",hdu.name
                if hdu.name or ("extname" in hdu.header): 
                    namedext = True
                    #print "AD1040: Named", hdu.header["extname"]
                else:
                    #print "AD1042: Not Named"
                    pass
            if namedext == False:
                #print "AD1046: No named extension"
                l = len(hdul) # len w/phu
                #print "AD1048: len of hdulist ",l
                # nhdul = [hdul[0]]
                # nhdulist = pyfits.HDUList(nhdul)
                for i in range(1, l):
                    hdu = hdul[i]
                    kafter = "GCOUNT"
                    if hdu.header.get("TFIELDS"): kafter = "TFIELDS"
                    hdu.header.update("EXTNAME", "SCI", \
                        "added by AstroData", after=kafter)
                    hdu.header.update("EXTVER", i, \
                        "added by AstroData", after="EXTNAME")
                    hdu.name = SCI
                    hdu._extver = i
            else:
                # infer bad extvers to a number.
                inferEXT = max(hdul, key= lambda arg: arg.header.get("EXTVER"))
                inferEV = int(inferEXT.header.get("EXTVER"))
                if inferEV < 1:
                    inferEV = 0
                    
                numhdu = len(hdul)
                for i in range(1, numhdu):
                    hdu = hdul[i]
                    ev = hdu.header.get("EXTVER")
                    inferEV += 1
                    if not ev or int(ev)< 1:
                        hdu.header.update("EXTVER", inferEV ,after="EXTNAME")
                        hdu._extver = inferEV
                        
    
    def rename_ext(self, name, ver=None, force=True):
        """
        :param name: New "EXTNAME" for the given extension.
        :type name: string
        
        :param ver: New "EXTVER" for the given extension
        :type ver: int

        Note: This member only works on single extension AstroData instances.

        The rename_ext() function is used in order to rename an HDU with a new
        EXTNAME and EXTVER based identifier.  Merely changing the EXTNAME and 
        EXTEVER values in the extensions pyfits.Header are not sufficient.
        Though the values change in the pyfits.Header object, there are special
        HDU class members which are not updated. 
        
        :warning:   This function maniplates private (or somewhat private)  HDU
                    members, specifically "name" and "_extver". STSCI has been
                    informed of the issue and
                    has made a special HDU function for performing the renaming. 
                    When generally available, this new function will be used instead of
                    manipulating the  HDU's properties directly, and this function will 
                    call the new pyfits.HDUList(..) function.
        """
        # @@TODO: change to use STSCI provided function.
        if force != True and self.borrowed_hdulist:
            raise Errors.AstroDataError("cannot setExtname on subdata")
        if not self.has_single_hdu():
            raise Errors.SingleHDUMemberExcept()
        rename_hdu(name=name, ver=ver, hdu=self.hdulist[1])    
        # print "AD553:", repr(hdu.__class__)
    #alias
    setExtname = rename_ext
   
    def replace(self, index, data=None, header=None, phu=None):
        """
        :param index: the extension index, either an int or (EXTNAME, EXTVER)
            pair before which the extension is to be inserted. Note, the 
            first data extension is [0], you cannot insert before the PHU.
        :type index: integer or (EXTNAME,EXTVER) tuple
        
        :param data: data and header should both be set and are used to 
            construct a new HDU which is then added to the AstroData instance.
        :type data: numarray.numaraycore.NumArray

        :param header: data and header should both be set and are used to 
            construct a new HDU which is then added to the AstroData instance.
        :type header: pyfits.Header
        
        :param phu: primary header unit  
        :type phu: pyfits.core.PrimaryHDU, pyfits.core.Header 
        
        :param ai: auto-increment appends to match existing extname - extver 
            convention.
        :type ai: boolean
        Sets up a call to insert with the replace flag set, but also handles 
        direct replacement of header, data or phu
        """
        hdul = self.gethdul()
        if type(index) == tuple:
            index = self.get_int_ext(index)
        if phu is None:
            if data is None and header is not None:
                hdul[index+1].header = header
            elif data is not None and header is None:
                hdul[index+1].data = data
            else:
                #insert replace algorithm here
                old_extname = hdul[index].header['EXTNAME']
                old_extver = hdul[index].header['EXTVER']
                hdul.__delitem__(index)
                if old_extname != extname:
                    hdul.insert(index,pyfits.ImageHDU(data=data, \
                        header=header))
                    hdul[index].header['EXTVER'] = last_extver
                else:
                    hdul.insert(index,pyfits.ImageHDU(data=data, \
                        header=header))
                    hdul[index].header['EXTVER'] = old_extver
                
        else:
            hdul.__delitem__(0)
            hdul.insert(0, phu)

  
    def verify_header(self, extname=None, extver=None, header=None):
        """
        :param extname: extension name (ex, 'SCI', 'VAR', 'DQ')
        :type extname: string
        
        :param extver: extension version
        :type extname: integer
        
        :param header: a valid pyfits.Header object
        :type header: pyfits.core.Header
        
        This is a helper function for insert, append and replace that compares
        the extname argument with the extname in the header. If the key does
        not exist it adds it, if its different, it changes it to match the 
        argument
        :returns header: a validated pyfits.Header object
        :rtype: pyfits.core.Header
        """
        if header is None:
            ihdu = pyfits.ImageHDU()
            header = ihdu.header
            if extname is None:
                raise Errors.AstroDataError("cannot resolve extname")
            else: 
                header.update("EXTNAME", extname, "Added by AstroData")
            if extver is None:
                header.update("EXTVER", 1, "Added by AstroData")
            else:
                header.update("EXTVER", extver, "Added by AstroData")
        else:
            if extver and header.has_key("EXTVER"):
                if extver != header["EXTVER"]:
                    header.update("EXTVER", extver, "Added by AstroData")
            if extname and header.has_key("EXTNAME"):
                if extver != header["EXTNAME"]:
                    header.update("EXTNAME", extname, "Added by AstroData")
        return header
        
    def write(self, filename=None, clobber=False, rename=None):
        """
        :param fname: file name to write to, optional if instance already has
                      name, which might not be the case for new AstroData
                      instances created in memory.
        :type fname: string
        :param clobber: This flag drives if AstroData will overwrite an existing
                    file.
        :type clobber: bool
        :param rename: This flag allows you to write the AstroData instance to
            a new filename, but leave the "current" name in tact.
        :type rename: bool

        The write function acts similarly to the pyfits HDUList.writeto(..)
        function if a filename is given, or like pyfits.HDUList.update(..) if 
        no name is given, using whatever the current name is set to. When a name
        is given, this becomes the new on-disk name of the AstroData object and
        will be used on subsequent calls to  write for which a filename is not
        provided. If the clobber flag is False (the default) then write(..)
        throws an exception if the file already exists.

        """
        
        if (self.mode == "readonly" and not clobber):
            if rename == True  or rename == None:
                if filename != None or filename != self.filename:
                    msg =  "Cannot use AstroData.write(..) on this instance,"
                    msg += "file opened in readonly mode, either open for "
                    msg += "update/writing or rename the file."
                    raise Errors.AstroDataReadonlyError(msg)
            else:
                if filename == None or filename == self.filename:
                    msg = "Attemt to write out readonly AstroData instance."
                    raise Errors.AstroDataError(msg)
        if rename == None:
            if filename == None:
                rename = False
            else:
                rename = True
        fname = filename
        hdul = self.gethdul()
        if fname == None:
            if rename == True:
                mes = ("Option rename=True but filename is None")
                raise Errors.AstroDataError(mes)
            fname = self.filename
        else:
            if rename == True:
                self.filename = fname
        # by here fname is either the name passed in, or if None,
        #    it is self.filename
        if (fname == None):
            # @@FUTURE:
            # perhaps create tempfile name and use it?
            raise Errors.AstroDataError("fname is None")
        if os.path.exists(fname):
            if clobber:
                os.remove(fname)
            else:
                raise Errors.OutputExists(fname)
        hdul.writeto(fname)
    
    def get_hdulist(self):
        """
        This function retrieves the HDUList. NOTE: The HDUList should also be
        "released" by calling L{release_hdulist}, as access is reference
        counted. This function is also aliased to L{gethdul(..)<gethdul>}.
        
        :return: The AstroData's HDUList as returned by pyfits.open()
        :rtype: pyfits.HDUList
        """
        self.hdurefcount = self.hdurefcount + 1
        return self.hdulist
    gethdul = get_hdulist # function alias
    
    def release_hdulist(self):
        """
        This function will release a reference to the HDUList... don't call 
        unless you have called L{get_hdulist} at some prior point. 
        (Note, release_hdulist is aliased to L{relhdul(..)<relhdul>})
        """
        self.hdurefcount = self.hdurefcount - 1
        return
    relhdul = release_hdulist # function alias
            
    def get_classification_library(self):
        """
        This function will return a handle to the ClassificationLibrary.  
        NOTE: the ClassificationLibrary is a singleton, this call will either
        return the currently extant instance or, if not extant,
        will create the classification library (using the default context).
        
        :return: A reference to the system classification library
        :rtype: L{ClassificationLibrary}
        """
        if (self.classification_library == None):
            try:
                self.classification_library = ClassificationLibrary()
            except CLAlreadyExists, s:
                self.classification_library = s.clInstance
        return self.classification_library
    
    def prune_typelist(self, typelist):
        cl = self.get_classification_library()
        retary = typelist;
        pary = []
        for typ in retary:
            notSuper = True
            for supertype in retary:
                sto = cl.get_type_obj(supertype)
                if sto.is_subtype_of(typ):
                    notSuper = False
            if notSuper:
                pary.append(typ)
        return pary
        
    def refresh_types(self):
        self.types = None
        self.discover_types()
        
    def get_types(self, prune=False):
        """
        :param prune: flag which controls 'pruning' the returned type list 
            so that only the leaf node type for a given set of related types
            is returned.
        :type prune: bool
        :returns: a list of classification names that apply to this data
        :rtype: list of strings

        The get_types(..) function returns a list of type names, where type 
        names are as always, strings. It is possible to "prune" the list so
        that only leaf nodes are returned, which is useful when features (such
        as descriptors) are set and leaf node settings take priority.  
         
        Note: types are divided into two categories, one intended for types
        which represent processing status (i.e. RAW vs PREPARED), and another
        which contains a more traditional "typology" consisting of a 
        heirarchical tree of datastypes. This latter tree maps roughly to
        instrument-modes, with instrument types branching from the general
        observatory type, (e.g. "GEMINI"). 
        
        To retrieve only status types, use get_status(..); to retreive just
        typological types use get_typology(..).  Note that the system does not
        enforce what checks are actually performed by types in each category,
        that is, one could miscategorize a type when authoring a configuration
        package. Both classifications use the same DataClassification objects
        to classify datasets. It is up to  those implementing the
        type-specific configuration package to ensure types related to status
        appear in the correct part of the configuration space.
        
        Currently the distinction betwen status and typology is not used by the
        system (e.g. in type-specific default recipe assignments) and is
        provided as a service for higher level code, e.g. primitives and
        scripts which make use of the distinction.
        """
        retary = self.discover_types()
        if prune :
            # since there is no particular order to identifying types, 
            # I've deced to do this here rather than try to build the list
            # with this in mind (i.e. passing prune to
            # ClassificationLibrary.discover_types()
            #  basic algo: run through types, if one is a supertype of another, 
            #  remove the supertype
            retary = self.prune_typelist(retary)
        return retary
        
    def discover_types(self, all  = False):
        """
        :param all: a flag which  controls how the classes are returned... if
            True, then the function will return a dictionary of three lists,
            "all", "status", and "typology".  If False, the return value is a
            list which is in fact the "all" list, containing all the status and
            typology related types together.
        :return: a list of DataClassification objects, or a dictionary of lists
            if the C{all} flag is set.
        :rtype: list | dict

        This function provides a list of classifications of both processing
        status and typology which apply to the data encapsulated by this
        instance,  identified by their string names.
        """
        if (self.types == None):
            cl = self.get_classification_library()

            alltypes = cl.discover_types(self, all=True)
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
        
    def get_status(self, prune=False):
        """
        This function returns the set of type names (strings) which apply to
        this dataset and which come from the status section of the AstroData
        Type library. "Status" classifications are those which tend to change
        during the reduction of a dataset based on the amount of processing as
        opposed to instrument-mode classifications, such as GMOS_MOS,  which
        will tend to persist. e.g. RAW vs PREPARED.  Strictly a "status" type 
        is any type defined in or below the status part of the classification
        configuration, i.e. in the Gemini type configuration any type 
        definition files in or below the 
        "astrodata_Gemini/ADCONFIG/classification/status" directory.

        :returns: a list of string classification names
        :rtype: list of strings
        """
        retary = self.discover_status()
        if prune:
            retary = self.prune_typelist(retary)
        return retary
    
    def discover_status(self):
        """
        This function returns the set of processing types applicable to 
        this dataset.
        :returns: a list of classification name strings
        :rtype: list of strings
        """
        if (self.typesStatus == None):
            cl = self.get_classification_library()
            self.typesStatus = cl.discover_status(self)
        return self.typesStatus

    def get_typology(self):
        """
        This function returns the set of type names (strings) which apply to
        this dataset and which come from the typology section of the AstroData
        Type library. "Typology" classifications are those which tend to remain
        with the data in spite of reduction status, e.g. those related to the
        instrument-mode of the dataset or of the datasets used to produce
        it. Strictly these consist of any type defined in or below
        the correct configuration directory, i.e. in Gemini's configuration,
        "astrodata_Gemini/ADCONFIG/classification/types"  directory.
        
        :returns: a list of classification name strings
        :rtype: list of strings"""
        
        retary = self.discover_typology()
        if prune:
            retary = self.prune_typelist(retary)
        return retary

    def discover_typology(self):
        """
        This function returns a list of classification names
        for typology related classifications, as apply to this
        dataset.
        :return: DataClassification objects in a list
        :rtype: list
        """
        if (self.typesTypology == None):
            cl = self.get_classification_library()
            self.typesTypology = cl.discover_typology(self)
        return self.typesTypology
        
    def is_type(self, *typenames):
        """
        :param typename: specifies the type name to check.
        :type typename: string
        :returns: True if the given type applies to this dataset,
            False otherwise
        :rtype: Bool

        This function checks the AstroData object to see if it is the
        given type(s) and returns True if so.
        
        :note: "AstroData.check_type(..)" is an alias for 
            "AstroData.is_type(..)".
        
        """
        if (self.types == None):
            cl = self.get_classification_library()
            self.types = cl.discover_types(self)
            typestrs = self.get_types()
        for typen in typenames:
            if typen in self.types:
                pass
            else:
                return False
        return True
    check_type = is_type

    def re_phukeys(self, rekey):
        """
        :param rekey: A regular expression
        :type rekey: string
        :returns: a list of keys from the PHU that matched C{rekey}
        :rtype: list
        
        The re_phukeys(..) function returns all keys in this dataset's PHU 
        which match the given  regular expression.
        """
        phuh = self.hdulist[0].header
        retset = re_header_keys(rekey, phuh)
        return retset
            
    # PHU manipulations
    def phu_get_key_value(self, key):
        """
        :param key: name of header value to retrieve
        :type key: string
        :rtype: string
        :returns: the key's value as string or None if not present.

        The phu_get_key_value(..) function returns the value associated with the
        given key within the primary header unit
        of the dataset. The value is returned as a string (storage format)
        and must be converted as necessary by the caller.
        
        """
        try:
            hdus = self.get_hdulist()
            retval = hdus[0].header[key]
            if isinstance(retval, pyfits.core.Undefined):
                raise Errors.UndefinedKeyError()
            if retval == "" or retval == " ":
                raise Errors.EmptyKeyError()
            self.relhdul()
            return retval
        except:
            setattr(self, "exception_info", sys.exc_info()[1])
            return None
    phuValue = phu_get_key_value
    phuHeader = phuValue
    
    def phu_set_key_value(self, key, value, comment = None):
        """
        :param key: name of PHU header value to set
        :type key: string
        :param value: value to apply to PHU header
        :type value: string (or can be converted to string)
        :param comment: value to be put in the comment part of the header key
        :type comment: string
        
        The phu_set_key_value(..) function is used to set the value  (and
        optionally the comment) associated with a given key in the primary
        header unit of the dataset. The value argument will be converted to
        string, so it must have a string operator member function or be passed
        in as string. 
        """
        hdus = self.hdulist
        hdus[0].header.update(key, value, comment)
        return
        
    def get_phu(self):
        return self.hdulist[0]
    
    def set_phu(self, phu):
        self.hdulist[0] = phu
        return
    phu = property(get_phu, set_phu)

    def translate_int_ext(self, integer):
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
        return integer + 1
# for the life of me I can't remember why I'm using self.extensions...
# @@TODO: remove self.extensions completely?  might be useful to know
# the hdu's extension in the original file... ?
#        if (self.extensions == None):
#            return integer+1
#        else:
#            print "AD874:", repr(self.extensions)
#            return self.extensions[integer]
    
    def get_key_value(self, key):
        """
        :param key: name of header value to set
        :type key: string
        :returns: the specified value
        :rtype: string

        The get_key_value(..) function is used to get the value associated
        with a given key in the data-header unit of a single-HDU
        AstroData instance (such as returned by iteration). The value argument
        will be converted to string, so it must have a string operator member
        function or be passed in as string. 
        
        :note: 
        
            Single extension AstroData objects are those with only a single
            header-data unit besides the PHU.  They may exist if a single
            extension file is loaded, but in general are produced by indexing or
            iteration instructions, i.e.:
        
                sead = ad[("SCI",1)]
            
                for sead in ad["SCI"]:
                    ...
                
            The variable "sead" above is ensured to hold a single extension
            AstroData object, and can be used more convieniently.
            
                        
        """
        if len(self.hdulist) == 2:
            return self.ext_get_key_value(0,key)
        else:
            mes = "getHeaderValue must be called on single extension instance"
            raise Errors.AstroDataError(mes)
    getHeaderValue = get_key_value

    def set_key_value(self, key, value, comment=None):
        """
        :param key: name of data header value to set
        :type key: string
        :param value: value to apply to header
        :type value: string (or can be converted to string)
        :param comment: value to be put in the comment part of the header key
        :type comment: string
        
        The set_key_value(..) function is used to set the value (and optionally
        the comment) associated
        with a given key in the data-header of a single-HDU AstroData instance.
                
        :note: 
        
            Single extension AstroData objects are those with only a single
            header-data unit besides the PHU.  They may exist if a single
            extension file is loaded, but in general are produced by indexing or
            iteration instructions, i.e.:
        
                sead = ad[("SCI",1)]
            
                for sead in ad["SCI"]:
                    ...
                
            The variable "sead" above is ensured to hold a single extension
            AstroData object, and can be used more convieniently.
            
        """
        if len(self.hdulist) == 2:
            self.ext_set_key_value(0, key, value, comment)
        else:
            mes = "set_key_value must be called on single extension instance"
            raise Errors.AstroDataError(mes)
           
    def ext_get_key_value(self, extension, key):
        """
        :param extension: identifies which extension, either an integer index 
            or (EXTNAME, EXTVER) tuple
        :type extension: int or (EXTNAME, EXTVER) tuple
        :param key: name of header entry to retrieve
        :type key: string
        :rtype: string
        :returns: the value associated with the key, or None if not present

        This function returns the value from the given extension's
        header, with "0" being the first data extension.  To get
        values from the PHU use phu_get_key_value(..).
        """
        
        if type(extension) == int:
            extension = self.translate_int_ext(extension)
        #make sure extension is in the extensions list
        #@@TODO: remove these self.extensions lists
        
        # if (self.extensions != None) and (not extension in self.extensions):
        #    return None
        hdul = self.gethdul()
        try:
            exthd = hdul[extension]
        except KeyError:
            mes = "No such extension: %s" % str(extension)
            raise Errors.AstroDataError(mes)
        try:
            retval = exthd.header[key]
            if isinstance(retval, pyfits.core.Undefined):
                raise Errors.UndefinedKeyError()
            if retval == "" or retval == " ":
                raise Errors.EmptyKeyError()
            return retval
        except:
            setattr(self, "exception_info", sys.exc_info()[1])
            return None
    
    def ext_set_key_value(self, extension, key, value, comment=None):
        """
        :param extension: identifies which extension, either an integer index 
                          or (EXTNAME, EXTVER) tuple
        :type extension: int or (EXTNAME, EXTVER) tuple
        :param key: name of PHU header value to set
        :type key: string
        :param value: value to apply to PHU header
        :type value: string (or can be converted to string)
        :param comment: value to be put in the comment part of the header key
        :type comment: string

        The ext_set_key_value(..) function is used to set the value (and optionally
        the comment) associated with a given key in the header unit of the given
        extension within the dataset. This function sets the value in the
        given extension's header, with "0" being the first data extension.  To
        set values in the PHU use phusetKeyValue(..).
        """
        origextension = extension
        if type(extension) == int:
            # this translates ints from our 0-relative base of AstroData to the 
            #  1-relative base of the hdulist, but leaves tuple extensions
            #  as is.
            #print "AD892: pre-ext", extension
            extension = self.translate_int_ext(extension)
            #print "AD892: ext", extension
            
        #make sure extension is in the extensions list if present
        #if (self.extensions != None) and (not extension in self.extensions):
        #    print "AD1538:", self.extensions
        try:
            tx = self.hdulist[extension]
        except:
            mes = "Extension %s not present in AstroData instance" % \
                str(origextension)
            raise Errors.AstroDataError(mes)
        hdul = self.gethdul()
        hdul[extension].header.update(key, value, comment)
        self.relhdul()
        return 
   
    def info(self, verbose=False, table=False):
        """The info(..) function prints self.infostr() and 
        is maintained for convienience and low level debugging.
        """
        print self.infostr(verbose=verbose, table=table)       

    def display_id(self):
        import IDFactory
        return IDFactory.generate_stackable_id(self)
    
    # MID LEVEL MEF INFORMATION
    def count_exts(self, extname):
        """
        :param extname: the name of the extension, equivalent to the
                       value associated with the "EXTNAME" key in the extension 
                       header.
        :type extname: string
        :returns: number of extensions of that name
        :rtype: int
        
        The count_exts(..) function counts the extensions of a given name
        (as stored in the HDUs "EXTVER" header). 
        """
        hdul = self.gethdul()
        maxl = len(hdul)
        count = 0
        for i in range(1,maxl):
            try:
                # note, only count extension in our subdata extension list
                if (self.extensions == None) or \
                    ((extname, i) in self.extensions):
                    if (hdul[i].header["EXTNAME"] == extname):
                        count += 1
            except KeyError:
                #no biggie if some extention has no EXTNAME
                if extname == None:
                    count += 1  # in this case we are counting when there is no
                                # EXTNAME in the header
        self.relhdul()
        return count
        
    def get_hdu(self, extid):
        """
        :param extid: specifies the extention (pyfits.HDU) to return.
        :type extid: int | tuple
        :returns:the extension specified
        :rtype:pyfits.HDU
        
        This function returns the HDU identified by the C{extid} argument. This
        argument can be an integer or (EXTNAME, EXTVER) tuple.
        """
        return self.hdulist[extid]
        
    def get_phuheader(self):
        return self.get_hdu(0).header
            
    def history_mark(self, key=None, comment=None, stomp=True):
        """
        This function will add the timestamp type keys to the astrodata 
        instance's PHU.  The default will be to update the GEM-TLM key by just
        calling ad.history_mark() without any input vals. Value stored is the
        UT time in the same format as the CL scripts.  The GEM-TLM key will be
        updated along with the specified key automatically.
        
        param key: header keyword to be changed/added
        type key: string
        param comment: comment for the keyword in the PHU, keep it short
                    default if key is provided is 'UT Time stamp for '+key 
        type comment: string
        param stomp: if True, use the current time; if False, use the latest 
                    saved time
        type stomp: boolean (True/False)
        """
        if stomp:
            self.tlm = datetime.now().isoformat()[0:-7]
        elif (stomp == False) and (self.tlm == None):
            self.tlm = datetime.now().isoformat()[0:-7]
        if comment == None and key != None:
            comment = "UT Time stamp for " + key
        
        # Updating PHU with specified key and GEM-TLM    
        if key !=None:
            self.phu_set_key_value(key,self.tlm,comment)
            self.phu_set_key_value("GEM-TLM", self.tlm, 
                "UT Last modification with GEMINI")
        # Only updating the GEM-TLM PHU key
        else:
             self.phu_set_key_value("GEM-TLM", self.tlm,
                "UT Last modification with GEMINI")     
        # Returning the current time for logging if desired
        return self.tlm        
    
    def store_original_name(self):
        """
        This function will add the key 'ORIGNAME' to PHU of an astrodata object 
        containing the filename when object was instantiated (without any 
        directory info, ie. the basename).
        
        If key has all ready been added (ie. has undergone processing where
        store_original_name was performed before), then the value original 
        filename is just returned.  If the key is there, but does not match
        the original filename of the object, then the original name is 
        returned, NOT the value in the PHU. The value in the PHU can always be
        found using ad.phu_get_key_value('ORIGNAME').
        """
        phuOrigFilename = self.phu_get_key_value("ORIGNAME")
        origFilename = self.__origFilename
        
        if origFilename != None:
            origFilename = os.path.basename(self.__origFilename)
        
        if origFilename == phuOrigFilename == None:
            # No private member value was found so throw an exception
            mes = "failed to have its original filename stored when astrodata"
            mes += "instantiated it"
            raise Errors.AstroDataError(self.filename + mes)
        elif (phuOrigFilename is None) and (origFilename is not None):
            # phu key doesn't exist yet, so add it
            mes = "Original name of file prior to processing"
            self.phu_set_key_value("ORIGNME", origFilename, mes)
        # The check could be useful in the future
        #elif (phuOrigFilename is not None) and (origFilename is not None):
            # phu key exists, so check if it matches private members value
            #if phuOrigFilename != origFilename:
                #$$ They don't match, do something?
        else:
            # last choice, private one is None, but phu one isn't
            origFilename = phuOrigFilename
        return origFilename

    def div(self, denominator):
        return arith.div(self,denominator)

    def mult(self, input_b):
        return arith.mult(self,input_b)
    
    def add(self, input_b):
        return arith.add(self, input_b)
    
    def sub(self, input_b):
        return arith.sub(self, input_b)
    
# SERVICE FUNCTIONS and FACTORIES
def correlate(*iary):
    """
    :param iary: A list of AstroData instances for which a correlation dictionary
        will be constructed to produce a correlation dict.
    :type iary: list of AstroData instance
    :returns: a list of tuples containing correlated extensions from the arguments. 
    :rtype: list of tuples

    The *correlate(..)* function is a module-level helper function which returns
    a list of tuples of Single Extension AstroData instances which associate
    extensions from each listed AstroData object, to identically named
    extensions among the rest of the input array. The *correlate(..)* function
    accepts a variable number of arguments, all of which should be AstroData
    instances.
    
    The function returns a structured dictionary of dictories of lists of
    AstroData objects. For example, given three inputs, *ad*, *bd* and *cd*, all
    with three "SCI", "VAR" and "DQ" extensions. Given *adlist = [ad, bd,
    cd]*, then *corstruct = correlate(adlist)* will return to *corstruct* a
    dictionary first keyed by the EXTNAME, then keyed by tuple. The contents
    (e.g. of *corstruct["SCI"][1]*) are just a list of AstroData instances each
    containing a header-data unit from *ad*, *bd*, and *cd* respectively.
        
    :info: to appear in the list, all the given arguments must have an extension
        with the given (EXTNAME,EXTVER) for that tuple.
    """
    numinputs = len(iary)
    if numinputs < 1:
        raise Errors.AstroDataError("Inputs for correlate method < 1")
    outlist = []
    outrow = []
    baseGD = iary[0]
    for extinbase in baseGD:
        try:
            extname = extinbase.header["EXTNAME"]
        except:
            extname = "NONE"
        try:
            extver  = extinbase.header["EXTVER"]
        except:
            extver  = 0
        outrow = [extinbase]
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

def prep_output(input_ary=None, name=None, clobber=False):
    """
    :param input_ary: The input array from which propagated content (such as
        the  source PHU) will be taken. Note: the zero-th element in the list
        is  used as the reference dataset, for PHU or other items which require
        a particular reference.
    :type input_ary: list of AstroData Instances
    
    :param name: File name to use for returned AstroData, optional.
    
    :param clobber: By default prep_output(..) checks to see if a file of the
        given name already exists, and will raise an exception if found.
        Set *clobber* to *True* to override this behavior and potentially
        overwrite the extant file.  The datset on disk will not be overwritten
        as a direct result of prep_output, which only prepares the object
        in memory, but will occur when the AstroData object returned is 
        written (i.e. *ad.write()*)). 
    :type clobber: bool
        
    :returns: an AstroData instance initialized with appropriate
        header-data units such as the PHU, Standard Gemini headers
        and with type-specific associated  data-header units such as
        binary table Mask Definition tables (aka MDF).
        
        ..info: File will not have been written to disk by prep_output(..).
    :rtype: AstroData

    The prep_output(..) function creates a new AstroData object ready for
    appending output information (i.e. *ad.append(..)*).  While you can also
    create an empty AstroData object by giving no arguments to the AstroData
    constructor  (i.e. *ad = AstroData()*), *prep_output(..)* exists for the
    common case where a new dataset object is intended as the output of
    come combinatorial process on a list of source dataset, and some information
    from the source inputs must be propagated. 
    
    The *prep_output(..)* function makes use of this knowledge to ensure the
    file meets standards in what is considered a complete output file given
    such a combination.  In the future this function can make use of dataset
    history and structure definitions in the ADCONFIG configuration space. As
    *AstroData.prepOutpu(..)* it improves, scripts and primtiives that use it
    will benefit in a forward compatible way, in that their output datasets will
    benefit from more automatic propagation, validations, and data flow control,
    such as the emergence of history database propagation.
    
    Presently, it already provides the following:
    
    + Ensures that all standard headers  are in place in the new file, using the
      configuration .
    + Copy the PHU of the reference image (input_ary[0]). 
    + Propagate associated information such as the MDF in the case of a MOS 
        observation, configurable by the Astrodata Structures system. 
    """ 
    if input_ary == None: 
        raise Errors.AstroDataError("prep_output input is None") 
        return None
    if type(input_ary) != list:
        iary = [input_ary]
    else:
        iary = input_ary
    
    #get PHU from input_ary[0].hdulist
    hdl = iary[0].gethdul()
    outphu = copy(hdl[0])
    outphu.header = outphu.header.copy()
        
    # make outlist the complete hdulist
    outlist = [outphu]

    #perform extension propagation
    newhdulist = pyfits.HDUList(outlist)
    retgd = AstroData(newhdulist, mode = "update")
    
    # Ensuring the prepared output has the __origFilename private variable
    retgd._AstroData__origFilename = input_ary._AstroData__origFilename
    if name != None:
        if os.path.exists(name):
            if clobber == False:
                raise Errors.OutputExists(name)
            else:
                os.remove(name)
        retgd.filename = name
    return retgd
