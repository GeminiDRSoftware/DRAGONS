#
#                                                                      astrodata
#                                                                   AstroData.py
# ------------------------------------------------------------------------------
# $Id$
# ------------------------------------------------------------------------------
__version__      = '$Revision$'[11:-2]
__version_date__ = '$Date$'[7:-2]
# ------------------------------------------------------------------------------
__docform__ = "restructuredtext" #for epydoc
"""
The AstroData class abstracts datasets stored in MEF files
and provides uniform interfaces for working on datasets from different
instruments and modes.  Configuration packages are used to describe
the specific data characteristics, layout, and to store type-specific
implementations.

MEFs can be generalized as lists of header-data units (HDU), with key-value
pairs populating headers, and pixel values populating the data array.
AstroData interprets a MEF as a single complex entity.  The individual
"extensions" within the MEF are available using Python list ("[]") syntax;
they are wrapped in AstroData objects. (See __getitem__()).

AstroData uses ``pyfits`` for MEF I/O and ``numpy`` for pixel manipulations.

While the ``pyfits`` and ``numpy`` objects are available to the programmer, 
``AstroData`` provides analogous methods for most ``pyfits`` functionalities
which allows it to maintain the dataset  as a cohesive whole. The programmer
does however use the ``numpy.ndarrays`` directly for pixel manipulation.

In order to identify types of dataset and provide type-specific behavior,
``AstroData`` relies on configuration packages either in the ``PYTHONPATH``
environment variable or the ``Astrodata`` package environment variables,
``ADCONFIGPATH`` and ``RECIPEPATH``. A configuration package 
(eg. ``astrodata_Gemini``) contains definitions for all instruments and
modes. A configuration package contains type definitions, meta-data 
functions, information lookup tables, and any other code
or information needed to handle specific types of dataset.

This allows ``AstroData`` to manage access to the dataset for convenience
and consistency. For example, ``AstroData`` is able:

  - to allow reduction scripts to have easy access to dataset classification
    information in a consistent way across all instruments and modes;
  - to provide consistent interfaces for obtaining common meta-data across all
    instruments and modes;
  - to relate internal extensions, e.g. discriminate between science and 
    variance arrays and associate them properly;
  - to help propagate header-data units important to the given instrument mode,
    but unknown to general purpose transformations.

In general, the purpose of ``AstroData`` is to provide smart dataset-oriented
interfaces that adapt to dataset type. The primary interfaces are for file
handling, dataset-type checking, and managing meta-data, but ``AstroData`` 
also integrates other functionalities.

"""
import os
import re
import sys
import numpy
import pyfits

from copy import copy, deepcopy
from urllib2 import HTTPError

from astrodata import Errors
from astrodata import new_pyfits_version
from astrodata.IDFactory import generate_stackable_id
from astrodata.mkcalciface import get_calculator_interface

from astrodata.adutils import arith
from astrodata.adutils.netutil import urlfetch
from astrodata.adutils.gemutil import rename_hdu
from astrodata.adutils.docstrings import SUBDATA_INFO_STRING

from AstroDataType import *  # Not recommended -> ClassificationLibrary

# ------------------------------------------------------------------------------
verbt = False
verbose = False
verboseLoadTypes = True
CalculatorInterface = get_calculator_interface()

# ------------------------------------------------------------------------------
class AstroData(CalculatorInterface):
    """
    The AstroData constructor constructs an in-memory representation of a
    dataset. If given a filename it uses pyfits to open the dataset, reads
    the header and detects applicable types. Binary data, such as pixel
    data, is left on disk until referenced.

    :param dataset: the dataset to load, either a filename (string) path
                    or URL, an 'AstroData' instance, or a 'pyfits.HDUList'
    :type dataset:  string, AstroData, HDUList
        
    :param phu: Primary Header Unit. This object is propagated to all 
                astrodata sub-data ImageHDUs. Special handling is made 
                for header instances that are passed in as this arg., 
                where a phu will be created and the '.header' will be 
                assigned (ex. hdulist[0], ad.phu, ad[0].hdulist[0], 
                ad['SCI',1].hdulist[0], ad[0].phu, ad['SCI',1].phu, 
                and all the previous with .header appended) 
    :type phu: pyfits.core.PrimaryHDU, pyfits.core.Header 

    :param header: extension header for images (eg. 'hdulist[1].header',
                   'ad[0].hdulist[1].header', 'ad['SCI',1].hdulist[1].header')
    :type phu: pyfits.core.Header
    
    :param data: the image pixel array (eg. 'hdulist[1].data',
                 'ad[0].hdulist[1].data', 'ad['SCI',1].hdulist[1].data')
    :type data: numpy.ndarray

    :param exts: (advanced) a list of extension indexes in the parent
                 'HDUList' that this instance should refer to, given  integer or 
                 (EXTNAME, EXTVER) tuples specifying each extension in the pyfits
                 index space where the PHU is at index 0, the first data extension
                 is at index 1, and so on. I.e. This is primarily intended for 
                 internal use when creating "sub-data", which are AstroData instances
                 that represent a slice, or subset, of some other AstroData instance.
            
                 NOTE: if present, this option will override and obscure the 
                 <extInsts> argument, in other word <extInsts> will be ignored.
    
                 Example of sub-data:

                    sci_subdata = ad["SCI"]

                 The sub-data is created by passing "SCI" as an argument to the
                 constructor. The 'sci_subdata' object would consist of its own 
                 'AstroData' instance referring to it's own ``HDUList``, but the 
                 HDUs in this list would still be shared (in memory) with the 'ad'
                 object, and appear in its ``HDUList`` as well.
    :type exts: list
        
    :param extInsts: (advanced) A list of extensions this instance should 
                     contain, specified as actual pyfits.HDU instances. NOTE: if the 
                     'exts' argument is also set, ``extInsts`` is ignored.
    :type extInsts: list of pyfits.HDU objects

    :param store: directory where a copy of the original file will be 
                  stored.  This is used in the special case where the
                  filename is an URL to a remote fits file.  Otherwise it has
                  no effect.
    :type store: string

    :param mode: IO access mode, same as pyfits mode ("readonly", "update",
                  or "append") with one additional AstroData-specific mode, "new".
                  If the mode is "new", and a filename is provided, the constructor
                  checks that the named file does not exist on disk,
                  and if it does not it creates an empty ``AstroData`` of that name 
                  but does not write it to disk. Such an ``AstroData`` 
                  instance is ready to have HDUs appended, and to be written to disk
                  at the user's command with ``ad.write()``.
    :type mode: string

    """
    def __init__(self, dataset=None, phu=None, header=None, data=None, 
                 exts=None, extInsts=None, store=None, mode="readonly"):

        self._filename = None
        self._hdulist  = None
        self._types    = None
        self._typology = None
        self._status   = None
        self.index     = 0           # index iterator
        self.mode      = "readonly"
        self.__origFilename = None
        self.descriptor_calculator = None

        # subdata
        self.borrowed_hdulist = False  # hdul from another AD instance
        self.container        = None   # AstroData instance of hdul
        
        # None means "all", else an array of extensions
        self.extensions = None

        # CODE FOLLOWING THIS COMMENT IS REQUIRED BY DESIGN
        # "extensions" first so other initialization code knows this is 
        # subdata.
        self.extInsts   = extInsts
        self.extensions = exts

        # alias ClassificationLibrary class method get_classification_library
        #
        # **ADX1 former: 
        self.get_classification_library = \
            ClassificationLibrary.get_classification_library
        # **ADX1 former: 
        self.classification_library = self.get_classification_library()
        # ClassificationLibrary Singleton, must be retrieved through
        #   get_classification_library()
        #self.classification_library = None

        fname   = None
        headers = None
        if type(dataset) is str:
            parts = dataset.split(":")
            if len(parts) > 1:
                if parts[0] == "file":
                    remoteFile = False
                    dataset = parts[1][2:]
                else:
                    remoteFile = True
            else:
                remoteFile = False
            if remoteFile:
                # string is a URL, retrieve it
                savename = os.path.basename(dataset)
                try:
                    if store:
                        fname = urlfetch(dataset, store=store, clobber= True)
                    else:
                        fname = urlfetch(dataset, clobber=True)
                    dataset = savename
                except HTTPError, error:
                    admsg = "AstroData could not load via http: %s"
                    raise Errors.AstroDataError(admsg % dataset)
            elif store:
                from shutil import copy as sh_copy
                if (os.path.abspath(os.path.dirname(dataset)) != 
                    os.path.abspath(store)):
                    sh_copy(dataset, store)
                    dataset = os.path.join(store,dataset)
            
        if dataset is None:
            if (type(data) is list):
                raise TypeError("cannot accept data as a list")

            if phu is None:
                hdu = pyfits.PrimaryHDU()
                dataset = pyfits.HDUList(hdu)
                if data is not None:
                    dataset.append(pyfits.ImageHDU(data=data, header=header))
            else: 
                hdu = pyfits.PrimaryHDU()
                dataset = pyfits.HDUList(hdu)
                if type(phu) is pyfits.core.PrimaryHDU:
                    dataset[0] = phu

                # if phu is a header, then assigned to a new phu,
                # PHU is the 0'th in the array, which is ad.phu
                elif phu.__class__ == pyfits.core.Header:
                    dataset[0].header = phu
                else:
                    raise TypeError("phu is of an unsupported type")

                if data is not None:
                    dataset.append(pyfits.ImageHDU(data=data, header=header))

        if fname is None:
            self.open(dataset, mode)
        else:
            # fname is set when retrieving a url
            self.open(fname, mode)
            if store is None:
                os.remove(fname)
    
    def __del__(self):
        """ 
        This is the destructor for AstroData. It performs reference 
        counting and behaves differently when this instance is subdata, since
        in that case some other instance "owns" the pyfits HDUs instance.
        """
        if (self.hdulist != None):
            self.hdulist = None
        return

    def __contains__(self, ext):
        try:
            val = self[ext]
            if val is None:
                return False
        except:
            return False
        return True
                    
    def __getitem__(self, ext):
        """
        AstroData instances behave as list-like objects and therefore pythonic
        slicing operations may be performed on instances of this class.
        This method provides support for list slicing with the "[]" syntax.
        Slicing is used to create AstroData objects associated with "subdata" 
        of the parent AstroData object, that is, consisting of
        an HDUList made up of some subset of the parent MEF.

            E.g.,

            *datasetA = AstroData(dataset="datasetMEF.fits")*

            *datasetB = datasetA['SCI']*

            *datasetC = datasetA[2]*

            *datasetD = datasetA[("SCI",1)]*

            etc.
            
        In this case, after the operations, datasetB is an ``AstroData`` object
        associated with the same MEF, sharing some of the the same actual HDUs
        in memory as ``datasetA``. The object in ``datasetB`` will behave as if
        the SCI extensions are its only members, and it does in fact have its 
        own pyfits.HDUList. Note that 'datasetA' and 'datasetB' share the 
        PHU and also the data structures of the HDUs they have in common, so 
        that a change to 'datasetA[('SCI',1)].data' will change the 
        'datasetB[('SCI',1)].data' member and vice versa. They are in fact both
        references to the same numpy array in memory. The 'HDUList' is a 
        different list, however, that references common HDUs. If a subdata 
        related 'AstroData' object is written to disk, the resulting MEF will
        contain only the extensions in the subdata's 'HDUList'.

        Note: Integer extensions start at 0 for the data-containing 
        extensions, not at the PHU as with pyfits.  This is important:
        'ad[0]' is the first content extension, in a traditional MEF 
        perspective, the extension AFTER the PHU; it is not the PHU!  In
        ``AstroData`` instances, the PHU is purely a header, and not counted
        as an extension in the way that headers generally are not counted
        as their own elements in the array they contain meta-data for.
        The PHU can be accessed via the 'phu' member.

        :param ext: Integer index, an index tuple (EXTNAME, EXTVER),
                    or EXTNAME name. If an int or tuple, the single
                    extension identified is wrapped with an AstroData instance,
                    and single-extension members of the AstroData object can 
                    be used. A string 'EXTNAME' results in all extensions with 
                    the given EXTNAME wrapped by the new instance.
        :type ext: <str>, <int>, or <tuple>

        :returns: AstroData instance associated with the subset of data.
        :rtype: <AstroData>
        :raises: KeyError, IndexError

        """
        from astrodata import Structures
        hdul = self.hdulist
        exs  = []

        # 'ext' can be <tuple>,<int>, or <str> ("EXTNAME")
        if type(ext) is str:
            # iterate on extensions, find correct extname
            count = 0
            extname = ext
            for i in range(0, len(hdul)):
                try:
                    # note, only count extension in our extension
                    if (hdul[i].header["EXTNAME"] == extname):
                        try:
                            extver = int(hdul[i].header["EXTVER"])
                        except KeyError:
                            extver = 1
                        exs.append((extname, extver))
                except KeyError:
                    pass
            
            if len(exs):
                return AstroData(self, exts=exs)
            else:
                return None
            
        elif type(ext) is tuple or type(ext) is int:
            if type(ext) is int:
                if ext >= 0:
                    ext = ext + 1        # 0th content ext, not PHU
                else:
                    try:                 # handle negative slicing
                        assert abs(ext) < len(hdul)
                    except AssertionError:
                        raise IndexError("list index out of range")
            try:
                self.hdulist[ext]
            except KeyError:
                return None

            gdpart = AstroData(self, exts=[ext])
            return gdpart

        elif isinstance(ext, Structures.Structure):
            return Structures.get_structured_slice(self, structure=ext)

        else:
            raise KeyError

        return
            
    def __len__(self):
        """
        Length operator for AstroData.

        :returns: number of extensions minus the PHU
        :rtype:   <int>
        """
        return len(self.hdulist) - 1
    
    # ITERATOR PROTOCOL FUNCTIONS
    def __iter__(self):
        """
        Override nominal iterator for AstroData. Initializes the iteration 
        process, resetting the index of the 'current' extension to the first 
        data extension.

        :returns: self
        :rtype:   AstroData
        """
        self.index = 0
        return self

    def __deepcopy__(self, memo):
        # pyfits throws exception on deepcopy
        lohdus = []
        for hdu in self.hdulist:
            nhdu = copy(hdu)
            nhdu.header = nhdu.header.copy()
            lohdus.append(nhdu)
        hdulist = pyfits.HDUList(lohdus)
        adReturn = AstroData(hdulist)
        adReturn.__origFilename = self.__origFilename
        adReturn._filename = self._filename
        adReturn.mode = self.mode
        return adReturn

    def next(self):
        """
        This function exists so that AstroData can be used as an iterator.
        This function returns the objects "ext" in the following line:
        
            for ext in ad:
        
        If this AstroData instance is associated with a subset of the data in
        the MEF to which it refers, then this iterator goes through that subset
        in order.
        
        :returns: a single extension AstroData instance representing the
                  'current' extension in the AstroData iteration loop.
        :rtype:   AstroData
        :raises: StopIteration
        """
        try:
            if self.extensions is None:
                ext = self.index
            else:
                ext = self.index
        except IndexError:
            raise StopIteration
        self.index += 1
        try:
            retobj = self[ext]
        except IndexError:
            raise StopIteration
        return retobj

    # ---------------------------- Defined properties --------------------------
    @property
    def data(self):
        """
        Property: The data property can only be used for single-HDU AstroData
        instances, such as those returned during iteration. To set the 
        data member, use *ad.data = newdata*, where *newdata* must be a 
        numpy array. To get the data member, use *npdata = ad.data*.

        The "data" member returns appropriate HDU's data member(s) specifically
        for the case in which the AstroData instance has ONE HDU (in 
        addition to the PHU). This allows a single-extension AstroData, 
        such as AstroData generates through iteration, to be used as though 
        it simply is just the one extension. One is dealing with single 
        extension AstroData instances when iterating over the AstroData 
        extensions  and when picking out an extension by integer or tuple 
        indexing. 
        Eg.,

            for ad in dataset[SCI]:
                # ad is a single-HDU index
                  ad.data = newdata

        :returns: data array associated with the single extension
        :rtype:  <ndarray>
        :raises:  Errors.SingleHDUMemberExcept

        """
        self._except_if_single()
        return self.hdulist[1].data

    @data.setter
    def data(self, newdata):
        """
        Sets the data member of a data section of an AstroData object, 
        specifically for the case in which the AstroData instance has
        ONE header-data unit (in addition to PHU).

        :param newdata: new data objects
        :type newdata: numpy.ndarray
        :raises: Errors.SingleHDUMemberExcept

        """
        self._except_if_single()
        self.hdulist[1].data = newdata
        return 

    @property
    def descriptors(self):
        """
        Property: Returns a dictionary of all registered metadata descriptor 
        functions defined on the instance.

        Eg.,

        {descriptor_function_name : descriptor value (dv)}

        :returns: dict of descriptor functions
        :rtype: <dict>

        """
        return self._all_descriptors()

    @property
    def filename(self):
        """ 
        Property: 'filename' is monitored so that the mode can be changed 
        from 'readonly' when 'filename' is changed.

        """
        return self._filename

    @filename.setter
    def filename(self, newfn):
        if self.mode == "readonly":
            self.mode = "update"
        self._filename = newfn
        return

    @property
    def header(self):
        """
        Property: Returns the header member for Single-HDU AstroData instances. 

        The header property can only be used for single-HDU AstroData
        instances, such as those returned during iteration. It is a
        property attribute which uses *get_header(..)* and
        *set_header(..)* to access the header member with the "=" syntax.
        To set the header member, use *ad.header = newheader*, where
        *newheader* must be a pyfits.Header object. To get the header
        member, use *hduheader = ad.header*.

        :return: header
        :rtype:  pyfits.Header
        :raises:  Errors.SingleHDUMemberExcept

        """
        self._except_if_single()
        return self.hdulist[1].header

    @header.setter
    def header(self, header):
        """
        Sets the header member for single extension instance

        :param header: pyfits Header to set for given extension
        :type header: pyfits.Header
        :raises: Errors.SingleHDUMemberExcept

        """
        self._except_if_single()
        self.hdulist[1].header = header
        return

    @property
    def headers(self):
        """
        Property: Returns header member(s) for all extension (except PHU).

        :returns: list of pyfits.Header instances
        :rtype:  <list>

        """
        retary = []
        for hdu in self.hdulist:
            retary.append(hdu.header)
        return retary

    @property
    def hdulist(self):
        """
        Property: Returns a list of header-data units on the instance.
        
        :returns: The AstroData's HDUList as returned by pyfits.open()
        :rtype:  <pyfits.HDUList>

        """
        return self._hdulist

    @hdulist.setter
    def hdulist(self, new_list):
        self._hdulist = new_list

    @hdulist.deleter
    def hdulist(self):
        if self._hdulist is not None:
            del self._hdulist
            self._hdulist = None
        return

    @property
    def phu(self):
        """
        Property: Returns the instance's primary HDU.

        :returns: The instance "phu"
        :rtype: <PrimaryHDU>

        """
        return self._hdulist[0]
    
    @phu.setter
    def phu(self, phu):
        self._hdulist[0] = phu
        return

    # -------------------------------- File Ops --------------------------------
    def append(self, moredata=None, data=None, header=None, extname=None, 
               extver=None, auto_number=False, do_deepcopy=False):
        """
        Appends header-data units (HDUs) to the AstroData instance.

        :param moredata: either an AstroData instance, an HDUList instance, 
                         or an HDU instance to add to this AstroData object.
                         When present, data and header arguments will be 
                         ignored.
        :type  moredata: pyfits.HDU, pyfits.HDUList, or AstroData

        :param data: 'data' and 'header' are used to construct a new HDU which
                     is then added to the ``HDUList`` associated to the 
                     AstroData instance. The 'data' argument should be set to 
                     a valid numpy array. If 'modedata' is not specified, 
                     'data' and 'header' must both be set.
        :type  data: numpy.ndarray
        
        :param header: 'data' and 'header' are used to construct a new 
                       HDU which is then added to the 'HDUList' associated to 
                       AstroData instance. The 'header' argument should be set 
                       to a valid pyfits.Header object.
        :type header: pyfits.Header

        :param auto_number: auto-increment the extension version, 'EXTVER', 
                            to fit file convention
        :type auto_number: <bool>
        
        :param extname: extension name as set in keyword 'EXTNAME' 
                        (eg. 'SCI', 'VAR', 'DQ'). This is used only when 
                        'header' and 'data' are used.
        :type extname: <str>

        :param extver: extension version as set in keyword 'EXTVER'. This is 
                       used only when 'header' and 'data' are used.
        :type extver: <int>

        :param do_deepcopy: deepcopy the input before appending. May be useful
                            when auto_number is True and the input comes from 
                            another AD object.
        :type do_deepcopy: <bool>

        """
        hdulist = None
        if moredata:
            hdulist = self._moredata_check(md=moredata, append=True)
            if not hdulist:
                return
            
            if do_deepcopy:
                # To avoid inarvertedly corrupting the data deepcopy the hdulist
                # to append, then append it.
                ad_from_input = AstroData(hdulist)
                deepcopy_of_input = deepcopy(ad_from_input)
                hdulist_to_append = deepcopy_of_input.hdulist
            else:
                hdulist_to_append = hdulist
            
            self._moredata_work(append=True, autonum=auto_number,
                               md=moredata, hdul=hdulist_to_append)
        else:
            self._onehdu_work(append=True, header=header, data=data,
                             extname=extname, extver=extver, autonum=auto_number)
        return

    def close(self):
        """ Method will close the 'HDUList' on this instance. """
        if self.borrowed_hdulist:
            self.hdulist = None
        else:
            if self.hdulist != None:
                self.hdulist.close()
                self.hdulist = None
        return

    def insert(self, index, moredata=None, data=None, header=None, extname=None,
               extver=None, auto_number=False, do_deepcopy=False):
        """
        Insert a header-data unit (HDUs) into the AstroData instance.

        :param index: the extension index, either an int or (EXTNAME, EXTVER)
                      pair before which the extension is to be inserted.
                      Note: the first data extension is [0]; cannot insert
                      before the PHU. 'index' is the  Astrodata index, where
                      0 is the 1st extension.
        :type index: <int> or <tuple> (EXTNAME,EXTVER)
        
        :param moredata: An AstroData instance, an HDUList instance, or
                         an HDU instance. When present, data and header will be
                         ignored.
        :type moredata: pyfits.HDU, pyfits.HDUList, or AstroData
        
        :param data: 'data' and 'header' are used in conjunction to construct a
                     new HDU which is then added to the HDUList of the AstroData
                     instance. 'data' should be set to a valid numpy array. 
                     If 'modedata' is not specified, 'data' and 'header' both
                     must be set.
        :type data: numpy.ndarray
        
        :param header: 'data' and 'header' are used in conjunction to construct
                       a new HDU which is then added to the HDUList of the
                       instance. The 'header' argument should be set to a valid
                       pyfits.Header object. If 'moredata' is not specified,
                       'data' and 'header' both must be set.
        :type header: pyfits.Header

        :param extname: extension name (eg. 'SCI', 'VAR', 'DQ')
        :type extname: <str>
        
        :param extver: extension version (eg. 1, 2, 3)
        :type extver: <int>

        :param auto_number: auto-increment the extension version, 'EXTVER',
                            to fit file convention. If set to True, this will
                            override the 'extver' and 'extname' arguments
                            settings.
        :type auto_number:  <bool>
        
        :param do_deepcopy: deepcopy the input before appending. May be useful
                            when auto_number is True and the input comes from
                            another AD object.
        :type do_deepcopy: <bool>

        """
        hdulist = None
        hdu_index = None
        if type(index) is tuple:
            hdu_index = self.ext_index(index, hduref=True)
        else:    
            hdu_index = index + 1

        if hdu_index > len(self.hdulist):
            raise Errors.AstroDataError("Index out of range")

        if moredata:
            hdulist = self._moredata_check(md=moredata, insert=True, index=index)
            if not hdulist:
                return
            
            if do_deepcopy:
                ad_from_input = AstroData(hdulist)
                deepcopy_of_input = deepcopy(ad_from_input)
                hdulist_to_insert = deepcopy_of_input.hdulist
            else:
                hdulist_to_insert = hdulist

            self._moredata_work(md=moredata, insert=True, autonum=auto_number,
                               hdul=hdulist_to_insert, hduindx=hdu_index)
        else:
            if header is None or data is None: 
                raise Errors.AstroDataError("both header and data are required")

            if extname is None and header.get("EXTNAME") is None:
                raise KeyError("No EXTNAME keyword. Set EXTNAME in the passed " 
                               "header OR pass the extname= parameter.")

            self._onehdu_work(header=header, data=data, insert=True, 
                              extname=extname, extver=extver, 
                              autonum=auto_number, hduindx=hdu_index)
        return

    def open(self, source, mode="readonly"):
        """
        Method wraps a source dataset, which can be in memory as another
        AstroData or pyfits HDUList, or on disk, given as the string filename.
        
        NOTE: In general, users should not use 'open' directly, but pass
        the filename to the AstroData constructor. The constructor uses
        open(..) however. Users should use the constructor, which may 
        perform extra operations.

        :param source: source contains some reference for the dataset to 
                       be opened and associated with this instance. Generally
                       it would be a filename, but can also be
                       an AstroData instance or a pyfits.HDUList instance.
        :type source: <str> | <AstroData> | <pyfits.HDUList>
        
        :param mode: IO access mode, same as the pyfits open mode, 'readonly,
                     'update', or 'append'.  The mode is passed to pyfits so
                     if it is an illegal mode name, pyfits will be the
                     subsystem reporting the error. 
        :type mode: <str>

        """
        inferRAW = True
        # if AstroData instance, then it has opened or gets to open the data...
        if isinstance(source, AstroData):
            inferRAW = False
            self.filename = source.filename
            self.__origFilename = source.filename
            self.borrowed_hdulist = True
            self.container = source
            
            # @@REVISIT: should this cache copy of types be here?
            # probably not... works now where type is PHU dependent, but
            # this may not remain the case... left for now.
            if source.types:
                self._types = source.types
                self._status = source._status
                self._typology = source._typology

            #self.refresh_types(reset=True)

            chdu = source.hdulist
            sublist = [chdu[0]]
            
            if self.extensions is not None:
                for extn in self.extensions:
                    sublist.append(chdu[extn])
            elif self.extInsts is not None:
                sublist += self.extInsts
            else:
                for extn in chdu:
                    sublist.append(extn)            
            self.hdulist = pyfits.HDUList(sublist)

        elif isinstance(source, pyfits.HDUList) or self._ishdu(source):
            self.hdulist = self._check_for_simple_fits_file(source)

        else:
            if source is None:
                phu = pyfits.PrimaryHDU()
                self.hdulist = pyfits.HDUList([phu])
            elif not os.path.exists(source):
                raise IOError("Cannot open '%s'" % source)

            self.filename = source
            self.__origFilename = source

            # if somehow filename is set to None from source,
            # pyfits raises a ValueError on open().
            try:
                source = pyfits.open(self.filename, mode=mode)
            except ValueError, err:
                raise Errors.AstroDataError("Attempt to open {0}".format(
                    str(err) + " failed."))
            except IOError:
                raise IOError("Cannot open {0}, mode={1}".format(
                    self.filename, mode))
                
            self.hdulist = self._check_for_simple_fits_file(source)
            self.mode = mode
                    
        if len(self.hdulist):
            try:
                self._discover_types()
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except:
                raise Errors.AstroDataError("discover types failed")
        
        if inferRAW and "RAW" in self.status():
            hdul = self.hdulist
            namedext = False
            for hdu in hdul[1:]:
                if hdu.name or ("EXTNAME" in hdu.header): 
                    namedext = True
                else:
                    pass
            if namedext == False:
                l = len(hdul)
                for i in range(1, l):
                    hdu = hdul[i]
                    _pyfits_update_compatible(hdu)
                    kafter = "GCOUNT"
                    if hdu.header.get("TFIELDS"): kafter = "TFIELDS"
                    hdu.header.update("EXTNAME", "SCI", 
                                      "ad1406: added by AstroData",
                                      after=kafter)
                    if 'EXTVER' in hdu.header:
                        del hdu.header['EXTVER']
                    hdu.header.update("EXTVER", i, 
                                      "ad1410: added by AstroData", 
                                      after="EXTNAME")
                    hdu.name = 'SCI'
                    hdu._extver = i
            else:
                # infer bad extvers to a number.
                inferEXT = max(hdul, key=lambda arg: arg.header.get("EXTVER"))
                inferEV = inferEXT.header.get("EXTVER")
                if inferEV is None:
                    inferEV = 0
                else:
                    inferEV = int(inferEV)
                if inferEV < 1:
                    inferEV = 0
                    
                numhdu = len(hdul)
                for i in range(1, numhdu):
                    hdu = hdul[i]
                    _pyfits_update_compatible(hdu)
                    ev = hdu.header.get("EXTVER")
                    inferEV += 1
                    if not ev or int(ev)< 1:
                        if 'EXTVER' in hdu.header:
                            del hdu.header['EXTVER']
                        hdu.header.update("EXTVER", inferEV ,after="EXTNAME")
                        hdu._extver = inferEV
        return

    def remove(self, index, hdui=False):
        """
        :param index: the extension index, either an int or (EXTNAME, EXTVER)
                      pair before which the extension is to be removed.
                      Note: the first data extension is [0], you cannot remove 
                      before the PHU. Index always refers to Astrodata Numbering 
                      system, 0 = HDU
        :type index: <int>, or <tuple> (EXTNAME,EXTVER)
        """
        if type(index) == tuple:
            index = self.ext_index(index, hduref=True)
            if hdui:
                raise Errors.AstroDataError("Must provide HDUList index")
            self.hdulist.__delitem__(index)
        else:    
            if index > len(self) - 1:
                raise Errors.AstroDataError("Index out of range")
            if hdui:
                if index == 0:
                    raise Errors.AstroDataError("ERROR: Cannot remove PHU.")
                self.hdulist.__delitem__(index)
            else:
                if index + 1 == 0:
                    raise Errors.AstroDataError("ERROR: Cannot remove PHU.")
                self.hdulist.__delitem__(index + 1)
        return

    def store_original_name(self):
        """
        Method adds the key 'ORIGNAME' to PHU of an astrodata object 
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
        origFilename    = self.__origFilename
        
        if origFilename != None:
            origFilename = os.path.basename(self.__origFilename)
        
        if origFilename == phuOrigFilename == None:
            mes = " failed to have its original filename stored when astrodata"
            mes += " instantiated it"
            raise Errors.AstroDataError(self.filename + mes)
        elif (phuOrigFilename is None) and (origFilename is not None):
            mes = "Original filename prior to processing"
            self.phu_set_key_value("ORIGNAME", origFilename, mes)
        else:
            origFilename = phuOrigFilename
        return origFilename

    def write(self, filename=None, clobber=False, rename=None, prefix=None, 
              suffix=None):
        """
        The write method acts similarly to the 'pyfits HDUList.writeto(..)'
        function if a filename is given, or like 'pyfits.HDUList.update(..)' if 
        no name is given, using whatever the current name is set to. When a name
        is given, this becomes the new name of the ``AstroData`` object and
        will be used on subsequent calls to  write for which a filename is not
        provided. If the ``clobber`` flag is ``False`` (the default) then 
        'write(..)' throws an exception if the file already exists.

        :param filename: name of the file to write to. Optional if the instance
                         already has a filename defined, which might not be the 
                         case for new AstroData instances created in memory.
        :type filename: <str>

        :param clobber: This flag drives if AstroData will overwrite an existing
                        file.
        :type clobber: <bool>

        :param rename: This flag allows you to write the AstroData instance to
                       a new filename, but leave the 'current' name in memory.
        :type rename: <bool>

        :param prefix: Add a prefix to ``filename``.
        :type prefix: <str>

        :param suffix: Add a suffix to ``filename``.
        :type suffix: <str>

        """
        # apply prefix or suffix
        fname = filename if filename else self.filename
        if prefix or suffix:
            fpath = os.path.dirname(fname)
            fname = os.path.basename(fname)
            base,ext = os.path.splitext(fname)
            pfix = prefix if prefix else ""
            sfix = suffix if suffix else ""
            fname = os.path.join(fpath, pfix+base+sfix+ext)
            filename = fname
            
        filenamechange = self.filename != filename
        
        if (filename and (rename==True or rename == None)):
            self.filename = filename
        if (self.mode == "readonly" and not clobber and not filenamechange):
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

        hdul = self.hdulist
        if fname == None:
            if rename == True:
                mes = ("Option rename=True but filename is None")
                raise Errors.AstroDataError(mes)
            fname = self.filename
            
        # postfix and suffix
        if fname != self.filename and rename == True:
            self.filename = fname

        if (fname == None):
            raise Errors.AstroDataError("fname is None")

        if self.mode == 'update' or self.mode == 'append':
            clobber = True

        if os.path.exists(fname):
            if clobber:
                os.remove(fname)
            else:
                raise Errors.OutputExists(fname)
        hdul.writeto(fname)
        return

    # ------------------------- Classification Members -------------------------
    @property
    def types(self):
        """
        Property: Returns the composite list of AstroData classifications. I.e. 
        the instance's qualified type and status classifications.
        
        :returns: a list of types and status strings
        :rtype: <list> 

        """
        if not self._types:
            self._discover_types()
        return self._types

    @types.setter
    def types(self, new_types):
        self._types = new_types
        return

    def type(self, prune=False):
        """
        Returns a list of type classifications. It is possible to 'prune' 
        the list so that only leaf nodes are returned, which is 
        useful when leaf nodes take precedence such as for descriptors.

        Note: types consist of a hierarchical tree of dataset types.
        This latter tree maps roughly to instrument-modes, with instrument 
        types branching from the general observatory type, (e.g. 'GEMINI'). 
        
        Currently the distinction betwen status and type is not used by the
        system (e.g. in type-specific default recipe assignments) and is
        provided as a service for higher level code, e.g. primitives and
        scripts which make use of the distinction.

        :param prune: flag which controls 'pruning' the returned type list 
                      so that only the leaf node type for a given set of 
                      related types is returned.
        :type prune: <bool>

        :returns: list of classification names
        :rtype:   <list> of strings


        """
        if self._typology is None:
            self._typology = self._discover_types(all=True)['typology']
        if prune:
            return self._prune_typelist(self._typology)
        return self._typology

    def status(self, prune=False):
        """
        Returns the set of 'status' classifications, which are those that 
        tend to change during the reduction of a dataset based on 
        the amount of processing, e.g. RAW vs PREPARED.  Strictly, a 'status' 
        type is any type defined in or below the status part of the 
        'classification' directory within the configuration package. For 
        example, in the Gemini type configuration this means any type definition
        files in or below the 'astrodata_Gemini/ADCONFIG/classification/status'
        directory.

        :param prune: flag which controls 'pruning' the returned type list 
                      so that only the leaf node type for a given set of 
                      related status types is returned.
        :type prune: <bool>

        :returns: list of classification names
        :rtype:   <list> of strings

        """
        if self._status is None:
            self._status = self._discover_types(all=True)['status']
        if prune:
            return self._prune_typelist(self._status)
        return self._status

    def refresh_types(self):
        self._types = None
        self._discover_types(all=True)
        return

    # ------------------------ Inspection/Modificaton --------------------------
    def count_exts(self, extname=None):
        """
        The count_exts() function returns the number of extensions matching the
        passed <extname> (as stored in the HDUs "EXTNAME" header).

        :param extname: the name of the extension, equivalent to the
                        value associated with the "EXTNAME" key in the extension
                        header.
        :type extname: <str>

        :returns: number of <extname> extensions
        :rtype:   <int>

        """
        hdul = self.hdulist
        maxl = len(hdul)
        if extname is None:
            return maxl - 1
        count = 0
        for i in range(1, maxl):
            try:
                if (hdul[i].header["EXTNAME"] == extname):
                    count += 1
            except KeyError:
                if extname == None:
                    count += 1 
        return count

    def ext_index(self, extension, hduref=False):
        """
        Takes an extension index, either an integer or (EXTNAME, EXTVER) 
        tuple, and returns the index location of the extension.  If hduref is 
        set to True, then the index returned is relative to the HDUList 
        (0=PHU, 1=First non-PHU extension). If hduref is False (the default) 
        then the index returned is relative to the AstroData numbering 
        convention, where index=0 is the first extension in the MEF file.

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

    def rename_ext(self, name, ver=None, force=True):
        """
        The rename_ext(..) function is used in order to rename an 
        HDU with a new EXTNAME and EXTVER identifier.  Merely changing 
        the EXTNAME and EXTVER values in the extensions pyfits.Header 
        is not sufficient. Though the values change in the pyfits.Header 
        object, there are special HDU class members which are not updated. 
        
        :warning: This function manipulates private (or somewhat private) 
                  HDU members, specifically 'name' and '_extver'. STSCI 
                  has been informed of the issue and has made a special 
                  HDU function for performing the renaming. 
                  When generally available, this new function will be used 
                  instead of manipulating the  HDU's properties directly, 
                  and this function will call the new pyfits.HDUList(..) 
                  function.

        :note: Works only on single extension instances.

        :param name: New 'EXTNAME' for the given extension.
        :type name: <str>
        
        :param ver: New 'EXTVER' for the given extension
        :type ver: <int>
        
        :param force: Will update even on subdata, or shared hdulist.
                      Default=True
        :type force: <bool>

        """
        # @@TODO: change to use STSCI provided function.
        if force != True and self.borrowed_hdulist:
            raise Errors.AstroDataError("cannot setExtname on subdata")
        self._except_if_single()
        rename_hdu(name=name, ver=ver, hdu=self.hdulist[1])    
        return

    def extname(self):
        self._except_if_single()
        return self.hdulist[1].header.get("EXTNAME", None)
        
    def extver(self):
        self._except_if_single()
        retv = self.hdulist[1].header.get("EXTVER", None)
        if retv:
            retv = int(retv)
        return retv

    def info(self, oid=False, table=False, help=False):
        """
        Prints to stdout information about the phu and extensions found 
        in the current instance.
        """
        print self._infostr(oid=oid, table=table, help=help)
        return

    def get_key_value(self, key):
        """
        The get_key_value() function is used to get the value associated
        with a given key in the data-header unit of a single-HDU
        AstroData instance (such as returned by iteration).
        
        :note: 
          Single extension AstroData objects are those with only a single
          header-data unit besides the PHU.  They may exist if a single
          extension file is loaded, but in general are produced by indexing or
          iteration instructions, Eg.:
        
              sead = ad[("SCI",1)]
            
              for sead in ad["SCI"]:
                  ...
                
          The variable "sead" above is ensured to hold a single extension
          AstroData object, and can be used more convieniently.

        :param key: name of header keyword to set
        :type key: <str> header keyword

        :returns: header keyword value
        :rtype:  <int>, or <float>, or <str>
        :raises: SingleHDUMemberExcept

        """
        self._except_if_single()
        return self._ext_get_key_value(0, key)

    def set_key_value(self, key, value, comment=None):
        """
        The set_key_value() function is used to set the value (and optionally
        the comment) associated with a given key in the data-header of a 
        single-HDU AstroData instance. The value argument will be converted to 
        string, so it must have a string operator member function or be passed 
        in as string. 

        :note: 
          Single extension AstroData objects are those with only a single
          header-data unit besides the PHU.  They may exist if a single
          extension file is loaded, but in general are produced by indexing 
          or iteration instructions.Eg.:
        
            sead = ad[("SCI",1)]

            for sead in ad["SCI"]:
              ...
                
          The variable "sead" above is ensured to hold a single extension
          AstroData object, and can be used more convieniently.

        :param key: header keyword
        :type  key: <str>

        :param value: header keyword value
        :type  value: <int>, or <float>, or <str>

        :param comment: header keyword comment
        :type  comment: <str>

        """
        self._except_if_single()
        self._ext_set_key_value(0, key, value, comment)
        return

    # PHU manipulations
    def phu_get_key_value(self, key):
        """
        The phu_get_key_value(..) function returns the value associated 
        with the given key within the primary header unit of the dataset.
        The value is returned as a string (storage format) and must be 
        converted as necessary by the caller.

        :param key: name of header value to retrieve
        :type  key: <str>

        :returns: keyword value as string or None if not present.
        :rtype: <str>

        """
        try:
            retval = self.phu.header[key]
            if isinstance(retval, pyfits.core.Undefined):
                raise Errors.UndefinedKeyError()
            if retval == "" or retval == " ":
                raise Errors.EmptyKeyError()
            return retval
        except:
            setattr(self, "exception_info", sys.exc_info()[1])
            return None

    def phu_set_key_value(self, keyword=None, value=None, comment=None):
        """
        Add or update a keyword in the PHU of the AstroData object with a
        specific value and, optionally, a comment
        
        :param keyword: Name of the keyword to add or update in the PHU
        :type  keyword: <str>

        :param value: Value of the keyword to add or update in the PHU
        :type  value: <int>, <float>, or <str>

        :param comment: Comment of the keyword to add or update in the PHU
        :type  comment: string

        """
        if keyword is None:
            raise Errors.AstroDataError("No keyword provided")
        if value is None:
            raise Errors.AstroDataError("No keyword value provided")
        
        history_comment = None
        add_history = True
        if keyword.endswith("-TLM"):
            add_history = False
        original_value = self.phu_get_key_value(keyword)
        
        if original_value is None:
            history_comment = ("New keyword %s=%s was written to the PHU" %
                               (keyword, value))
            comment_prefix = "(NEW)"
        else:
            if original_value == value:
                if comment is not None:
                    if add_history:
                        history_comment = ("The comment for the keyword %s "
                                           "was updated" % keyword)
                    comment_prefix = "(UPDATED)"
            else:
                if add_history:
                    history_comment = ("The keyword %s=%s was overwritten in "
                                       "the PHU with new value %s" %
                                       (keyword, original_value, value))
                comment_prefix = "(UPDATED)"
        
        if comment is None:
            final_comment = None
        else:
            full_comment = "%s %s" % (comment_prefix, comment)
            # Truncate if necessary
            if len(str(value)) >= 65:
                final_comment = ""
            elif len(comment) > 47:
                final_comment = full_comment[:47]
            else:
                final_comment = full_comment[:65-len(str(value))]
        
        _pyfits_update_compatible(self.phu)
        self.phu.header.update(keyword, value, final_comment)
        
        # Add history comment
        if history_comment is not None:
            self.phu.header.add_history(history_comment)
        return

    # --------------------------------- prive ----------------------------------
    def _all_descriptors(self):
        funs = self._all_descriptor_names()
        rdict = {}
        for fun in funs:
            try:
                member = eval("self.%s"% fun)
                if callable(member):                
                    val = eval("self.%s(asList=True)" % fun)
                else:
                    continue
            except AttributeError:
                val = 'ERROR: No Descriptor Function Named "%s"' % fun  
            except:
                val = "ERROR: %s" % repr(sys.exc_info()[1])
            rdict.update({fun:val})
        return rdict

    def _all_descriptor_names(self):
        funs = dir(CalculatorInterface)
        descs = []
        for fun in funs:
            if "_" != fun[0] and (fun.lower() == fun) :
                descs.append(fun)
        return descs

    def _ishdu(self, md=None):
        """Checks to see if md (moredata) is acutally an hdu.

        :returns: True or False
        :rtype:  <bool>
        """
        if hasattr(pyfits, "hdu") \
           and  hasattr(pyfits.hdu, "base") \
           and  hasattr(pyfits.hdu.base, "_BaseHDU"):
            return isinstance(md, pyfits.hdu.base._BaseHDU)
        elif hasattr(pyfits, "core") \
             and  hasattr(pyfits.core, "_AllHDU"):
            return isinstance(md, pyfits.core._AllHDU)
        else:
            return False

    def _moredata_check(self, md=None, append=False, insert=False,
                        replace=False, index=None):
        if isinstance(md, AstroData):
            return md.hdulist
        elif type(md) is pyfits.HDUList:
            return md
        elif self._ishdu(md):
            try:
                if append:
                    self.hdulist.append(md)
                    print "WARNING: Appending unknown HDU type"
                    return False
                elif insert or replace:
                    if not index:
                        raise Errors.AstroDataError("index required to insert")
                    else:
                        if replace:
                            self.remove(index)
                        self.hdulist.insert(index + 1, md)
                        print "WARNING: Inserting unknown HDU type"
                        return False
            except:
                raise Errors.AstroDataError(
                   "cannot operate on pyfits instance " + repr(md))
        else:
            raise Errors.AstroDataError(
                "The 'moredata' argument is of an unsupported type")
        return

    def _moredata_work(self, append=False, insert=False, autonum=False,
                       md=None, hduindx=None, hdul=None): 
        """
        create a master table out of the host and update the EXTVER 
        for the guest as it is being updated in the table
        """
        et_host = _ExtTable(hdul=self.hdulist)
        if isinstance(md, pyfits.HDUList):
            et_guest = _ExtTable(hdul=md)
        else:
            et_guest = _ExtTable(ad=md)
        if autonum:
            for row in et_guest.rows():
                host_bigver = et_host.largest_extver()
                for tup in row:
                    if None in et_guest.xdict[tup[0]].keys():
                        et_host.putAD(extname=tup[0], extver=None,
                                      ad=tup[1])
                        rename_hdu(name=tup[0], ver=None,
                                   hdu=hdul[tup[1][1]])
                    else:
                        et_host.putAD(extname=tup[0],
                                      extver=host_bigver + 1, ad=tup[1])
                        rename_hdu(name=tup[0], ver=host_bigver + 1,
                                   hdu=hdul[tup[1][1]])

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
                            print "AD626:",ver,ext
                            print "--"*20
                            print repr(et_guest.xdict)
                            print "--"*20
                            print repr(et_host.xdict)
                            print "--"*20
                            raise Errors.AstroDataError(
                                "EXTNAME, EXTVER conflict, use auto_number")
            for hdu in hdul[1:]:
                if append:
                    self.hdulist.append(hdu)
                elif insert:
                    self.hdulist.insert(hduindx, hdu)
        return
    
    def _onehdu_work(self, append=False, insert=False, replace=False,
                     header=None, data=None, extver=None, extname=None, 
                     autonum=False, hduindx=None):
        """ Does extension work for one HDU """
        if header is None or data is None: 
            raise Errors.AstroDataError("Required parameters: header *and* data")

        header = self._verify_header(extname=extname, extver=extver, header=header)
        xver = header.get("EXTVER")

        et_host = _ExtTable(self)
        if not autonum:
            for ext in et_host.xdict.keys():
                if header["EXTNAME"] == ext:
                    if xver in et_host.xdict[ext].keys():
                        raise Errors.AstroDataError(
                            "EXTNAME EXTVER conflict, use auto_number") 
        host_bigver = et_host.largest_extver()
        if new_pyfits_version:
            header.update = header.set

        if extver:
            xver = extver
            header.update("EXTVER", xver, "Added by AstroData", after="EXTNAME")
        elif xver <=  host_bigver and xver is not None:
            xver = host_bigver + 1
            header.update("EXTVER", xver, "Added by AstroData", after="EXTNAME")

        if append:
            if isinstance(data, pyfits.core.FITS_rec):
                self.hdulist.append(pyfits.BinTableHDU(data=data, header=header))
            else:
                self.hdulist.append(pyfits.ImageHDU(data=data, header=header))
        elif replace or insert:
            if replace:
                self.remove(hduindx, hdui=True)
            if isinstance(data, pyfits.core.FITS_rec):
                self.hdulist.insert(hduindx, pyfits.BinTableHDU(data=data,
                                                                header=header))
            else:
                self.hdulist.insert(hduindx, pyfits.ImageHDU(data=data,
                                                             header=header))
        return

    def _except_if_single(self):
        if len(self.hdulist) != 2:
            raise Errors.SingleHDUMemberExcept()
        return

            
    def _verify_header(self, extname=None, extver=None, header=None):
        """
        This is a helper function for insert, append and replace that compares
        the extname argument with the extname in the header. If the key does
        not exist it adds it, if its different, it changes it to match the 
        argument

        :param extname: extension name (eg., 'SCI', 'VAR', 'DQ')
        :type  extname: <str>
        
        :param extver: extension version
        :type  extver: <int>
        
        :param header: a valid pyfits.Header object
        :type  header: pyfits.core.Header

        :returns header: a validated pyfits.Header object
        :rtype: <pyfits.core.Header>
        """
        if header is None:
            ihdu = pyfits.ImageHDU()
            header = ihdu.header

            _pyfits_update_compatible(ihdu)

            if extname is None:
                raise Errors.AstroDataError("cannot resolve extname")
            else: 
                ihdu.header.update("EXTNAME", value=extname, 
                                   comment="Added by AstroData")

            if extver is None:
                ihdu.header.update("EXTVER", value=1, after="EXTNAME",
                                   comment="Added by AstroData")
            else:
                ihdu.header_update("EXTVER", value=extver, after="EXTNAME",
                                   comment="Added by AstroData")
        else:
            if new_pyfits_version:
                header.update = header.set

            if extname and header.get("EXTNAME"):
                header.update("EXTNAME", value=extname, 
                              comment="Added by AstroData")
            elif extname and not header.get("EXTNAME"):
                header.update("EXTNAME", value=extname, after='GCOUNT',
                              comment="Added by AstroData")

            if extver and header.get("EXTVER"):
                header.update("EXTVER", value=extver, after="EXTNAME",
                              comment="Added by AstroData")
                              
        return header

    def _discover_types(self, all=False):
        """
        Method provides a list of classifications of both processing
        status and typology which apply to the data encapsulated by this
        instance, identified by their string names.

        :param all: controls how the classes are returned. If True, 
                    returns a <dict> of three lists, 'all', 'status', and 
                    'typology'. If False, returns  a list which is in fact the 
                    'all' list containing all the status and typology related 
                    types together.
        :type  all: <bool>

        :returns: <list> DataClassification objects, or <dict> of lists
                        if the C{alltypes} flag is set.
        :rtype:  <list> or <dict>
        """
        alltypes = None
        if self._types is None:
            cl = self.get_classification_library()
            alltypes = cl.discover_types(self, all=True)
            self._types = alltypes["all"]
            self._status = alltypes['status']
            self._typology = alltypes['typology']

        if alltypes is None and all is True:
            alltypes = {}
            alltypes['all'] = self._types
            alltypes['status'] = self._status
            alltypes['typology'] = self._typology
            return alltypes
        else:
            return self._types

    def _prune_typelist(self, typelist):
        cl = self.get_classification_library()
        retary = typelist
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
           
    def _ext_get_key_value(self, extension, key):
        """
        Method returns the value from the given extension's header, 
        with "0" being the first data extension.  To get values from 
        the PHU use phu_get_key_value(..).

        :param extension: identifies extension
        :type  extension: <int> or <tuple> (EXTNAME, EXTVER)

        :param key: name of header entry to retrieve
        :type  key: <str>

        :returns: keyword value, or None if not present
        :rtype: <str> or None
        """
        
        if type(extension) == int:
            extension = extension + 1
        #make sure extension is in the extensions list
        #@@TODO: remove these self.extensions lists
        
        # if (self.extensions != None) and (not extension in self.extensions):
        #    return None
        hdul = self.hdulist
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
    
    def _ext_set_key_value(self, extension=None, keyword=None, value=None,
                          comment=None):
        """
        Add or update a keyword in the header of an extension of the AstroData
        object with a specific value and, optionally, a comment. To add or
        update a keyword in the PHU of the AstroData object, use
        phu_set_key_value().
        
        :param extension: Name of the extension to add or update. The index [0]
                          refers to the first extension in the AstroData
                          object.
        :type  extension: <int> or <tuple> (EXTNAME, EXTVER)

        :param keyword: Name of the keyword to add or update in the extension
        :type  keyword: <str>

        :param value:   Value of the keyword to add or update in the extension
        :type  value:   <int>, <float>, or <str>

        :param comment: Comment of the keyword to add or update in the
                        extension 
        :type  comment: <str>
        """
        origextension = extension
        if type(extension) == int:
            extension = extension + 1

        try:
            tx = self.hdulist[extension]
        except:
            mes = ("Extension %s not present in AstroData instance" %
                   str(origextension))
            raise Errors.AstroDataError(mes)
        
        hdul = self.hdulist
        ext = hdul[extension]
        extname = ext.header.get("EXTNAME")
        extver = ext.header.get("EXTVER")
        
        if keyword is None:
            raise Errors.AstroDataError("No keyword provided")
        if value is None:
            raise Errors.AstroDataError("No keyword value provided")
        
        # Check to see whether the keyword is already in the extension
        history_comment = None
        original_value = ext.header.get(keyword)
        
        if original_value is None:
            history_comment = ("New keyword %s=%s was written to extension "
                               "%s,%s" % (keyword, value, extname, extver))
            comment_prefix = "(NEW)"
        else:
            if original_value == value:
                if comment is not None:
                    history_comment = ("The comment for the keyword %s was "
                                       "updated" % keyword)
                    comment_prefix = "(UPDATED)"
            else:
                history_comment = ("The keyword %s=%s was overwritten in "
                                   "extension %s,%s with new value %s" %
                                   (keyword, original_value, extname, extver,
                                    value))
                comment_prefix = "(UPDATED)"
        
        if comment is None:
            final_comment = None
        else:
            full_comment = "%s %s" % (comment_prefix, comment)
            if len(str(value)) >= 65:
                final_comment = ""
            elif len(comment) > 47:
                final_comment = full_comment[:47]
            else:
                final_comment = full_comment[:65-len(str(value))]
        
        _pyfits_update_compatible(ext)
        ext.header.update(keyword, value, final_comment)
        
        # Add history comment to the PHU
        if history_comment is not None:
            self.phu.header.add_history(history_comment)
        return

    def _check_for_simple_fits_file(self, source):
        """
        Check for a simple fits file, e.g., as a HDU instance or as HDUList.

        If `source` is a simple FITS image the header is copied to the PHU and
        the input appended to the HDUList containing the new PHU.

        If `source` is a simple FITS table the header is copied
        to the PHU and the input appended to the HDUList containing the new
        PHU.

        If `source` is a PHU with data a new PHU is created and a new ImageHDU
        is created from the header and data from `source` and both are
        returned as a PHUList.

        If `source` is a PHUList of length greater than 1 and the first
        extension contains data, an AstroDataError is raised.

        If `source` is a PHUList of length greater than 1 and the first
        extension doesn't contain data, a check that the first extension in
        `source` is a PrimaryHDU is made. If the first extension is not a
        PrimaryPHU and doesn't contain data a new HDUList is returned with the
        first extension converted to a PrimaryHDU.

        If `source` is a valid MEF HDUList, `source` is returned untouched.

        :param source: Input HDU or HDUList to check for simple FITS and / or
                  PrimaryHDU
        :type source: pyfits.HDU; pyfits.HDUList

        :returns: HDUList with PrimaryHDU as first extension. If `source`[0] or
                 just `source` contained data the data are appended to the
                 HDUList.
        :rtype: pyfits.HDUList
        :raises: AstroDataError

        """
        is_hdulist = False
        if isinstance(source, pyfits.HDUList):
            # Extract the first extension and flag that it is an HDUList
            is_hdulist = True
            first_extension = source[0]
        else:
            first_extension = source

        # Chcek if the first extension has data
        if first_extension.data is not None:
            #TODO: Possibly update these to be PyFITS isimage etc, tests?
            __HDUS_TO_MANIPULATE__ = (pyfits.PrimaryHDU,
                                      pyfits.ImageHDU,
                                      pyfits.TableHDU,
                                      pyfits.BinTableHDU)

            if isinstance(first_extension, __HDUS_TO_MANIPULATE__):
                # If the length of the source is greater than one and the first
                # extension in the list has data raise an exception. Only
                # perform this check if source is an HDUList
                if is_hdulist and len(source) > 1:
                    raise Errors.AstroDataError("PHU contains data and "
                                                "HDUList has length %d" %
                                                len(source))

                # Single extension or HDUList of length 1 supplied
                header = copy(first_extension.header)

                # Check if first_extension is a PrimaryHDU; set the data
                # value to decide whether to append source or create a new
                # HDU from source
                if isinstance(first_extension, pyfits.PrimaryHDU):
                    data = copy(first_extension.data)
                else:
                    data = None

                # Create PrimaryHDU with copy of input hedear even if it
                # was a PrimrayHDU
                hdu = pyfits.PrimaryHDU(header=header)

                # Create HDUList
                dataset = pyfits.HDUList(hdu)

                # Reset hdu variable
                if data is not None:
                    # Data is only not None if input was a Primary HDU
                    # If data and/or header is None, pyfits will allow it
                    hdu = pyfits.ImageHDU(data=data, header=header)
                else:
                    # Just append the input source
                    hdu = first_extension

                dataset.append(hdu)

                # Add the EXTNAME keyword
                # TODO:
                # This has been left for compatibility. There should only be
                # one function that does this - MS
                if "EXTNAME" not in hdu.header:
                    kafter = "GCOUNT"
                    if not kafter in hdu.header:
                        kafter = None
                    if hdu.header.get("TFIELDS"):
                        kafter = "TFIELDS"

                    _pyfits_update_compatible(hdu)
                    hdu.header.update("EXTNAME", "SCI",
                                      "ad4549: added by AstroData", after=kafter)

        else:
            # Source or first extension of source doesn't have data associated
            # with it.

            # Check first_extension is a PrimaryHDU. Create a PrimaryHDU if not
            first_extension_was_phu = True
            if not isinstance(first_extension, pyfits.PrimaryHDU):
                first_extension_was_phu = False
                first_extension = pyfits.PrimaryHDU(
                                      header=first_extension.header)

            # first_extesnion is a PrimaryHDU by this point

            if not is_hdulist:
                # An HDU with no data has been supplied
                # Form the HDUList
                dataset = pyfits.HDUList(first_extension)
            elif not first_extension_was_phu:
                # An HDUList has been supplied and the the first extension
                # doesn't have data associated with it but wasn't a PrimaryHDU

                # Update first extension to be a PrimaryHDU
                dataset = copy(source)
                dataset[0] = first_extension
            else:
                # HDUList supplied and first extension is a PHU.
                dataset = source

        return dataset

    def _infostr(self, as_html=False, oid=False, table=False, help=False):
        """
        The infostr(..) function is used to get a string ready for display
        either as plain text or HTML.  It provides AstroData-relative
        information.  

        :param as_html: return as HTML formatted string
        :type  as_html: <bool>
        
        :param oid: include object id 
        :type  oid: <bool>
        
        :param help: include sub-data reference information
        :type  help: <bool>

        :returns: instance information string
        :rtype:   <str>
        """
        if not as_html:
            hdulisttype = ""
            phutype = None
            #Check basic structure of ad
            if isinstance(self, AstroData):
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
            if oid:
                rets += "\n Obj. ID: %s" % str(id(self))
            rets += "\n    Type: %s" % selftype
            rets += "\n    Mode: %s" % str(self.mode)

            if new_pyfits_version:
                lencards = len(self.phu._header.cards)
            else:
                lencards = len(self.phu._header.ascard)

            if oid:
                rets += "\n\nAD No.    Name          Type      MEF No."
                rets += "  Cards    Dimensions   Format   ObjectID   "

                rets += "\n%shdulist%s%s%s%s" % (" "*8, " "*7, \
                    hdulisttype, " "*45, str(id(self.hdulist)))

                rets += "\n%sphu%s%s    0%s%d%s%s" % (" "*8, " "*11, \
                    phutype, " "*7, lencards," "*27, str(id(self.phu)))

                rets += "\n%sphu.header%s%s%s%s" % (" "*8, " "*4, \
                    phuHeaderType, " "*46, str(id(self.phu.header)))
            else:
                rets += "\n\nAD No.    Name          Type      MEF No."
                rets += "  Cards    Dimensions   Format   "
                rets += "\n%shdulist%s%s" % (" "*8, " "*7, hdulisttype)
                rets += "\n%sphu%s%s    0%s%d" % (" "*8, " "*11, \
                    phutype, " "*7, lencards)
                rets += "\n%sphu.header%s%s" % (" "*8, " "*4, phuHeaderType)
            hdu_indx = 1

            for hdu in self.hdulist[1:]:
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
                    if not "EXTVER" in hdu.header:
                        name_ = hdu.header["EXTNAME"]
                    else:
                        name_ = "('" + hdu.header['EXTNAME'] + "', "
                        name_ += str(hdu.header['EXTVER']) + ")"
                    if new_pyfits_version:
                        cards_ = len(self.hdulist[hdu_indx]._header.cards)
                    else:
                        cards_ = len(self.hdulist[hdu_indx]._header.ascard)

                except:
                    pass
                if extType == "ImageHDU":
                    if self.hdulist[hdu_indx].data == None:
                        dimention_ = None
                        form_ = None
                    else:
                        dimention_ = self.hdulist[hdu_indx].data.shape
                        form_ = self.hdulist[hdu_indx].data.dtype.name
                else:
                    dimention_ = ""
                    form_ = ""
                if oid:
                    rets += "\n%-7s %-13s %-13s %-8s %-5s %-13s %s  %s" % \
                        (adno_, name_, extType, str(hdu_indx), str(cards_), \
                            dimention_, form_, \
                            str(id(self.hdulist[hdu_indx])))
                    if extType == "ImageHDU" or extType == "BinTableHDU":
                        rets += "\n           .header    %s%s%s" % \
                            (extHeaderType, " "*46, \
                            str(id(self.hdulist[hdu_indx].header)))
                        rets += "\n           .data      %s%s%s" % \
                            (extDataType, " "*45, \
                            str(id(self.hdulist[hdu_indx].data)))
                else:
                    rets += "\n%-7s %-13s %-13s %-8s %-5s %-13s %s" % \
                        (adno_, name_, extType, str(hdu_indx), str(cards_), \
                            dimention_, form_)
                    if extType == "ImageHDU" or extType == "BinTableHDU":
                        rets += "\n           .header    %s" % extHeaderType 
                        rets += "\n           .data      %s" % extDataType
                hdu_indx += 1
            if table:
                for i, ext in enumerate(self):
                    if isinstance(ext.hdulist[1], pyfits.core.BinTableHDU):
                        rets += "\n" + "="*79 + "\nAD[" + str(i) + "]" 
                        rets += ", BinTableHDU: " + ext.extname() + \
                            "\n" + "="*79
                        rets += "\n      Name            Value" + \
                            " "*25 + "Format"
                        rets += "\n" + "-"*79
                        fitsrec = ext.hdulist[1].data
                        for j in range(len(fitsrec.names)):
                            rets += "\n%-15s : %-25s         %3s" % \
                            (fitsrec.names[j], fitsrec[0][j],
                             fitsrec.formats[j])
                        rets += "\n" + "="*79 + "\n\n"
            if help:
                s = " "*24
                rets += SUBDATA_INFO_STRING
        else:
            rets = "<b>Extension List</b>: %d in file" % len(self)
            rets += "<ul>"
            for ext in self:
                rets += "<li>(%s, %s)</li>" % (ext.extname(), 
                                               str(ext.extver()))
            rets += "</ul>"
        return rets

    def div(self, denominator):
        return arith.div(self, denominator)
    div.__doc__ = arith.div.__doc__
    
    def mult(self, input_b):
        return arith.mult(self, input_b)
    mult.__doc__ = arith.mult.__doc__
    
    def add(self, input_b):
        return arith.add(self, input_b)
    add.__doc__ = arith.add.__doc__
    
    def sub(self, input_b):
        return arith.sub(self, input_b)
    sub.__doc__ = arith.sub.__doc__

# ------------------------------------------------------------------------------
class _ExtTable(object):
    """
    _ExtTable will create a dictionary structure keyed on 'EXTNAME' with an
    internal dictionary keyed on 'EXTVER' with ad as values.
    """
    def __init__(self, ad=None, hdul=None):
        if ad and hdul:
            raise Errors.ExtTableError("Pass ONE of 'ad' OR 'hdul', not both")
        if not ad and not hdul:
            raise Errors.ExtTableError("Constructor requires AstroData OR hdulist")

        self.ad    = ad
        self.hdul  = hdul
        self.xdict = {}

        if hdul:
            app_hdul = AstroData(hdul)
            self.hdul = app_hdul.hdulist
        self.create_xdict()

    def create_xdict(self):
        hdulist = None
        if isinstance(self.hdul, pyfits.core.HDUList):
            hdulist = self.hdul
        elif isinstance(self.ad, AstroData):
            hdulist = self.ad.hdulist
        extnames = []
        for i in range(1,len(hdulist)):
            xname = None
            xver = None
            hdu = hdulist[i]
            if 'EXTNAME' in hdu.header:
                xname = hdu.header['EXTNAME']
                newname = True
                if xname in extnames:
                    newname=False
                else:
                    extnames.append(xname)
            if 'EXTVER' in hdu.header:
                xver = hdu.header['EXTVER']
            if newname:
                if self.ad is None:
                    self.xdict.update({xname:{xver:(True,i)}})
                else:
                    self.xdict.update({xname:{xver:(self.ad,i)}})
            else:
                if self.ad is None:
                    self.xdict[xname].update({xver:(True,i)})
                else:
                    self.xdict[xname].update({xver:(self.ad,i)})
        return

    def putAD(self, extname=None, extver=None, ad=None, auto_inc=False):
        if extname is None or ad is None:
            raise Errors.ExtTableError("At least extname and ad required")
        if extname in self.xdict.keys():
            if extver in self.xdict[extname].keys():
                # deal with collision
                if auto_inc is True:
                    extver_list = self.xdict[extname].keys()
                    extver_list.sort()
                    extver = extver_list.pop() + 1
                    self.xdict[extname].update({extver:ad})
                else:
                    raise Errors.ExtTableError(\
                        "Table already has %s, %s" % (extname, extver))
            else:
                # the extver is open, put in the AD!
                self.xdict[extname].update({extver:ad})
        else:
            #extname not in table, going to add it, then put in the AD!
            if extver is None and auto_inc:
                extver = 1
            self.xdict.update({extname:{extver:ad}})
        return

    def getAD(self, extname=None, extver=None, asext=False):
        if extname is None or extver is None:
            raise Errors.ExtTableError("extname and extver are required")
        if extname in self.xdict.keys():
            if extver in self.xdict[extname].keys():
                if asext:
                    rad = self.xdict[extname][extver]
                    return rad[extname,extver]
                return self.xdict[extname][extver]
        print "Warning: Cannot find ad in %s, %s" % (extname,extver)
        return None
        
    def rows(self, asext=False):
        # find the largest extver out of all extnames
        bigver = 0
        for xnam in self.xdict.keys(): 
            for ver in self.xdict[xnam].keys():
                if ver > bigver:
                    bigver = ver
        index = 1
        # generator will keep yielding a table row until bigver hit
        while(index <= bigver):
            namlis = self.xdict.keys()
            rlist = []
            for xnam in self.xdict.keys():
                if index in self.xdict[xnam].keys():
                    if asext:
                        ad = self.xdict[xnam][index]
                        rlist.append((xnam,ad[xnam,index]))
                    else:
                        rlist.append((xnam,self.xdict[xnam][index]))
                    yield rlist
            index += 1
        return
                
    def largest_extver(self):
        bigver = 0
        for xnam in self.xdict.keys(): 
            for ver in self.xdict[xnam].keys():
                if ver > bigver:
                    bigver = ver
        return bigver

# ------------------------------------------------------------------------------
def _pyfits_update_compatible(hdu):
    """
    Creates a member with the header function pointer set or update if 
    new_pyfits_version is True or False.

    The new_pyfits_version member is defined in astrodata/__init__.py
    """
    if new_pyfits_version:
        hdu.header.update = hdu.header.set
    return

def re_header_keys(rekey, header):
    """
    This utility function returns a list of keys from the input header that 
    match the regular expression.
    
    :param rekey: a regex to match keys in header
    :type rekey:  string
    
    :param header: pyfits.Header object from 'ad[("SCI",1)].header'
    :type  header: pyfits.Header
    
    :returns: a list of strings; matching keywords
    :rtype:  <list>

    """
    retset = []
    if new_pyfits_version:
        for k in header.keys():
            #print "gd278: key=%s" % k
            if re.match(rekey, k):
                retset.append(k)
    else:
        for k in header.ascardlist().keys():
            # print "gd278: key=%s" % k
            if re.match(rekey, k):
                retset.append(k)

    if len(retset) == 0:
        retset = None
    return retset
