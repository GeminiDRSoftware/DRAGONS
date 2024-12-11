#
#                                                                  gemini_python
#
#                                                           test_AstroDataAPI.py
# ------------------------------------------------------------------------------
# $Id$
# ------------------------------------------------------------------------------
__version__      = '$Revision$'[11:-2]
__version_date__ = '$Date$'[7:-2]
# ------------------------------------------------------------------------------
""" Initial AstroData API tests.

    To run theses tests w/ pytest:

    $ py.test [-v] -rxX

      -v is verbose. See pytest help for more options.
"""
import os
import sys
import pytest
import numpy  as np
import pyfits as pf
from   cStringIO import StringIO

from astrodata import AstroData
from astrodata.utils.Errors import OutputExists
from astrodata.utils.Errors import AstroDataError
from astrodata.utils.Errors import SingleHDUMemberExcept
# ==================================== Set up  =================================
# TESTFILEs are located under gemini_python/test_data/astrodata_bench/
# TESTURL specifies a non-extistent file, but request on a actual service.
GEM0 = None
GEM1 = None
# OR a user location for benchmark datasets
GEM0 = os.environ.get("ADTESTDATA")

# If no env var, look for 'test_data' under gemini_python
if GEM0 is None:
    for path in sys.path:
        if 'gemini_python' in path:
            GEM1 = os.path.join(path.split('gemini_python')[0],
                               'gemini_python', 'test_data')
            break

if GEM0:
    TESTFILE  = os.path.join(GEM0, 'GS_GMOS_IMAGE.fits')   # 3 'SCI'
    TESTFILE2 = os.path.join(GEM0, 'GS_GMOS_IMAGE_2.fits')  # 1 'SCI'
    TESTPHU   = os.path.join(GEM0, 'GN_GNIRS_IMAGE_PHU.fits')  # PHU
elif GEM1:
    TESTFILE  = os.path.join(GEM1, 'astrodata_bench/GS_GMOS_IMAGE.fits')   # 3 'SCI'
    TESTFILE2 = os.path.join(GEM1, 'astrodata_bench/GS_GMOS_IMAGE_2.fits')  # 1 'SCI'
    TESTPHU   = os.path.join(GEM1, 'astrodata_bench/GN_GNIRS_IMAGE_PHU.fits')  # PHU
else:
    SystemExit("Cannot find astrodata benchmark test_data in available paths")

# Do TESTFILEs exist?
try:
    assert os.path.isfile(TESTFILE)
    assert os.path.isfile(TESTFILE2)
    assert os.path.isfile(TESTPHU)
except AssertionError:
    SystemExit("Cannot find astrodata benchmark datasets in available paths")
# ==============================================================================
TESTURL   = 'http://fits/file/GS_GMOS_IMAGE.fits'
KNOWNTYPE = 'GMOS_IMAGE'
KNOWNSTAT = 'GMOS_RAW'
PHUTYPE   = 'GNIRS_IMAGE'
NULLTYPES = ['RAW', 'UNPREPARED']
BADFILE   = 'FOO.fits'

pfob    = pf.open(TESTFILE)      # <'HDUList'>
phu     = pfob[0]                # <'PrimaryHDU'>
hdu1    = pfob[1]                # <'HDU'>
hdu2    = pfob[2]                # <'HDU'>
data1   = pfob[1].data           # <'ndarray'>
data2   = pfob[2].data           # <'ndarray'>
header1 = pfob[1].header         # <'Header'>
header2 = pfob[2].header         # <'Header'>
# ==================================== tests  ==================================
xfail = pytest.mark.xfail
# ==============================================================================
# Constructor
def test_constructor_0():
    """ Good filename """
    assert AstroData(dataset=TESTFILE)

def test_constructor_1():
    """ Good filename, def mode """
    ad = AstroData(dataset=TESTFILE)
    assert ad.mode == 'readonly'

def test_constructor_2():
    """ filename w mode option """
    ad = AstroData(dataset=TESTFILE, mode="update")
    assert ad.mode == 'update'

def test_constructor_3():
    """ Non-existant file """
    with pytest.raises(IOError):
        AstroData(dataset=BADFILE)

def test_constructor_4():
    """ dataset as bad URL """
    with pytest.raises(AstroDataError):
        assert AstroData(dataset=TESTURL)

def test_constructor_5():
    """ HDUList """
    assert AstroData(dataset=pfob)

def test_constructor_6():
    """ header <pyfits.Header>, data <ndarray> """
    assert AstroData(header=header1, data=data1)

def test_constructor_7():
    """ phu, header, data """
    assert AstroData(phu=phu, header=header1, data=data1)

def test_constructor_8():
    """ dataset, exts """
    ad = AstroData(dataset=pfob)
    sub_ad = AstroData(dataset=ad, exts=[2])
    assert len(sub_ad) == 1

def test_constructor_9():
    """ dataset, exts as <int> """
    ad = AstroData(dataset=pfob)
    with pytest.raises(TypeError):
        sub_ad = AstroData(dataset=ad, exts=2)

def test_constructor_10():
    """ dataset, extInsts """
    extenstion_list = [hdu1, hdu2]
    ad = AstroData(dataset=TESTFILE)
    sub_ad = AstroData(ad, extInsts=extenstion_list)
    assert len(sub_ad) == 2

def test_constructor_11():
    """ Simple FITS, PHU only """
    with pytest.raises(AssertionError):
        assert AstroData(dataset=TESTPHU)

def test_constructor_12():
    """ Simple FITS, PHU only """
    ad = AstroData(dataset=TESTPHU)
    assert isinstance(ad, AstroData)

def test_constructor_13():
    """ Simple FITS, PHU only """
    ad = AstroData(dataset=TESTPHU)
    assert len(ad) == 0

# ==============================================================================
# Slice Operator
def test_slice_1():
    """ Slice on index, <int> """
    ad = AstroData(dataset=TESTFILE)
    sub_ad = ad[1]
    assert len(sub_ad) == 1

def test_slice_2():
    """ Slice on EXTNAME <str> """
    ad = AstroData(dataset=TESTFILE)
    sub_ad = ad['SCI']
    assert len(sub_ad) == 3

def test_slice_3():
    """ Slice on tuple, (<str>, <int>) """
    ad = AstroData(dataset=TESTFILE)
    sub_ad = ad[('SCI',1)]
    assert len(sub_ad) == 1

def test_slice_4():
    """ Null slice """
    ad = AstroData(dataset=TESTFILE)
    sub_ad = ad['FOO']
    assert sub_ad is None

def test_slice_5():
    """ Bad Index """
    ad = AstroData(dataset=TESTFILE)
    with pytest.raises(IndexError):
        ad[99]

def test_slice_6():
    """ Slice on negative index """
    ad = AstroData(dataset=TESTFILE)
    assert isinstance(ad[-1], AstroData)

def test_slice_7():
    """ Bad negative index """
    ad = AstroData(dataset=TESTFILE)
    with pytest.raises(IndexError):
        ad[-5]

#   __len__
def test_len_1():
    ad = AstroData(dataset=TESTFILE)
    assert len(ad) == (len(ad.hdulist) - 1)

# ==============================================================================
# Iterator
#   Uncertain how to test that the iterator is delivering the correct,
#   i.e. next(), HDU in ad.hdulist. This test confirms that an 'ad' of
#   len 1 is delivered in every iteration.
def test_iterate_1():
    ad = AstroData(dataset=TESTFILE)
    for item in ad:
        assert isinstance(item, AstroData)
        assert len(item) == 1

# ==============================================================================
# Attributes
#   @property filename
def test_attr_filename_1():
    """ getter """
    ad = AstroData(TESTFILE)
    assert ad.filename == TESTFILE

def test_attr_filename_2():
    """ setter """
    ad = AstroData(TESTFILE)
    ad.filename = "FOO.fits"
    assert ad.filename == "FOO.fits"

#   @property data
def test_attr_data_1():
    ad = AstroData(header=header1, data=data1)
    assert isinstance(ad.data, np.ndarray)

def test_attr_data_2():
    ad = AstroData(header=header1, data=data1)
    assert ad.data.dtype == data1.dtype

def test_attr_data_3():
    ad = AstroData(header=header1, data=data1)
    assert ad.data.shape == data1.shape

def test_attr_data_4():
    ad = AstroData(header=header1, data=data1)
    null_array = ad.data - data1
    assert not null_array.all()

#   @property descriptors
#   N.B. This interface is not yet implemented.
#
#   ad.descriptors is the new property and replacement for all_descriptors()
def test_attr_descriptors_1():
    ad = AstroData(header=header1, data=data1)
    assert isinstance(ad.descriptors, dict)

#   @property header
def test_attr_header_1():
    """ getter on singleHDU """
    ad = AstroData(header=header1, data=data1)
    assert ad.header

def test_attr_header_2():
    """ setter """
    ad = AstroData(header=header1, data=data1)
    ad.header = header2
    assert ad.header == header2

def test_attr_header_3():
    """ type Okay """
    ad = AstroData(header=header1, data=data1)
    assert type(ad.header) == type(header1)

def test_attr_header_4():
    ad = AstroData(dataset=TESTFILE)
    with pytest.raises(SingleHDUMemberExcept):
        assert ad.header

def test_attr_header_5():
    ad = AstroData(header=header1, data=data1)
    assert ad.header['EXTNAME']

def test_attr_header_6():
    ad = AstroData(header=header1, data=data1)
    assert ad.header['EXTVER']

#   @property headers
#   N.B. This interface is not yet implemented.
#
#   ad.headers is the new property and replacement for the get_headers()
#   method. For now, we shall call get_headers(), which will fail on the new
#   interface.
def test_attr_headers_1():
    ad = AstroData(TESTFILE)
    assert isinstance(ad.headers, list)

def test_attr_headers_2():
    ad = AstroData(TESTFILE)
    assert len(ad.headers) == len(ad) + 1

#   @property hdulist
def test_attr_hdulist_1():
    """ getter """
    ad = AstroData(TESTFILE)
    assert ad.hdulist

def test_attr_hdulist_1():
    """ type """
    ad = AstroData(TESTFILE)
    assert type(ad.hdulist) == type(pfob)

def test_attr_hdulist_2():
    ad = AstroData(TESTFILE)
    assert len(ad.hdulist) == len(pfob)

def test_attr_hdulist_3():
    """ equivalence """
    ad = AstroData(TESTFILE)
    assert ad.hdulist[1].verify_checksum() == hdu1.verify_checksum()

def test_attr_hdulist_4():
    """ PHU Only dataset """
    ad = AstroData(TESTPHU)
    assert hasattr(ad, 'hdulist')

def test_attr_hdulist_5():
    """ PHU Only dataset """
    ad = AstroData(TESTPHU)
    assert len(ad.hdulist) == 1

#   @property phu
def test_attr_phu_1():
    """ getter """
    ad = AstroData(TESTFILE)
    assert ad.phu

def test_attr_phu_2():
    """ type """
    ad = AstroData(TESTFILE)
    assert type(ad.phu) == type(phu)

def test_attr_phu_3():
    """ setter """
    ad = AstroData(TESTFILE)
    ad.phu = phu
    assert ad.phu == phu

def test_attr_phu_4():
    """ PHU Only dataset """
    ad = AstroData(TESTPHU)
    assert hasattr(ad, 'phu')

def test_attr_phu_5():
    """ PHU Only dataset """
    ad = AstroData(TESTPHU)
    assert ad.phu == ad.hdulist[0]

# ==============================================================================
# File Operations
#   append()
def test_method_append_1():
    ad = AstroData(TESTFILE)
    initial_len = len(ad)
    ad2 = AstroData(TESTFILE2)
    ad.append(moredata=ad2, auto_number=True)
    assert len(ad) == initial_len + len(ad2)

def test_method_append_2():
    ad = AstroData(TESTFILE)
    ad2 = AstroData(TESTFILE2)
    with pytest.raises(AstroDataError):
        ad.append(moredata=ad2)

def test_method_append_3():
    ad = AstroData(TESTFILE)
    initial_len = len(ad)
    ad.append(moredata=pfob, auto_number=True)
    assert len(ad) == initial_len + len(pfob) - 1

def test_method_append_4():
    ad = AstroData(TESTFILE)
    initial_len = len(ad)
    ad.append(moredata=hdu1, auto_number=True)
    assert len(ad) == initial_len + 1

def test_method_append_5():
    ad = AstroData(TESTFILE)
    ad.append(moredata=hdu1, auto_number=True)
    assert ad.hdulist[-1] == hdu1

def test_method_append_6():
    ad = AstroData(TESTFILE)
    initial_len = len(ad)
    ad.append(header=header1, data=data1, auto_number=True)

def test_method_append_7():
    ad = AstroData(TESTFILE)
    ad.append(header=header1, data=data1, extname='TESTH', extver=1)
    assert ad[3].header['EXTNAME'] == 'TESTH'

def test_method_append_8():
    ad = AstroData(TESTFILE)
    ad.append(header=header1, data=data1, extname='TESTH', extver=2)
    assert ad[3].header['EXTVER'] == 2

def test_method_append_9():
    ad = AstroData(TESTFILE)
    ad.append(header=header1, data=data1, extname='TESTH', extver=1)
    assert ad[('TESTH', 1)]


#   close()
def test_method_close_1():
    ad = AstroData(TESTFILE)
    ad.close()
    assert not ad.hdulist

def test_method_close_2():
    ad = AstroData(TESTFILE)
    ad.close()
    with pytest.raises(AstroDataError):
        ad.append(moredata=hdu1)

def test_method_close_3():
    ad = AstroData(TESTFILE)
    ad.close()
    with pytest.raises(TypeError):
        ad[0]

#   insert()
def test_method_insert_1():
    ad = AstroData(TESTFILE)
    ad2 = AstroData(TESTFILE2)
    initial_len = len(ad)
    ad.insert(1, moredata=ad2, auto_number=True)
    assert len(ad) == initial_len + 1

def test_method_insert_2():
    ad = AstroData(TESTFILE)
    initial_len = len(ad)
    ad.insert(1, moredata=hdu1, auto_number=True)
    assert len(ad) == initial_len + 1

def test_method_insert_3():
    ad = AstroData(TESTFILE)
    initial_len = len(ad)
    ad.insert(1, moredata=hdu1, extname='TEST', auto_number=True)
    with pytest.raises(KeyError):
        assert ad[1].header['TEST']

def test_method_insert_4():
    ad = AstroData(TESTFILE)
    initial_len = len(ad)
    ad.insert(1, moredata=pfob, auto_number=True)
    assert len(ad) == initial_len + len(pfob) - 1

def test_method_insert_5():
    ad = AstroData(TESTFILE)
    del header1['EXTNAME']
    with pytest.raises(KeyError):
        ad.insert(1, header=header1, data=data1, auto_number=True)

def test_method_insert_6():
    ad = AstroData(TESTFILE)
    del header1['EXTNAME']
    ad.insert(1, header=header1, data=data1, extname="TEST",
              auto_number=True)
    assert ad[1].header.get("EXTNAME") == "TEST"

def test_method_insert_7():
    ad = AstroData(TESTFILE)
    xname = "TEST"
    xver  = 99
    del header1['EXTNAME']
    ad.insert(1, header=header1, data=data1, extname=xname,
              extver=xver, auto_number=True)
    assert ad[1].header.get("EXTNAME") == xname
    assert ad[1].header.get("EXTVER") == xver

def test_method_insert_8():
    ad = AstroData(TESTFILE)
    initial_len = len(ad)
    header1['EXTNAME'] = 'TEST'
    ad.insert(1, header=header1, data=data1, auto_number=True)
    assert ad[1].header['EXTNAME'] == 'TEST'

def test_method_insert_9():
    ad = AstroData(TESTFILE)
    initial_len = len(ad)
    header1['EXTNAME'] = "TEST"
    with pytest.raises(AstroDataError):
        ad.insert(1, header=header1, auto_number=True)

def test_method_insert_10():
    ad = AstroData(TESTFILE)
    initial_len = len(ad)
    del header1['EXTNAME']
    with pytest.raises(AstroDataError):
        ad.insert(1, data=data1, auto_number=True)

#   open()
def test_method_open_1():
    ad = AstroData()
    ad.open(TESTFILE)
    assert len(ad) > 0

def test_method_open_2():
    ad = AstroData()
    ad.open(TESTFILE)
    assert set(ad.types) == set(NULLTYPES)

#   remove()
def test_method_remove_1():
    ad = AstroData(TESTFILE)
    initial_len = len(ad)
    ad.remove(1)
    assert len(ad) == initial_len - 1

def test_method_remove_2():
    ad = AstroData(TESTFILE)
    initial_len = len(ad)
    ad.remove(('SCI', 1))
    assert len(ad) == initial_len - 1

#   store_original_name()
def test_method_store_orig_1():
    ad = AstroData(TESTFILE)
    orig_name = ad.store_original_name()
    assert orig_name == os.path.basename(TESTFILE)

def test_method_store_orig_2():
    ad = AstroData(TESTFILE)
    orig_name = ad.store_original_name()
    assert ad.phu.header['ORIGNAME']

def test_method_store_orig_3():
    ad = AstroData(TESTFILE)
    orig_name = ad.store_original_name()
    assert orig_name == ad.phu.header['ORIGNAME']

#   write()
def test_method_write_1():
    ad = AstroData(TESTFILE)
    os.remove(TESTFILE)
    ad.write()
    assert os.path.isfile(TESTFILE)

def test_method_write_2():
    ad = AstroData(TESTFILE)
    ad.write(filename="New_File.test", rename=True)
    assert os.path.isfile("New_File.test")

def test_method_write_3():
    ad = AstroData(TESTFILE)
    assert os.path.isfile("New_File.test")
    ad.write(filename="New_File.test", rename=True)
    assert os.path.isfile("New_File.test")

def test_method_write_4():
    ad = AstroData(TESTFILE)
    ad.write(filename="New_File.test", rename=True, clobber=True)
    assert os.path.isfile("New_File.test")
    os.remove("New_File.test")         # Clean it up for next run.

# ==============================================================================
#   Classifications
#
#   Attr types
def test_attr_types_1():
    ad = AstroData(TESTFILE)
    assert isinstance(ad.types, list)

def test_attr_types_2():
    ad = AstroData(TESTFILE)
    assert KNOWNTYPE in ad.types

def test_attr_types_3():
    ad = AstroData(TESTFILE)
    assert KNOWNSTAT in ad.types

def test_attr_types_4():
    ad = AstroData()
    assert set(ad.types) == set(NULLTYPES)

def test_attr_types_5():
    """ Simple FITS, PHU only """
    ad = AstroData(TESTPHU)
    assert isinstance(ad.types, list)

def test_attr_types_6():
    """ Simple FITS, PHU only """
    ad = AstroData(TESTPHU)
    assert PHUTYPE in ad.types

#   type()
def test_method_type_1():
    ad = AstroData(TESTFILE)
    assert isinstance(ad.type(), list)

def test_method_type_2():
    ad = AstroData(TESTFILE)
    assert isinstance(ad.type(prune=True), list)

def test_method_type_3():
    ad = AstroData(TESTFILE)
    assert KNOWNTYPE in ad.type()

def test_method_type_4():
    ad = AstroData(TESTFILE)
    assert KNOWNTYPE in ad.type(prune=True)

#   status()
def test_method_status_1():
    ad = AstroData(TESTFILE)
    assert isinstance(ad.status(), list)

def test_method_status_2():
    ad = AstroData(TESTFILE)
    assert isinstance(ad.status(prune=True), list)

def test_method_status_3():
    ad = AstroData(TESTFILE)
    assert KNOWNSTAT in ad.status()

def test_method_status_4():
    ad = AstroData(TESTFILE)
    assert KNOWNSTAT in ad.status(prune=True)

#   refresh_types()
def test_method_refresh_1():
    ad = AstroData(TESTFILE)
    ad.types.append("FOO")
    assert "FOO" in ad.types
    ad.refresh_types()
    assert "FOO" not in ad.types

def test_method_refresh_2():
    ad = AstroData()
    ad.open(TESTFILE)
    ad.refresh_types()
    assert KNOWNTYPE in ad.types

def test_method_refresh_3():
    ad = AstroData()
    ad.open(TESTFILE)
    ad.refresh_types()
    assert KNOWNSTAT in ad.types

# ==============================================================================
# Inspection and Modification

#   count_exts()
def test_method_count_exts_1():
    ad = AstroData(TESTFILE)
    ad_len = len(ad)
    assert ad_len == ad.count_exts()

def test_method_count_exts_2():
    ad = AstroData(TESTFILE)
    sci_exts = 3
    assert sci_exts == ad.count_exts('SCI')

def test_method_count_exts_3():
    ad = AstroData(TESTFILE2)
    sci_exts = 1
    assert sci_exts == ad.count_exts('SCI')

#   ext_index() was get_int_ext()
def test_method_ext_index_1():
    ad = AstroData(TESTFILE)
    assert isinstance(ad.ext_index(('SCI',1)), int)

def test_method_ext_index_2():
    ad = AstroData(TESTFILE)
    assert ad.ext_index(('SCI',1)) == 0

def test_method_ext_index_3():
    ad = AstroData(TESTFILE)
    assert ad.ext_index(('SCI',3)) == 2

#   rename_ext()
def test_method_rename_ext_1():    # Raise on multi-ext
    ad = AstroData(TESTFILE)
    with pytest.raises(SingleHDUMemberExcept):
        ad.rename_ext("SCI", ver=99)

def test_method_rename_ext_2():
    ad = AstroData(TESTFILE)
    with pytest.raises(SingleHDUMemberExcept):
        ad.rename_ext("FOO")

def test_method_rename_ext_3():
    ad = AstroData(TESTFILE2)      # Single 'SCI' ext
    ad.rename_ext("FOO")
    assert ad.extname() == "FOO"

def test_method_rename_ext_4():
    ad = AstroData(TESTFILE2)
    ad.rename_ext("FOO", ver=99)
    assert ad.extname() == "FOO"
    assert ad.extver() == 99

def test_method_rename_ext_5():    # TypeError w/ only 'ver=' param
    ad = AstroData(TESTFILE2)
    with pytest.raises(TypeError):
        ad.rename_ext(ver=2)

#   extname()
def test_method_extname_1():
    ad = AstroData(TESTFILE2)      # Single 'SCI' ext
    assert isinstance(ad.extname(), str)

def test_method_extname_2():
    ad = AstroData(TESTFILE2)
    assert ad.extname() == 'SCI'

def test_method_extname_3():
    ad = AstroData(TESTFILE)
    with pytest.raises(SingleHDUMemberExcept):
        ad.extname()

#   extver()
def test_method_extver_1():
    ad = AstroData(TESTFILE2)      # Single 'SCI' ext
    assert isinstance(ad.extver(), int)

def test_method_extver_2():
    ad = AstroData(TESTFILE2)
    assert ad.extver() == 1

def test_method_extver_3():
    ad = AstroData(TESTFILE)
    with pytest.raises(SingleHDUMemberExcept):
        ad.extver()

#   info()
def test_method_info_1():
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    ad = AstroData(TESTFILE)
    ad.info()
    sys.stdout = old_stdout
    assert isinstance(mystdout.getvalue(), str)

def test_method_info_2():
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    ad = AstroData(TESTFILE)
    ad.info(table=True)
    sys.stdout = old_stdout
    assert isinstance(mystdout.getvalue(), str)

def test_method_info_3():
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    ad = AstroData(TESTFILE)
    ad.info(oid=True)
    sys.stdout = old_stdout
    assert isinstance(mystdout.getvalue(), str)

def test_method_info_4():
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    ad = AstroData(TESTFILE)
    ad.info(oid=True, table=True)
    sys.stdout = old_stdout
    assert isinstance(mystdout.getvalue(), str)

#   get_key_val()
#   Using get_key_value() until refactor to new name
def test_method_get_key_val_1():
    ad = AstroData(TESTFILE2)
    assert isinstance(ad.get_key_value('EXTNAME'), str)

def test_method_get_key_val_2():
    ad = AstroData(TESTFILE2)
    assert isinstance(ad.get_key_value('BITPIX'), int)

def test_method_get_key_val_3():
    ad = AstroData(TESTFILE2)
    assert isinstance(ad.get_key_value('CRVAL1'), float)

def test_method_get_key_val_4():
    ad = AstroData(TESTFILE2)
    assert ad.get_key_value('DATATYP') is None

def test_method_get_key_val_5():
    ad = AstroData(TESTFILE2)
    assert ad.get_key_value('FOO') is None

def test_method_get_key_val_6():
    ad = AstroData(TESTFILE2)
    assert ad.get_key_value('FOO') is None

def test_method_get_key_val_7():
    ad = AstroData(TESTFILE2)
    with pytest.raises(TypeError):
        assert ad.get_key_value()

def test_method_get_key_val_8():
    ad = AstroData(TESTFILE)
    with pytest.raises(AstroDataError):
        assert ad.get_key_value('BITPIX')

#   set_key_val()
#   Using set_key_value() until refactor to new name
def test_method_set_key_val_1():
    ad = AstroData(TESTFILE2)
    ad.set_key_value('TESTKEY', 'TESTVALUE')
    assert ad.header['TESTKEY']

def test_method_set_key_val_2():
    ad = AstroData(TESTFILE2)
    ad.set_key_value('BITPIX', 999)
    assert isinstance(ad.header['BITPIX'], int)

def test_method_set_key_val_3():
    ad = AstroData(TESTFILE2)
    ad.set_key_value('CRVAL1', 1.0)
    assert isinstance(ad.header['CRVAL1'], float)

def test_method_set_key_val_4():
    ad = AstroData(TESTFILE2)
    test_comment = "This is a TESTKEY"
    ad.set_key_value('TESTKEY', 'TESTVALUE', comment=test_comment)
    assert ad.header.cards['TESTKEY'].comment.endswith(test_comment)

def test_method_set_key_val_5():
    ad = AstroData(TESTFILE2)
    with pytest.raises(TypeError):
        assert ad.set_key_value()

def test_method_set_key_val_6():
    ad = AstroData(TESTFILE2)
    with pytest.raises(TypeError):
        assert ad.set_key_value('FOO')

def test_method_set_key_val_7():
    ad = AstroData(TESTFILE2)
    with pytest.raises(AstroDataError):
        assert ad.set_key_value('FOO', None)

def test_method_set_key_val_8():
    ad = AstroData(TESTFILE)
    with pytest.raises(AstroDataError):
        assert ad.set_key_value('BITPIX', 1)

#   phu_get_key_value()
def test_method_phu_get_key_val_1():
    ad = AstroData(TESTFILE2)
    assert isinstance(ad.phu_get_key_value('SIMPLE'), bool)

def test_method_phu_get_key_val_2():
    ad = AstroData(TESTFILE2)
    assert isinstance(ad.phu_get_key_value('BITPIX'), int)

def test_method_phu_get_key_val_3():
    ad = AstroData(TESTFILE2)
    assert isinstance(ad.phu_get_key_value('RA'), float)

def test_method_phu_get_key_val_5():
    ad = AstroData(TESTFILE2)
    assert ad.phu_get_key_value('FOO') is None

def test_method_phu_get_key_val_6():
    ad = AstroData(TESTFILE2)
    with pytest.raises(TypeError):
        assert ad.phu_get_key_value()

#   phu_set_key_val()
#   Using set_key_value() until refactor to new name
def test_method_phu_set_key_val_1():
    ad = AstroData(TESTFILE2)
    test_val = 'TESTVALUE'
    ad.phu_set_key_value('TESTKEY', test_val)
    assert ad.phu.header['TESTKEY'] == test_val

def test_method_phu_set_key_val_2():
    ad = AstroData(TESTFILE2)
    ad.phu_set_key_value('BITPIX', 999)
    assert isinstance(ad.phu.header['BITPIX'], int)
    assert ad.phu.header['BITPIX'] == 999

def test_method_phu_set_key_val_3():
    ad = AstroData(TESTFILE2)
    ad.phu_set_key_value('CRVAL1', 1.0)
    assert isinstance(ad.phu.header['CRVAL1'], float)

def test_method_phu_set_key_val_4():
    ad = AstroData(TESTFILE2)
    test_comment = "This is a TESTKEY"
    ad.phu_set_key_value('TESTKEY', 'TESTVALUE', comment=test_comment)
    assert ad.phu.header.cards['TESTKEY'].comment.endswith(test_comment)

def test_method_phu_set_key_val_5():
    ad = AstroData(TESTFILE2)
    with pytest.raises(AstroDataError):
        assert ad.phu_set_key_value()

def test_method_phu_set_key_val_6():
    ad = AstroData(TESTFILE2)
    with pytest.raises(AstroDataError):
        assert ad.phu_set_key_value('FOO')

def test_method_phu_set_key_val_7():
    ad = AstroData(TESTFILE2)
    with pytest.raises(AstroDataError):
        assert ad.phu_set_key_value('FOO', None)
