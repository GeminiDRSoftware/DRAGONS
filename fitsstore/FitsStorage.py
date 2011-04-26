"""
This module contains the ORM classes for the tables in the fits storage
database.
"""
import sys

sys.path.append('/opt/gemini_python')

import sqlalchemy
import sqlalchemy.orm
import os
import datetime
import dateutil.parser
import zlib
import re

from sqlalchemy import Table, Column, MetaData, ForeignKey
from sqlalchemy import desc, func, extract
from sqlalchemy import Integer, String, Boolean, Text, DateTime, Time, Date, Numeric, BigInteger

from sqlalchemy.orm import relation, backref, join

from sqlalchemy.ext.declarative import declarative_base

import FitsVerify
import CadcCRC
import CadcWMD

from FitsStorageConfig import *

from astrodata import Errors
from astrodata.AstroData import AstroData

# This was to debug the number of open database sessions.
#import logging
#logging.basicConfig(filename='/data/autoingest/debug.log', level=logging.DEBUG)
#logging.getLogger('sqlalchemy.pool').setLevel(logging.INFO)


Base = declarative_base()

# We need to handle the database connection in here too so that the
# orm can properly handle the relations defined in the database
# at this level rather than in the main script

# Create a database engine connection to the postgres database
# and an sqlalchemy session to go with it
pg_db = sqlalchemy.create_engine(fits_database, echo = False)
sessionfactory = sqlalchemy.orm.sessionmaker(pg_db)

# Do not create the session here, these are not supposed to be global
#session = sessionfactory()


class File(Base):
  """
  This is the ORM class for the file table
  """
  __tablename__ = 'file'

  id = Column(Integer, primary_key=True)
  filename = Column(Text, nullable=False, unique=True, index=True)
  path = Column(Text)

  def __init__(self, filename, path):
    self.filename = filename
    self.path = path

  def __repr__(self):
    return "<File('%s', '%s')>" %(self.id, self.filename)

  def fullpath(self):
    fullpath = os.path.join(storage_root, self.path, self.filename)
    # print "FS74:", fullpath
    return fullpath

  def exists(self):
    exists = os.access(self.fullpath(), os.F_OK | os.R_OK)
    isfile = os.path.isfile(self.fullpath())
    return (exists and isfile)

  def size(self):
    return os.path.getsize(self.fullpath())

  def ccrc(self):
    return CadcCRC.cadcCRC(self.fullpath())

  def md5(self):
    return CadcCRC.md5sum(self.fullpath())

  def lastmod(self):
    return datetime.datetime.fromtimestamp(os.path.getmtime(self.fullpath()))

class DiskFile(Base):
  """
  This is the ORM class for the diskfile table.
  """
  __tablename__ = 'diskfile'

  id = Column(Integer, primary_key=True)
  file_id = Column(Integer, ForeignKey('file.id'), nullable=False, index=True)
  file = relation(File, order_by=id)
  present = Column(Boolean, index=True)
  canonical = Column(Boolean, index=True)
  ccrc = Column(Text)
  md5 = Column(Text)
  size = Column(Integer)
  lastmod = Column(DateTime(timezone=True), index=True)
  entrytime = Column(DateTime(timezone=True), index=True)
  isfits = Column(Boolean)
  fvwarnings = Column(Integer)
  fverrors = Column(Integer)
  wmdready = Column(Boolean)

  def __init__(self, file):
    self.file_id = file.id
    self.present = True
    self.canonical = True
    self.entrytime = datetime.datetime.now()
    self.size = file.size()
    self.ccrc = file.ccrc()
    self.md5 = file.md5()
    self.lastmod = file.lastmod()

  def __repr__(self):
    return "<DiskFile('%s', '%s')>" %(self.id, self.file_id)


class DiskFileReport(Base):
  """
  This is the ORM object for DiskFileReport.
  Contains the Fits Verify and WMD reports for a diskfile
  These can be fairly large chunks of text, so we split this
  out from the DiskFile table for DB performance reasons

  When we instantiate this class, we pass it the diskfile object.
  This class will update that diskfile object with the fverrors and wmdready
  values, but will not commit the changes.
  """
  __tablename__ = 'diskfilereport'

  id = Column(Integer, primary_key=True)
  diskfile_id = Column(Integer, ForeignKey('diskfile.id'), nullable=False, index=True)
  fvreport = Column(Text)
  wmdreport = Column(Text)


  def __init__(self, diskfile, skip_fv, skip_wmd):
    self.diskfile_id = diskfile.id
    if(skip_fv or fsc_localmode == True):
      diskfile.fverrors=0
    else:
      self.fits_verify(diskfile)
    if(skip_wmd or fsc_localmode == True):
      diskfile.wmdready = True
    else:
      self.wmd(diskfile)

  def fits_verify(self, diskfile):
    """
    Calls the FitsVerify module and records the results.
    - Populates the isfits, fverrors and fvwarnings in the diskfile object
      passed in
    - Populates the fvreport in self
    """
    list = FitsVerify.fitsverify(diskfile.file.fullpath())
    diskfile.isfits = bool(list[0])
    diskfile.fvwarnings = list[1]
    diskfile.fverrors = list[2]
    # If the FITS file has bad strings in it, fitsverify will quote them in 
    # the report, and the database will object to the bad characters in 
    # the unicode string - errors=ignore makes it ignore these.
    self.fvreport = unicode(list[3], errors='replace')

  def wmd(self, diskfile):
    """
    Calls the CadcWMD module and records the results
    - Populates the wmdready flag in the diskfile object passed in
    - Populates the wmdreport text in self
    """
    list = CadcWMD.cadcWMD(diskfile.file.fullpath())
    diskfile.wmdready = bool(list[0])
    self.wmdreport = list[1]


class Header(Base):
  """
  This is the ORM class for the Header table
  """
  __tablename__ = 'header'

  id = Column(Integer, primary_key=True)
  diskfile_id = Column(Integer, ForeignKey('diskfile.id'), nullable=False, index=True)
  diskfile = relation(DiskFile, order_by=id)
  program_id = Column(Text, index=True)
  observation_id = Column(Text, index=True)
  data_label = Column(Text, index=True)
  telescope = Column(Text)
  instrument = Column(Text, index=True)
  ut_datetime = Column(DateTime(timezone=False), index=True)
  local_time = Column(Time(timezone=False))
  observation_type = Column(Text, index=True)
  observation_class = Column(Text, index=True)
  object = Column(Text)
  ra = Column(Numeric(precision=16, scale=12))
  dec = Column(Numeric(precision=16, scale=12))
  azimuth = Column(Numeric(precision=16, scale=12))
  elevation = Column(Numeric(precision=16, scale=12))
  cass_rotator_pa = Column(Numeric(precision=16, scale=12))
  airmass = Column(Numeric(precision=8, scale=6))
  filter_name = Column(Text)
  exposure_time = Column(Numeric(precision=8, scale=4))
  disperser = Column(Text)
  central_wavelength = Column(Numeric(precision=8, scale=6))
  focal_plane_mask = Column(Text)
  spectroscopy = Column(Boolean)
  adaptive_optics = Column(Boolean)
  raw_iq = Column(Text)
  raw_cc = Column(Text)
  raw_wv = Column(Text)
  raw_bg = Column(Text)
  qa_state = Column(Text)
  release = Column(Date(TimeZone=False))
  reduction = Column(Text)

  def __init__(self, diskfile):
    self.diskfile_id = diskfile.id
    self.populate_fits(diskfile)

  def __repr__(self):
    return "<Header('%s', '%s')>" %(self.id, self.diskfile_id)

  def populate_fits(self, diskfile):
    """
    Populates header table values from the FITS headers of the file.
    Uses the AstroData object to access the file.
    """
    fullpath = diskfile.file.fullpath()
    # Try and open it as a fits file
    ad=0
    try:
      ad=AstroData(fullpath, mode='readonly')
      ad.descriptorFormat = "db"

      # Basic data identification part
      try:
        self.program_id = ad.program_id().forDB()
      except (KeyError, ValueError, Errors.InvalidValueError, Errors.EmptyKeyError):
        pass
      try:
        self.observation_id = ad.observation_id().forDB()
      except (KeyError, ValueError, Errors.InvalidValueError, Errors.EmptyKeyError):
        pass
      try:
        self.data_label = ad.data_label().forDB()
      except (KeyError, ValueError, Errors.InvalidValueError, Errors.EmptyKeyError):
        pass
      try:
        self.telescope = ad.telescope().forDB()
      except (KeyError, ValueError, Errors.InvalidValueError, Errors.EmptyKeyError):
        pass
      try:
        self.instrument = ad.instrument().forDB()
      except (KeyError, ValueError, Errors.InvalidValueError, Errors.EmptyKeyError):
        pass

      # Date and times part
      try:
        self.ut_datetime = ad.ut_datetime().forDB()
      except:
        raise

      try:
        localtime_string = ad.local_time().forDB()
        if(localtime_string):
          # This is a bit of a hack so as to use the nice parser
          self.local_time = dateutil.parser.parse("2000-01-01 %s" % (localtime_string)).time()
      except (KeyError, ValueError, Errors.InvalidValueError, Errors.EmptyKeyError):
        pass

      # Data Types
      try:
        self.observation_type = ad.observation_type().forDB()
        if('GNIRS_PINHOLE' in ad.types):
          self.observation_type='PINHOLE'
        if('NIFS_RONCHI' in ad.types):
          self.observation_type='RONCHI'
      except (KeyError, ValueError, Errors.InvalidValueError, Errors.EmptyKeyError):
        pass
      try:
        self.observation_class = ad.observation_class().forDB()
      except (KeyError, ValueError, Errors.InvalidValueError, Errors.EmptyKeyError):
        pass
      try:
        self.object = ad.object().forDB()
      except (KeyError, ValueError, Errors.InvalidValueError, Errors.EmptyKeyError):
        pass
      try:
        self.ra = ad.ra().forDB()
      except (KeyError, ValueError, Errors.InvalidValueError, Errors.EmptyKeyError):
        pass
      try:
        self.dec = ad.dec().forDB()
      except (KeyError, ValueError, Errors.InvalidValueError, Errors.EmptyKeyError):
        pass
      try:
        self.azimuth = ad.azimuth().forDB()
      except (KeyError, ValueError, Errors.InvalidValueError, Errors.EmptyKeyError):
        pass
      try:
        self.elevation = ad.elevation().forDB()
      except (KeyError, ValueError, Errors.InvalidValueError, Errors.EmptyKeyError):
        pass
      try:
        self.cass_rotator_pa = ad.cass_rotator_pa().forDB()
      except (KeyError, ValueError, Errors.InvalidValueError, Errors.EmptyKeyError):
        pass
      try:
        self.airmass = ad.airmass().forDB()
      except (KeyError, ValueError, Errors.InvalidValueError, Errors.EmptyKeyError):
        pass
      try:
        self.raw_iq = ad.raw_iq().forDB()
      except (KeyError, ValueError, Errors.InvalidValueError, Errors.EmptyKeyError):
        pass
      try:
        self.raw_cc = ad.raw_cc().forDB()
      except (KeyError, ValueError, Errors.InvalidValueError, Errors.EmptyKeyError):
        pass
      try:
        self.raw_wv = ad.raw_wv().forDB()
      except (KeyError, ValueError, Errors.InvalidValueError, Errors.EmptyKeyError):
        pass
      try:
        self.raw_bg = ad.raw_bg().forDB()
      except (KeyError, ValueError, Errors.InvalidValueError, Errors.EmptyKeyError):
        pass
      try:
        self.filter_name = ad.filter_name(pretty=True).forDB()
      except (KeyError, ValueError, Errors.InvalidValueError, Errors.EmptyKeyError):
        pass
      try:
        self.exposure_time = ad.exposure_time().forDB()
      except (KeyError, ValueError, Errors.InvalidValueError, Errors.EmptyKeyError):
        pass
      try:
        self.disperser = ad.disperser(pretty=True).forDB()
      except (KeyError, ValueError, Errors.InvalidValueError, Errors.EmptyKeyError):
        pass
      try:
        self.central_wavelength = ad.central_wavelength(asMicrometers=True).forDB()
      except (KeyError, ValueError, Errors.InvalidValueError, Errors.EmptyKeyError, Errors.DescriptorTypeError):
        pass
      try:
        self.focal_plane_mask = ad.focal_plane_mask(pretty=True).forDB()
      except (KeyError, ValueError, Errors.InvalidValueError, Errors.EmptyKeyError):
        pass

      # Hack the AO header for now
      aofold = ad.phuHeader('AOFOLD')
      self.adaptive_optics = (aofold == 'IN')

      # And the Spectroscopy header
      self.spectroscopy = False
      if('SPECT' in ad.types):
        self.spectroscopy = True

      # Set the derived QA state and release date
      try:
        self.qa_state = ad.qa_state().forDB()
      except (KeyError, ValueError, Errors.InvalidValueError, Errors.EmptyKeyError, AttributeError):
        pass
      try:
        reldatestring = ad.phuHeader('RELEASE')
        if(reldatestring):
          reldts = "%s 00:00:00" % reldatestring
          self.release = dateutil.parser.parse(reldts).date()
      except (KeyError, ValueError, Errors.InvalidValueError, Errors.EmptyKeyError):
        pass


      # Set the reduction state
      self.reduction = 'RAW'
      if('PREPARED' in ad.types):
        self.reduction = 'PREPARED'
      if('PROCESSED_FLAT' in ad.types):
        self.reduction = 'PROCESSED_FLAT'
      if('PROCESSED_BIAS' in ad.types):
        self.reduction = 'PROCESSED_BIAS'

      ad.close()
    except:
      # Astrodata open failed or there was some other exception
      ad.close()
      raise

class FullTextHeader(Base):
  """
  This is the ORM object for the Full Text of the header.
  We keep this is a separate table from Header to improve DB performance
  """
  __tablename__ = 'fulltextheader'

  id = Column(Integer, primary_key=True)
  diskfile_id = Column(Integer, ForeignKey('diskfile.id'), nullable=False, index=True)
  fulltext = Column(Text)

  def __init__(self, diskfile):
    self.diskfile_id = diskfile.id
    self.populate(diskfile)

  def populate(self, diskfile):
    fullpath = diskfile.file.fullpath()
    # Try and open it as a fits file
    ad=0
    try:
      ad=AstroData(fullpath, mode='readonly')
      self.fulltext = ""
      self.fulltext += "Full Path Filename: " +  diskfile.file.fullpath() + "\n\n"
      self.fulltext += "AstroData Types: " +str(ad.types) + "\n\n"
      for i in range(len(ad.hdulist)):
        self.fulltext += "\n--- HDU %s ---\n" % i
        self.fulltext += unicode(str(ad.hdulist[i].header.ascardlist()), errors='replace')
        self.fulltext += '\n'
      ad.close()

    except:
      # Astrodata open failed or there was some other exception
      ad.close()
      raise


class IngestQueue(Base):
  """
  This is the ORM object for the IngestQueue table
  """
  __tablename__ = 'ingestqueue'

  id = Column(Integer, primary_key=True)
  filename = Column(Text, nullable=False, unique=False, index=True)
  path = Column(Text)
  inprogress = Column(Boolean, index=True)
  added = Column(DateTime)

  def __init__(self, filename, path):
    self.filename = filename
    self.path = path
    self.added = datetime.datetime.now()
    self.inprogress = False

  def __repr__(self):
    return "<IngestQueue('%s', '%s')>" %(self.id, self.filename)

class Tape(Base):
  """
  This is the ORM object for the Tape table
  Each row in this table represents a data tape
  """
  __tablename__ = 'tape'

  id = Column(Integer, primary_key=True)
  label = Column(Text, nullable=False, index=True)
  firstwrite = Column(DateTime(timezone=False))
  lastwrite = Column(DateTime(timezone=False))
  lastverified = Column(DateTime(timezone=False))
  location = Column(Text)
  lastmoved = Column(DateTime(timezone=False))
  active = Column(Boolean, index=True)
  full = Column(Boolean, index=True)
  set = Column(Integer, index=True)
  fate = Column(Text)

  def __init__(self, label):
    self.label = label
    self.active = True
    self.full = False
    self.set = 0

class TapeWrite(Base):
  """
  This is the ORM object for the TapeWrite table
  Each row in this table represents a tape writing session
  """

  __tablename__ = 'tapewrite'

  id = Column(Integer, primary_key=True)
  tape_id = Column(Integer, ForeignKey('tape.id'), nullable=False, index=True)
  tape = relation(Tape, order_by=id)
  filenum = Column(Integer, index=True)
  startdate = Column(DateTime(timezone=False))
  enddate = Column(DateTime(timezone=False))
  suceeded = Column(Boolean, index=True)
  size = Column(BigInteger)
  beforestatus = Column(Text)
  afterstatus = Column(Text)
  hostname = Column(Text)
  tapedrive = Column(Text)
  notes = Column(Text)

  
class TapeFile(Base):
  """
  This is the ORM object for the TapeFile table
  """
  __tablename__ = 'tapefile'

  id = Column(Integer, primary_key=True)
  tapewrite_id = Column(Integer, ForeignKey('tapewrite.id'), nullable=False, index=True)
  tapewrite = relation(TapeWrite, order_by=id)
  filename = Column(Text, index=True)
  size = Column(Integer, index=True)
  ccrc = Column(Text)
  md5 = Column(Text, index=True)
  lastmod = Column(DateTime(timezone=True))

class TapeRead(Base):
  """
  This is the ORM object for the TapeRead table
  """
  __tablename__ = 'taperead'

  id = Column(Integer, primary_key=True)
  tapefile_id = Column(Integer, ForeignKey('tapefile.id'), nullable=False, index=True)
  tapefile = relation(TapeFile, order_by=id)
  filename = Column(Text, index=True)
  md5 = Column(Text, index=True)
  tape_id = Column(Integer, index=True)
  tape_label = Column(Text, index=True)
  filenum = Column(Integer, index=True)
  requester = Column(Text)

class Gmos(Base):
  """
  This is the ORM object for the GMOS details.
  This is used for both GMOS-N and GMOS-S
  """
  __tablename__ = 'gmos'

  id = Column(Integer, primary_key=True)
  header_id = Column(Integer, ForeignKey('header.id'), nullable=False, index=True)
  header = relation(Header, order_by=id)
  disperser = Column(Text, index=True)
  filter_name = Column(Text, index=True)
  detector_x_bin = Column(Integer, index=True)
  detector_y_bin = Column(Integer, index=True)
  amp_read_area = Column(Text, index=True)
  read_speed_setting = Column(Text, index=True)
  gain_setting = Column(Text, index=True)
  focal_plane_mask = Column(Text, index=True)
  nodandshuffle = Column(Boolean, index=True)
  nod_count = Column(Integer, index=True)
  nod_pixels = Column(Integer, index=True)

  def __init__(self, header):
    self.header = header

    # Populate from the astrodata object
    self.populate()

  def populate(self):
    # Get an AstroData object on it
    try:
      ad = AstroData(self.header.diskfile.file.fullpath(), mode="readonly")
      ad.descriptorFormat = "db"
      # Populate values
      try:
        self.disperser = ad.disperser().forDB()
      except (KeyError, ValueError, Errors.InvalidValueError, Errors.EmptyKeyError):
        pass
      try:
        self.filter_name = ad.filter_name().forDB()
      except (KeyError, ValueError, Errors.InvalidValueError, Errors.EmptyKeyError):
        pass
      try:
        self.detector_x_bin = ad.detector_x_bin().forDB()
      except (KeyError, ValueError, Errors.InvalidValueError, Errors.EmptyKeyError, IndexError):
        pass
      try:
        self.detector_y_bin = ad.detector_y_bin().forDB()
      except (KeyError, ValueError, Errors.InvalidValueError, Errors.EmptyKeyError, IndexError):
        pass
      try:
        self.amp_read_area = ad.amp_read_area().forDB()
      except (KeyError, ValueError, Errors.InvalidValueError, Errors.EmptyKeyError, IndexError):
        pass
      try:
        self.read_speed_setting = ad.read_speed_setting().forDB()
      except (KeyError, ValueError, Errors.InvalidValueError, Errors.EmptyKeyError, IndexError):
        pass
      try:
        self.gain_setting = ad.gain_setting().forDB()
      except (KeyError, ValueError, Errors.InvalidValueError, Errors.EmptyKeyError, IndexError):
        pass
      try:
        self.focal_plane_mask = ad.focal_plane_mask().forDB()
      except (KeyError, ValueError, Errors.InvalidValueError, Errors.EmptyKeyError):
        pass
      try:
        self.nodandshuffle = ad.isType('GMOS_NODANDSHUFFLE')
      except (KeyError, ValueError, Errors.InvalidValueError, Errors.EmptyKeyError):
        pass
      if(self.nodandshuffle):
        try:
          self.nod_count = ad.nod_count().forDB()
        except (KeyError, ValueError, Errors.InvalidValueError, Errors.EmptyKeyError, Errors.DescriptorTypeError):
          pass
        try:
          self.nod_pixels = ad.nod_pixels().forDB()
        except (KeyError, ValueError, Errors.InvalidValueError, Errors.EmptyKeyError, Errors.DescriptorTypeError):
          pass
      ad.close()
    except:
      # Astrodata open failed or there was some other exception
      ad.close()
      raise

class Niri(Base):
  """
  This is the ORM object for the NIRI details
  """
  __tablename__ = 'niri'

  id = Column(Integer, primary_key=True)
  header_id = Column(Integer, ForeignKey('header.id'), nullable=False, index=True)
  header = relation(Header, order_by=id)
  disperser = Column(Text, index=True)
  filter_name = Column(Text, index=True)
  read_mode = Column(Text, index=True)
  well_depth_setting = Column(Text, index=True)
  data_section = Column(Text, index=True)
  coadds = Column(Integer, index=True)
  camera = Column(Text, index=True)
  focal_plane_mask = Column(Text)

  def __init__(self, header):
    self.header = header

    # Populate from an astrodata object
    self.populate()

  def populate(self):
    # Get an AstroData object on it
    try:
      ad = AstroData(self.header.diskfile.file.fullpath(), mode="readonly")
      ad.descriptorFormat = "db"
      # Populate values
      try:
        self.disperser = ad.disperser().forDB()
      except (KeyError, ValueError, Errors.InvalidValueError, Errors.EmptyKeyError):
        pass
      try:
        self.filter_name = ad.filter_name().forDB()
      except (KeyError, ValueError, Errors.InvalidValueError, Errors.EmptyKeyError):
        pass
      try:
        self.read_mode = ad.read_mode().forDB()
      except (KeyError, ValueError, Errors.InvalidValueError, Errors.EmptyKeyError):
        pass
      try:
        self.well_depth_setting = ad.well_depth_setting().forDB()
      except (KeyError, ValueError, Errors.InvalidValueError, Errors.EmptyKeyError):
        pass
      try:
        # the str() is a temp workaround 20110404 PH
        self.data_section = str(ad.data_section().forDB())
      except (KeyError, ValueError, Errors.InvalidValueError, Errors.EmptyKeyError, IndexError):
        pass
      try:
        self.coadds = ad.coadds().forDB()
      except (KeyError, ValueError, Errors.InvalidValueError, Errors.EmptyKeyError):
        pass
      try:
        self.camera = ad.camera().forDB()
      except (KeyError, ValueError, Errors.InvalidValueError, Errors.EmptyKeyError):
        pass
      try:
        self.focal_plane_mask = ad.focal_plane_mask().forDB()
      except (KeyError, ValueError, Errors.InvalidValueError, Errors.EmptyKeyError):
        pass
      ad.close()
    except:
      # Astrodata open failed or there was some other exception
      ad.close()
      raise

class Gnirs(Base):
  """
  This is the ORM object for the GNIRS details
  """
  __tablename__ = 'gnirs'

  id = Column(Integer, primary_key=True)
  header_id = Column(Integer, ForeignKey('header.id'), nullable=False, index=True)
  header = relation(Header, order_by=id)
  disperser = Column(Text, index=True)
  filter_name = Column(Text, index=True)
  read_mode = Column(Text, index=True)
  well_depth_setting = Column(Text, index=True)
  coadds = Column(Integer, index=True)
  camera = Column(Text, index=True)
  focal_plane_mask = Column(Text)

  def __init__(self, header):
    self.header = header

    # Populate from an astrodata object
    self.populate()

  def populate(self):
    # Get an AstroData object on it
    try:
      ad = AstroData(self.header.diskfile.file.fullpath(), mode="readonly")
      ad.descriptorFormat = "db"
      # Populate values
      try:
        self.disperser = ad.disperser().forDB()
      except (KeyError, ValueError, Errors.InvalidValueError, Errors.EmptyKeyError):
        pass
      try:
        self.filter_name = ad.filter_name().forDB()
      except (KeyError, ValueError, Errors.InvalidValueError, Errors.EmptyKeyError):
        pass
      try:
        self.read_mode = ad.read_mode().forDB()
      except (KeyError, ValueError, Errors.InvalidValueError, Errors.EmptyKeyError):
        pass
      try:
        self.well_depth_setting = ad.well_depth_setting().forDB()
      except (KeyError, ValueError, Errors.InvalidValueError, Errors.EmptyKeyError):
        pass
      try:
        self.coadds = ad.coadds().forDB()
      except (KeyError, ValueError, Errors.InvalidValueError, Errors.EmptyKeyError):
        pass
      try:
        self.camera = ad.camera().forDB()
      except (KeyError, ValueError, Errors.InvalidValueError, Errors.EmptyKeyError):
        pass
      try:
        self.focal_plane_mask = ad.focal_plane_mask().forDB()
      except (KeyError, ValueError, Errors.InvalidValueError, Errors.EmptyKeyError):
        pass
      ad.close()
    except:
      # Astrodata open failed or there was some other exception
      ad.close()
      raise

class Nifs(Base):
  """
  This is the ORM object for the NIFS details
  """
  __tablename__ = 'nifs'

  id = Column(Integer, primary_key=True)
  header_id = Column(Integer, ForeignKey('header.id'), nullable=False, index=True)
  header = relation(Header, order_by=id)
  disperser = Column(Text, index=True)
  filter_name = Column(Text, index=True)
  read_mode = Column(Text, index=True)
  coadds = Column(Integer, index=True)
  focal_plane_mask = Column(Text)
  
  def __init__(self, header):
    self.header = header

    # Populate from an astrodata object
    self.populate()

  def populate(self):
    # Get an AstroData object on it
    try:
      ad = AstroData(self.header.diskfile.file.fullpath(), mode="readonly")
      ad.descriptorFormat = "db"
      # Populate values
      try:
        self.disperser = ad.disperser().forDB()
      except (KeyError, ValueError, Errors.InvalidValueError, Errors.EmptyKeyError):
        pass
      try:
        self.filter_name = ad.filter_name().forDB()
      except (KeyError, ValueError, Errors.InvalidValueError, Errors.EmptyKeyError):
        pass
      try:
        self.read_mode = ad.read_mode().forDB()
      except (KeyError, ValueError, Errors.InvalidValueError, Errors.EmptyKeyError):
        pass
      try:
        self.coadds = ad.coadds().forDB()
      except (KeyError, ValueError, Errors.InvalidValueError, Errors.EmptyKeyError):
        pass
      try:
        self.focal_plane_mask = ad.focal_plane_mask().forDB()
      except (KeyError, ValueError, Errors.InvalidValueError, Errors.EmptyKeyError):
        pass
      ad.close()
    except:
      # Astrodata open failed or there was some other exception
      ad.close()
      raise

class Michelle(Base):
  """
  This is the ORM object for the MICHELLE details
  """
  __tablename__ = 'michelle'

  id = Column(Integer, primary_key=True)
  header_id = Column(Integer, ForeignKey('header.id'), nullable=False, index=True)
  header = relation(Header, order_by=id)
  disperser = Column(Text, index=True)
  filter_name = Column(Text, index=True)
  read_mode = Column(Text, index=True)
  coadds = Column(Integer, index=True)
  focal_plane_mask = Column(Text)
  
  def __init__(self, header):
    self.header = header

    # Populate from an astrodata object
    self.populate()

  def populate(self):
    # Get an AstroData object on it
    try:
      ad = AstroData(self.header.diskfile.file.fullpath(), mode="readonly")
      ad.descriptorFormat = "db"
      # Populate values
      try:
        self.disperser = ad.disperser().forDB()
      except (KeyError, ValueError, Errors.InvalidValueError, Errors.EmptyKeyError):
        pass
      try:
        self.filter_name = ad.filter_name().forDB()
      except (KeyError, ValueError, Errors.InvalidValueError, Errors.EmptyKeyError):
        pass
      try:
        self.read_mode = ad.read_mode().forDB()
      except (KeyError, ValueError, Errors.InvalidValueError, Errors.EmptyKeyError):
        pass
      try:
        self.coadds = ad.coadds().forDB()
      except (KeyError, ValueError, Errors.InvalidValueError, Errors.EmptyKeyError):
        pass
      try:
        self.focal_plane_mask = ad.focal_plane_mask().forDB()
      except (KeyError, ValueError, Errors.InvalidValueError, Errors.EmptyKeyError):
        pass
      ad.close()
    except:
      # Astrodata open failed or there was some other exception
      ad.close()
      raise

class PhotStandard(Base):
  """
  This is the ORM class for the table holding the standard star list for the instrument monitoring
  """
  __tablename__ = 'standards'

  id = Column(Integer, primary_key=True)
  name = Column(Text)
  field = Column(Text)
  ra = Column(Numeric(precision=16, scale=12), index=True)
  dec = Column(Numeric(precision=16, scale=12), index=True)
  u_mag = Column(Numeric(precision=6, scale=4))
  v_mag = Column(Numeric(precision=6, scale=4))
  g_mag = Column(Numeric(precision=6, scale=4))
  r_mag = Column(Numeric(precision=6, scale=4))
  i_mag = Column(Numeric(precision=6, scale=4))
  z_mag = Column(Numeric(precision=6, scale=4))
  y_mag = Column(Numeric(precision=6, scale=4))
  j_mag = Column(Numeric(precision=6, scale=4))
  h_mag = Column(Numeric(precision=6, scale=4))
  k_mag = Column(Numeric(precision=6, scale=4))
  lprime_mag = Column(Numeric(precision=6, scale=4))
  m_mag = Column(Numeric(precision=6, scale=4))

class Notification(Base):
  """
  This is the ORM class for the table holding the email notification list for this server.
  """
  __tablename__ = 'notification'

  id = Column(Integer, primary_key=True)
  label = Column(Text)
  selection = Column(Text)
  to = Column(Text)
  cc = Column(Text)
  internal = Column(Boolean)

  def __init__(self, label):
    self.label = label
