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
  progid = Column(Text, index=True)
  obsid = Column(Text, index=True)
  datalab = Column(Text, index=True)
  telescope = Column(Text)
  instrument = Column(Text, index=True)
  utdatetime = Column(DateTime(timezone=False), index=True)
  localtime = Column(Time(timezone=False))
  obstype = Column(Text, index=True)
  obsclass = Column(Text, index=True)
  object = Column(Text)
  ra = Column(Numeric(precision=16, scale=12))
  dec = Column(Numeric(precision=16, scale=12))
  azimuth = Column(Numeric(precision=16, scale=12))
  elevation = Column(Numeric(precision=16, scale=12))
  crpa = Column(Numeric(precision=16, scale=12))
  airmass = Column(Numeric(precision=8, scale=6))
  filter = Column(Text)
  exptime = Column(Numeric(precision=8, scale=4))
  disperser = Column(Text)
  cwave = Column(Numeric(precision=8, scale=6))
  fpmask = Column(Text)
  spectroscopy = Column(Boolean)
  adaptive_optics = Column(Boolean)
  rawiq = Column(Text)
  rawcc = Column(Text)
  rawwv = Column(Text)
  rawbg = Column(Text)
  qastate = Column(Text)
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

      # Basic data identification part
      try:
        self.progid = ad.program_id()
      except (KeyError, ValueError):
        pass
      try:
        self.obsid = ad.observation_id()
      except (KeyError, ValueError):
        pass
      try:
        self.datalab = ad.data_label()
      except (KeyError, ValueError):
        pass
      try:
        self.telescope = ad.telescope()
      except (KeyError, ValueError):
        pass
      try:
        self.instrument = ad.instrument()
      except (KeyError, ValueError):
        pass

      # Date and times part
      try:
        self.utdatetime = ad.ut_datetime()
      except:
        raise

      try:
        localtime_string = ad.local_time()
        if(localtime_string):
          # This is a bit of a hack so as to use the nice parser
          self.localtime = dateutil.parser.parse("2000-01-01 %s" % (localtime_string)).time()
      except (KeyError, ValueError):
        pass

      # Data Types
      try:
        self.obstype = ad.observation_type()
        if('GNIRS_PINHOLE' in ad.types):
          self.obstype='PINHOLE'
        if('NIFS_RONCHI' in ad.types):
          self.obstype='RONCHI'
      except (KeyError, ValueError):
        pass
      try:
        self.obsclass = ad.observation_class()
      except (KeyError, ValueError):
        pass
      try:
        self.object = ad.object()
      except (KeyError, ValueError):
        pass
      try:
        self.ra = float(ad.ra())
      except (KeyError, ValueError, TypeError):
        pass
      try:
        self.dec = float(ad.dec())
      except (KeyError, ValueError, TypeError):
        pass
      try:
        self.azimuth = float(ad.azimuth())
      except (KeyError, ValueError, TypeError):
        pass
      try:
        self.elevation = float(ad.elevation())
      except (KeyError, ValueError, TypeError):
        pass
      try:
        self.crpa = float(ad.cass_rotator_pa())
      except (KeyError, ValueError, TypeError):
        pass
      try:
        self.airmass = float(ad.airmass())
      except (KeyError, ValueError, TypeError):
        pass
      try:
        self.rawiq = ad.raw_iq()
      except (KeyError, ValueError):
        pass
      try:
        self.rawcc = ad.raw_cc()
      except (KeyError, ValueError):
        pass
      try:
        self.rawwv = ad.raw_wv()
      except (KeyError, ValueError):
        pass
      try:
        self.rawbg = ad.raw_bg()
      except (KeyError, ValueError):
        pass
      try:
        self.filter = ad.filter_name(pretty=True)
      except (KeyError, ValueError):
        pass
      try:
        self.exptime = float(ad.exposure_time())
      except (KeyError, ValueError, TypeError):
        pass
      try:
        self.disperser = ad.disperser(pretty=True)
      except (KeyError, ValueError):
        pass
      try:
        self.cwave = float(ad.central_wavelength(asMicrometers=True))
      except (KeyError, ValueError, TypeError):
        pass
      try:
        self.fpmask = ad.focal_plane_mask(pretty=True)
      except (KeyError, ValueError):
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
        self.qastate = ad.qa_state()
      except (KeyError, ValueError, AttributeError):
        pass
      try:
        reldatestring = ad.phuHeader('RELEASE')
        if(reldatestring):
          reldts = "%s 00:00:00" % reldatestring
          self.release = dateutil.parser.parse(reldts).date()
      except (KeyError, ValueError):
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
      # Astrodata open or any of the above failed
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

    except:
      # Astrodata open or header reference failed
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
  active = Column(Boolean)
  full = Column(Boolean)
  set = Column(Integer)
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
  filenum = Column(Integer)
  startdate = Column(DateTime(timezone=False))
  enddate = Column(DateTime(timezone=False))
  suceeded = Column(Boolean)
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
  tapewrite_id = Column(Integer, ForeignKey('tapewrite.id'), nullable=False)
  tapewrite = relation(TapeWrite, order_by=id)
  filename = Column(Text)
  size = Column(Integer)
  ccrc = Column(Text)
  md5 = Column(Text)
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
  filtername = Column(Text, index=True)
  xccdbin = Column(Integer, index=True)
  yccdbin = Column(Integer, index=True)
  amproa = Column(Text, index=True)
  readspeedmode = Column(Text, index=True)
  gainmode = Column(Text, index=True)
  fpmask = Column(Text, index=True)
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
      # Populate values
      try:
        self.disperser = ad.disperser()
      except (KeyError, ValueError):
        pass
      try:
        self.filtername = ad.filter_name()
      except (KeyError, ValueError):
        pass
      try:
        self.xccdbin = ad.detector_x_bin()
      except (KeyError, IndexError, ValueError):
        pass
      try:
        self.yccdbin = ad.detector_y_bin()
      except (KeyError, IndexError, ValueError):
        pass
      try:
        self.amproa = str(ad.amp_read_area(asList=True))
      except (KeyError, IndexError, ValueError):
        pass
      try:
        self.readspeedmode = ad.read_speed_setting()
      except (KeyError, IndexError, ValueError):
        pass
      try:
        self.gainmode = ad.gain_setting()
      except (KeyError, IndexError, ValueError):
        pass
      try:
        self.fpmask = ad.focal_plane_mask()
      except (KeyError, ValueError):
        pass
      try:
        self.nod_count = ad.nod_count()
      except (KeyError, ValueError):
        pass
      try:
        self.nod_pixels = ad.nod_pixels()
      except (KeyError, ValueError):
        pass
      try:
        self.nodandshuffle = ad.isType('GMOS_NODANDSHUFFLE')
      except (KeyError, ValueError):
        pass
      ad.close()
    except:
      # Astrodata open failed
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
  filtername = Column(Text, index=True)
  readmode = Column(Text, index=True)
  welldepthmode = Column(Text, index=True)
  detsec = Column(Text, index=True)
  coadds = Column(Integer, index=True)
  camera = Column(Text, index=True)
  fpmask = Column(Text)

  def __init__(self, header):
    self.header = header

    # Populate from an astrodata object
    self.populate()

  def populate(self):
    # Get an AstroData object on it
    try:
      ad = AstroData(self.header.diskfile.file.fullpath(), mode="readonly")
      # Populate values
      try:
        self.disperser = ad.disperser()
      except (KeyError, ValueError):
        pass
      try:
        self.filtername = ad.filter_name()
      except (KeyError, ValueError):
        pass
      try:
        self.readmode = ad.read_mode()
      except (KeyError, ValueError):
        pass
      try:
        self.welldepthmode = ad.well_depth_setting()
      except (KeyError, ValueError):
        pass
      try:
        self.detsec = ad.data_section()
      except (KeyError, ValueError):
        pass
      try:
        self.coadds = ad.coadds()
      except (KeyError, ValueError):
        pass
      try:
        self.camera = ad.camera()
      except (KeyError, ValueError):
        pass
      try:
        self.fpmask = ad.focal_plane_mask()
      except (KeyError, ValueError):
        pass
      ad.close()
    except:
      # Astrodata open failed
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
  filtername = Column(Text, index=True)
  readmode = Column(Text, index=True)
  welldepthmode = Column(Text, index=True)
  coadds = Column(Integer, index=True)
  camera = Column(Text, index=True)
  fpmask = Column(Text)
  
  def __init__(self, header):
    self.header = header

    # Populate from an astrodata object
    self.populate()

  def populate(self):
    # Get an AstroData object on it
    try:
      ad = AstroData(self.header.diskfile.file.fullpath(), mode="readonly")
      # Populate values
      try:
        self.disperser = ad.disperser()
      except (KeyError, ValueError):
        pass
      try:
        self.filtername = ad.filter_name()
      except (KeyError, ValueError):
        pass
      try:
        self.readmode = ad.read_mode()
      except (KeyError, ValueError):
        pass
      try:
        self.welldepthmode = ad.well_depth_setting()
      except (KeyError, ValueError):
        pass
      try:
        self.coadds = ad.coadds()
      except (KeyError, ValueError):
        pass
      try:
        self.camera = ad.camera()
      except (KeyError, ValueError):
        pass
      try:
        self.fpmask = ad.focal_plane_mask()
      except (KeyError, ValueError):
        pass
      ad.close()
    except:
      # Astrodata open failed
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
  filtername = Column(Text, index=True)
  readmode = Column(Text, index=True)
  coadds = Column(Integer, index=True)
  fpmask = Column(Text)
  
  def __init__(self, header):
    self.header = header

    # Populate from an astrodata object
    self.populate()

  def populate(self):
    # Get an AstroData object on it
    try:
      ad = AstroData(self.header.diskfile.file.fullpath(), mode="readonly")
      # Populate values
      try:
        self.disperser = ad.disperser()
      except (KeyError, ValueError):
        pass
      try:
        self.filtername = ad.filter_name()
      except (KeyError, ValueError):
        pass
      try:
        self.readmode = ad.read_mode()
      except (KeyError, ValueError):
        pass
      try:
        self.coadds = ad.coadds()
      except (KeyError, ValueError):
        pass
      try:
        self.fpmask = ad.focal_plane_mask()
      except (KeyError, ValueError):
        pass
      ad.close()
    except:
      # Astrodata open failed
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
  filtername = Column(Text, index=True)
  readmode = Column(Text, index=True)
  coadds = Column(Integer, index=True)
  fpmask = Column(Text)
  
  def __init__(self, header):
    self.header = header

    # Populate from an astrodata object
    self.populate()

  def populate(self):
    # Get an AstroData object on it
    try:
      ad = AstroData(self.header.diskfile.file.fullpath(), mode="readonly")
      # Populate values
      try:
        self.disperser = ad.disperser()
      except (KeyError, ValueError):
        pass
      try:
        self.filtername = ad.filter_name()
      except (KeyError, ValueError):
        pass
      try:
        self.readmode = ad.read_mode()
      except (KeyError, ValueError):
        pass
      try:
        self.coadds = ad.coadds()
      except (KeyError, ValueError):
        pass
      try:
        self.fpmask = ad.focal_plane_mask()
      except (KeyError, ValueError):
        pass
      ad.close()
    except:
      # Astrodata open failed
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
