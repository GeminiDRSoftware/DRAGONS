"""
This module provides calibration handling
"""

from FitsStorage import *
import FitsStorageConfig
import GeminiMetadataUtils

def get_cal_object(session, filename, header=None, descriptors=None, types=None):
  """
  This function returns an appropriate calibration object for the given dataset
  Need to pass in a sqlalchemy session that should already be open, the class will not close it
  Also pass either a filename or a header object instance
  """

  # Did we get a header?
  if(header==None and descriptors==None):
    # Get the header object from the filename
    query = session.query(Header).select_from(join(Header, join(DiskFile, File))).filter(File.filename==filename).order_by(desc(DiskFile.lastmod)).limit(1)
    header = query.first()

  # OK, now instantiate the appropriate Calibration object and return it
  c = None
  if(header):
    instrument = header.instrument
  else:
    instrument = descriptors['instrument']
  if('GMOS' in instrument):
    c = CalibrationGMOS(session, header, descriptors, types)
  if(instrument == 'NIRI'):
    c = CalibrationNIRI(session, header, descriptors, types)
  if(instrument == 'GNIRS'):
    c = CalibrationGNIRS(session, header, descriptors, types)
  if(instrument == 'NIFS'):
    c = CalibrationNIFS(session, header, descriptors, types)
  if(instrument == 'michelle'):
    c = CalibrationMICHELLE(session, header, descriptors, types)
  # Add other instruments here
  if(c==None):
    c = Calibration(session, header, descriptors, types)

  return c

class Calibration():
  """
  This class provides a basic Calibration Manager
  This is the superclass from which the instrument specific variants subclass
  """

  session = None
  header = None
  descriptors = None
  types = None
  required = []

  def __init__(self, session, header, descriptors, types):
    """
    Initialise a calibration manager for a given header object (ie data file)
    Need to pass in an sqlalchemy session that should already be open, this class will not close it
    Also pass in a header object
    """
    self.session = session
    self.header = header
    self.descriptors = descriptors
    self.types = types

  def arc(self):
    return "arc method not defined for this instrument"

  def bias(self):
    return "bias method not defined for this instrument"

class CalibrationGMOS(Calibration):
  """
  This class implements a calibration manager for GMOS.
  It is a subclass of Calibration
  """
  gmos = None

  def __init__(self, session, header, descriptors, types):
    # Init the superclass
    Calibration.__init__(self, session, header, descriptors, types)

    # if header based, Find the gmosheader
    if(header):
      query = session.query(Gmos).filter(Gmos.header_id==self.header.id)
      self.gmos = query.first()

    # Set the list of required calibrations
    self.required = self.required()

  def required(self):
    # Return a list of the calibrations required for this GMOS dataset
    list=[]

    if(self.header):
      # BIASes do not require a bias. 
      if(self.header.obstype != 'BIAS'):
        list.append('bias')

      # If it (is spectroscopy) and (is an OBJECT) and (is not a Twilight) then it needs an arc
      if((self.header.spectroscopy==True) and (self.header.obstype=='OBJECT') and (self.header.object!='Twilight')):
        list.append('arc')

    return list

  def arc(self, sameprog=False):
    query = self.session.query(Header).select_from(join(join(Gmos, Header), DiskFile))
    query = query.filter(Header.obstype=='ARC')

    # Search only the canonical (latest) entries
    query = query.filter(DiskFile.canonical==True)

    # Knock out the FAILs
    query = query.filter(Header.qastate!='Fail')

    # Must Totally Match: Instrument, disperser
    query = query.filter(Header.instrument==self.header.instrument).filter(Gmos.disperser==self.gmos.disperser)

    # Must match filter (from KR 20100423)
    query = query.filter(Gmos.filtername==self.gmos.filtername)

    # Must Match cwave 
    query = query.filter(Header.cwave==self.header.cwave)

    # Must match fpmask only if it's not the 5.0arcsec slit in the target, otherwise any longslit is OK
    if(self.gmos.fpmask != '5.0arcsec'):
      query = query.filter(Gmos.fpmask==self.gmos.fpmask)
    else:
      query = query.filter(Gmos.fpmask.like('%arcsec'))

    # Must match ccd binning
    query = query.filter(Gmos.xccdbin==self.gmos.xccdbin).filter(Gmos.yccdbin==self.gmos.yccdbin)

    # The science amproa must be equal or substring of the arc amproa
    query = query.filter(Gmos.amproa.like('%'+self.gmos.amproa+'%'))

    # Should we insist on the program ID matching?
    if(sameprog):
      query = query.filter(Header.progid==self.header.progid)

    # Order by absolute time separation. 
    query = query.order_by(func.abs(extract('epoch', Header.utdatetime - self.header.utdatetime)).asc())

    # For now, we only want one result - the closest in time
    query = query.limit(1)

    return query.first()

  def dark(self):
    query = self.session.query(Header).select_from(join(join(Gmos, Header), DiskFile))
    query = query.filter(Header.obstype=='DARK')

     # Search only the canonical (latest) entries
    query = query.filter(DiskFile.canonical==True)

    # Knock out the FAILs
    query = query.filter(Header.qastate!='Fail')

    # Must totally match instrument, xccdbin, yccdbin, readspeedmode, gainmode, exptime, nodandshuffle
    query = query.filter(Header.instrument==self.header.instrument)
    query = query.filter(Gmos.xccdbin==self.gmos.xccdbin).filter(Gmos.yccdbin==self.gmos.yccdbin)
    query = query.filter(Gmos.readspeedmode==self.gmos.readspeedmode).filter(Gmos.gainmode==self.gmos.gainmode)
    query = query.filter(Header.exptime==self.header.exptime)
    query = query.filter(Gmos.nodandshuffle==self.gmos.nodandshuffle)
    if(self.gmos.nodandshuffle):
      query = query.filter(Gmos.nod_count==self.gmos.nod_count)
      query = query.filter(Gmos.nod_pixels==self.gmos.nod_pixels)

    # The science amproa must be equal or substring of the arc amproa
    query = query.filter(Gmos.amproa.like('%'+self.gmos.amproa+'%'))

    # Order by absolute time separation. 
    query = query.order_by(func.abs(extract('epoch', Header.utdatetime - self.header.utdatetime)).asc())

    # For now, we only want one result - the closest in time
    query = query.limit(1)

    return query.first()

  def bias(self):
    query = self.session.query(Header).select_from(join(join(Gmos, Header), DiskFile))
    query = query.filter(Header.obstype=='BIAS')

     # Search only the canonical (latest) entries
    query = query.filter(DiskFile.canonical==True)

    # Knock out the FAILs
    query = query.filter(Header.qastate!='Fail')

    # Must totally match instrument, xccdbin, yccdbin, readspeedmode, gainmode
    query = query.filter(Header.instrument==self.header.instrument)
    query = query.filter(Gmos.xccdbin==self.gmos.xccdbin).filter(Gmos.yccdbin==self.gmos.yccdbin)
    query = query.filter(Gmos.readspeedmode==self.gmos.readspeedmode).filter(Gmos.gainmode==self.gmos.gainmode)

    # The science amproa must be equal or substring of the arc amproa
    query = query.filter(Gmos.amproa.like('%'+self.gmos.amproa+'%'))

    # Order by absolute time separation. 
    query = query.order_by(func.abs(extract('epoch', Header.utdatetime - self.header.utdatetime)).asc())

    # For now, we only want one result - the closest in time
    query = query.limit(1)

    return query.first()

  def processed_bias(self):
    # The basic PROCESSED_BIAS search
    query = self.session.query(Header).select_from(join(join(Gmos, Header), DiskFile))
    query = query.filter(Header.obstype=='BIAS')
    query = query.filter(Header.reduction=='PROCESSED_BIAS')

    # Search only the canonical (latest) entries
    query = query.filter(DiskFile.canonical==True)

    # Knock out the FAILs
    query = query.filter(Header.qastate!='Fail')

    if(self.descriptors==None):
      self.descriptors = {}
      self.descriptors['instrument']=self.header.instrument
      self.descriptors['detector_x_bin']=self.gmos.xccdbin
      self.descriptors['detector_y_bin']=self.gmos.yccdbin
      self.descriptors['read_speed_mode']=self.gmos.readspeedmode
      self.descriptors['gain_mode']=self.gmos.gainmode
      self.descriptors['amp_read_area']=self.gmos.amproa
      self.descriptors['ut_datetime']=self.header.utdatetime
    else:
      datetime_string = "%s %s" % (self.descriptors['ut_date'], self.descriptors['ut_time'])
      self.descriptors['ut_datetime'] = dateutil.parser.parse(datetime_string)

    # Must totally match instrument, xccdbin, yccdbin, readspeedmode, gainmode
    query = query.filter(Header.instrument==self.descriptors['instrument'])
    query = query.filter(Gmos.xccdbin==self.descriptors['detector_x_bin'])
    query = query.filter(Gmos.yccdbin==self.descriptors['detector_y_bin'])
    query = query.filter(Gmos.readspeedmode==self.descriptors['read_speed_mode'])
    query = query.filter(Gmos.gainmode==self.descriptors['gain_mode'])

    # The science amproa must be equal or substring of the bias amproa
    query = query.filter(Gmos.amproa.like('%'+str(self.descriptors['amp_read_area'])+'%'))

    # Order by absolute time separation.
    if fsc_localmode:
        # note: double check if this even works, we suspect it doesn't
        # but it's hard to notice as a somewhat fitting cal will be returned
        # but perhaps not the most recent.
        query = query.order_by(func.abs(Header.utdatetime - self.header.utdatetime))
    else:
        query = query.order_by(func.abs(extract('epoch', Header.utdatetime - self.descriptors['ut_datetime'])).asc())

    # For now, we only want one result - the closest in time
    query = query.limit(1)

    return query.first()

  def flat(self):
    query = self.session.query(Header).select_from(join(join(Gmos, Header), DiskFile))
    query = query.filter(Header.reduction=='FLAT')

    # Search only the canonical (latest) entries
    query = query.filter(DiskFile.canonical==True)

    # Knock out the FAILs
    query = query.filter(Header.qastate!='Fail')

    # Must totally match instrument, xccdbin, yccdbin, filter
    query = query.filter(Header.instrument==self.header.instrument)
    query = query.filter(Gmos.xccdbin==self.gmos.xccdbin).filter(Gmos.yccdbin==self.gmos.yccdbin)
    query = query.filter(Gmos.filtername==self.gmos.filtername)
    #query = query.filter(Gmos.readspeedmode==self.gmos.readspeedmode).filter(Gmos.gainmode==self.gmos.gainmode)
    query = query.filter(Header.spectroscopy==self.header.spectroscopy)
    if(self.header.spectroscopy):
      query = query.filter(Gmos.disperser==self.gmos.disperser)
      query = query.filter(Header.cwave==self.header.cwave)
      query = query.filter(Gmos.fpmask==self.gmos.fpmask)

    # The science amproa must be equal or substring of the flat amproa
    query = query.filter(Gmos.amproa.like('%'+self.gmos.amproa+'%'))

    # Order by absolute time separation.
    query = query.order_by(func.abs(extract('epoch', Header.utdatetime - self.header.utdatetime)).asc())

    # For now, we only want one result - the closest in time
    query = query.limit(1)

    return query.first()

  def processed_flat(self):
    query = self.session.query(Header).select_from(join(join(Gmos, Header), DiskFile))
    query = query.filter(Header.reduction=='PROCESSED_FLAT')

    # Search only the canonical (latest) entries
    query = query.filter(DiskFile.canonical==True)

    # Knock out the FAILs
    query = query.filter(Header.qastate!='Fail')

    # Must totally match instrument, xccdbin, yccdbin, filter
    query = query.filter(Header.instrument==self.header.instrument)
    query = query.filter(Gmos.xccdbin==self.gmos.xccdbin).filter(Gmos.yccdbin==self.gmos.yccdbin)
    query = query.filter(Gmos.filtername==self.gmos.filtername)
    #query = query.filter(Gmos.readspeedmode==self.gmos.readspeedmode).filter(Gmos.gainmode==self.gmos.gainmode)
    query = query.filter(Header.spectroscopy==self.header.spectroscopy)
    if(self.header.spectroscopy):
      query = query.filter(Gmos.disperser==self.gmos.disperser)
      query = query.filter(Header.cwave==self.header.cwave)
      query = query.filter(Gmos.fpmask==self.gmos.fpmask)

    # The science amproa must be equal or substring of the flat amproa
    query = query.filter(Gmos.amproa.like('%'+self.gmos.amproa+'%'))

    # Order by absolute time separation.
    if fsc_localmode:
        # note: double check if this even works, we suspect it doesn't
        # but it's hard to notice as a somewhat fitting cal will be returned
        # but perhaps not the most recent.
        query = query.order_by(func.abs(Header.utdatetime - self.header.utdatetime))
    else:
        query = query.order_by(func.abs(extract('epoch', Header.utdatetime - self.descriptors['ut_datetime'])).asc())

    # For now, we only want one result - the closest in time
    query = query.limit(1)

    return query.first()

  def processed_fringe(self):
    query = self.session.query(Header).select_from(join(join(Gmos, Header), DiskFile))
    query = query.filter(Header.reduction=='PROCESSED_FRINGE')

    # Search only the canonical (latest) entries
    query = query.filter(DiskFile.canonical==True)

    # Knock out the FAILs
    query = query.filter(Header.qastate!='Fail')

    # Must totally match instrument, xccdbin, yccdbin, filter
    query = query.filter(Header.instrument==self.header.instrument)
    query = query.filter(Gmos.xccdbin==self.gmos.xccdbin).filter(Gmos.yccdbin==self.gmos.yccdbin)
    query = query.filter(Gmos.filtername==self.gmos.filtername)
    #query = query.filter(Gmos.readspeedmode==self.gmos.readspeedmode).filter(Gmos.gainmode==self.gmos.gainmode)

    # The science amproa must be equal or substring of the flat amproa
    query = query.filter(Gmos.amproa.like('%'+self.gmos.amproa+'%'))

    # Order by absolute time separation
    query = query.order_by(func.abs(extract('epoch', Header.utdatetime - self.header.utdatetime)).asc())

    # For now, we only want one result - the closest in time
    query = query.limit(1)

    return query.first()



class CalibrationNIRI(Calibration):
  """
  This class implements a calibration manager for NIRI.
  It is a subclass of Calibration
  """
  niri = None

  def __init__(self, session, header, descriptors, types):
    # Init the superclass
    Calibration.__init__(self, session, header, descriptors, types)

    # Find the niriheader
    query = session.query(Niri).filter(Niri.header_id==self.header.id)
    self.niri = query.first()

    # Set the list of required calibrations
    self.required = self.required()

  def required(self):
    # Return a list of the calibrations required for this NIRI dataset
    list=[]

    # Science Imaging OBJECTs require a DARK and FLAT
    if((self.header.obstype=='OBJECT') and (self.header.spectroscopy==False) and (self.header.obsclass=='science')):
      list.append('dark')
      list.append('flat')

    return list

  def dark(self):
    query = self.session.query(Header).select_from(join(join(Niri, Header), DiskFile))
    query = query.filter(Header.obstype=='DARK')

    # Search only canonical entries
    query = query.filter(DiskFile.canonical==True)

    # Knock out the FAILs
    query = query.filter(Header.qastate!='Fail')

    # Must totally match: detsec, readmode, welldepthmode, exptime, coadds
    query = query.filter(Niri.detsec==self.niri.detsec)
    query = query.filter(Niri.readmode==self.niri.readmode).filter(Niri.welldepthmode==self.niri.welldepthmode)
    query = query.filter(Header.exptime==self.header.exptime).filter(Niri.coadds==self.niri.coadds)

    # Order by absolute time separation
    query = query.order_by(func.abs(extract('epoch', Header.utdatetime - self.header.utdatetime)).asc())

    # For now, we only want one result - the closest in time
    query = query.limit(1)

    return query.first()

  def flat(self):
    query = self.session.query(Header).select_from(join(join(Niri, Header), DiskFile))
    query = query.filter(Header.obstype=='FLAT')

    # Search only canonical entries
    query = query.filter(DiskFile.canonical==True)

    # Knock out the FAILs
    query = query.filter(Header.qastate!='Fail')

    # Must totally match: detsec, readmode, welldepthmode, filtername, camera
    query = query.filter(Niri.detsec==self.niri.detsec)
    query = query.filter(Niri.readmode==self.niri.readmode).filter(Niri.welldepthmode==self.niri.welldepthmode)
    query = query.filter(Niri.filtername==self.niri.filtername).filter(Niri.camera==self.niri.camera)

    # Order by absolute time separation
    query = query.order_by(func.abs(extract('epoch', Header.utdatetime - self.header.utdatetime)).asc())

    # For now, we only want one result - the closest in time
    query = query.limit(1)

    return query.first()



class CalibrationGNIRS(Calibration):
  """
  This class implements a calibration manager for GNIRS.
  It is a subclass of Calibration
  """
  gnirs = None

  def __init__(self, session, header, descriptors, types):
    # Init the superclass
    Calibration.__init__(self, session, header, descriptors, types)

    # Find the gnirsheader
    query = session.query(Gnirs).filter(Gnirs.header_id==self.header.id)
    self.gnirs = query.first()

    # Set the list of required calibrations
    self.required = self.required()

  def required(self):
    # Return a list of the calibrations required for this GNIRS dataset
    list=[]

    # Science Imaging OBJECTs require a DARK
    if((self.header.obstype=='OBJECT') and (self.header.spectroscopy==False)):
      list.append('dark')
    if((self.header.obstype=='OBJECT') and (self.header.spectroscopy==True)):
      list.append('flat')
      list.append('arc')

    return list

  def dark(self):
    query = self.session.query(Header).select_from(join(join(Gnirs, Header), DiskFile))
    query = query.filter(Header.obstype=='DARK')

    # Search only canonical entries
    query = query.filter(DiskFile.canonical==True)

    # Knock out the FAILs
    query = query.filter(Header.qastate!='Fail')

    # Must totally match: readmode, welldepthmode, exptime, coadds
    query = query.filter(Gnirs.readmode==self.gnirs.readmode).filter(Gnirs.welldepthmode==self.gnirs.welldepthmode)
    query = query.filter(Header.exptime==self.header.exptime).filter(Gnirs.coadds==self.gnirs.coadds)

    # Order by absolute time separation
    query = query.order_by(func.abs(extract('epoch', Header.utdatetime - self.header.utdatetime)).asc())
    
    # For now, we only want one result - the closest in time
    query = query.limit(1)

    return query.first()

  def flat(self):
    query = self.session.query(Header).select_from(join(join(Gnirs, Header), DiskFile))
    query = query.filter(Header.obstype=='FLAT')

    # Search only canonical entries
    query = query.filter(DiskFile.canonical==True)

    # Knock out the FAILs
    query = query.filter(Header.qastate!='Fail')

    # Must totally match: disperser, cwave, fpmask, camera, filtername, readmode, welldepthmode
    query = query.filter(Gnirs.disperser==self.gnirs.disperser)
    query = query.filter(Header.cwave==self.header.cwave)
    query = query.filter(Gnirs.fpmask==self.gnirs.fpmask)
    query = query.filter(Gnirs.camera==self.gnirs.camera)
    query = query.filter(Gnirs.filtername==self.gnirs.filtername)
    query = query.filter(Gnirs.readmode==self.gnirs.readmode)
    query = query.filter(Gnirs.welldepthmode==self.gnirs.welldepthmode)

    # Order by absolute time separation
    query = query.order_by(func.abs(extract('epoch', Header.utdatetime - self.header.utdatetime)).asc())

    # For now, we only want one result - the closest in time
    query = query.limit(1)

    return query.first()

  def arc(self, sameprog=False):
    query = self.session.query(Header).select_from(join(join(Gnirs, Header), DiskFile))
    query = query.filter(Header.obstype=='ARC')

    # Search only the canonical (latest) entries
    query = query.filter(DiskFile.canonical==True)

    # Knock out the FAILs
    query = query.filter(Header.qastate!='Fail')

    # Must Totally Match: disperser, cwave, fpmask, filtername, camera
    query = query.filter(Gnirs.disperser==self.gnirs.disperser)
    query = query.filter(Header.cwave==self.header.cwave)
    query = query.filter(Gnirs.fpmask==self.gnirs.fpmask)
    query = query.filter(Gnirs.filtername==self.gnirs.filtername)
    query = query.filter(Gnirs.camera==self.gnirs.camera)

    # Order by absolute time separation
    query = query.order_by(func.abs(extract('epoch', Header.utdatetime - self.header.utdatetime)).asc())

    # For now, we only want one result - the closest in time
    query = query.limit(1)

    return query.first()

  def pinhole_mask(self):
    query = self.session.query(Header).select_from(join(join(Gnirs, Header), DiskFile))
    query = query.filter(Header.obstype=='PINHOLE_MASK')

    # Search only the canonical (latest) entries
    query = query.filter(DiskFile.canonical==True)

    # Knock out the FAILs
    query = query.filter(Header.qastate!='Fail')

    # Must totally match: disperser, cwave, camera, (only for cross dispersed mode?)
    query = query.filter(Gnirs.disperser==self.gnirs.disperser)
    query = query.filter(Header.cwave==self.header.cwave)
    query = query.filter(Gnirs.camera==self.gnirs.camera)

    # Order by absolute time separation
    query = query.order_by(func.abs(extract('epoch', Header.utdatetime - self.header.utdatetime)).asc())

    # For now, we only want one result - the closest in time
    query = query.limit(1)

    return query.first()


class CalibrationNIFS(Calibration):
  """
  This class implements a calibration manager for NIFS.
  It is a subclass of Calibration
  """
  nifs = None

  def __init__(self, session, header, descriptors, types):
    # Init the superclass
    Calibration.__init__(self, session, header, descriptors, types)

    # Find the nifsheader
    query = session.query(Nifs).filter(Nifs.header_id==self.header.id)
    self.nifs = query.first()

    # Set the list of required calibrations
    self.required = self.required()

  def required(self):
    # Return a list of the calibrations required for this NIFS dataset
    list=[]

    # Science Imaging OBJECTs require a DARK
    if((self.header.obstype == 'OBJECT') and (self.header.spectroscopy == False) and (self.header.obsclass=='science')):
      list.append('dark')

    return list

  def dark(self):
    query = self.session.query(Header).select_from(join(join(Nifs, Header), DiskFile))
    query = query.filter(Header.obstype=='DARK')

    # Search only canonical entries
    query = query.filter(DiskFile.canonical == True)

    # Knock out the FAILs
    query = query.filter(Header.qastate!='Fail')

    # Must totally match: readmode, exptime, coadds, disperser
    query = query.filter(Nifs.readmode==self.nifs.readmode)
    query = query.filter(Header.exptime==self.header.exptime)
    query = query.filter(Nifs.coadds==self.nifs.coadds)
    query = query.filter(Nifs.disperser==self.nifs.disperser)

    # Order by absolute time separation.
    query = query.order_by(func.abs(extract('epoch', Header.utdatetime - self.header.utdatetime)).asc())

    # For now, we only want one result - the closest in time
    query = query.limit(1)

    return query.first()

  def flat(self):
    query = self.session.query(Header).select_from(join(join(Nifs, Header), DiskFile))
    query = query.filter(Header.obstype=='FLAT')

    # Search only canonical entries
    query = query.filter(DiskFile.canonical == True)

    # Knock out the FAILs
    query = query.filter(Header.qastate!='Fail')

    # Must totally match: disperser, cwave, fpmask, filter, readmode
    query = query.filter(Nifs.disperser==self.nifs.disperser)
    query = query.filter(Header.cwave==self.header.cwave)
    query = query.filter(Nifs.fpmask==self.nifs.fpmask)
    query = query.filter(Nifs.filter==self.nifs.filter)
    query = query.filter(Nifs.readmode==self.nifs.readmode)

    # Order by absolute time separation
    query = query.order_by(func.abs(extract('epoch', Header.utdatetime - self.header.utdatetime)).asc())

    # For now, we only want one result - the closest in time
    query = query.limit(1)

    return query.first()

  def arc(self, sameprog=False):
    query = self.session.query(Header).select_from(join(join(Nifs, Header), DiskFile))
    query = query.filter(Header.obstype=='ARC')

    # Search only the canonical (latest) entries
    query = query.filter(DiskFile.canonical==True)

    # Knock out the FAILs
    query = query.filter(Header.qastate!='Fail')

    # Must Totally Match: disperser, cwave, fpmask, filter
    query = query.filter(Nifs.disperser==self.nifs.disperser)
    query = query.filter(Header.cwave==self.header.cwave)
    query = query.filter(Nifs.fpmask==self.nifs.fpmask)
    query = query.filter(Nifs.filter==self.nifs.filter)

    # Order by absolute time separation
    query = query.order_by(func.abs(extract('epoch', Header.utdatetime - self.header.utdatetime)).asc())

    # For now, we only want one result - the closest in time
    query = query.limit(1)

    return query.first()

  def ronchi_mask(self):
    query = self.session.query(Header).select_from(join(join(Nifs, Header), DiskFile))
    query = query.filter(Header.obstype=='RONCHI_MASK')

    # Search only the canonical (latest) entries
    query = query.filter(DiskFile.canonical==True)

    # Knock out the FAILs
    query = query.filter(Header.qastate!='Fail')

    # Must totally match: disperser, cwave
    query = query.filter(Nifs.disperser==self.nifs.disperser)
    query = query.filter(Header.cwave==self.header.cwave)

    # Order by absolute time separation
    query = query.order_by(func.abs(extract('epoch', Header.utdatetime - self.header.utdatetime)).asc())

    # For now, we only want one result - the closest in time
    query = query.limit(1)

    return query.first()


class CalibrationMICHELLE(Calibration):
  """
  This class implements a calibration manager for MICHELLE.
  It is a subclass of Calibration
  """
  michelle = None

  def __init__(self, session, header, descriptors, types):
    # Init the superclass
    Calibration.__init__(self, session, header, descriptors, types)

    # Find the michelleheader
    query = session.query(Michelle).filter(Michelle.header_id==self.header.id)
    self.michelle = query.first()

    # Set the list of required calibrations
    self.required = self.required()

  def required(self):
    # Return a list of the calibrations required for this MICHELLE dataset
    list=[]

    # Science Imaging OBJECTs require a DARK
    if((self.header.obstype == 'OBJECT') and (self.header.spectroscopy == False) and (self.header.obsclass=='science')):
      list.append('dark')

    return list

  def dark(self):
    query = self.session.query(Header).select_from(join(join(Michelle, Header), DiskFile))
    query = query.filter(Header.obstype=='DARK')

    # Search only canonical entries
    query = query.filter(DiskFile.canonical == True)

    # Knock out the FAILs
    query = query.filter(Header.qastate!='Fail')

    # Must totally match: readmode, exptime, coadds
    query = query.filter(Michelle.readmode == self.michelle.readmode)
    query = query.filter(Header.exptime == self.header.exptime).filter(Michelle.coadds == self.michelle.coadds)

    # Order by absolute time separation
    query = query.order_by(func.abs(extract('epoch', Header.utdatetime - self.header.utdatetime)).asc())

    # For now, we only want one result - the closest in time
    query = query.limit(1)

    return query.first()

