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
    self.from_descriptors = False

    # Populate the descriptors dictionary for header
    if(self.descriptors==None):
      self.from_descriptors=True
      self.descriptors = {}
      self.descriptors['header_id']=self.header.id
      self.descriptors['observation_type']=self.header.observation_type
      self.descriptors['spectroscopy']=self.header.spectroscopy
      self.descriptors['object']=self.header.object
      self.descriptors['instrument']=self.header.instrument
      self.descriptors['central_wavelength']=self.header.central_wavelength
      self.descriptors['program_id']=self.header.program_id
      self.descriptors['ut_datetime']=self.header.ut_datetime
      self.descriptors['exposure_time']=self.header.exposure_time
      self.descriptors['observation_class']=self.header.observation_class

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
      query = session.query(Gmos).filter(Gmos.header_id==self.descriptors['header_id'])
      self.gmos = query.first()

    # Populate the descriptors dictionary for GMOS
    if(self.from_descriptors):
      self.descriptors['disperser']=self.gmos.disperser
      self.descriptors['filter_name']=self.gmos.filter_name
      self.descriptors['focal_plane_mask']=self.gmos.focal_plane_mask
      self.descriptors['detector_x_bin']=self.gmos.detector_x_bin
      self.descriptors['detector_y_bin']=self.gmos.detector_y_bin
      self.descriptors['amp_read_area']=self.gmos.amp_read_area
      self.descriptors['read_speed_setting']=self.gmos.read_speed_setting
      self.descriptors['gain_setting']=self.gmos.gain_setting
      self.descriptors['nodandshuffle']=self.gmos.nodandshuffle
      self.descriptors['nod_count']=self.gmos.nod_count
      self.descriptors['nod_pixels']=self.gmos.nod_pixels

    # Set the list of required calibrations
    self.required = self.required()

  def required(self):
    # Return a list of the calibrations required for this GMOS dataset
    list=[]

    if(self.descriptors):
      # BIASes do not require a bias. 
      if(self.descriptors['observation_type'] != 'BIAS'):
        list.append('bias')
        list.append('processed_bias')

      # If it (is spectroscopy) and (is an OBJECT) and (is not a Twilight) then it needs an arc
      if((self.descriptors['spectroscopy']==True) and (self.descriptors['observation_type']=='OBJECT') and (self.descriptors['object']!='Twilight')):
        list.append('arc')
        #list.append('dark')
        #list.append('flat')
        #list.append('processed_flat')
        #list.append('processed_fringe')

    return list

  def arc(self, sameprog=False):
    query = self.session.query(Header).select_from(join(join(Gmos, Header), DiskFile))
    query = query.filter(Header.observation_type=='ARC')

    # Search only the canonical (latest) entries
    query = query.filter(DiskFile.canonical==True)

    # Knock out the FAILs
    query = query.filter(Header.qa_state!='Fail')

    # Must Totally Match: Instrument, disperser
    query = query.filter(Header.instrument==self.descriptors['instrument']).filter(Gmos.disperser==self.descriptors['disperser'])

    # Must match filter (from KR 20100423)
    query = query.filter(Gmos.filter_name==self.descriptors['filter_name'])

    # Must Match central_wavelength 
    query = query.filter(Header.central_wavelength==self.descriptors['central_wavelength'])

    # Must match focal_plane_mask only if it's not the 5.0arcsec slit in the target, otherwise any longslit is OK
    if(self.descriptors['focal_plane_mask'] != '5.0arcsec'):
      query = query.filter(Gmos.focal_plane_mask==self.descriptors['focal_plane_mask'])
    else:
      query = query.filter(Gmos.focal_plane_mask.like('%arcsec'))

    # Must match ccd binning
    query = query.filter(Gmos.detector_x_bin==self.descriptors['detector_x_bin']).filter(Gmos.detector_y_bin==self.descriptors['detector_y_bin'])

    # The science amp_read_area must be equal or substring of the arc amp_read_area
    query = query.filter(Gmos.amp_read_area.like('%'+self.descriptors['amp_read_area']+'%'))

    # Should we insist on the program ID matching?
    if(sameprog):
      query = query.filter(Header.program_id==self.descriptors['program_id'])

    # Order by absolute time separation. 
    query = query.order_by(func.abs(extract('epoch', Header.ut_datetime - self.descriptors['ut_datetime'])).asc())

    # For now, we only want one result - the closest in time
    query = query.limit(1)

    return query.first()

  def dark(self):
    query = self.session.query(Header).select_from(join(join(Gmos, Header), DiskFile))
    query = query.filter(Header.observation_type=='DARK')

     # Search only the canonical (latest) entries
    query = query.filter(DiskFile.canonical==True)

    # Knock out the FAILs
    query = query.filter(Header.qa_state!='Fail')

    # Must totally match instrument, detector_x_bin, detector_y_bin, read_speed_setting, gain_setting, exposure_time, nodandshuffle
    query = query.filter(Header.instrument==self.descriptors['instrument'])
    query = query.filter(Gmos.detector_x_bin==self.descriptors['detector_x_bin']).filter(Gmos.detector_y_bin==self.descriptors['detector_y_bin'])
    query = query.filter(Gmos.read_speed_setting==self.descriptors['read_speed_setting'])
    query = query.filter(Gmos.gain_setting==self.descriptors['gain_setting'])
    query = query.filter(Header.exposure_time==self.descriptors['exposure_time'])
    query = query.filter(Gmos.nodandshuffle==self.descriptors['nodandshuffle'])
    if(self.descriptors['nodandshuffle']):
      query = query.filter(Gmos.nod_count==self.descriptors['nod_count'])
      query = query.filter(Gmos.nod_pixels==self.descriptors['nod_pixels'])

    # The science amp_read_area must be equal or substring of the arc amp_read_area
    query = query.filter(Gmos.amp_read_area.like('%'+self.descriptors['amp_read_area']+'%'))

    # Order by absolute time separation. 
    query = query.order_by(func.abs(extract('epoch', Header.ut_datetime - self.descriptors['ut_datetime'])).asc())

    # For now, we only want one result - the closest in time
    query = query.limit(1)

    return query.first()

  def bias(self):
    query = self.session.query(Header).select_from(join(join(Gmos, Header), DiskFile))
    query = query.filter(Header.observation_type=='BIAS')

     # Search only the canonical (latest) entries
    query = query.filter(DiskFile.canonical==True)

    # Knock out the FAILs
    query = query.filter(Header.qa_state!='Fail')

    # Must totally match instrument, detector_x_bin, detector_y_bin, read_speed_setting, gain_setting
    query = query.filter(Header.instrument==self.descriptors['instrument'])
    query = query.filter(Gmos.detector_x_bin==self.descriptors['detector_x_bin']).filter(Gmos.detector_y_bin==self.descriptors['detector_y_bin'])
    query = query.filter(Gmos.read_speed_setting==self.descriptors['read_speed_setting']).filter(Gmos.gain_setting==self.descriptors['gain_setting'])

    # The science amp_read_area must be equal or substring of the arc amp_read_area
    query = query.filter(Gmos.amp_read_area.like('%'+self.descriptors['amp_read_area']+'%'))

    # Order by absolute time separation. 
    query = query.order_by(func.abs(extract('epoch', Header.ut_datetime - self.descriptors['ut_datetime'])).asc())

    # For now, we only want one result - the closest in time
    query = query.limit(1)

    return query.first()

  def processed_bias(self):
    # The basic PROCESSED_BIAS search
    query = self.session.query(Header).select_from(join(join(Gmos, Header), DiskFile))
    query = query.filter(Header.observation_type=='BIAS')
    query = query.filter(Header.reduction=='PROCESSED_BIAS')

    # Search only the canonical (latest) entries
    query = query.filter(DiskFile.canonical==True)

    # Knock out the FAILs
    query = query.filter(Header.qa_state!='Fail')

    # Must totally match instrument, detector_x_bin, detector_y_bin, read_speed_setting, gain_setting
    query = query.filter(Header.instrument==self.descriptors['instrument'])
    query = query.filter(Gmos.detector_x_bin==self.descriptors['detector_x_bin']).filter(Gmos.detector_y_bin==self.descriptors['detector_y_bin'])
    query = query.filter(Gmos.read_speed_setting==self.descriptors['read_speed_setting']).filter(Gmos.gain_setting==self.descriptors['gain_setting'])

    # The science amp_read_area must be equal or substring of the bias amp_read_area
    query = query.filter(Gmos.amp_read_area.like('%'+str(self.descriptors['amp_read_area'])+'%'))

    # Order by absolute time separation.
    if fsc_localmode:
        # note: double check if this even works, we suspect it doesn't
        # but it's hard to notice as a somewhat fitting cal will be returned
        # but perhaps not the most recent.
        query = query.order_by(func.abs(Header.ut_datetime - self.descriptors['ut_datetime']))
    else:
        query = query.order_by(func.abs(extract('epoch', Header.ut_datetime - self.descriptors['ut_datetime'])).asc())

    # For now, we only want one result - the closest in time
    query = query.limit(1)

    return query.first()

  def flat(self):
    query = self.session.query(Header).select_from(join(join(Gmos, Header), DiskFile))
    query = query.filter(Header.reduction=='FLAT')

    # Search only the canonical (latest) entries
    query = query.filter(DiskFile.canonical==True)

    # Knock out the FAILs
    query = query.filter(Header.qa_state!='Fail')

    # Must totally match instrument, detector_x_bin, detector_y_bin, filter
    query = query.filter(Header.instrument==self.descriptors['instrument'])
    query = query.filter(Gmos.detector_x_bin==self.descriptors['detector_x_bin']).filter(Gmos.detector_y_bin==self.descriptors['detector_y_bin'])
    query = query.filter(Gmos.filter_name==self.descriptors['filter_name'])
    #query = query.filter(Gmos.read_speed_setting==self.descriptors['read_speed_setting']).filter(Gmos.gain_setting==self.descriptors['gain_setting'])
    query = query.filter(Header.spectroscopy==self.descriptors['spectroscopy'])
    if(self.descriptors['spectroscopy']):
      query = query.filter(Gmos.disperser==self.descriptors['disperser'])
      query = query.filter(Header.central_wavelength==self.descriptors['central_wavelength'])
      query = query.filter(Gmos.focal_plane_mask==self.descriptors['focal_plane_mask'])

    # The science amp_read_area must be equal or substring of the flat amp_read_area
    query = query.filter(Gmos.amp_read_area.like('%'+self.descriptors['amp_read_area']+'%'))

    # Order by absolute time separation.
    query = query.order_by(func.abs(extract('epoch', Header.ut_datetime - self.descriptors['ut_datetime'])).asc())

    # For now, we only want one result - the closest in time
    query = query.limit(1)

    return query.first()

  def processed_flat(self):
    query = self.session.query(Header).select_from(join(join(Gmos, Header), DiskFile))
    query = query.filter(Header.reduction=='PROCESSED_FLAT')

    # Search only the canonical (latest) entries
    query = query.filter(DiskFile.canonical==True)

    # Knock out the FAILs
    query = query.filter(Header.qa_state!='Fail')

    # Must totally match instrument, detector_x_bin, detector_y_bin, filter
    query = query.filter(Header.instrument==self.descriptors['instrument'])
    query = query.filter(Gmos.detector_x_bin==self.descriptors['detector_x_bin']).filter(Gmos.detector_y_bin==self.descriptors['detector_y_bin'])
    query = query.filter(Gmos.filter_name==self.descriptors['filter_name'])
    #query = query.filter(Gmos.read_speed_setting==self.descriptors['read_speed_setting']).filter(Gmos.gain_setting==self.descriptors['gain_setting'])
    query = query.filter(Header.spectroscopy==self.descriptors['spectroscopy'])
    if(self.descriptors['spectroscopy']):
      query = query.filter(Gmos.disperser==self.descriptors['disperser'])
      query = query.filter(Header.central_wavelength==self.descriptors['central_wavelength'])
      query = query.filter(Gmos.focal_plane_mask==self.descriptors['focal_plane_mask'])

    # The science amp_read_area must be equal or substring of the flat amp_read_area
    query = query.filter(Gmos.amp_read_area.like('%'+self.descriptors['amp_read_area']+'%'))

    # Order by absolute time separation.
    if fsc_localmode:
        # note: double check if this even works, we suspect it doesn't
        # but it's hard to notice as a somewhat fitting cal will be returned
        # but perhaps not the most recent.
        query = query.order_by(func.abs(Header.ut_datetime - self.descriptors['ut_datetime']))
    else:
        query = query.order_by(func.abs(extract('epoch', Header.ut_datetime - self.descriptors['ut_datetime'])).asc())

    # For now, we only want one result - the closest in time
    query = query.limit(1)

    return query.first()

  def processed_fringe(self):
    query = self.session.query(Header).select_from(join(join(Gmos, Header), DiskFile))
    query = query.filter(Header.reduction=='PROCESSED_FRINGE')

    # Search only the canonical (latest) entries
    query = query.filter(DiskFile.canonical==True)

    # Knock out the FAILs
    query = query.filter(Header.qa_state!='Fail')

    # Must totally match instrument, detector_x_bin, detector_y_bin, filter
    query = query.filter(Header.instrument==self.descriptors['instrument'])
    query = query.filter(Gmos.detector_x_bin==self.descriptors['detector_x_bin']).filter(Gmos.detector_y_bin==self.descriptors['detector_y_bin'])
    query = query.filter(Gmos.filter_name==self.descriptors['filter_name'])
    #query = query.filter(Gmos.read_speed_setting==self.descriptors['read_speed_setting']).filter(Gmos.gain_setting==self.descriptors['gain_setting'])

    # The science amp_read_area must be equal or substring of the flat amp_read_area
    query = query.filter(Gmos.amp_read_area.like('%'+self.descriptors['amp_read_area']+'%'))

    # Order by absolute time separation
    query = query.order_by(func.abs(extract('epoch', Header.ut_datetime - self.descriptors['ut_datetime'])).asc())

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
    query = session.query(Niri).filter(Niri.header_id==self.descriptors['header_id'])
    self.niri = query.first()

    # Populate the descriptors dictionary for NIRI
    if(self.from_descriptors):
      self.descriptors['data_section']=self.niri.data_section
      self.descriptors['read_mode']=self.niri.read_mode
      self.descriptors['well_depth_setting']=self.niri.well_depth_setting
      self.descriptors['coadds']=self.niri.coadds
      self.descriptors['filter_name']=self.niri.filter_name
      self.descriptors['camera']=self.niri.camera

    # Set the list of required calibrations
    self.required = self.required()

  def required(self):
    # Return a list of the calibrations required for this NIRI dataset
    list=[]

    # Science Imaging OBJECTs require a DARK and FLAT
    if((self.descriptors['observation_type']=='OBJECT') and (self.descriptors['spectroscopy']==False) and (self.descriptors['observation_class']=='science')):
      list.append('dark')
      list.append('flat')

    return list

  def dark(self):
    query = self.session.query(Header).select_from(join(join(Niri, Header), DiskFile))
    query = query.filter(Header.observation_type=='DARK')

    # Search only canonical entries
    query = query.filter(DiskFile.canonical==True)

    # Knock out the FAILs
    query = query.filter(Header.qa_state!='Fail')

    # Must totally match: data_section, read_mode, well_depth_setting, exposure_time, coadds
    query = query.filter(Niri.data_section==self.descriptors['data_section'])
    query = query.filter(Niri.read_mode==self.descriptors['read_mode']).filter(Niri.well_depth_setting==self.descriptors['well_depth_setting'])
    query = query.filter(Header.exposure_time==self.descriptors['exposure_time']).filter(Niri.coadds==self.descriptors['coadds'])

    # Order by absolute time separation
    query = query.order_by(func.abs(extract('epoch', Header.ut_datetime - self.descriptors['ut_datetime'])).asc())

    # For now, we only want one result - the closest in time
    query = query.limit(1)

    return query.first()

  def flat(self):
    query = self.session.query(Header).select_from(join(join(Niri, Header), DiskFile))
    query = query.filter(Header.observation_type=='FLAT')

    # Search only canonical entries
    query = query.filter(DiskFile.canonical==True)

    # Knock out the FAILs
    query = query.filter(Header.qa_state!='Fail')

    # Must totally match: data_section, read_mode, well_depth_setting, filter_name, camera
    query = query.filter(Niri.data_section==self.descriptors['data_section'])
    query = query.filter(Niri.read_mode==self.descriptors['read_mode'])
    query = query.filter(Niri.well_depth_setting==self.descriptors['well_depth_setting'])
    query = query.filter(Niri.filter_name==self.descriptors['filter_name'])
    query = query.filter(Niri.camera==self.descriptors['camera'])

    # Order by absolute time separation
    query = query.order_by(func.abs(extract('epoch', Header.ut_datetime - self.descriptors['ut_datetime'])).asc())

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
    query = session.query(Gnirs).filter(Gnirs.header_id==self.descriptors['header_id'])
    self.gnirs = query.first()

    # Populate the descriptors dictionary for GNIRS
    if(self.from_descriptors):
      self.descriptors['read_mode']=self.gnirs.read_mode
      self.descriptors['well_depth_setting']=self.gnirs.well_depth_setting
      self.descriptors['coadds']=self.gnirs.coadds
      self.descriptors['disperser']=self.gnirs.disperser
      self.descriptors['focal_plane_mask']=self.gnirs.focal_plane_mask
      self.descriptors['camera']=self.gnirs.camera
      self.descriptors['filter_name']=self.gnirs.filter_name

    # Set the list of required calibrations
    self.required = self.required()

  def required(self):
    # Return a list of the calibrations required for this GNIRS dataset
    list=[]

    # Science Imaging OBJECTs require a DARK
    if((self.descriptors['observation_type']=='OBJECT') and (self.descriptors['spectroscopy']==False)):
      list.append('dark')
    if((self.descriptors['observation_type']=='OBJECT') and (self.descriptors['spectroscopy']==True)):
      list.append('flat')
      list.append('arc')
      #list.append('pinhole_mask')

    return list

  def dark(self):
    query = self.session.query(Header).select_from(join(join(Gnirs, Header), DiskFile))
    query = query.filter(Header.observation_type=='DARK')

    # Search only canonical entries
    query = query.filter(DiskFile.canonical==True)

    # Knock out the FAILs
    query = query.filter(Header.qa_state!='Fail')

    # Must totally match: read_mode, well_depth_setting, exposure_time, coadds
    query = query.filter(Gnirs.read_mode==self.descriptors['read_mode'])
    query = query.filter(Gnirs.well_depth_setting==self.descriptors['well_depth_setting'])
    query = query.filter(Header.exposure_time==self.descriptors['exposure_time'])
    query = query.filter(Gnirs.coadds==self.descriptors['coadds'])

    # Order by absolute time separation
    query = query.order_by(func.abs(extract('epoch', Header.ut_datetime - self.descriptors['ut_datetime'])).asc())
    
    # For now, we only want one result - the closest in time
    query = query.limit(1)

    return query.first()

  def flat(self):
    query = self.session.query(Header).select_from(join(join(Gnirs, Header), DiskFile))
    query = query.filter(Header.observation_type=='FLAT')

    # Search only canonical entries
    query = query.filter(DiskFile.canonical==True)

    # Knock out the FAILs
    query = query.filter(Header.qa_state!='Fail')

    # Must totally match: disperser, central_wavelength, focal_plane_mask, camera, filter_name, read_mode, well_depth_setting
    query = query.filter(Gnirs.disperser==self.descriptors['disperser'])
    query = query.filter(Header.central_wavelength==self.descriptors['central_wavelength'])
    query = query.filter(Gnirs.focal_plane_mask==self.descriptors['focal_plane_mask'])
    query = query.filter(Gnirs.camera==self.descriptors['camera'])
    query = query.filter(Gnirs.filter_name==self.descriptors['filter_name'])
    query = query.filter(Gnirs.read_mode==self.descriptors['read_mode'])
    query = query.filter(Gnirs.well_depth_setting==self.descriptors['well_depth_setting'])

    # Order by absolute time separation
    query = query.order_by(func.abs(extract('epoch', Header.ut_datetime - self.descriptors['ut_datetime'])).asc())

    # For now, we only want one result - the closest in time
    query = query.limit(1)

    return query.first()

  def arc(self, sameprog=False):
    query = self.session.query(Header).select_from(join(join(Gnirs, Header), DiskFile))
    query = query.filter(Header.observation_type=='ARC')

    # Search only the canonical (latest) entries
    query = query.filter(DiskFile.canonical==True)

    # Knock out the FAILs
    query = query.filter(Header.qa_state!='Fail')

    # Must Totally Match: disperser, central_wavelength, focal_plane_mask, filter_name, camera
    query = query.filter(Gnirs.disperser==self.descriptors['disperser'])
    query = query.filter(Header.central_wavelength==self.descriptors['central_wavelength'])
    query = query.filter(Gnirs.focal_plane_mask==self.descriptors['focal_plane_mask'])
    query = query.filter(Gnirs.filter_name==self.descriptors['filter_name'])
    query = query.filter(Gnirs.camera==self.descriptors['camera'])

    # Order by absolute time separation
    query = query.order_by(func.abs(extract('epoch', Header.ut_datetime - self.descriptors['ut_datetime'])).asc())

    # For now, we only want one result - the closest in time
    query = query.limit(1)

    return query.first()

  def pinhole_mask(self):
    query = self.session.query(Header).select_from(join(join(Gnirs, Header), DiskFile))
    query = query.filter(Header.observation_type=='PINHOLE_MASK')

    # Search only the canonical (latest) entries
    query = query.filter(DiskFile.canonical==True)

    # Knock out the FAILs
    query = query.filter(Header.qa_state!='Fail')

    # Must totally match: disperser, central_wavelength, camera, (only for cross dispersed mode?)
    query = query.filter(Gnirs.disperser==self.descriptors['disperser'])
    query = query.filter(Header.central_wavelength==self.descriptors['central_wavelength'])
    query = query.filter(Gnirs.camera==self.descriptors['camera'])

    # Order by absolute time separation
    query = query.order_by(func.abs(extract('epoch', Header.ut_datetime - self.descriptors['ut_datetime'])).asc())

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
    query = session.query(Nifs).filter(Nifs.header_id==self.descriptors['header_id'])
    self.nifs = query.first()

    # Populate the descriptors dictionary for NIFS
    if(self.from_descriptors):
      self.descriptors['read_mode']=self.nifs.read_mode
      self.descriptors['coadds']=self.nifs.coadds
      self.descriptors['disperser']=self.nifs.disperser
      self.descriptors['focal_plane_mask']=self.nifs.focal_plane_mask
      self.descriptors['filter_name']=self.nifs.filter_name

    # Set the list of required calibrations
    self.required = self.required()

  def required(self):
    # Return a list of the calibrations required for this NIFS dataset
    list=[]

    # Science Imaging OBJECTs require a DARK
    if((self.descriptors['observation_type'] == 'OBJECT') and (self.descriptors['spectroscopy'] == False) and (self.descriptors['observation_class']=='science')):
      list.append('dark')
    if((self.descriptors['observation_type']=='OBJECT') and (self.descriptors['spectroscopy']==True)):
      list.append('flat')
      list.append('arc')
      list.append('ronchi_mask')

    return list

  def dark(self):
    query = self.session.query(Header).select_from(join(join(Nifs, Header), DiskFile))
    query = query.filter(Header.observation_type=='DARK')

    # Search only canonical entries
    query = query.filter(DiskFile.canonical == True)

    # Knock out the FAILs
    query = query.filter(Header.qa_state!='Fail')

    # Must totally match: read_mode, exposure_time, coadds, disperser
    query = query.filter(Nifs.read_mode==self.descriptors['read_mode'])
    query = query.filter(Header.exposure_time==self.descriptors['exposure_time'])
    query = query.filter(Nifs.coadds==self.descriptors['coadds'])
    query = query.filter(Nifs.disperser==self.descriptors['disperser'])

    # Order by absolute time separation.
    query = query.order_by(func.abs(extract('epoch', Header.ut_datetime - self.descriptors['ut_datetime'])).asc())

    # For now, we only want one result - the closest in time
    query = query.limit(1)

    return query.first()

  def flat(self):
    query = self.session.query(Header).select_from(join(join(Nifs, Header), DiskFile))
    query = query.filter(Header.observation_type=='FLAT')

    # Search only canonical entries
    query = query.filter(DiskFile.canonical == True)

    # Knock out the FAILs
    query = query.filter(Header.qa_state!='Fail')

    # Must totally match: disperser, central_wavelength, focal_plane_mask, filter, read_mode
    query = query.filter(Nifs.disperser==self.descriptors['disperser'])
    query = query.filter(Header.central_wavelength==self.descriptors['central_wavelength'])
    query = query.filter(Nifs.focal_plane_mask==self.descriptors['focal_plane_mask'])
    query = query.filter(Nifs.filter_name==self.descriptors['filter_name'])
    query = query.filter(Nifs.read_mode==self.descriptors['read_mode'])

    # Order by absolute time separation
    query = query.order_by(func.abs(extract('epoch', Header.ut_datetime - self.descriptors['ut_datetime'])).asc())

    # For now, we only want one result - the closest in time
    query = query.limit(1)

    return query.first()

  def arc(self, sameprog=False):
    query = self.session.query(Header).select_from(join(join(Nifs, Header), DiskFile))
    query = query.filter(Header.observation_type=='ARC')

    # Search only the canonical (latest) entries
    query = query.filter(DiskFile.canonical==True)

    # Knock out the FAILs
    query = query.filter(Header.qa_state!='Fail')

    # Must Totally Match: disperser, central_wavelength, focal_plane_mask, filter
    query = query.filter(Nifs.disperser==self.descriptors['disperser'])
    query = query.filter(Header.central_wavelength==self.descriptors['central_wavelength'])
    query = query.filter(Nifs.focal_plane_mask==self.descriptors['focal_plane_mask'])
    query = query.filter(Nifs.filter_name==self.descriptors['filter_name'])

    # Order by absolute time separation
    query = query.order_by(func.abs(extract('epoch', Header.ut_datetime - self.descriptors['ut_datetime'])).asc())

    # For now, we only want one result - the closest in time
    query = query.limit(1)

    return query.first()

  def ronchi_mask(self):
    query = self.session.query(Header).select_from(join(join(Nifs, Header), DiskFile))
    query = query.filter(Header.observation_type=='RONCHI_MASK')

    # Search only the canonical (latest) entries
    query = query.filter(DiskFile.canonical==True)

    # Knock out the FAILs
    query = query.filter(Header.qa_state!='Fail')

    # Must totally match: disperser, central_wavelength
    query = query.filter(Nifs.disperser==self.descriptors['disperser'])
    query = query.filter(Header.central_wavelength==self.descriptors['central_wavelength'])

    # Order by absolute time separation
    query = query.order_by(func.abs(extract('epoch', Header.ut_datetime - self.descriptors['ut_datetime'])).asc())

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
    query = session.query(Michelle).filter(Michelle.header_id==self.descriptors['header_id'])
    self.michelle = query.first()

    # Populate the descriptors dictionary for MICHELLE
    if(self.from_descriptors):
      self.descriptors['read_mode']=self.michelle.read_mode
      self.descriptors['coadds']=self.michelle.coadds

    # Set the list of required calibrations
    self.required = self.required()

  def required(self):
    # Return a list of the calibrations required for this MICHELLE dataset
    list=[]

    # Science Imaging OBJECTs require a DARK
    if((self.descriptors['observation_type'] == 'OBJECT') and (self.descriptors['spectroscopy'] == False) and (self.descriptors['observation_class']=='science')):
      list.append('dark')

    return list

  def dark(self):
    query = self.session.query(Header).select_from(join(join(Michelle, Header), DiskFile))
    query = query.filter(Header.observation_type=='DARK')

    # Search only canonical entries
    query = query.filter(DiskFile.canonical == True)

    # Knock out the FAILs
    query = query.filter(Header.qa_state!='Fail')

    # Must totally match: read_mode, exposure_time, coadds
    query = query.filter(Michelle.read_mode == self.descriptors['read_mode'])
    query = query.filter(Header.exposure_time == self.descriptors['exposure_time'])
    query = query.filter(Michelle.coadds == self.descriptors['coadds'])

    # Order by absolute time separation
    query = query.order_by(func.abs(extract('epoch', Header.ut_datetime - self.descriptors['ut_datetime'])).asc())

    # For now, we only want one result - the closest in time
    query = query.limit(1)

    return query.first()

