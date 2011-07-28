"""
This module holds the CalibrationGNIRS class
"""

from fitsstore import FitsStorageConfig
from fitsstore import GeminiMetadataUtils
from fitsstore.FitsStorage import DiskFile, Header, Nifs
from fitsstore.FitsStorageCal.Calibration import Calibration

from sqlalchemy.orm import join
from sqlalchemy import func, extract


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

  def dark(self, List=None):
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
    
    # For now, we only want one result - the closest in time, unless otherwise indicated
    if(List):
      query = query.limit(List)
      return  query.all()
    else:
      query = query.limit(1)
      return query.first()

  def flat(self, List=None):
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

    # For now, we only want one result - the closest in time, unless otherwise indicated
    if(List):
      query = query.limit(List)
      return  query.all()
    else:
      query = query.limit(1)
      return query.first()

  def arc(self, sameprog=False, List=None):
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

    # For now, we only want one result - the closest in time, unless otherwise indicated
    if(List):
      query = query.limit(List)
      return  query.all()
    else:
      query = query.limit(1)
      return query.first()

  def pinhole_mask(self, List=None):
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

    # For now, we only want one result - the closest in time, unless otherwise indicated
    if(List):
      query = query.limit(List)
      return  query.all()
    else:
      query = query.limit(1)
      return query.first()

