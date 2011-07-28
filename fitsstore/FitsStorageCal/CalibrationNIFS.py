"""
This module holds the CalibrationNIFS class
"""

from fitsstore import FitsStorageConfig
from fitsstore import GeminiMetadataUtils
from fitsstore.FitsStorage import DiskFile, Header, Nifs
from fitsstore.FitsStorageCal.Calibration import Calibration

from sqlalchemy.orm import join
from sqlalchemy import func, extract


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

  def dark(self, List=None):
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

    # For now, we only want one result - the closest in time, unless otherwise indicated
    if(List):
      query = query.limit(List)
      return  query.all()
    else:
      query = query.limit(1)
      return query.first()

  def flat(self, List=None):
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

    # For now, we only want one result - the closest in time, unless otherwise indicated
    if(List):
      query = query.limit(List)
      return  query.all()
    else:
      query = query.limit(1)
      return query.first()

  def arc(self, sameprog=False, List=None):
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

    # For now, we only want one result - the closest in time, unless otherwise indicated
    if(List):
      query = query.limit(List)
      return  query.all()
    else:
      query = query.limit(1)
      return query.first()

  def ronchi_mask(self, List=None):
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

    # For now, we only want one result - the closest in time, unless otherwise indicated
    if(List):
      query = query.limit(List)
      return  query.all()
    else:
      query = query.limit(1)
      return query.first()

