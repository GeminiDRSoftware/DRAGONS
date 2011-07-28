"""
This module holds the CalibrationNIRI class
"""

from fitsstore import FitsStorageConfig
from fitsstore import GeminiMetadataUtils
from fitsstore.FitsStorage import DiskFile, Header, Niri
from fitsstore.FitsStorageCal.Calibration import Calibration

from sqlalchemy.orm import join
from sqlalchemy import func, extract


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

  def dark(self, List=None):
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

    # For now, we only want one result - the closest in time, unless otherwise indicated
    if(List):
      query = query.limit(List)
      return  query.all()
    else:
      query = query.limit(1)
      return query.first()

  def flat(self, List=None):
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

    # For now, we only want one result - the closest in time, unless otherwise indicated
    if(List):
      query = query.limit(List)
      return  query.all()
    else:
      query = query.limit(1)
      return query.first()


