"""
This module holds the CalibrationMICHELLE class
"""

from fitsstore import FitsStorageConfig
from fitsstore import GeminiMetadataUtils
from fitsstore.FitsStorage import DiskFile, Header, Nifs
from fitsstore.FitsStorageCal.Calibration import Calibration

from sqlalchemy.orm import join
from sqlalchemy import func, extract


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

  def dark(self, List=None):
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

    # For now, we only want one result - the closest in time, unless otherwise indicated
    if(List):
      query = query.limit(List)
      return  query.all()
    else:
      query = query.limit(1)
      return query.first()

