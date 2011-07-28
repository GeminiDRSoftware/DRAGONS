"""
This module holds the CalibrationGMOS class
"""

from fitsstore.FitsStorageConfig import fsc_localmode
from fitsstore import GeminiMetadataUtils
from fitsstore.FitsStorage import DiskFile, Header, Gmos
from fitsstore.FitsStorageCal.Calibration import Calibration

from sqlalchemy.orm import join
from sqlalchemy import func, extract


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
      print "CGMOS27:", repr(self.descriptors)
      query = session.query(Gmos).filter(Gmos.header_id==self.descriptors['header_id'])
      self.gmos = query.first()
      print "CGMOS30:", repr(self.gmos)
      print "CGMOS31: self.from_descriptors =", self.from_descriptors

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
      print "CM46:", repr(self.descriptors)

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

      # If it (is imaging) and (is an OBJECT) and (is not a Twilight) then it needs a processed_flat
      if((self.descriptors['spectroscopy']==False) and (self.descriptors['observation_type']=='OBJECT') and (self.descriptors['object']!='Twilight')):
        list.append('processed_flat')
 
      # If it (is imaging) and (is an OBJECT) and (is not a Twilight) then it maybe needs a processed_fringe
      if((self.descriptors['spectroscopy']==False) and (self.descriptors['observation_type']=='OBJECT') and (self.descriptors['object']!='Twilight')):
        list.append('processed_fringe')

        #list.append('dark')
        #list.append('flat')
        #list.append('processed_flat')
        #list.append('processed_fringe')

    return list

  def arc(self, sameprog=False, List=None):
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

    # For now, we only want one result - the closest in time, unless otherwise indicated
    if(List):
      query = query.limit(List)
      return  query.all()
    else:
      query = query.limit(1)
      return query.first()

  def dark(self, List=None):
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

    # The science amp_read_area must be equal or substring of the dark amp_read_area
    query = query.filter(Gmos.amp_read_area.like('%'+self.descriptors['amp_read_area']+'%'))

    # Order by absolute time separation. 
    query = query.order_by(func.abs(extract('epoch', Header.ut_datetime - self.descriptors['ut_datetime'])).asc())

    # For now, we only want one result - the closest in time, unless otherwise indicated
    if(List):
      query = query.limit(List)
      return  query.all()
    else:
      query = query.limit(1)
      return query.first()

  def bias(self, processed=False, List=None):
    query = self.session.query(Header).select_from(join(join(Gmos, Header), DiskFile))
    
    # @@DEBUGQUERY
    
    query = query.filter(Header.observation_type=='BIAS')
    if(processed):
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
    if(processed):
      query = query.filter(Gmos.amp_read_area.like('%'+str(self.descriptors['amp_read_area'])+'%'))
    else:
      query = query.filter(Gmos.amp_read_area.like('%'+self.descriptors['amp_read_area']+'%'))

    # The data section should be a equal or substring match, - so that we get the correctly trimmed bias, but also can select only CCD2 out of a 3 chip bias
    # This needs adding to FLAT and FRINGE once the DB supports it.
    # query.filter(Gmos.data_section.like('%'+self.descriptors['data_section']+'%'))

    # Order by absolute time separation.
    if(processed and fsc_localmode):
      # note: double check if this even works, we suspect it doesn't
      # but it's hard to notice as a somewhat fitting cal will be returned
      # but perhaps not the most recent.
      query = query.order_by(func.abs(Header.ut_datetime - self.descriptors['ut_datetime']))
    else:
      query = query.order_by(func.abs(extract('epoch', Header.ut_datetime - self.descriptors['ut_datetime'])).asc())

    d2 = query.statement.compile()
    d2.visit_bindparam=d2.render_literal_bindparam
    print "CG190:", str(d2)
    
    # For now, we only want one result - the closest in time, unless otherwise indicated
    if(List):
      query = query.limit(List)
      return  query.all()
    else:
      query = query.limit(1)
      return query.first()

  def flat(self, processed=False, List=None):
    query = self.session.query(Header).select_from(join(join(Gmos, Header), DiskFile))
    if(processed):
      query = query.filter(Header.reduction=='PROCESSED_FLAT')
    else:
      query = query.filter(Header.reduction=='FLAT')

    # Search only the canonical (latest) entries
    query = query.filter(DiskFile.canonical==True)

    # Knock out the FAILs
    query = query.filter(Header.qa_state!='Fail')

    # Must totally match instrument, detector_x_bin, detector_y_bin, filter
    query = query.filter(Header.instrument==self.descriptors['instrument'])
    query = query.filter(Gmos.detector_x_bin==self.descriptors['detector_x_bin']).filter(Gmos.detector_y_bin==self.descriptors['detector_y_bin'])
    query = query.filter(Gmos.filter_name==self.descriptors['filter_name'])
    query = query.filter(Gmos.read_speed_setting==self.descriptors['read_speed_setting']).filter(Gmos.gain_setting==self.descriptors['gain_setting'])
    query = query.filter(Header.spectroscopy==self.descriptors['spectroscopy'])

    # Focal plane mask must match for imaging too... To avoid daytime thru-MOS mask imaging "flats"
    query = query.filter(Gmos.focal_plane_mask==self.descriptors['focal_plane_mask'])

    if(self.descriptors['spectroscopy']):
      query = query.filter(Gmos.disperser==self.descriptors['disperser'])
      query = query.filter(Header.central_wavelength==self.descriptors['central_wavelength'])

    # The science amp_read_area must be equal or substring of the flat amp_read_area
    query = query.filter(Gmos.amp_read_area.like('%'+self.descriptors['amp_read_area']+'%'))

    # Order by absolute time separation.
    if(processed and fsc_localmode):
      # note: double check if this even works, we suspect it doesn't
      # but it's hard to notice as a somewhat fitting cal will be returned
      # but perhaps not the most recent.
      query = query.order_by(func.abs(Header.ut_datetime - self.descriptors['ut_datetime']))
    else:
      query = query.order_by(func.abs(extract('epoch', Header.ut_datetime - self.descriptors['ut_datetime'])).asc())

    # For now, we only want one result - the closest in time, unless otherwise indicated
    if(List):
      query = query.limit(List)
      return  query.all()
    else:
      query = query.limit(1)
      return query.first()

  def processed_fringe(self, List=None):
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

    # For now, we only want one result - the closest in time, unless otherwise indicated
    if(List):
      query = query.limit(List)
      return  query.all()
    else:
      query = query.limit(1)
      return query.first()

