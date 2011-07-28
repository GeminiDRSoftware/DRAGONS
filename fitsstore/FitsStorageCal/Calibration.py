"""
This module holds the Calibration superclass
"""


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
    Initialize a calibration manager for a given header object (ie data file)
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
