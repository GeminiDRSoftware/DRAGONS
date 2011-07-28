# init file for package
from Calibration import Calibration
from CalibrationGMOS import CalibrationGMOS
from CalibrationNIRI import CalibrationNIRI
from CalibrationGNIRS import CalibrationGNIRS
from CalibrationNIFS import CalibrationNIFS
from CalibrationMICHELLE import CalibrationMICHELLE


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
    print "FSCi27:",instrument
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
