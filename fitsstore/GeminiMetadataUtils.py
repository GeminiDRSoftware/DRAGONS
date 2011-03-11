"""
This is the GeminiMetadataUtils module. It provides a number of utility
classes and functions for parsing the metadata in Gemini FITS files.
"""
import re
import datetime

# This first block of regexps are compiled here but used elsewhere
percentilecre=re.compile('^\d\d-percentile$')

# Compile some regular expressions here. This is fairly complex, so I've
# split it up in substrings to make it easier to follow.
# Also these substrings are used directly by the classes

# This re matches a program id like GN-CAL20091020 with no groups
calengre='G[NS]-(?:(?:CAL)|(?:ENG))20\d\d[01]\d[0123]\d'
# This re matches a program id like GN-2009A-Q-23 with no groups
scire='G[NS]-20\d\d[AB]-[A-Z]*-\d*'

# This matches a program id
progre='(?:^%s$)|(?:^%s$)' % (calengre, scire)

# This matches an observation id with the project id and obsnum as groups
obsre='((?:^%s)|(?:^%s))-(\d*)$' % (calengre, scire)

# A utility function for matching instrument names
niricre=re.compile('[Nn][Ii][Rr][Ii]')
nifscre=re.compile('[Nn][Ii][Ff][Ss]')
gmosncre=re.compile('[Gg][Mm][Oo][Ss]-[Nn]')
gmosscre=re.compile('[Gg][Mm][Oo][Ss]-[Ss]')
gmoscre=re.compile('[Gg][Mm][Oo][Ss]')
michellecre=re.compile('[Mm][Ii][Cc][Hh][Ee][Ll][Ll][Ee]')
gnirscre=re.compile('[Gg][Nn][Ii][Rr][Ss]')
phoenixcre = re.compile('[Pp][Hh][Oo][Ee][Nn][Ii][Xx]')
trecscre = re.compile('[Tt][Rr][Ee][Cc][Ss]')
nicicre = re.compile('[Nn][Ii][Cc][Ii]')
hqcre = re.compile('[Hh][Oo][Kk][Uu][Pp]([Aa])+(\+)*[Qq][Uu][Ii][Rr][Cc]')
gsaoicre = re.compile('[Gg][Ss][Aa][Oo][Ii]')
oscircre = re.compile('[Oo][Ss][Cc][Ii][Rr]')

def gemini_instrument(string, gmos=False):
  """
  If the string argument matches a gemini instrument name, 
  then returns the "official" (ie same as in the fits headers) 
  name of the instrument. Otherwise returns an empty string.
  If the gmos argument is true, this also recognises GMOS as
  a valid instruemnt name.
  """
  retary=''
  if(niricre.match(string)):
    retary='NIRI'
  if(nifscre.match(string)):
    retary='NIFS'
  if(gmosncre.match(string)):
    retary='GMOS-N'
  if(gmosscre.match(string)):
    retary='GMOS-S'
  if(michellecre.match(string)):
    retary='michelle'
  if(gnirscre.match(string)):
    retary='GNIRS'
  if(phoenixcre.match(string)):
    retary='PHOENIX'
  if(trecscre.match(string)):
    retary='TReCS'
  if(nicicre.match(string)):
    retary='NICI'
  if(hqcre.match(string)):
    retary='Hokupaa+QUIRC'
  if(gsaoicre.match(string)):
    retary='GSAOI'
  if(oscircre.match(string)):
    retary='oscir'
  if(gmos):
    if(gmoscre.match(string)):
      retary='GMOS'
  return retary

datecre=re.compile('^20\d\d[01]\d[0123]\d$')
def gemini_date(string):
  """
  A utility function for matching dates of the form YYYYMMDD
  also supports today, yesterday
  returns the YYYYMMDD string, or '' if not a date
  May need modification to make today and yesterday work usefully 
  for Chile
  """
  if(datecre.match(string)):
    return string
  if(string == 'today'):
    now=datetime.datetime.utcnow().date()
    return now.strftime('%Y%m%d')
  if(string == 'yesterday'):
    then=datetime.datetime.utcnow() - datetime.timedelta(days=1)
    return then.date().strftime('%Y%m%d')
  return ''

daterangecre=re.compile('(20\d\d[01]\d[0123]\d)-(20\d\d[01]\d[0123]\d)')
def gemini_daterange(string):
  """
  A utility function for matching date ranges of the form YYYYMMDD-YYYYMMDD
  Could make this support today and yesterday, but right now it doesn't
  Also this does not yet check for sensible date ordering
  returns the YYYYMMDD-YYYYMMDD string, or '' if not a daterange
  """
  if(daterangecre.match(string)):
    return string
  else:
    return ''

def gemini_obstype(string):
  """
  A utility function for matching Gemini ObsTypes
  If the string argument matches a gemini ObsType 
  then we return the obstype
  Otherwise return an empty string
  We add the unofficial values PINHOLE for GNIRS pinhole mask observations and RONCHI for NIFS Ronchi mask observations here too
  """
  list = ['DARK', 'ARC', 'FLAT', 'BIAS', 'OBJECT', 'PINHOLE', 'RONCHI']
  retary = ''
  if(string in list):
    retary = string
  return retary
  
def gemini_obsclass(string):
  """
  A utility function matching Gemini ObsClasses
  If the string argument matches a gemini ObsClass then we return 
  the obsclass
  Otherwise we return an empty string
  """
  list = ['dayCal', 'partnerCal', 'acqCal', 'acq', 'science', 'progCal']
  retary = ''
  if (string in list):
    retary = string
  return retary

def gemini_caltype(string):
  """
  A utility function matching Gemini calibration types.
  If the string argument matches a gemini calibration type then we return
  the calibration type, otherwise we return an empty string

  The list of calibration types is somewhat arbitrary, it's not coupled
  to the DHS or ODB, it's more or less defined by the Fits Storage project

  These should all be lower case so as to avoid conflict with gemini_obstype
  """
  list = ['bias', 'dark', 'flat', 'arc', 'processed_bias', 'processed_flat']
  retary = ''
  if (string in list):
    retary = string
  return retary

def gmos_gratingname(string):
  """
  A utility function matching a GMOS Grating name. This could be expanded to
  other instruments, but for many instruments the grating name is too ambiguous and 
  could be confused with a filter or band (eg 'H'). Also most of the use cases 
  for this are where gratings are swapped.

  This function doesn't match or return the component ID.

  If the string argument matches a grating, we return the official name for
  that grating.
  """
  retary = ''
  list = ['MIRROR', 'B600', 'R600', 'R400', 'R831', 'R150', 'B1200']
  if (string in list):
    retary = string
  return retary

gmosfpmaskcre = re.compile('^G[NS](20\d\d)[AB](.)(\d\d\d)-(\d\d)$')
def gmos_fpmask(string):
  """
  A utility function matching gmos focal plane mask names. This could be expanded to
  other instruments. Most of the uses cases for this are for masks that are swapped.
  This function knows the names of the facility masks (long slits, NSlongslits and IFUs)
  Also it knows the form of the MOS mask names and will return a mosmask name if the string
  matches that format, even if that maskname does not actually exist

  If the string matches an fpmask, we return the fpmask.
  """

  retary = ''
  facility = ['NS2.0arcsec', 'IFU-R', 'focus_array_new', 'Imaging', 'IFU', '2.0arcsec', 'NS1.0arcsec', 'NS0.75arcsec', '5.0arcsec', '1.5arcsec', 'IFU-2', 'NS1.5arcsec', '0.75arcsec', '1.0arcsec', '0.5arcsec']

  retary = None
  if(string in facility):
    retary = string
  elif(gmosfpmaskcre.match(string)):
    retary = string

  return retary
 
fitsfilenamecre = re.compile('^([NS])(20\d\d)([01]\d[0123]\d)(S)(\d\d\d\d)([\d-]*)(?P<fits>.fits)?$')
vfitsfilenamecre = re.compile('^(20)?(\d\d)(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)(\d\d)_(\d+)(?P<fits>.fits)?$')
def gemini_fitsfilename(string):
  """
  A utility function matching Gemini raw data fits filenames
  If the string argument matches the format of a gemini
  data filename, with or without the .fits on the end, this
  function will return the filename, with the .fits on the end.

  If the string does not look like a filename, we return an empty string.
  """
  retval = ''
  m = fitsfilenamecre.match(string) or vfitsfilenamecre.match(string)
  if(m):
    # Yes, but does it not have a .fits?
    if(m.group('fits') == None):
      retval = "%s.fits" % string
    else:
      retval = string

  return retval
    
  
# The Gemini Data Label Class

# This re matches progid-obsum-dlnum - ie a datalabel,
# With 3 groups - progid, obsnum, dlnum
dlcre=re.compile('^((?:%s)|(?:%s))-(\d*)-(\d*)$' % (calengre, scire))

class GeminiDataLabel:
  """
  The Gemini Data Label Class. This class parses various
  useful things from a Gemini Data Label.

  Simply instantiate the class with the datalabel, then
  reference the following data members:

  * datalabel: The datalabel provided. If the class could cannot
               make sense of the datalabel string passed in,
               this field will be empty.
  * projectid: The Project ID
  * obsid: The Observation ID
  * obsnum: The Observation Number within the project
  * dlnum: The Dataset Number within the observation
  * project: A GeminiProject object for the project this is part of
  """
  datalabel = ''
  projectid = ''
  obsid = ''
  obsnum = ''
  dlnum = ''
  project = ''

  def __init__(self, dl):
    self.datalabel = dl
    self.projectid = ''
    self.obsid = ''
    self.obsnum = ''
    self.dlnum = ''
    if(self.datalabel):
      self.parse()

  def parse(self):
    dlm=dlcre.match(self.datalabel)
    if(dlm):
      self.projectid = dlm.group(1)
      self.obsnum = dlm.group(2)
      self.dlnum = dlm.group(3)
      self.project = GeminiProject(self.projectid)
      self.obsid='%s-%s' % (self.projectid, self.obsnum)
    else:
      # Match failed - Null the datalabel field
      self.datalabel=''

# This matches an observation id
obscre = re.compile(obsre)

class GeminiObservation:
  """
  The GeminiObservation class parses an observation ID

  Simply instantiate the class with an observation id string
  then reference the following data members:

  * obsid: The observation ID provided. If the class cannot
           make sense of the string passed in, this field will
           be empty
  * project: A GeminiProject object for the project this is part of
  * obsnum: The observation numer within the project
  """
  obsid = ''
  project = ''
  obsnum =''

  def __init__(self, obsid):
    if(obsid):
      match = obscre.match(obsid)
      if(match):
        self.obsid = obsid
        self.project = GeminiProject(match.group(1))
        self.obsnum = match.group(2)
      else:
        self.obsid = ''
        self.project=''
        self.obsnum=''
    else:
      self.obsid = ''

# This matches a program id
progcre=re.compile(progre)

# this matches a cal or eng projectid with CAL or ENG and the date as matched groups
cecre=re.compile('G[NS]-((?:CAL)|(?:ENG))(20\d\d[01]\d[0123]\d)')

class GeminiProject:
  """
  The GeminiProject class parses a Gemini Project ID and provides
  various useful information deduced from it.

  Simply instantiate the class with a project ID string, then
  referernce the following data members:

  * progid: The program ID passed in. If the class could not 
            make sense of the string, this will be empty.
  * iscal: a Boolean that is true if this is a CAL project
  * iseng: a Boolean that is true if this is an ENG project
  """
  progid = ''
  iscal = ''
  iseng = ''

  def __init__(self, progid):
    if(progcre.match(progid)):
      self.progid = progid
      self.parse()
    else:
      self.progid=''
      iscal = False
      iseng = False

  def parse(self):
    cem=cecre.match(self.progid)
    if(cem):
      caleng = cem.group(1)
      self.iseng = (caleng == 'ENG')
      self.iscal = (caleng == 'CAL')

