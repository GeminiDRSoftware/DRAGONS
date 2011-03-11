"""
This is the Fits Storage Web Summary module. It provides the functions
which query the database and generate html for the web header
summaries.
"""
from sqlalchemy import or_
from sqlalchemy.sql.expression import alias, cast
from FitsStorage import *
from GeminiMetadataUtils import *
from FitsStorageConfig import *

import os

class stub:
    pass
    
if fsc_localmode:
    apache = stub()
    apache.OK = True
    
try:
    from mod_python import apache, util
except ImportError:
    pass
    
import FitsStorageCal
import FitsStorageConfig
import urllib

def summary(req, type, selection, orderby, links=True):
  """
  This is the main summary generator.
  req is an apache request handler request object
  type is the summary type required
  selection is an array of items to select on, simply passed
    through to the webhdrsummary function
  orderby specifies how to order the output table, simply
    passed through to the webhdrsummary function

  returns an apache request status code

  This function outputs header and footer for the html page,
  and calls the webhdrsummary function to actually generate
  the html table containing the actually summary information.
  """
  req.content_type = "text/html"
  req.write("<html>")
  title = "FITS header %s table %s" % (type, sayselection(selection))
  req.write("<head>")
  req.write("<title>%s</title>" % (title))
  req.write('<link rel="stylesheet" href="/htmldocs/table.css">')
  req.write("</head>\n")
  req.write("<body>")
  if (fits_system_status == "development"):
    req.write('<h1>This is the development system, please use <a href="http://fits/">fits</a> for operational use</h1>')
  req.write("<H1>%s</H1>" % (title))

  # If this is a diskfiles summary, select even ones that are not canonical
  if(type != 'diskfiles'):
    # Usually, we want to only select headers with diskfiles that are canonical
    selection['canonical']=True

  session = sessionfactory()
  try:
    webhdrsummary(session, req, type, list_headers(session, selection, orderby), links)
  except IOError:
    pass
  finally:
    session.close()

  req.write("</body></html>")
  return apache.OK

def list_headers(session, selection, orderby):
  """
  This function queries the database for a list of header table 
  entries that satsify the selection criteria.

  session is an sqlalchemy session on the database
  selection is a dictionary containing fields to select on
  orderby is a list of fields to sort the results by

  Returns a list of Header objects
  """
  localmode = fsc_localmode
  # The basic query...
  if localmode:
    query = session.query(Header).select_from(Header, DiskFile, File)
    query = query.filter(Header.diskfile_id == DiskFile.id)
    query = query.filter(DiskFile.file_id == File.id)
  else:
    query = session.query(Header).select_from(join(Header, join(DiskFile, File)))
  query = queryselection(query, selection)

  # Do we have any order by arguments?

  whichorderby = ['instrument', 'data_label', 'observation_class', 'airmass', 'ut_datetime', 'local_time', 'raw_iq', 'raw_cc', 'raw_bg', 'raw_wv', 'qa_state', 'filter_name', 'exposure_time', 'object']
  if (orderby):
    for i in range(len(orderby)):
      if '_desc' in orderby[i]:
        orderby[i] = orderby[i].replace('_desc', '')
        if (orderby[i] == 'filename'):
          query = query.order_by(desc('File.%s' % orderby[i]))
        if orderby[i] in whichorderby:
          query = query.order_by(desc('Header.%s' % orderby[i]))
      else:
        if '_asc' in orderby[i]:
          orderby[i] = orderby[i].replace('_asc', '')
        if (orderby[i] == 'filename'):
          query = query.order_by('File.%s' % orderby[i])
        if orderby[i] in whichorderby:
          query = query.order_by('Header.%s' % orderby[i])


  # If this is an open query, we should reverse sort by filename
  # and limit the number of responses
  if(openquery(selection)):
    query = query.order_by(desc(File.filename))
    query = query.limit(1000)
  else:
    # By default we should order by filename
    query = query.order_by(File.filename)

  if localmode:
    results = query.all()
    # print "FSWS126:", repr(results)
    # print "FSWS127:", str(results)
    headers = results
    # headers, diskfiles, files = zip(*results)
  else:
    headers = query.all()
  
  # Return the list of DiskFile objects
  return headers

def webhdrsummary(session, req, type, headers, links=True):
  """
  Generates an HTML header summary table of the specified type from
  the list of header objects provided. Writes that table to an apache
  request object.

  session: sqlalchemy database session
  req: the apache request object to write the output
  type: the summary type required
  headers: the list of header objects to include in the summary
  """
  # Get the uri to use for the re-sort links
  try:
    myuri = req.uri
  except AttributeError:
    myuri = "localmode"
  # A certain amount of parsing the summary type...
  want=[]
  if(type == 'summary'):
    want.append('obs')
    want.append('qa')
    want.append('expamlt')
  if(type == 'ssummary'):
    want.append('obs')
    want.append('qa')
  if(type == 'diskfiles'):
    want.append('diskfiles')

  # Output the start of the table including column headings
  # First part included in all summary types
  req.write('<TABLE border=0>')
  req.write('<TR class=tr_head>')
  if(links):
    req.write('<TH>Filename <a href="%s?orderby=filename_asc">&uarr</a><a href="%s?orderby=filename_desc">&darr</a></TH>' % (myuri, myuri))
    req.write('<TH>Data Label <a href="%s?orderby=data_label_asc">&uarr</a><a href="%s?orderby=data_label_desc">&darr</a></TH>' % (myuri, myuri))
    req.write('<TH>UT Date Time <a href="%s?orderby=ut_datetime_asc">&uarr</a><a href="%s?orderby=ut_datetime_desc">&darr</a></TH>' % (myuri, myuri))
    req.write('<TH><abbr title="Instrument">Inst</abbr> <a href="%s?orderby=instrument_asc">&uarr</a><a href="%s?orderby=instrument_desc">&darr</a></TH>' % (myuri, myuri))
  else:
    req.write('<TH>Filename</TH>')
    req.write('<TH>Data Label</TH>')
    req.write('<TH>UT Date Time</TH>')
    req.write('<TH><abbr title="Instrument">Inst</abbr></TH>')

  # This is the 'obs', 'expamlt' and 'qa' parts
  wants = ['obs', 'expamlt', 'qa']
  for w in wants:
    if w in want:
      if(w == 'obs'):
        vals = [['ObsClass', 'Class', 'observation_class'], ['ObsType', 'Type', 'observation_type'], ['Object Name', 'Object', 'object']]
      elif(w == 'expamlt'):
        vals = [['Exposure Time', 'ExpT', 'exposure_time'], ['AirMass', 'AM', 'airmass'], ['Localtime', 'Lcltime', 'local_time']]
      elif(w == 'qa'):
        vals = [['QA State', 'QA', 'qa_state'], ['Raw IQ', 'IQ', 'raw_iq'], ['Raw CC', 'CC', 'raw_cc'], ['Raw WV', 'WV', 'raw_wv'], ['Raw BG', 'BG', 'raw_bg']]

      if(links):
        for i in range(len(vals)):
          req.write('<TH><abbr title="%s">%s</abbr> <a href="%s?orderby=%s_asc">&uarr</a><a href="%s?orderby=%s_desc">&darr</a></TH>' % (vals[i][0], vals[i][1], myuri, vals[i][2], myuri, vals[i][2]))
      else:
        for i in range(len(vals)):
          req.write('<TH><abbr title="%s">%s</abbr></TH>' % (vals[i][0], vals[i][1]))
      if(w == 'obs'):
        req.write('<TH><abbr title="Imaging Filter or Spectroscopy Wavelength and Disperser">WaveBand<abbr></TH>')
 

  # This is the 'diskfiles' part
  if('diskfiles' in want):
    req.write('<TH>Present</TH>')
    req.write('<TH>Entry</TH>')
    req.write('<TH>Lastmod</TH>')
    req.write('<TH>Size</TH>')
    req.write('<TH>CCRC</TH>')

  # Last bit included in all summary types
  req.write('</TR>')

  # Loop through the header list, outputing table rows
  even=0
  bytecount = 0
  filecount = 0
  for h in headers:
    even = not even
    if(even):
      cs = "tr_even"
    else:
      cs = "tr_odd"
    # Again, the first part included in all summary types
    req.write("<TR class=%s>" % (cs))

    # Parse the datalabel first
    dl = GeminiDataLabel(h.data_label)

    # The filename cell, with the link to the full headers and the optional WMD and FITS error flags
    if(h.diskfile.fverrors):
      if(links):
        fve='<a href="/fitsverify/%d">- fits!</a>' % (h.diskfile.id)
      else:
        fve='- fits!'
    else:
      fve=''
    # Do not raise the WMD flag on ENG data
    iseng = bool(dl.datalabel) and dl.project.iseng
    if((not iseng) and (not h.diskfile.wmdready)):
      if(links):
        wmd='<a href="/wmdreport/%d">- md!</a>' % (h.diskfile.id)
      else:
        wmd='- md!'
    else:
      wmd=''

    if(links):
      req.write('<TD><A HREF="/fullheader/%s">%s</A> %s %s</TD>' % (h.diskfile.file.filename, h.diskfile.file.filename, fve, wmd))
    else:
      req.write('<TD>%s %s %s</TD>' % (h.diskfile.file.filename, fve, wmd))


    # The datalabel, parsed to link to the program_id and observation_id,
    if(dl.datalabel):
      if(links):
        req.write('<TD><NOBR><a href="/summary/%s">%s</a>-<a href="/summary/%s">%s</a>-<a href="/summary/%s">%s</a></NOBR></TD>' % (dl.projectid, dl.projectid, dl.observation_id, dl.obsnum, dl.datalabel, dl.dlnum))
      else:
        req.write('<TD><NOBR>%s-%s-%s</NOBR></TD>' % (dl.projectid, dl.obsnum, dl.dlnum))
    else:
      req.write('<TD>%s</TD>' % h.data_label)

    if(h.ut_datetime):
      req.write("<TD><NORB>%s</NOBR></TD>" % (h.ut_datetime.strftime("%Y-%m-%d %H:%M:%S")))
    else:
      req.write("<TD>%s</TD>" % ("None"))

    inst = h.instrument
    if(h.adaptive_optics):
      inst += " + AO"
    req.write("<TD>%s</TD>" % (inst))

    # Now the 'obs' part
    if('obs' in want):
      req.write("<TD>%s</TD>" % (h.observation_class))
      req.write("<TD>%s</TD>" % (h.observation_type))
      if (h.object and len(h.object)>12):
        req.write('<TD><abbr title="%s">%s</abbr></TD>' % (h.object, (h.object)[0:12]))
      else:
        req.write("<TD>%s</TD>" % (h.object))

      if(h.spectroscopy):
        try:
          req.write("<TD>%s : %.3f</TD>" % (h.disperser, h.central_wavelength))
        except:
          req.write("<TD>%s : </TD>" % (h.disperser))
      else:
        req.write("<TD>%s</TD>" % (h.filter_name))

    # Now the 'expamlt' part
    if ('expamlt' in want):
      try:
        req.write("<TD>%.2f</TD>" % h.exposure_time)
      except:
        req.write("<TD></TD>")
 
      try:
        req.write("<TD>%.2f</TD>" % h.airmass)
      except:
        req.write("<TD></TD>")

      if(h.local_time):
        req.write("<TD>%s</TD>" % (h.local_time.strftime("%H:%M:%S")))
      else:
        req.write("<TD>%s</TD>" % ("None"))

    # Now the 'qa' part
    # Abreviate the raw XX values to 4 characters
    if('qa' in want):
      req.write('<TD>%s</TD>' % (h.qa_state))

      if(h.raw_iq and percentilecre.match(h.raw_iq)):
        req.write('<TD><abbr title="%s">%s</abbr></TD>' % (h.raw_iq, h.raw_iq[0:4]))
      else:
        req.write('<TD>%s</TD>' % (h.raw_iq))

      if(h.raw_cc and percentilecre.match(h.raw_cc)):
        req.write('<TD><abbr title="%s">%s</abbr></TD>' % (h.raw_cc, h.raw_cc[0:4]))
      else:
        req.write('<TD>%s</TD>' % (h.raw_cc))

      if(h.raw_wv and percentilecre.match(h.raw_wv)):
        req.write('<TD><abbr title="%s">%s</abbr></TD>' % (h.raw_wv, h.raw_wv[0:4]))
      else:
        req.write('<TD>%s</TD>' % (h.raw_wv))
 
      if(h.raw_bg and percentilecre.match(h.raw_bg)):
        req.write('<TD><abbr title="%s">%s</abbr></TD>' % (h.raw_bg, h.raw_bg[0:4]))
      else:
        req.write('<TD>%s</TD>' % (h.raw_bg))

    # the 'diskfiles' part
    if('diskfiles' in want):
      req.write("<TD>%s</TD>" % (h.diskfile.present))
      req.write("<TD>%s</TD>" % (h.diskfile.entrytime))
      req.write("<TD>%s</TD>" % (h.diskfile.lastmod))
      req.write("<TD>%s</TD>" % (h.diskfile.size))
      req.write("<TD>%s</TD>" % (h.diskfile.ccrc))

    # And again last bit included in all summary types
    req.write("</TR>\n")

    bytecount += h.diskfile.size
    filecount += 1
  req.write("</TABLE>\n")
  req.write("<P>%d files totalling %.2f GB</P>" % (filecount, bytecount/1.0E9))

def xmlfilelist(req, selection):
  """
  This generates an xml list of the files that met the selection
  """
  req.content_type = "text/xml"
  req.write('<?xml version="1.0" ?>')
  req.write("<file_list>")

  session = sessionfactory()
  orderby = ['filename_asc']
  selection['present']=True
  try:
    headers = list_headers(session, selection, orderby)
    for h in headers:
      req.write("<file>")
      req.write("<filename>%s</filename>" % h.diskfile.file.filename)
      req.write("<size>%d</size>" % h.diskfile.size)
      req.write("<md5>%s</md5>" % h.diskfile.md5)
      req.write("<ccrc>%s</ccrc>" % h.diskfile.ccrc)
      req.write("<lastmod>%s</lastmod>" % h.diskfile.lastmod)
      req.write("</file>")
  finally:
    session.close()
  req.write("</file_list>")
  return apache.OK

def fileontape(req, things):
  """
  Outputs xml describing the tapes that the specified file is on
  """
  req.content_type = "text/xml"
  req.write('<?xml version="1.0" ?>')
  req.write("<file_list>")

  filename = things[0]

  session = sessionfactory()
  try:
    query = session.query(TapeFile).select_from(join(TapeFile, join(TapeWrite, Tape)))
    query = query.filter(Tape.active == True).filter(TapeWrite.suceeded == True)
    query = query.filter(TapeFile.filename == filename)
    list = query.all()

    for tf in list:
      req.write("<file>")
      req.write("<filename>%s</filename>" % tf.filename)
      req.write("<size>%d</size>" % tf.size)
      req.write("<md5>%s</md5>" % tf.md5)
      req.write("<ccrc>%s</ccrc>" % tf.ccrc)
      req.write("<lastmod>%s</lastmod>" % tf.lastmod)
      req.write("<tapeid>%d</tapeid>" % tf.tapewrite.tape.id)
      req.write("<tapeset>%d</tapeset>" % tf.tapewrite.tape.set)
      req.write("</file>")

  finally:
    session.close()

  req.write("</file_list>")
  return apache.OK



def progsobserved(req, selection):
  """
  This function generates a list of programs observed on a given night
  """

  # Get a database session
  session = sessionfactory()
  try:
    # the basic query in this case
    query = session.query(Header.program_id).select_from(join(Header, join(DiskFile, File)))

    # Add the selection criteria
    query = queryselection(query, selection)

    # And the group by clause
    query = query.group_by(Header.program_id)

    list = query.all()
    title = "Programs Observed: %s" % sayselection(selection)
    req.content_type = "text/html"
    req.write('<html><head><title>%s</title></head><body><h1>%s</h1>' % (title, title))
    req.write('<H2>To paste into nightlog: </H2>')
    req.write('<P>')
    for row in list:
      p = row[0]
      if(p):
        req.write('%s ' % p)
    req.write('</P>')

    req.write('<H2>With more detail: </H2>')
    req.write('<UL>')
    for row in list:
      p = row[0]
      if(p):
        req.write('<LI><a href="/summary/%s/%s">%s</a></LI> ' % (p, '/'.join(selection.values()), p))
    req.write('</UL>')
    req.write('</body></html>')
    return apache.OK


  except IOError:
    pass
  finally:
    session.close()


def gmoscal(req, selection):
   """
   This generates a GMOS imaging twilight flat report.
   And a BIAS report.
   If no date or daterange is given, tries to find last processing date
   """

   title = "GMOS Cal (Imaging Twilight Flats and Biases) Report %s" % sayselection(selection)
   req.content_type = "text/html"
   req.write('<html><head><title>%s</title><link rel="stylesheet" href="/htmldocs/table.css"></head><body><h1>%s</h1>' % (title, title))
   if(fits_system_status == 'development'):
     req.write("<H1>This is the Development Server, not the operational system. If you're not sure why you're seeing this message, please consult PH</H1>")
 
   # If no date or daterange, look on endor or josie to get the last processing date
   if(('date' not in selection) and ('daterange' not in selection)):
     base_dir=das_calproc_path
     checkfile = 'Basecalib/biasall.list'
     enddate = datetime.datetime.now().date()
     oneday = datetime.timedelta(days=1)
     date = enddate
     found = -1000
     startdate = None
     while(found < 0):
       datestr = date.strftime("%Y%b%d").lower()
       file = os.path.join(base_dir, datestr, checkfile)
       if(os.path.exists(file)):
         found = 1
         startdate = date
       date -= oneday
       found += 1

       if(startdate):
         # Start the day after the last reduction
         startdate += oneday
         selection['daterange']="%s-%s" % (startdate.strftime("%Y%m%d"), enddate.strftime("%Y%m%d"))
         req.write("<H2>Auto-detecting Last Processing Date: %s<H2>" % selection['daterange'])

   # Get a database session
   session = sessionfactory()
   try:
     # First the Twilight Flats part
     req.write('<H2>Twilight Flats</H2>')

     # We do this twice, first for the science data, then for the twilight flat data
     # These are differentiated by being science or dayCal

     # Put the results into dictionaries, which we can then combine into one html table
     sci = {}
     tlf = {}
     for observation_class in (['science', 'dayCal']):

       # The basic query for this
       query = session.query(func.count(1), Header.filter_name, Gmos.detector_x_bin, Gmos.detector_y_bin).select_from(join(Gmos, join(Header, join(DiskFile, File))))
       query = query.filter(DiskFile.canonical == True)

       # Fudge and add the selection criteria
       selection['observation_class']=observation_class
       selection['observation_type']='OBJECT'
       selection['spectroscopy']=False
       selection['inst']='GMOS'
       if(observation_class == 'science'):
         selection['qa_state']='Win'
       else:
         selection['qa_state']='Pass'
         # Only select full frame dayCals
         query = query.filter(or_(Gmos.amp_read_area == '''["'EEV 9273-16-03, right':[1:2048,1:4608]", "'EEV 9273-20-04, right':[2049:4096,1:4608]", "'EEV 9273-20-03, left':[4097:6144,1:4608]"]''', Gmos.amp_read_area == '''["'EEV 2037-06-03, left':[1:2048,1:4608]", "'EEV 8194-19-04, left':[2049:4096,1:4608]", "'EEV 8261-07-04, right':[4097:6144,1:4608]"]'''))

       query = queryselection(query, selection)
  
       # Knock out ENG programs
       query = query.filter(~Header.program_id.like('%ENG%'))

       # Group by clause
       query = query.group_by(Header.filter_name, Gmos.detector_x_bin, Gmos.detector_y_bin).order_by(Gmos.detector_x_bin, Header.filter_name)

       list = query.all()

       # Populate the dictionary
       # as {'i-2x2':[10, 'i', '2x2'], ...}  ie [number, filter_name, binning]
       if(observation_class == 'science'):
         dict = sci
       else:
         dict = tlf

       for row in list:
         binning = "%dx%d" % (row[2], row[3])
         key = "%s-%s" % (row[1], binning)
         dict[key]=[row[0], row[1], binning]

     # Make the master dictionary
     # as {'i-2x2':[10, 20, 'i', '2x2'], ...}   [n_sci, n_tlf, filter_name, binning]
     all = {}
     for key in sci.keys():
       nsci = sci[key][0]
       ntlf = 0
       filter_name = sci[key][1]
       binning = sci[key][2]
       all[key] = [nsci, ntlf, filter_name, binning]
     for key in tlf.keys():
       if (key in all.keys()):
         all[key][1] = tlf[key][0]
       else:
         nsci = 0
         ntlf = tlf[key][0]
         filter_name = tlf[key][1]
         binning = tlf[key][2]
         all[key] = [nsci, ntlf, filter_name, binning]
     
     
     # Output the HTML table and links to summaries etc
     req.write('<TABLE border=0>')
     req.write('<TR class=tr_head>')
     req.write('<TH>Number of Science Frames</TH>')
     req.write('<TH>Number of Twilight Frames</TH>')
     req.write('<TH>Filter</TH>')
     req.write('<TH>Binning</TH>')
     req.write('</TR>')
     
     even=False
     keys = all.keys()
     keys.sort(reverse=True)
     for key in keys:
       even = not even
       if(even):
         if((all[key][0] > 0) and (all[key][1] == 0)):
           cs = "tr_warneven"
         else:
           cs = "tr_even"
       else:
         if((all[key][0] > 0) and (all[key][1] == 0)):
           cs = "tr_warnodd"
         else:
           cs = "tr_odd"

       req.write("<TR class=%s>" % cs)

       for i in range(4):
         req.write("<TD>%d</TD>" % all[key][i])

       req.write("</TR>")
     req.write("</TABLE>")
     datething=''
     if('date' in selection):
       datething = selection['date']
     if('daterange' in selection):
       datething = selection['daterange']
     req.write('<P><a href="/summary/GMOS/imaging/OBJECT/science/Win/%s">Science Frames Summary Table</a></P>' % datething)
     req.write('<P><a href="/summary/GMOS/imaging/OBJECT/dayCal/Pass/%s">Twilight Flat Summary Table</a></P>' % datething)
     req.write('<P>NB. Summary tables will show ENG program data not reflected in the counts above.</P>')

     # Now the BIAS report
     req.write('<H2>Biases</H2>')

     #tzoffset = datetime.timedelta(seconds=time.timezone)
     #hack to make hbffits1 look chilean for tmp work Dec 2010
     tzoffset = datetime.timedelta(seconds=14400)
     
     oneday = datetime.timedelta(days=1)
     offset = sqlalchemy.sql.expression.literal(tzoffset - oneday, sqlalchemy.types.Interval)
     query = session.query(func.count(1), cast((Header.ut_datetime + offset), sqlalchemy.types.DATE).label('utdate'), Gmos.detector_x_bin, Gmos.detector_y_bin, Gmos.amp_read_area).select_from(join(Gmos, join(Header, join(DiskFile, File))))

     query = query.filter(DiskFile.canonical == True)

     # Fudge and add the selection criteria
     selection['observation_type']='BIAS'
     selection['inst']='GMOS'
     selection['qa_state']='Pass'
     query = queryselection(query, selection)

     query = query.group_by('utdate', Gmos.detector_x_bin, Gmos.detector_y_bin, Gmos.amp_read_area).order_by('utdate', Gmos.detector_x_bin, Gmos.detector_y_bin, Gmos.amp_read_area)

     list = query.all()

     # OK, re-organise results into tally table dict
     # dict is: {utdate: {binning: {roi: Number}}
     dict={}
     for row in list:
       # Parse the element numbers for simplicity
       num = row[0]
       utdate = row[1]
       binning = "%dx%d" % (row[2], row[3])
       roi = row[4]
       if(roi == '''["'EEV 9273-16-03, right':[1:2048,1:4608]", "'EEV 9273-20-04, right':[2049:4096,1:4608]", "'EEV 9273-20-03, left':[4097:6144,1:4608]"]'''):
         roi = "Full"
       if(roi == '''["'EEV 9273-16-03, right':[1:2048,1792:2815]", "'EEV 9273-20-04, right':[2049:4096,1792:2815]", "'EEV 9273-20-03, left':[4097:6144,1792:2815]"]'''):
         roi = "Cent"
       if(roi == '''["'EEV 2037-06-03, left':[1:2048,1:4608]", "'EEV 8194-19-04, left':[2049:4096,1:4608]", "'EEV 8261-07-04, right':[4097:6144,1:4608]"]'''):
         roi = "Full"
       if(roi == '''["'EEV 2037-06-03, left':[1:2048,1792:2815]", "'EEV 8194-19-04, left':[2049:4096,1792:2815]", "'EEV 8261-07-04, right':[4097:6144,1792:2815]"]'''):
         roi = "Cent"

       if(utdate not in dict.keys()):
         dict[utdate]={}
       if(binning not in dict[utdate].keys()):
         dict[utdate][binning] = {}
       if(roi not in dict[utdate][binning].keys()):
         dict[utdate][binning][roi] = num

     # Output the HTML table 
     # While we do it, add up the totals as a simply column tally
     binlist = ['1x1', '2x2', '2x1', '1x2', '2x4', '4x2', '4x1', '1x4', '4x4']
     roilist = ['Full', 'Cent']
     req.write('<TABLE border=0>')
     req.write('<TR class=tr_head>')
     req.write('<TH rowspan=2>UT Date</TH>')
     for b in binlist:
       req.write('<TH colspan=2>%s</TH>' %b)
     req.write('</TR>')
     req.write('<TR class=tr_head>')
     for b in binlist:
       for r in roilist:
         req.write('<TH>%s</TH>'% r)
     req.write('</TR>')

     even=False
     utdates = dict.keys()
     utdates.sort(reverse=True)
     total=[]
     for i in range(0, len(binlist)*len(roilist)):
       total.append(0)

     for utdate in utdates:
       even = not even
       if(even):
         cs = "tr_even"
       else:
         cs = "tr_odd"

       req.write("<TR class=%s>" % cs)
       req.write("<TD>%s</TD>" % utdate)
       i=0
       for b in binlist:
         for r in roilist:
           try:
             num = dict[utdate][b][r]
           except KeyError:
             num = 0
           total[i] += num
           i += 1
           req.write("<TD>%d</TD>" % num)
       req.write("</TR>")

     req.write("<TR class=tr_head>")
     req.write("<TH>%s</TH>" % 'Total')
     for t in total:
       req.write("<TH>%d</TH>" % t)
     req.write("</TR>")
     req.write("</TABLE>")

     # OK, find if there were dates for which there were no biases...
     # Can only do this if we got a daterange selection, otherwise it's broken if there's none on the first or last day
     # utdates is a reverse sorted list for which there were biases.
     if('daterange' in selection):
       # Parse the date to start and end datetime objects
       daterangecre=re.compile('(20\d\d[01]\d[0123]\d)-(20\d\d[01]\d[0123]\d)')
       m = daterangecre.match(selection['daterange'])
       startdate = m.group(1)
       enddate = m.group(2)
       tzoffset = datetime.timedelta(seconds=time.timezone)
       oneday = datetime.timedelta(days=1)
       startdt = dateutil.parser.parse("%s 14:00:00" % startdate)
       startdt = startdt + tzoffset - oneday
       enddt = dateutil.parser.parse("%s 14:00:00" % enddate)
       enddt = enddt + tzoffset - oneday
       enddt = enddt + oneday
       # Flip them round if reversed
       if(startdt > enddt):
         tmp = enddt
         enddt = startdt
         startdt = tmp
       startdate = startdt.date()
       enddate = enddt.date()

       nobiases = []
       date = startdate
       while(date <= enddate):
         if(date not in utdates): 
           nobiases.append(str(date))
         date += oneday

       req.write('<P>There were %d dates with no biases: ' % len(nobiases))
       if(len(nobiases)>0):
         req.write(', '.join(nobiases))
       req.write('</P>')

     req.write("</body></html>")
     return apache.OK


   except IOError:
     pass
   finally:
     session.close()


def tape(req, things):
  """
  This is the tape list function
  """
  req.content_type="text/html"
  req.write("<html>")
  req.write("<head><title>FITS Storage tape information</title></head>")
  req.write("<body>")
  req.write("<h1>FITS Storage tape information</h1>")

  session = sessionfactory()
  try:
    # Process form data first
    formdata = util.FieldStorage(req)
    #req.write(str(formdata) )
    for key in formdata.keys():
      field=key.split('-')[0]
      tapeid=int(key.split('-')[1])
      value = formdata[key].value
      if(tapeid):
        tape=session.query(Tape).filter(Tape.id==tapeid).one()
        if(field == 'moveto'):
          tape.location = value
          tape.lastmoved = datetime.datetime.utcnow()
        if(field == 'active'):
          if(value == 'Yes'):
            tape.active = True
          if(value == 'No'):
            tape.active = False
        if(field == 'full'):
          if(value == 'Yes'):
            tape.full = True
          if(value == 'No'):
            tape.full = False
        if(field == 'set'):
          tape.set = value
        if(field == 'fate'):
          tape.fate = value
      if(field == 'newlabel'):
        # Add a new tape to the database
        newtape = Tape(value)
        session.add(newtape)

      session.commit()
    
    query = session.query(Tape)
    # Get a list of the tapes that apply
    if(len(things)):
      searchstring = '%'+things[0]+'%'
      query = query.filter(Tape.label.like(searchstring))
    query=query.order_by(desc(Tape.id))
    list = query.all()

    req.write("<HR>")
    for tape in list:
      req.write("<H2>ID: %d, Label: %s, Set: %d</H2>" % (tape.id, tape.label, tape.set))
      req.write("<UL>")
      req.write("<LI>First Write: %s UTC - Last Write: %s UTC</LI>" % (tape.firstwrite, tape.lastwrite))
      req.write("<LI>Last Verified: %s UTC</LI>" % tape.lastverified)
      req.write("<LI>Location: %s; Last Moved: %s UTC</LI>" % (tape.location, tape.lastmoved))
      req.write("<LI>Active: %s</LI>" % tape.active)
      req.write("<LI>Full: %s</LI>" % tape.full)
      req.write("<LI>Fate: %s</LI>" % tape.fate)
  
      # Count Writes
      twqtotal = session.query(TapeWrite).filter(TapeWrite.tape_id == tape.id)
      twq = session.query(TapeWrite).filter(TapeWrite.tape_id == tape.id).filter(TapeWrite.suceeded == True)
      # Count Bytes
      if(twq.count()):
        bytesquery = session.query(func.sum(TapeWrite.size)).filter(TapeWrite.tape_id == tape.id).filter(TapeWrite.suceeded == True)
        bytes = bytesquery.one()[0]
        if(not bytes):
          bytes=0
      else:
        bytes=0
      req.write('<LI>Sucessfull/Total Writes: <A HREF="/tapewrite/%d">%d/%d</A>. %.2f GB Sucessfully written</LI>' % (tape.id, twq.count(), twqtotal.count(), float(bytes)/1.0E9))
        
      req.write("</UL>")

      # The form for modifications
      req.write('<FORM action="/tape" method="post">')
      req.write('<TABLE>')
      # Row 1
      req.write('<TR>')
      movekey = "moveto-%d" % tape.id
      req.write('<TD><LABEL for="%s">Move to new location:</LABEL></TD>' % movekey)
      req.write('<TD><INPUT type="text" size=32 name="%s"></INPUT></TD>' % movekey)
      req.write('</TR>')
      # Row 2
      req.write('<TR>')
      setkey = "set-%d" % tape.id
      req.write('<TD><LABEL for="%s">Change Set Number to:</LABEL></TD>' % setkey)
      req.write('<TD><INPUT type="text" size=4 name="%s"></INPUT></TD>' % setkey)
      req.write('</TR>')
      # Row 3
      activekey = "active-%d" % tape.id
      req.write('<TR>')
      req.write('<TD><LABEL for="%s">Active:</LABEL></TD>' % activekey)
      yeschecked = ""
      nochecked = ""
      if(tape.active):
        yeschecked="checked"
      else:
        nochecked="checked"
      req.write('<TD><INPUT type="radio" name="%s" value="Yes" %s>Yes</INPUT> ' % (activekey, yeschecked))
      req.write('<INPUT type="radio" name="%s" value="No" %s>No</INPUT></TD>' % (activekey, nochecked))
      req.write('</TR>')
      # Row 4
      fullkey = "full-%d" % tape.id
      req.write('<TR>')
      req.write('<TD><LABEL for="%s">Full:</LABEL></TD>' % fullkey)
      yeschecked = ""
      nochecked = ""
      if(tape.full):
        yeschecked="checked"
      else:
        nochecked="checked"
      req.write('<TD><INPUT type="radio" name="%s" value="Yes" %s>Yes</INPUT> ' % (fullkey, yeschecked))
      req.write('<INPUT type="radio" name="%s" value="No" %s>No</INPUT></TD>' % (fullkey, nochecked))
      req.write('</TR>')
      # Row 5
      req.write('<TR>')
      fatekey = "fate-%d" % tape.id
      req.write('<TD><LABEL for="%s">Fate:</LABEL></TD>' % fatekey)
      req.write('<TD><INPUT type="text" name="%s" size=32></INPUT></TD>' % fatekey)
      req.write('</TR>')
      # End of form
      req.write('</TABLE>')
      req.write('<INPUT type="submit" value="Save"></INPUT> <INPUT type="reset"></INPUT>')
      req.write('</FORM>')
      req.write('<HR>')

    req.write('<HR>')
    req.write('<H2>Add a New Tape</H2>')
    req.write('<FORM action="/tape" method="post">')
    req.write('<LABEL for=newlabel-0>Label</LABEL> <INPUT type="text" size=32 name=newlabel-0></INPUT> <INPUT type="submit" value="Save"></INPUT> <INPUT type="reset"></INPUT>')
    req.write('</FORM>')

    req.write("</body></html>")
    return apache.OK
  except IOError:
    pass
  finally:
    session.close()

def tapewrite(req, things):
  """
  This is the tapewrite list function
  """
  req.content_type="text/html"
  req.write("<html>")
  req.write("<head><title>FITS Storage tapewrite information</title></head>")
  req.write("<body>")
  req.write("<h1>FITS Storage tapewrite information</h1>")

  session = sessionfactory()
  try:

    # Find the appropriate TapeWrite entries
    query = session.query(TapeWrite)

    # Can give a tape id (numeric) or label as an argument
    if(len(things)):
      thing = things[0]
      tapeid=0
      try:
        tapeid = int(thing)
      except:
        pass
      if(tapeid):
        query = query.filter(TapeWrite.tape_id == tapeid)
      else:
        thing = '%'+thing+'%'
        tapequery = session.query(Tape).filter(Tape.label.like(thing))
        if(tapequery.count() == 0):
          req.write("<P>Could not find tape by label search</P>")
          req.write("</body></html>")
          session.close()
          return apache.OK
        if(tapequery.count() > 1):
          req.write("<P>Found multiple tapes by label search. Please give the ID instead</P>")
          req.write("</body></html>")
          return apache.OK
        tape = query.one()
        query = query.filter(TapeWrite.tape_id == tape.id)

    query = query.order_by(desc(TapeWrite.startdate))
    tws = query.all()

    for tw in tws:
      req.write("<h2>ID: %d; Tape ID: %d; Tape Label: %s; File Number: %d</h2>" % (tw.id, tw.tape_id, tw.tape.label, tw.filenum))
      req.write("<UL>")
      req.write("<LI>Start Date: %s UTC - End Date: %s UTC</LI>" % (tw.startdate, tw.enddate))
      req.write("<LI>Suceeded: %s</LI>" % tw.suceeded)
      if(tw.size is None):
        req.write("<LI>Size: None")
      else:
        req.write("<LI>Size: %.2f GB</LI>" % (tw.size / 1.0E9))
      req.write("<LI>Status Before: <CODE>%s</CODE></LI>" % tw.beforestatus)
      req.write("<LI>Status After: <CODE>%s</CODE></LI>" % tw.afterstatus)
      req.write("<LI>Hostname: %s, Tape Device: %s</LI>" % (tw.hostname, tw.tapedrive))
      req.write("<LI>Notes: %s</LI>" % tw.notes)
      req.write('<LI>Files: <A HREF="/tapefile/%d">List</A></LI>' % tw.id)
      req.write("</UL>")
  
    req.write("</BODY></HTML>")
    return apache.OK
  except IOError:
    pass
  finally:
    session.close()

def tapefile(req, things):
  """
  This is the tapefile list function
  """
  req.content_type="text/html"
  req.write("<html>")
  req.write("<head>")
  req.write("<title>FITS Storage tapefile information</title>")
  req.write('<link rel="stylesheet" href="/htmldocs/table.css">')
  req.write("</head>")
  req.write("<body>")
  req.write("<h1>FITS Storage tapefile information</h1>")

  if(len(things) != 1):
    req.write("<P>Must supply one argument - tapewrite_id</P>")
    req.write("</body></html>")
    return apache.OK

  tapewrite_id = things[0]

  session = sessionfactory()
  try:
    query=session.query(TapeFile).filter(TapeFile.tapewrite_id == tapewrite_id).order_by(TapeFile.id)

    req.write('<TABLE border=0>')
    req.write('<TR class=tr_head>')
    req.write('<TH>TapeFile ID</TH>')
    req.write('<TH>TapeWrite ID</TH>')
    req.write('<TH>TapeWrite Start Date</TH>')
    req.write('<TH>Tape ID</TH>')
    req.write('<TH>Tape Label</TH>')
    req.write('<TH>File Num on Tape</TH>')
    req.write('<TH>Filename</TH>')
    req.write('<TH>Size</TH>')
    req.write('<TH>CCRC</TH>')
    req.write('<TH>MD5</TH>')
    req.write('<TH>Last Modified</TH>')
    req.write('</TR>')
  
    even=0
    for tf in query.all():
      even = not even
      if(even):
        cs = "tr_even"
      else:
        cs = "tr_odd"
      # Now the Table Row
      req.write("<TR class=%s>" % (cs))
      req.write("<TD>%d</TD>" % tf.id)
      req.write("<TD>%d</TD>" % tf.tapewrite_id)
      req.write("<TD>%s UTC</TD>" % tf.tapewrite.startdate)
      req.write("<TD>%d</TD>" % tf.tapewrite.tape.id)
      req.write("<TD>%s</TD>" % tf.tapewrite.tape.label)
      req.write("<TD>%d</TD>" % tf.tapewrite.filenum)
      req.write("<TD>%s</TD>" % tf.filename)
      req.write("<TD>%s</TD>" % tf.size)
      req.write("<TD>%s</TD>" % tf.ccrc)
      req.write("<TD>%s</TD>" % tf.md5)
      req.write("<TD>%s</TD>" % tf.lastmod)
      req.write("</TR>")

    req.write("</TABLE></BODY></HTML>")
    return apache.OK
  except IOError:
    pass
  finally:
    session.close()


def taperead(req, things):
  """
  This is the taperead list function
  """
  req.content_type="text/html"
  req.write("<html>")
  req.write("<head>")
  req.write("<title>FITS Storage taperead information</title>")
  req.write('<link rel="stylesheet" href="/htmldocs/table.css">')
  req.write("</head>")
  req.write("<body>")
  req.write("<h1>FITS Storage taperead information</h1>")

  session = sessionfactory()
  try:
    query = session.query(TapeRead).order_by(TapeRead.id)

    req.write('<TABLE border=0>')
    req.write('<TR class=tr_head>')
    req.write('<TH>Filename</TH>')
    req.write('<TH>MD5</TH>')
    req.write('<TH>Tape ID</TH>')
    req.write('<TH>Tape Label</TH>')
    req.write('<TH>File Num on Tape</TH>')
    req.write('<TH>Requester</TH>')
    req.write('</TR>')
  
    even=0
    for tr in query.all():
      even = not even
      if(even):
        cs = "tr_even"
      else:
        cs = "tr_odd"
      # Now the Table Row
      req.write("<TR class=%s>" % (cs))
      req.write("<TD>%s</TD>" % tr.filename)
      req.write("<TD>%s</TD>" % tr.md5)
      req.write("<TD>%d</TD>" % tr.tape_id)
      req.write("<TD>%s</TD>" % tr.tape_label)
      req.write("<TD>%d</TD>" % tr.filenum)
      req.write("<TD>%s</TD>" % tr.requester)
      req.write("</TR>")

    req.write("</TABLE></BODY></HTML>")
    return apache.OK
  except IOError:
    pass
  finally:
    session.close()


def notification(req, things):
  """
  This is the email notifications page. It's both to show the current notifcation list and to update it.
  """
  req.content_type="text/html"
  req.write("<html>")
  req.write("<head><title>FITS Storage new data email notification list</title></head>")
  req.write("<body>")
  req.write("<h1>FITS Storage new data email notification list</h1>")
  req.write("<P>There is a <a href='htmldocs/notificationhelp.html'>help page</a> if you're unsure how to use this.</P>")
  req.write("<HR>")

  session = sessionfactory()
  try:
    # Process form data first
    formdata = util.FieldStorage(req)
    # req.write(str(formdata))
    for key in formdata.keys():
      field=key.split('-')[0]
      id=int(key.split('-')[1])
      value = formdata[key].value
      if(id):
        notif=session.query(Notification).filter(Notification.id==id).first()
        if(field == 'delete' and value == 'Yes'):
          session.delete(notif)
        else:
          if(field == 'newlabel'):
            notif.label = value
          if(field == 'newsel'):
            notif.selection = value
          if(field == 'newto'):
            notif.to = value
          if(field == 'newcc'):
            notif.cc = value
          if(field == 'internal'):
            if(value == 'Yes'):
              notif.internal = True
            if(value == 'No'):
              notif.internal = False
   
      if(field == 'newone'):
        # Add a new notification to the database
        notif = Notification(value)
        session.add(notif)

      session.commit()

    # Get a list of the notifications in the table
    query = session.query(Notification).order_by(Notification.id)
    list = query.all()

    for notif in list:
      req.write("<H2>Notification ID: %d - %s</H2>" % (notif.id, notif.label))
      req.write("<UL>")
      req.write("<LI>Data Selection: %s</LI>" % notif.selection)
      req.write("<LI>Email To: %s</LI>" % notif.to)
      req.write("<LI>Email CC: %s</LI>" % notif.cc)
      req.write("<LI>Gemini Internal: %s</LI>" % notif.internal)
      req.write("</UL>")

      # The form for modifications
      req.write('<FORM action="/notification" method="post">')
      req.write('<TABLE>')

      mod_list = [['newlabel', 'Update notification label'], ['newsel', 'Update data selection'], ['newto', 'Update Email To'], ['newcc', 'Update Email Cc'], ['internal', 'Internal Email'], ['delete', 'Delete']]
      for key in range(len(mod_list)):
        user = mod_list[key][0]+"-%d" % notif.id
        req.write('<TR>')
        req.write('<TD><LABEL for="%s">%s:</LABEL></TD>' % (user, mod_list[key][1]))
        if (mod_list[key][0] == 'internal'):
          yeschecked = ""
          nochecked = ""
          if (notif.internal):
            yeschecked="checked"
          else:
            nochecked="checked"
          req.write('<TD><INPUT type="radio" name="%s" value="Yes" %s>Yes</INPUT> ' % (user, yeschecked))
          req.write('<INPUT type="radio" name="%s" value="No" %s>No</INPUT></TD>' % (user, nochecked))
        elif (mod_list[key][0] == 'delete'):
          yeschecked = ""
          nochecked = "checked"
          req.write('<TD><INPUT type="radio" name="%s" value="Yes" %s>Yes</INPUT> ' % (user, yeschecked))
          req.write('<INPUT type="radio" name="%s" value="No" %s>No</INPUT></TD>' % (user, nochecked))
        else:
          req.write('<TD><INPUT type="text" size=32 name="%s"></INPUT></TD>' % user)
        req.write('</TR>')

      req.write('</TABLE>')
      req.write('<INPUT type="submit" value="Save"></INPUT> <INPUT type="reset"></INPUT>')
      req.write('</FORM>')
      req.write('<HR>')

    req.write('<HR>')
    req.write('<H2>Add a New Notification</H2>')
    req.write('<FORM action="/notification" method="post">')
    req.write('<LABEL for=newone-0>Label</LABEL> <INPUT type="text" size=32 name=newone-0></INPUT> <INPUT type="submit" value="Save"></INPUT> <INPUT type="reset"></INPUT>')
    req.write('</FORM>')

    req.write("</body></html>")
    return apache.OK
  except IOError:
    pass
  finally:
    session.close()



def calmgr(req, selection):
  """
  This is the calibration manager. It implements a machine readable calibration association server
  req is an apache request handler object
  type is the summary type required
  selection is an array of items to select on, simply passed through to the webhdrsummary function
    - in this case, this will usually be a datalabel or filename

  if this code is called via an HTTP POST request rather than a GET, it expects to
  receive a string representation of a python dictionary containing descriptor values
  and a string representation of a python array containg astrodata types
  and it will use this data as the science target details with which to associate 
  the calibration.

  returns an apache request status code
  """
  # this allow me to force the flag back and forth to test integration attempts (we try to 
  # remove the localmode conditionals if possible). fsc_localmode comes from FitsStorageConfig.py
  localmode = fsc_localmode 
  
  session = sessionfactory()
  try:
    # Was the request for only one type of calibration?
    caltype=''
    if('caltype' in selection):
      caltype = selection['caltype']
    else:
      req.content_type="text/plain"
      req.write("<!-- Error: No calibration type specified-->\n")
      return apache.HTTP_NOT_ACCEPTABLE

    # Did we get called via an HTTP POST or HTTP GET?
    if(req.method == 'POST'):
      # OK, get the details from the POST data
      req.content_type = "text/plain"
      clientdata = req.read()
      #req.write("\nclient data: %s\n" % clientdata)
      clientstr = urllib.unquote_plus(clientdata)
      #req.write("\nclient str: %s\n" % clientstr)
      clientlist = clientstr.split('&')
      desc_str = clientlist[0].split('=')[1]
      type_str = clientlist[1].split('=')[1]
      #req.write("\ndesc_str: %s\n" % desc_str)
      #req.write("\ntype_str: %s\n" % type_str)
      descriptors = eval(desc_str)
      types = eval(type_str)
      #req.write("Descriptor Dictionary: %s\n" % descriptors)
      #req.write("Instrument Descriptor: %s\n\n" % descriptors['instrument'])
      #req.write("Types List: %s\n" % types)
      #gn = 'GMOS_N' in types
      #gs = 'GMOS_S' in types
      #req.write("IsType GMOS_N: %s\n" % gn)
      #req.write("IsType GMOS_S: %s\n" % gs)

      # Get a cal object for this target data
      c = FitsStorageCal.get_cal_object(session, None, header=None, descriptors=descriptors, types=types)
      req.content_type = "text/xml"
      req.write('<?xml version="1.0" ?>')
      req.write("<calibration_associations>\n")
      req.write("<dataset>\n")
      req.write("<datalabel>%s</datalabel>\n" % descriptors['data_label'])

      # Call the appropriate method depending what calibration type we want
      cal = None
      if(caltype == 'bias'):
        cal = c.bias()
      if(caltype == 'dark'):
        cal = c.dark()
      if(caltype == 'flat'):
        cal = c.flat()
      if(caltype == 'arc'):
        cal = c.arc()
      if(caltype == 'processed_bias'):
        cal = c.processed_bias()
      if(caltype == 'processed_flat'):
        cal = c.processed_flat()
      if(caltype == 'processed_fringe'):
        cal = c.processed_fringe()
      if(caltype == 'pinhole_mask'):
        cal = c.pinhole_mask()
      if(caltype == 'ronchi_mask'):
        cal = c.ronchi_mask()

      if(cal):
        req.write("<calibration>\n")
        req.write("<caltype>%s</caltype>\n" % caltype)
        req.write("<datalabel>%s</datalabel>\n" % cal.data_label)
        req.write("<filename>%s</filename>\n" % cal.diskfile.file.filename)
        req.write("<md5>%s</md5>\n" % cal.diskfile.md5)
        req.write("<ccrc>%s</ccrc>\n" % cal.diskfile.ccrc)
        req.write("<url>http://%s/file/%s</url>\n" % (fits_servername, cal.diskfile.file.filename))
        req.write("</calibration>\n")
      else:
        req.write("<!-- NO CALIBRATION FOUND-->\n")

      req.write("</dataset>\n");
      req.write("</calibration_associations>\n")


      return apache.OK

    else:
      # OK, we got called via a GET - find the science dataset in the database
      # The Basic Query
      if localmode:
        query = session.query(Header).select_from(Header, DiskFile, File)
        query = query.filter(Header.diskfile_id == DiskFile.id)
        query = query.filter(DiskFile.file_id == File.id)
      else:
        query = session.query(Header).select_from(join(Header, join(DiskFile, File)))

      # Only the canonical versions
      selection['canonical'] = True

      query = queryselection(query, selection)

      # Knock out the FAILs
      query = query.filter(Header.qa_state!='Fail')

      # Order by date, most recent first
      query = query.order_by(desc(Header.ut_datetime))

      # If openquery, limit number of responses
      if(openquery(selection)):
        query = query.limit(1000)

      # OK, do the query
      headers = query.all()

      req.content_type = "text/xml"
      req.write('<?xml version="1.0" ?>')
      req.write("<calibration_associations>\n")
      # Did we get anything?
      if(len(headers)>0):
        # Loop through targets frames we found
        for object in headers:
          req.write("<dataset>\n")
          req.write("<datalabel>%s</datalabel>\n" % object.data_label)
          req.write("<filename>%s</filename>\n" % object.diskfile.file.filename)
          req.write("<md5>%s</md5>\n" % object.diskfile.md5)
          req.write("<ccrc>%s</ccrc>\n" % object.diskfile.ccrc)

          # Get a cal object for this target data
          c = FitsStorageCal.get_cal_object(session, None, header=object)
   
          # Call the appropriate method depending what calibration type we want
          cal = None
          if(caltype == 'processed_bias'):
            cal = c.processed_bias()
          if(caltype == 'processed_flat'):
            cal = c.processed_flat()

          if(cal):
            # OK, say what we found
            req.write("<calibration>\n")
            req.write("<caltype>%s</caltype>\n" % caltype)
            req.write("<datalabel>%s</datalabel>\n" % cal.data_label)
            req.write("<filename>%s</filename>\n" % cal.diskfile.file.filename)
            req.write("<md5>%s</md5>\n" % cal.diskfile.md5)
            req.write("<ccrc>%s</ccrc>\n" % cal.diskfile.ccrc)
            req.write("<url>http://%s/file/%s</url>\n" % (req.server.server_hostname, cal.diskfile.file.filename))
            req.write("</calibration>\n")
          else:
            req.write("<!-- NO CALIBRATION FOUND-->\n")
          req.write("</dataset>\n")
      else:
        req.write("<!-- COULD NOT LOCATE METADATA FOR DATASET -->\n")

      req.write("</calibration_associations>\n")
      return apache.OK
  except IOError:
    pass
  finally:
    session.close()


def calibrations(req, selection):
  """
  This is the calibrations generator. It implements a human readable calibration association server

  req is an apache request handler request object
  selection is an array of items to select on, simply passed
    through to the webhdrsummary function

  returns an apache request status code
  """
  req.content_type = "text/html"
  req.write("<html>")
  title = "Calibrations %s" % sayselection(selection)

  req.write("<head>")
  req.write("<title>%s</title>" % (title))
  req.write('<link rel="stylesheet" href="/htmldocs/table.css">')
  req.write("</head>\n")
  req.write("<body>")
  if (fits_system_status == "development"):
    req.write('<h1>This is the development system, please use <a href="http://fits/">fits</a> for operational use</h1>')
  req.write("<H1>%s</H1>" % (title))

  session = sessionfactory()
  try:
    # OK, find the target files
    # The Basic Query
    query = session.query(Header).select_from(join(Header, join(DiskFile, File)))

    # Only the canonical versions
    selection['canonical'] = True

    query = queryselection(query, selection)

    # Knock out the FAILs
    query = query.filter(Header.qa_state!='Fail')

    # Order by date, most recent first
    query = query.order_by(desc(Header.ut_datetime))

    # If openquery, limit number of responses
    if(openquery(selection)):
      query = query.limit(1000)

    # OK, do the query
    headers = query.all()

    req.write("<H2>Found %d datasets to check for suitable calibrations</H2>" % len(headers))
    req.write("<HR>")

    # Was the request for only one type of calibration?
    caltype='all'
    if('caltype' in selection):
      caltype = selection['caltype']

    warnings = 0
    missings = 0
    for object in headers:
      # Accumulate the html in a string, so we can decide whether to display it all at once
      html=""
      takenow = False
      warning=False
      missing=False
      requires=False
      oldarc = None

      html+='<H3><a href="/fullheader/%s">%s</a> - <a href="/summary/%s">%s</a></H3>' % (object.diskfile.file.filename, object.diskfile.file.filename, object.data_label, object.data_label)

      c = FitsStorageCal.get_cal_object(session, None, header=object)
      if('arc' in c.required and (caltype=='all' or caltype=='arc')):
        requires=True

        # Look for an arc in the same program
        arc = c.arc(sameprog=True)

        if(arc):
          html += '<H4>ARC: <a href="/fullheader/%s">%s</a> - <a href="/summary/%s">%s</a></H4>' % (arc.diskfile.file.filename, arc.diskfile.file.filename, arc.data_label, arc.data_label)
          if(arc.ut_datetime and object.ut_datetime):
            html += "<P>arc was taken %s object</P>" % interval_string(arc, object)
            if(abs(interval_hours(arc, object)) > 24):
              html += '<P><FONT COLOR="Red">WARNING - this is more than 1 day different</FONT></P>'
              warning = True
              arc_a=arc.id
          else:
            html += '<P><FONT COLOR="Red">Hmmm, could not determine time delta...</FONT></P>'
            warning = True

        else:
          html += '<H3><FONT COLOR="Red">NO ARC FOUND!</FONT></H3>'
          warning = True
          missing = True

        # If we didn't find one in the same program or we did, but with warnings,
        # Re-do the search accross all program IDs
        if(warning):
          oldarc = arc
          arc = c.arc()
          # If we find a different one
          if(arc and oldarc and (arc.id != oldarc.id)):
            missing = False
            html += '<H4>ARC: <a href="/fullheader/%s">%s</a> - <a href="/summary/%s">%s</a></H4>' % (arc.diskfile.file.filename, arc.diskfile.file.filename, arc.data_label, arc.data_label)
            if(arc.ut_datetime and object.ut_datetime):
              html += "<P>arc was taken %s object</P>" % interval_string(arc, object)
              if(abs(interval_hours(arc, object)) > 24):
                html += '<P><FONT COLOR="Red">WARNING - this is more than 1 day different</FONT></P>'
                warning = True
            else:
              html += '<P><FONT COLOR="Red">Hmmm, could not determine time delta...</FONT></P>'
              warning = True
            if(arc.program_id != object.program_id):
              html += '<P><FONT COLOR="Red">WARNING: ARC and OBJECT come from different project IDs.</FONT></P>'
              warning = True

        # Handle the 'takenow' flag. This should get set to true if
        # no arc exists or 
        # all the arcs generate warnings, and
        # the time difference between 'now' and the science frame is 
        # less than the time difference between the science frame and the closest
        # arc to it that we currently have
        if(missing):
          takenow = True
        if(warning):
          # Is it worth re-taking?
          # Find the smallest interval between a valid arc and the science
          oldinterval = None
          newinterval = None
          smallestinterval = None
          if (oldarc):
            oldinterval = abs(interval_hours(oldarc, object))
            smallestinterval = oldinterval
          if (arc):
            newinterval = abs(interval_hours(arc, object))
            smallestinterval = newinterval
          if (oldinterval and newinterval):
            if (oldinterval > newinterval):
              smallestinterval = newinterval
            else:
              smallestinterval = oldinterval
          # Is the smallest interval larger than the interval between now and the science?
          now = datetime.datetime.utcnow()
          then = object.ut_datetime
          nowinterval = now - then
          nowhours = (nowinterval.days * 24.0) + (nowinterval.seconds / 3600.0)
          if(smallestinterval > nowhours):
            takenow=True

        html += "<HR>"

      if('dark' in c.required and (caltype=='all' or caltype=='dark')):
        requires=True
        dark = c.dark()
        if(dark):
          html += "<H4>DARK: %s - %s</H4>" % (dark.diskfile.file.filename, dark.data_label)
          if(dark.ut_datetime and object.ut_datetime):
            html += "<P>dark was taken %s object</P>" % interval_string(dark, object)
            if(abs(interval_hours(dark, object)) > 120):
              html += '<P><FONT COLOR="Red">WARNING - this is more than 5 days different</FONT></P>'
              warning = True
        else:
          html += '<H3><FONT COLOR="Red">NO DARK FOUND!</FONT></H3>'
          warning = True
          missing = True

      html += "<HR>"

      if('bias' in c.required and (caltype=='all' or caltype=='bias')):
        requires=True
        bias = c.bias()
        if(bias):
          html += "<H4>BIAS: %s - %s</H4>" % (bias.diskfile.file.filename, bias.data_label)
        else:
          html += '<H3><FONT COLOR="Red">NO BIAS FOUND!</FONT></H3>'
          warning = True
          missing = True

        html += "<HR>"

      if('flat' in c.required and (caltype=='all' or caltype=='flat')):
        requires=True
        flat = c.flat()
        if(flat):
          html += "<H4>FLAT: %s - %s</H4>" % (flat.diskfile.file.filename, flat.data_label)
        else:
          html += '<H3><FONT COLOR="Red">NO FLAT FOUND!</FONT></H3>'
          warning = True
          missing = True

        html += "<HR>"
   
      if('arc' in c.required and (caltype=='all' or caltype=='arc')):
        requires=True
        arc = c.arc()
        if(arc):
          html += "<H4>ARC: %s - %s</H4>" % (arc.diskfile.file.filename, arc.data_label)
        else:
          html += '<H3><FONT COLOR="Red">NO ARC FOUND!</FONT></H3>'
          warning = True
          missing = True

        html += "<HR>"
   
      if('processed_bias' in c.required and (caltype=='all' or caltype=='processed_bias')):
        requires=True
        processed_bias = c.processed_bias()
        if(processed_bias):
          html += "<H4>PROCESSED_BIAS: %s - %s</H4>" % (processed_bias.diskfile.file.filename, processed_bias.data_label)
        else:
          html += '<H3><FONT COLOR="Red">NO PROCESSED_BIAS FOUND!</FONT></H3>'
          warning = True
          missing = True

        html += "<HR>"
   
      if('processed_flat' in c.required and (caltype=='all' or caltype=='processed_flat')):
        requires=True
        processed_flat = c.processed_flat()
        if(processed_flat):
          html += "<H4>PROCESSED_FLAT: %s - %s</H4>" % (processed_flat.diskfile.file.filename, processed_flat.data_label)
        else:
          html += '<H3><FONT COLOR="Red">NO PROCESSED_FLAT FOUND!</FONT></H3>'
          warning = True
          missing = True

        html += "<HR>"
   
      if('processed_fringe' in c.required and (caltype=='all' or caltype=='processed_fringe')):
        requires=True
        processed_fringe = c.processed_fringe()
        if(processed_fringe):
          html += "<H4>PROCESSED_FRINGE: %s - %s</H4>" % (processed_fringe.diskfile.file.filename, processed_fringe.data_label)
        else:
          html += '<H3><FONT COLOR="Red">NO PROCESSED_FRINGE FOUND!</FONT></H3>'
          warning = True
          missing = True

        html += "<HR>"
   
      if('pinhole_mask' in c.required and (caltype=='all' or caltype=='pinhole_mask')):
        requires=True
        pinhole_mask = c.pinhole_mask()
        if(pinhole_mask):
          html += "<H4>PINHOLE_MASK: %s - %s</H4>" % (pinhole_mask.diskfile.file.filename, pinhole_mask.data_label)
        else:
          html += '<H3><FONT COLOR="Red">NO PINHOLE_MASK FOUND!</FONT></H3>'
          warning = True
          missing = True

        html += "<HR>"
   
      if('ronchi_mask' in c.required and (caltype=='all' or caltype=='ronchi_mask')):
        requires=True
        ronchi_mask = c.ronchi_mask()
        if(ronchi_mask):
          html += "<H4>RONCHI_MASK: %s - %s</H4>" % (ronchi_mask.diskfile.file.filename, ronchi_mask.data_label)
        else:
          html += '<H3><FONT COLOR="Red">NO RONCHI_MASK FOUND!</FONT></H3>'
          warning = True
          missing = True

        html += "<HR>"
   
      caloption=None
      if('caloption' in selection):
        caloption = selection['caloption']

      if(caloption=='warnings'):
        if(warning):
          req.write(html)
      elif(caloption=='missing'):
        if(missing):
          req.write(html)
      elif(caloption=='requires'):
        if(requires):
          req.write(html)
      elif(caloption=='takenow'):
        if(takenow):
          req.write(html)
      else:
        req.write(html)
      if(warning):
        warnings +=1
      if(missing):
        missings +=1

    req.write("<HR>")
    req.write("<H2>Counted %d potential missing Calibrations</H2>" % missings)
    req.write("<H2>Query generated %d warnings</H2>" % warnings)
    req.write("</body></html>")
    return apache.OK

  except IOError:
    pass
  finally:
    session.close()

def upload_processed_cal(req, filename):
  """
  This handles uploading processed calibrations.
  It has to be called via a POST request with a binary data payload
  We drop the data in a staging area, then call a (setuid) script to
  copy it into place and trigger the ingest.
  """

  if(req.method != 'POST'):
    return apache.HTTP_NOT_ACCEPTABLE

  # It's a bit brute force to read all the data in one chunk...
  clientdata = req.read()
  fullfilename = os.path.join(FitsStorageConfig.upload_staging_path, filename)

  f = open(fullfilename, 'w')
  f.write(clientdata)
  f.close()
  clientdata=None

  # Now invoke the setuid ingest program
  command="/opt/FitsStorage/invoke /opt/FitsStorage/ingest_uploaded_calibration.py %s" % filename
  os.system(command)

  return apache.OK


def interval_hours(a, b):
  """
  Given two header objects, returns the number of hours b was taken after a
  """
  interval = a.ut_datetime - b.ut_datetime
  tdelta = (interval.days * 24.0) + (interval.seconds / 3600.0)

  return tdelta

def interval_string(a, b):
  """
  Given two header objects, return a human readable string describing the time difference between them
  """
  t = interval_hours(a, b)

  word = "after"
  unit = "hours"

  if(t < 0.0):
    word = "before"
    t *= -1.0

  if(t > 48.0):
    t /= 24.0
    unit = "days"

  if(t < 1.0):
    t *= 60
    unit = "minutes"

  # 1.2 days after
  string = "%.1f %s %s" % (t, unit, word)

  return string

def sayselection(selection):
  """
  returns a string that describes the selection dictionary passed in
  suitable for pasting into html
  """
  string = ""

  defs = {'program_id': 'Program ID', 'observation_id': 'Observation ID', 'data_label': 'Data Label', 'date': 'Date', 'daterange': 'Daterange', 'inst':'Instrument', 'observation_type':'ObsType', 'observation_class': 'ObsClass', 'filename': 'Filename', 'gmos_grating': 'GMOS Grating', 'gmos_focal_plane_mask': 'GMOS FP Mask', 'caltype': 'Calibration Type', 'caloption': 'Calibration Option'}
  for key in defs:
    if key in selection:
      string += "; %s: %s" % (defs[key], selection[key])

  if('spectroscopy' in selection):
    if(selection['spectroscopy']):
      string += "; Spectroscopy"
    else:
      string += "; Imaging"
  if('qa_state' in selection):
    if(selection['qa_state']=='Win'):
      string += "; QA State: Win (Pass or Usable)"
    else:
      string += "; QA State: %s" % selection['qa_state']
  if('ao' in selection):
    if(selection['ao']=='AO'):
      string += "; Adaptive Optics in beam"
    else:
      string += "; No Adaptive Optics in beam"

  if('notrecognised' in selection):
    string += ". WARNING: I didn't understand these (case-sensitive) words: %s" % selection['notrecognised']

  return string

# import time module to get local timezone
import time
def queryselection(query, selection):
  """
  Given an sqlalchemy query object and a selection dictionary,
  add filters to the query for the items in the selection
  and return the query object
  """

  # Do want to select Header object for which diskfile.present is true?
  if('present' in selection):
    query = query.filter(DiskFile.present == selection['present'])

  if('canonical' in selection):
    query = query.filter(DiskFile.canonical == selection['canonical'])

  if('program_id' in selection):
    query = query.filter(Header.program_id==selection['program_id'])

  if('observation_id' in selection):
    query = query.filter(Header.observation_id==selection['observation_id'])

  if('data_label' in selection):
    query = query.filter(Header.data_label==selection['data_label'])

  # Should we query by date?
  if('date' in selection):
    # Parse the date to start and end datetime objects
    # We consider the night boundary to be 14:00 local time
    # This is midnight UTC in Hawaii, completely arbitrary in Chile
    startdt = dateutil.parser.parse("%s 14:00:00" % (selection['date']))
    tzoffset = datetime.timedelta(seconds=time.timezone)
    oneday = datetime.timedelta(days=1)
    startdt = startdt + tzoffset - oneday
    enddt = startdt + oneday
    # check it's between these two
    query = query.filter(Header.ut_datetime >= startdt).filter(Header.ut_datetime < enddt)

  # Should we query by daterange?
  if('daterange' in selection):
    # Parse the date to start and end datetime objects
    daterangecre=re.compile('(20\d\d[01]\d[0123]\d)-(20\d\d[01]\d[0123]\d)')
    m = daterangecre.match(selection['daterange'])
    startdate = m.group(1)
    enddate = m.group(2)
    tzoffset = datetime.timedelta(seconds=time.timezone)
    oneday = datetime.timedelta(days=1)
    startdt = dateutil.parser.parse("%s 14:00:00" % startdate)
    startdt = startdt + tzoffset - oneday
    enddt = dateutil.parser.parse("%s 14:00:00" % enddate)
    enddt = enddt + tzoffset - oneday
    enddt = enddt + oneday
    # Flip them round if reversed
    if(startdt > enddt):
      tmp = enddt
      enddt = startdt
      started = tmp
    # check it's between these two
    query = query.filter(Header.ut_datetime >= startdt).filter(Header.ut_datetime <= enddt)

  if('observation_type' in selection):
    query = query.filter(Header.observation_type==selection['observation_type'])

  if('observation_class' in selection):
    query = query.filter(Header.observation_class==selection['observation_class'])

  if('inst' in selection):
    if(selection['inst']=='GMOS'):
      query = query.filter(or_(Header.instrument == 'GMOS-N', Header.instrument == 'GMOS-S'))
    else:
      query = query.filter(Header.instrument==selection['inst'])

  if('filename' in selection):
    query = query.filter(File.filename == selection['filename'])

  if('gmos_grating' in selection):
    query = query.filter(Header.disperser == selection['gmos_grating'])

  if('gmos_focal_plane_mask' in selection):
    query = query.filter(Header.focal_plane_mask == selection['gmos_focal_plane_mask'])

  if('spectroscopy' in selection):
    query = query.filter(Header.spectroscopy == selection['spectroscopy'])

  if('qa_state' in selection):
    if(selection['qa_state']=='Win'):
      query = query.filter(or_(Header.qa_state=='Pass', Header.qa_state=='Usable'))
    else:
      query = query.filter(Header.qa_state==selection['qa_state'])

  if('ao' in selection):
    if(selection['ao']=='AO'):
      query = query.filter(Header.adaptive_optics==True)
    else:
      query = query.filter(Header.adaptive_optics==False)

  return query

def openquery(selection):
  """
  Returns a boolean to say if the selection is limited to a reasonable number of
  results - ie does it contain a date, daterange, prog_id, obs_id etc
  returns True if this selection will likely return a large number of results
  """
  openquery = True

  things = ['date', 'daterange', 'program_id', 'observation_id', 'data_label', 'filename']

  for thing in things:
    if(thing in selection):
      openquery = False

  return openquery

def curation_report(req, things):
  """
  Retrieves and prints out the desired values from the list created in 
  FitsStorageCuration.py in the hbffits3 browser.
  """
  req.content_type = 'text/html'
  req.write('<html>')
  req.write('<head>')
  req.write('<title>FITS Storage database curation report</title><link rel="stylesheet" href="/htmldocs/table.css">')
  req.write('</head>')
  req.write('<body>')
  req.write('<h1>FITS Storage database curation report</h1>')

  session = sessionfactory()
  try:
    from FitsStorageCuration import *
    checkonly = None
    exclude = None
    if len(things) != 0 and things[0] == 'noeng':
      exclude = 'ENG'    


    # Work for duplicate_datalabels
    dupdata = duplicate_datalabels(session, checkonly, exclude)
    previous_ans = ''
    even = 0
    req.write('<h2>Duplicate Datalabel Rows:</h2>')
    if dupdata != []:
      # Write the table headers
      req.write('<table border=0><tr class=tr_head><th>DiskFile ID</th><th>FileName</th><th>DataLabel</th></tr>')
      # Makes a list of diskfile ids such that every duplicate found has only one diskfile id
      for val in dupdata:
        this_ans = val
        if previous_ans != this_ans:
          header = session.query(Header).filter(Header.diskfile_id == this_ans).first()
          # Writes out the row for every duplicate in html
          if header:
            even = not even
            if(even):
              req.write('<tr class=tr_even>')
            else:
              req.write('<tr class=tr_odd>')
            req.write('<td>%s</td><td><a href="/summary/%s"> %s </a></td><td><a href="/summary/%s"> %s </a></td></tr>' %  (header.diskfile.id, header.diskfile.file.filename, header.diskfile.file.filename, header.data_label, header.data_label))        
        previous_ans = this_ans
      req.write('</table>')
    else:
      req.write("No rows with duplicate datalabels where canonical=True.<br/>")


    # Work for duplicate_canonicals
    dupcanon = duplicate_canonicals(session)
    previous_file = ''
    oneheader = 0
    empty = 0
    even = 0
    req.write('<h2>Duplicate Canonical Rows:</h2>')      
    # Makes a list of diskfile ids such that every duplicate row found has only one diskfile id
    for val in dupcanon:
      this_file = val.file_id
      if previous_file == this_file:
        # Writes the table headers
        if oneheader == 1:
          pass
        else: 
          req.write('<table border=0><tr class=tr_head><th>DiskFile id</th><th>FileName</th><th>Canonical</th></tr>')
          oneheader += 1
        # Writes out the row for every duplicate in html
        even = not even
        if(even):
          req.write('<tr class=tr_even>')
        else:
          req.write('<tr class=tr_odd>')
        req.write('<td>%s</td><td><a href="/summary/%s"> %s </a></td><td>%s</td></tr>' %  (val.id, val.file.filename, val.file.filename, val.canonical))
        empty += 1 
      previous_file = this_file
    req.write('</table>')
    if empty == 0:
      req.write("No rows with duplicate file ids where canonical=True.<br/>")


    # Work for duplicate_present
    duppres = duplicate_present(session)
    previous_file = ''
    oneheader = 0
    empty = 0
    even = 0
    req.write('<h2>Duplicate Present Rows:</h2>')
    # Makes a list of diskfile ids such that every duplicate row found has only one diskfile id
    for val in duppres: 
      this_file = val.file_id
      if previous_file == this_file:
        # Writes the table headers
        if oneheader == 1:
          pass
        else:
          req.write('<table border=0><tr class=tr_head><th>DiskFile id</th><th>FileName</th><th>Present</th></tr>')
          oneheader += 1
        # Writes out the row for every duplicate in html
        even = not even
        if(even):
          req.write('<tr class=tr_even>')
        else:
          req.write('<tr class=tr_odd>')
        req.write('<td>%s</td><td><a href="/summary/%s"> %s </a></td><td>%s</td></tr>' %  (val.id, val.file.filename, val.file.filename, val.present))
        empty += 1 
      previous_file = this_file
    req.write('</table>')
    if empty == 0:
      req.write("No rows with duplicate file ids where present=True.<br/>")


    # Work for present_not_canonical
    presnotcanon = present_not_canonical(session)
    previous_ans = ''
    even = 0
    req.write('<h2>Present Not Canonical Rows:</h2>')
    if presnotcanon != []:
      # Writes the table headers
      req.write('<table border=0><tr class=tr_head><th>DiskFile id</th><th>FileName</th><th>Present</th><th>Canonical</th></tr>')
      # Makes a list of diskfile ids such that every duplicate row found has only one diskfile id
      for val in presnotcanon:
        this_ans = val
        if previous_ans != this_ans:
          # Writes out the row for every duplicate in html
          even = not even
          if(even):
            req.write('<tr class=tr_even>')
          else:
            req.write('<tr class=tr_odd>')
          req.write('<td>%s</td><td><a href="/summary/%s"> %s </a></td><td>%s</td><td>%s</td></tr>' %  (val.id, val.file.filename, val.file.filename, val.present, val.canonical))         
        previous_ans = this_ans 
      req.write('</table>')
    else:
      req.write("No rows with the conditions present=True and canonical=False.<br/>")


    req.write('</body>')
    req.write('</html>')
    return apache.OK
  except IOError:
    pass
  finally:
    session.close()
  
