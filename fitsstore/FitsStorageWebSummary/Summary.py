"""
This module contains the main summary html generator function. 
"""
from FitsStorageWebSummary.Selection import *
from FitsStorageConfig import fsc_localmode


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
  the html table containing the actual summary information.
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

