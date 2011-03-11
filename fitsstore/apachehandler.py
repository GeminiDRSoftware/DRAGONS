# This is the apache python handler
# See /etc/httpd/conf.d/python.conf
# When a request comes in, handler(req) gets called by the apache server

from mod_python import apache
from mod_python import Cookie
from mod_python import util

import sys
import FitsStorage
from FitsStorageWebSummary import *

from GeminiMetadataUtils import *

import re
import datetime

import pyfits

# Compile regexps here

orderbycre=re.compile('orderby\=(\S*)')

# The top level handler. This essentially calls out to the specific
# handler function depending on the uri that we're handling
def handler(req):
  # Set the no_cache flag on all our output
  # no_cache is not writable, have to set the headers directly
  req.headers_out['Cache-Control'] = 'no-cache'
  req.headers_out['Expired'] = '-1'

  # Parse the uri we were given.
  # This gives everything from the uri below the handler
  # eg if we're handling /python and we're the client requests
  # http://server/python/a/b.fits then we get a/b.fits
  uri = req.uri
  
  # Split this about any /s
  things=uri.split('/')

  # Remove any blanks
  while(things.count('')):
    things.remove('')

  # Check if it's empty
  if(len(things)==0):
    # Empty request
    return usagemessage(req)

  # Before we process the request, parse any arguments into a list
  args=[]
  if(req.args):
    args = req.args.split('&')
    while(args.count('')):
      args.remove('')
 
  # OK, need to parse what we got.

  this = things.pop(0)

  # A debug util
  if(this == 'debug'):
    return debugmessage(req)

  # This is the header summary handler
  if((this == 'summary') or (this == 'diskfiles') or (this == 'ssummary')):

    links = True
    # the nolinks thing is for the external email notifications
    if 'nolinks' in things:
      links = False
      things.remove('nolinks')

    # Parse the rest of the uri here while we're at it
    # Expect some combination of program_id, observation_id, date and instrument name
    # We put the ones we got in a dictionary
    selection=getselection(things)

    # We should parse the arguments here too
    # All we have for now are order_by arguments
    # We form a list of order_by keywords
    # We should probably do more validation here
    orderby=[]
    for i in range(len(args)):
      match=orderbycre.match(args[i])
      if(match):
        orderby.append(match.group(1))

    retval = summary(req, this, selection, orderby, links)
    return retval

  # The calibrations handler
  if(this == 'calibrations'):
    # Parse the rest of the URL.
    selection=getselection(things)

    # If we want other arguments like order by
    # we should parse them here

    retval = calibrations(req, selection)
    return retval

  # The xml file list handler
  if(this == 'xmlfilelist'):
    selection = getselection(things)
    retval = xmlfilelist(req, selection)
    return retval

  # The fileontape handler
  if(this == 'fileontape'):
    retval = fileontape(req, things)
    return retval

  # The calmgr handler
  if(this == 'calmgr'):
    # Parse the rest of the URL.
    selection=getselection(things)

    # If we want other arguments like order by
    # we should parse them here

    retval = calmgr(req, selection)
    return retval

  # The processed_cal upload server
  if(this == 'upload_processed_cal'):
    retval = upload_processed_cal(req, things[0])
    return retval
    

  # This returns the fitsverify, wmdreport or fullheader text from the database
  # you can give it either a diskfile_id or a filename
  if(this == 'fitsverify' or this == 'wmdreport' or this == 'fullheader'):
    if(len(things)==0):
      req.content_type="text/plain"
      req.write("You must specify a filename or diskfile_id, eg: /fitsverify/N20091020S1234.fits\n")
      return apache.OK
    thing=things.pop(0)

    # OK, see if we got a filename
    fnthing = gemini_fitsfilename(thing)
    if(fnthing):
      # Now construct the query
      session = sessionfactory()
      try:
        query = session.query(File).filter(File.filename == fnthing)
        if(query.count()==0):
          req.content_type="text/plain"
          req.write("Cannot find file for: %s\n" % fnthing)
          return apache.OK
        file = query.one()
        # Query diskfiles to find the diskfile for file that is canonical
        query = session.query(DiskFile).filter(DiskFile.canonical == True).filter(DiskFile.file_id == file.id)
        diskfile = query.one()
        # Find the diskfilereport
        query = session.query(DiskFileReport).filter(DiskFileReport.diskfile_id == diskfile.id)
        diskfilereport = query.one()
        req.content_type="text/plain"
        if(this == 'fitsverify'):
          req.write(diskfilereport.fvreport)
        if(this == 'wmdreport'):
          req.write(diskfilereport.wmdreport)
        if(this == 'fullheader'):
          # Need to find the header associated with this diskfile
          query = session.query(FullTextHeader).filter(FullTextHeader.diskfile_id == diskfile.id)
          ftheader = query.one()
          req.write(ftheader.fulltext)
        return apache.OK
      except IOError:
        pass
      finally:
        session.close()
   
    # See if we got a diskfile_id
    match = re.match('\d+', thing)
    if(match):
      session = sessionfactory()
      try:
        query = session.query(DiskFile).filter(DiskFile.id == thing)
        if(query.count()==0):
          req.content_type="text/plain"
          req.write("Cannot find diskfile for id: %s\n" % thing)
          session.close()
          return apache.OK
        diskfile = query.one()
        # Find the diskfilereport
        query = session.query(DiskFileReport).filter(DiskFileReport.diskfile_id == diskfile.id)
        diskfilereport = query.one()
        req.content_type="text/plain"
        if(this == 'fitsverify'):
          req.write(diskfilereport.fvreport)
        if(this == 'wmdreport'):
          req.write(diskfilereport.wmdreport)
        return apache.OK
      except IOError:
        pass
      finally:
        session.close()

    # OK, they must have fed us garbage
    req.content_type="text/plain"
    req.write("Could not understand argument - You must specify a filename or diskfile_id, eg: /fitsverify/N20091020S1234.fits\n")
    return apache.OK


  # This is the fits file server
  if(this == 'file'):
    # OK, first find the file they asked for in the database
    # tart up the filename if possible
    if(len(things)==0):
      return apache.HTTP_NOT_FOUND
    filenamegiven=things.pop(0)
    filename = gemini_fitsfilename(filenamegiven)
    if(filename):
      pass
    else:
      filename = filenamegiven
    session = sessionfactory()
    try:
      query=session.query(File).filter(File.filename==filename)
      if(query.count()==0):
        return apache.HTTP_NOT_FOUND
      file=query.one()
      # OK, we should have the file record now.
      # Next, find the canonical diskfile for it
      query=session.query(DiskFile).filter(DiskFile.present==True).filter(DiskFile.file_id==file.id)
      diskfile = query.one()
      # And now find the header record...
      query=session.query(Header).filter(Header.diskfile_id==diskfile.id)
      header=query.one()

      # OK, now figure out if the data are public
      today = datetime.datetime.utcnow().date()
      canhaveit = False

      if((header.release) and (today > header.release)):
        # Yes, the data are public
        canhaveit = True
      else:
        # No, the data are not public. See if we got the magic cookie
        cookies = Cookie.get_cookies(req)
        if(cookies.has_key('gemini_fits_authorization')):
          auth = cookies['gemini_fits_authorization'].value
          if(auth=='good_to_go'):
            # OK, we got the magic cookie
            canhaveit = True

      if(canhaveit):
        # Send them the data
          req.sendfile(file.fullpath())
          return apache.OK
      else:
        # Refuse to send data
        return apache.HTTP_FORBIDDEN

    except IOError:
      pass
    finally:
      session.close()

  # This is the projects observed feature
  if(this == "programsobserved"):
    selection = getselection(things)
    if(("date" not in selection) and ("daterange" not in selection)):
      selection["date"]=gemini_date("today")
    retval =  progsobserved(req, selection)
    return retval
    
  # The GMOS twilight flat and bias report
  if(this == "gmoscal"):
    selection = getselection(things)
    retval = gmoscal(req, selection)
    return retval

  # Database Statistics
  if(this == "stats"):
    return stats(req)

  # Tape handler
  if(this == "tape"):
    return tape(req, things)

  # TapeWrite handler
  if(this == "tapewrite"):
    return tapewrite(req, things)

  # TapeFile handler
  if(this == "tapefile"):
    return tapefile(req, things)

  # TapeRead handler
  if(this == "taperead"):
    return taperead(req, things)

  # Emailnotification handler
  if(this == "notification"):
    return notification(req, things)

  # curation_report handler
  if(this == "curation"):
    return curation_report(req, things)

  # Some static files that the server should serve via a redirect.
  if((this == "robots.txt") or (this == "favicon.ico")):
    newurl = "/htmldocs/%s" % this
    util.redirect(req, newurl)
  
  # Last one on the list - if we haven't return(ed) out of this function
  # by one of the methods above, then we should send out the usage message
  return usagemessage(req)

# End of apache handler() function.
# Below are various helper functions called from above.
# The web summary has it's own module

def getselection(things):
  # this takes a list of things from the URL, and returns a
  # selection hash that is used by the html generators
  selection = {}
  while(len(things)):
    thing = things.pop(0)
    recognised=False
    if(gemini_date(thing)):
      selection['date']=gemini_date(thing)
      recognised=True
    if(gemini_daterange(thing)):
      selection['daterange']=gemini_daterange(thing)
      recognised=True
    gp=GeminiProject(thing)
    if(gp.program_id):
      selection['program_id']=thing
      recognised=True
    go=GeminiObservation(thing)
    if(go.observation_id):
      selection['observation_id']=thing
      recognised=True
    gdl=GeminiDataLabel(thing)
    if(gdl.datalabel):
      selection['data_label']=thing
      recognised=True
    if(gemini_instrument(thing, gmos=True)):
      selection['inst']=gemini_instrument(thing, gmos=True)
      recognised=True
    if(gemini_fitsfilename(thing)):
      selection['filename'] = gemini_fitsfilename(thing)
      recognised=True
    if(gemini_observation_type(thing)):
      selection['observation_type']=gemini_observation_type(thing)
      recognised=True
    if(gemini_observation_class(thing)):
      selection['observation_class']=gemini_observation_class(thing)
      recognised=True
    if(gemini_caltype(thing)):
      selection['caltype']=gemini_caltype(thing)
      recognised=True
    if(gmos_gratingname(thing)):
      selection['gmos_grating']=gmos_gratingname(thing)
      recognised=True
    if(gmos_focal_plane_mask(thing)):
      selection['gmos_focal_plane_mask']=gmos_focal_plane_mask(thing)
      recognised=True
    if(thing=='warnings' or thing=='missing' or thing=='requires' or thing=='takenow'):
      selection['caloption']=thing
      recognised=True
    if(thing=='imaging' or thing=='Imaging'):
      selection['spectroscopy']=False
      recognised=True
    if(thing=='spectroscopy' or thing=='Spectroscopy'):
      selection['spectroscopy']=True
      recognised=True
    if(thing=='Pass' or thing=='Usable' or thing=='Fail' or thing=='Win'):
      selection['qa_state']=thing
      recognised=True
    if(thing=='AO' or thing=='NOTAO'):
      selection['ao']=thing
      recognised=True
    if(thing=='present' or thing=='Present'):
      selection['present']=True
      recognised=True
    if(thing=='notpresent' or thing=='NotPresent'):
      selection['present']=False
      recognised=True
    if(thing=='canonical' or thing=='Canonical'):
      selection['canonical']=True
      recognised=True
    if(thing=='notcanonical' or thing=='NotCanonical'):
      selection['canonical']=False
      recognised=True

    if(not recognised):
      if('notrecognised' in selection):
        selection['notrecognised'] += " "+thing
      else:
        selection['notrecognised'] = thing
  return selection

# Send usage message to browser
def usagemessage(req):
  fp = open("/opt/FitsStorage/htmldocroot/htmldocs/usage.html", "r")
  stuff = fp.read()
  fp.close()
  req.content_type="text/html"
  req.write(stuff)
  return apache.OK

# Send debugging info to browser
def debugmessage(req):
  req.content_type = "text/plain"
  req.write("Debug info\n\n")
  req.write("python interpreter name: %s\n\n" % (str(req.interpreter)))
  req.write("Pythonpath: %s\n\n" % (str(sys.path)))
  req.write("uri: %s\n\n" % (str(req.uri)))
  req.write("unparsed_uri: %s\n\n" % (str(req.unparsed_uri)))
  req.write("the_request: %s\n\n" % (str(req.the_request)))
  req.write("filename: %s\n\n" % (str(req.filename)))
  req.write("path_info: %s\n\n" % (str(req.path_info)))
  req.write("args: %s\n\n" % (str(req.args)))
  
  return apache.OK

# Send database statistics to browser
def stats(req):
  req.content_type = "text/html"
  req.write("<html>")
  req.write("<head><title>FITS Storage database statistics</title></head>")
  req.write("<body>")
  req.write("<h1>FITS Storage database statistics</h1>")

  session = sessionfactory()
  try:

    # File table statistics
    query=session.query(File)
    req.write("<h2>File Table:</h2>")
    req.write("<ul>")
    req.write("<li>Total Rows: %d</li>" % query.count())
    req.write("</ul>")
  
    # DiskFile table statistics
    req.write("<h2>DiskFile Table:</h2>")
    req.write("<ul>")
    # Total rows
    query=session.query(DiskFile)
    totalrows=query.count()
    req.write("<li>Total Rows: %d</li>" % totalrows)
    # Present rows
    query=query.filter(DiskFile.present == True)
    presentrows = query.count()
    if totalrows != 0:
      percent = 100.0 * presentrows / totalrows
      req.write("<li>Present Rows: %d (%.2f %%)</li>" % (presentrows, percent))
    # Present size
    tpq = session.query(func.sum(DiskFile.size)).filter(DiskFile.present == True)
    tpsize=tpq.one()[0]
    if tpsize != None:
      req.write("<li>Total present size: %d bytes (%.02f GB)</li>" % (tpsize, (tpsize/1073741824.0)))
    # most recent entry
    query=session.query(func.max(DiskFile.entrytime))
    latest = query.one()[0]
    req.write("<li>Most recent diskfile entry was at: %s</li>" % latest)
    # Number of entries in last minute / hour / day
    mbefore = datetime.datetime.now() - datetime.timedelta(minutes=1)
    hbefore = datetime.datetime.now() - datetime.timedelta(hours=1)
    dbefore = datetime.datetime.now() - datetime.timedelta(days=1)
    mcount = session.query(DiskFile).filter(DiskFile.entrytime > mbefore).count()
    hcount = session.query(DiskFile).filter(DiskFile.entrytime > hbefore).count()
    dcount = session.query(DiskFile).filter(DiskFile.entrytime > dbefore).count()
    req.write('<LI>Number of DiskFile rows added in the last minute: %d</LI>' % mcount)
    req.write('<LI>Number of DiskFile rows added in the last hour: %d</LI>' % hcount)
    req.write('<LI>Number of DiskFile rows added in the last day: %d</LI>' % dcount)
    # Last 10 entries
    query = session.query(DiskFile).order_by(desc(DiskFile.entrytime)).limit(10)
    list = query.all()
    req.write('<LI>Last 10 diskfile entries added:<UL>')
    for i in list:
      req.write('<LI>%s : %s</LI>' % (i.file.filename, i.entrytime))
    req.write('</UL></LI>')
  
    req.write("</ul>")
  
    # Header table statistics
    query=session.query(Header)
    req.write("<h2>Header Table:</h2>")
    req.write("<ul>")
    req.write("<li>Total Rows: %d</li>" % query.count())
    req.write("</ul>")

    # Data rate statistics
    req.write("<h2>Data Rates</h2>")
    today = datetime.datetime.utcnow().date()
    zerohour = datetime.time(0,0,0)
    ddelta = datetime.timedelta(days=1)
    wdelta = datetime.timedelta(days=7)
    mdelta = datetime.timedelta(days=30)

    start = datetime.datetime.combine(today, zerohour)
    end = start + ddelta
 
    req.write("<h3>Last 10 days</h3><ul>")
    for i in range(10):
      query = session.query(func.sum(DiskFile.size)).select_from(join(Header, DiskFile)).filter(DiskFile.present==True).filter(Header.ut_datetime > start).filter(Header.ut_datetime < end)
      bytes = query.one()[0]
      if(not bytes):
        bytes = 0
      req.write("<li>%s: %.2f GB</li>" % (str(start.date()), bytes/1E9))
      start -= ddelta
      end -= ddelta
    req.write("</ul>")

    end = datetime.datetime.combine(today, zerohour)
    start = end - wdelta
    req.write("<h3>Last 6 weeks</h3><ul>")
    for i in range(6):
      query = session.query(func.sum(DiskFile.size)).select_from(join(Header, DiskFile)).filter(DiskFile.present==True).filter(Header.ut_datetime > start).filter(Header.ut_datetime < end)
      bytes = query.one()[0]
      if(not bytes):
        bytes = 0
      req.write("<li>%s - %s: %.2f GB</li>" % (str(start.date()), str(end.date()), bytes/1E9))
      start -= wdelta
      end -= wdelta
    req.write("</ul>")

    end = datetime.datetime.combine(today, zerohour)
    start = end - mdelta
    req.write("<h3>Last 6 pseudo-months</h3><ul>")
    for i in range(6):
      query = session.query(func.sum(DiskFile.size)).select_from(join(Header, DiskFile)).filter(DiskFile.present==True).filter(Header.ut_datetime > start).filter(Header.ut_datetime < end)
      bytes = query.one()[0]
      if(not bytes):
        bytes = 0
      req.write("<li>%s - %s: %.2f GB</li>" % (str(start.date()), str(end.date()), bytes/1E9))
      start -= mdelta
      end -= mdelta
    req.write("</ul>")

    req.write("</body></html>")
    return apache.OK
  except IOError:
    pass
  finally:
    session.close()
