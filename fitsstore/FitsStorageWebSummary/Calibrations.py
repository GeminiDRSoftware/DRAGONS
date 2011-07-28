"""
This module contains the calibrations html generator function. 
"""
from FitsStorageWebSummary.Selection import *
from FitsStorageConfig import fsc_localmode
from FitsStorageCal import get_cal_object


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

      c = get_cal_object(session, None, header=object)
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

      if('bias' in c.required and (caltype=='all' or caltype=='bias')):
        requires=True
        bias = c.bias()
        if(bias):
          html += "<H4>BIAS: %s - %s</H4>" % (bias.diskfile.file.filename, bias.data_label)
        else:
          html += '<H3><FONT COLOR="Red">NO BIAS FOUND!</FONT></H3>'
          warning = True
          missing = True

      if('flat' in c.required and (caltype=='all' or caltype=='flat')):
        requires=True
        flat = c.flat()
        if(flat):
          html += "<H4>FLAT: %s - %s</H4>" % (flat.diskfile.file.filename, flat.data_label)
        else:
          html += '<H3><FONT COLOR="Red">NO FLAT FOUND!</FONT></H3>'
          warning = True
          missing = True
      
      if('processed_bias' in c.required and (caltype=='all' or caltype=='processed_bias')):
        requires=True
        processed_bias = c.bias(processed=True)
        if(processed_bias):
          html += "<H4>PROCESSED_BIAS: %s - %s</H4>" % (processed_bias.diskfile.file.filename, processed_bias.data_label)
        else:
          html += '<H3><FONT COLOR="Red">NO PROCESSED_BIAS FOUND!</FONT></H3>'
          warning = True
          missing = True
  
      if('processed_flat' in c.required and (caltype=='all' or caltype=='processed_flat')):
        requires=True
        processed_flat = c.flat(processed=True)
        if(processed_flat):
          html += "<H4>PROCESSED_FLAT: %s - %s</H4>" % (processed_flat.diskfile.file.filename, processed_flat.data_label)
        else:
          html += '<H3><FONT COLOR="Red">NO PROCESSED_FLAT FOUND!</FONT></H3>'
          warning = True
          missing = True

      if('processed_fringe' in c.required and (caltype=='all' or caltype=='processed_fringe')):
        requires=True
        processed_fringe = c.processed_fringe()
        if(processed_fringe):
          html += "<H4>PROCESSED_FRINGE: %s - %s</H4>" % (processed_fringe.diskfile.file.filename, processed_fringe.data_label)
        else:
          html += '<H3><FONT COLOR="Red">NO PROCESSED_FRINGE FOUND!</FONT></H3>'
          warning = True
          missing = True

      if('pinhole_mask' in c.required and (caltype=='all' or caltype=='pinhole_mask')):
        requires=True
        pinhole_mask = c.pinhole_mask()
        if(pinhole_mask):
          html += "<H4>PINHOLE_MASK: %s - %s</H4>" % (pinhole_mask.diskfile.file.filename, pinhole_mask.data_label)
        else:
          html += '<H3><FONT COLOR="Red">NO PINHOLE_MASK FOUND!</FONT></H3>'
          warning = True
          missing = True

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

def interval_hours(a, b):
  """
  Given two header objects, returns the number of hours b was taken after a
  """
  interval = a.ut_datetime - b.ut_datetime
  tdelta = (interval.days * 24.0) + (interval.seconds / 3600.0)

  return tdelta
