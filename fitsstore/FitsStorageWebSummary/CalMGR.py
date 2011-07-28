"""
This module contains the calmgr html generator function. 
"""
from fitsstore.FitsStorageWebSummary.Selection import *
from fitsstore.FitsStorageCal import get_cal_object

import urllib


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
  print "CM26:", repr(selection)
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
      c = get_cal_object(session, None, header=None, descriptors=descriptors, types=types)
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
        cal = c.bias(processed=True)
      if(caltype == 'processed_flat'):
        cal = c.flat(processed=True)
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
      print "CM124:", repr(selection)
      query = queryselection(query, selection)

      # Knock out the FAILs
      #query = query.filter(Header.qa_state!='Fail')

      # Order by date, most recent first
      query = query.order_by(desc(Header.ut_datetime))

      # If openquery, limit number of responses
      if(openquery(selection)):
        query = query.limit(1000)

      # OK, do the query
      print "CM138:", query.statement
      headers = query.all()

      req.content_type = "text/xml"
      req.write('<?xml version="1.0" ?>')
      req.write("<calibration_associations>\n")
      # Did we get anything?
      if(len(headers)>0):
        # Loop through targets frames we found
        print "CM147:len headers = ", len(headers)
        for object in headers:
          req.write("<dataset>\n")
          req.write("<datalabel>%s</datalabel>\n" % object.data_label)
          req.write("<filename>%s</filename>\n" % object.diskfile.file.filename)
          req.write("<md5>%s</md5>\n" % object.diskfile.md5)
          req.write("<ccrc>%s</ccrc>\n" % object.diskfile.ccrc)

          # Get a cal object for this target data
          
          c = get_cal_object(session, None, header=object)
   
          # Call the appropriate method depending what calibration type we want
          cal = None
          if(caltype == 'processed_bias'):
            cal = c.bias(processed=True)
          if(caltype == 'processed_flat'):
            cal = c.flat(processed=True)

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
            req.write("<!-- NO CALIBRATION FOUND (GET)-->\n")
          req.write("</dataset>\n")
      else:
        req.write("<!-- COULD NOT LOCATE METADATA FOR DATASET -->\n")

      req.write("</calibration_associations>\n")
      return apache.OK
  except IOError:
    pass
  finally:
    session.close()

