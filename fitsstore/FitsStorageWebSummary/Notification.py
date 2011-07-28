"""
This module contains the notification html generator function. 
"""
from FitsStorage import *
from GeminiMetadataUtils import *
from FitsStorageConfig import *

class stub:
    pass
    
if fsc_localmode:
    apache = stub()
    apache.OK = True
    
try:
    from mod_python import apache, util
except ImportError:
    pass


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


