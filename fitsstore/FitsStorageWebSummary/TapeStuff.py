"""
This module contains the tape related html generator functions. 
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
      req.write("<LI>Succeeded: %s</LI>" % tw.suceeded)
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

