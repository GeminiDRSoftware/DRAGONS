"""
This module contains the tape related xml generator functions. 
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


def xmltape(req):
  """
  Outputs xml describing the tapes that the specified file is on
  """
  req.content_type = "text/xml"
  req.write('<?xml version="1.0" ?>')
  req.write("<on_tape>")

  session = sessionfactory()
  try:
    tape = session.query(Tape).filter(Tape.active==True).order_by(Tape.id).all()

    for t in tape:
      req.write("<tape>")
      req.write("<label>%s</label>" % t.label)
      req.write("<active>%s</active>" % t.active)
      req.write("<firstwrite>%s</firstwrite>" % t.firstwrite)
      req.write("<lastwrite>%s</lastwrite>" % t.lastwrite)
      req.write("<lastverified>%s</lastverified>" % t.lastverified)
      req.write("<location>%s</location>" % t.location)
      req.write("<lastmoved>%s</lastmoved>" % t.lastmoved)
      req.write("<full>%s</full>" % t.full)
      req.write("<set>%s</set>" % t.set)
      req.write("<fate>%s</fate>" % t.fate)


      tapewrite = session.query(TapeWrite).filter(TapeWrite.tape_id==t.id).filter(TapeWrite.suceeded==True).order_by(TapeWrite.id).all()

      for tw in tapewrite:
        req.write("<tapewrite>")
        req.write("<startdate>%s</startdate>" % tw.startdate)
        req.write("<filenum>%s</filenum>" % tw.filenum)
        req.write("<enddate>%s</enddate>" % tw.enddate)
        req.write("<suceeded>%s</suceeded>" % tw.tape_id)
        req.write("<size>%s</size>" % tw.size)
        req.write("<beforestatus>%s</beforestatus>" % tw.beforestatus)
        req.write("<afterstatus>%s</afterstatus>" % tw.afterstatus)
        req.write("<hostname>%s</hostname>" % tw.hostname)
        req.write("<tapedrive>%s</tapedrive>" % tw.tapedrive)
        req.write("<notes>%s</notes>" % tw.notes)


        tapefile = session.query(TapeFile).filter(TapeFile.tapewrite_id==tw.id).all()

        for tf in tapefile:
          req.write("<tapefile>")
          req.write("<filename>%s</filename>" % tf.filename)
          req.write("<size>%s</size>" % tf.size)
          req.write("<ccrc>%s</ccrc>" % tf.ccrc)
          req.write("<md5>%s</md5>" % tf.md5)
          req.write("<lastmod>%s</lastmod>" % tf.lastmod)
          req.write("</tapefile>")

        req.write("</tapewrite>")

      req.write("</tape>")

  finally:
    session.close()

  req.write("</on_tape>")
  return apache.OK

