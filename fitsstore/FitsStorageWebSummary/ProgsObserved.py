"""
This is the Fits Storage Web Summary module. It provides the functions
which query the database and generate html for the web header
summaries.
"""
from FitsStorageWebSummary.Selection import *


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
