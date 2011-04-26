# DEVELOPMEENT NOTE:
# This file was created as a minimal interface to web summaries
# able to provide request object proxies, etc, to call these
# functions in local mode without apache present
# not all functions were added through this file. 
# Cleanup required:
#   Either remove this file, or, move the other local features to also
#   come through this file.  Perhaps name the module, in that case,
#   something more appropriate.

from FitsStorage import *
from FitsStorageCal import get_cal_object


def openquery(selection):
  """
  Returns a boolean to say if the selection is limited to a reasonable number of
  results - ie does it contain a date, daterange, prog_id, obs_id etc
  returns True if this selection will likely return a large number of results
  """
  openquery = True

  things = ['date', 'daterange', 'progid', 'obsid', 'datalab', 'filename']

  for thing in things:
    if(thing in selection):
      openquery = False

  return openquery
  


class stub:
    pass
    
class reqProxy():
    server = None
    method = "GET"
    def __init__(self):
        self.server = stub()
        self.server.server_hostname = "LOCAL"
    buff = None
    def write(self, text):
        #print "req write:",text
        if self.buff == None:
            self.buff = ""
        self.buff += text

def search(criteria):
    from astrodata import AstroData
    # print "search:",repr(criteria)
    #ad = AstroData(sciencefile)
  
    try:
        req = reqProxy()
        from FitsStorageWebSummary.CalMGR import calmgr
        calmgr(req, criteria)
    except:
        print req.buff    
        raise

    return req.buff


def summary(parms):
    
    try:
        from FitsStorageWebSummary.Summary import summary
        req = reqProxy()
        req.uri="summary"
        if "orderby" in parms:
            orderby = parms["orderby"]
            print "s165:", orderby
        else:
            orderby = None
        summary(req, "summary", parms, orderby)
        
        return req.buff
    except:
        raise
        
#session = sessionfactory()

if __name__ == "__main__":
    buff = search ({"datalab":"GN-2009B-Q-28-148-003",
             "caltype":"processed_bias"})
    print "calsearch returned:",buff

    

