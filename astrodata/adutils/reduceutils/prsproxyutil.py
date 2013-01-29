
from astrodata import AstroData
from astrodata.usercalibrationservice import user_cal_service
from astrodata import Lookups

from xml.dom import minidom
import exceptions
import pprint
from pprint import pformat

calurldict = Lookups.get_lookup_table("Gemini/calurl_dict","calurl_dict")

CALMGR = calurldict["CALMGR"]
#CALMGR = "http://hbffits3.hi.gemini.edu/calmgr"
LOCALCALMGR = calurldict["LOCALCALMGR"] 

#LOCALCALMGR = "http://localhost:%(httpport)d/calsearch.xml?caltype=%(caltype)s&%(tokenstr)s"
#"None # needs to have adcc http port in
CALTYPEDICT = { "arc": "arc",
                "bias": "bias",
                "dark": "dark",
                "flat": "flat",
                "processed_arc":   "processed_arc",
                "processed_bias":   "processed_bias",
                "processed_dark":   "processed_dark",
                "processed_flat":   "processed_flat",
                "processed_fringe": "processed_fringe"}

def urljoin(*args):
    for arg in args:
        if arg[-1] == '/':
            arg = arg[-1]
    ret = "/".join(args)
    print "prs31:", repr(args), ret
    return ret

def upload_calibration(filename):
    import urllib, urllib2
    import httplib, mimetypes
    import os

    import sys
    import urllib, urllib2

    fpath = filename
    fn = os.path.basename(fpath)
    fd = open(fpath)
    d = fd.read()
    fd.close()
    
    #url = "http://hbffits3.hi.gemini.edu/upload_processed_cal/"+fn
    url = "http://fits/upload_processed_cal/"+fn

    postdata = d # urllib.urlencode(d)

    try:
        rq = urllib2.Request(url)
        u = urllib2.urlopen(rq, postdata)
        #fp = open("/tmp/tmpfile.txt")
        #raise urllib2.HTTPError("","","","",fp)
        
    except urllib2.HTTPError, error:
        contents = error.read()
        #print "ERROR:"
        #print contents
        raise

    response = u.read()
    #print "RESPONSE"
    #print response

calsearch_exc = None

def calibration_search(rq, fullResult = False):
    import urllib, urllib2
    calserv_msg = None
    print "\nppu68: calibration_search\n"
    from xmlrpclib import DateTime
    import datetime
    
    #if "ut_datetime" in rq:
    #    rq["ut_datetime"] = str(rq["ut_datetime"])
    #if not fss.is_setup():
    #    return None
    if "descriptors" in rq and "ut_datetime" in rq["descriptors"]:
        utc = rq["descriptors"]["ut_datetime"]
        pyutc = datetime.datetime.strptime(utc.value, "%Y%m%dT%H:%M:%S")
        print "ppu83",pyutc
        rq["descriptors"].update({"ut_datetime":pyutc} )
    
    
    
    if "source" not in rq:
        source = "central"
    else:
        source = rq["source"]
    
    token = "" # used for GETs, we're using the post method
    rqurl = None
    if source == "central" or source == "all":
        # print "ppu107: CENTRAL", token
        # print "ppu108: ", rq['caltype']
        
        rqurl = urljoin(CALMGR, CALTYPEDICT[rq['caltype']])
        print "ppu109: CENTRAL SEARCH: rqurl is "+ rqurl
        
    print "ppu112:", source
    if source == 'local' or (rqurl == None and source=="all"):
        rqurl = LOCALCALMGR % { "httpport": 8777,
                                "caltype":  CALTYPEDICT[rq['caltype']],
                                } # "tokenstr":tokenstr}
        print "ppu118: LOCAL SEARCH: rqurl is "+ rqurl

    rqurl = rqurl+"/%s"%rq["filename"]
    print "prs113:", pprint.pformat(rq)
    ### send request
    sequence = [("descriptors", rq["descriptors"]), ("types", rq["types"])]
    postdata = urllib.urlencode(sequence)
    response = "CALIBRATION_NOT_FOUND"    
    try:
        # print "ppu106:", repr(sequence)
        # print "ppu120:", pprint.pformat(rq["descriptors"])
        # print "ppu107: postdata",repr(postdata)
        calRQ = urllib2.Request(rqurl)
        if source == "local":
            u = urllib2.urlopen(calRQ, postdata)
        else:
            u = urllib2.urlopen(calRQ, postdata)
            
        response = u.read()
        # print "ppu129:", response
    except urllib2.HTTPError, error:
        
        calserv_msg = error.read()
        print "ppu131:HTTPError- server returns:", error.read()
        import traceback
        import sys
        traceback.print_exc()
        return (None, calserv_msg)
    #response = urllib.urlopen(rqurl).read()
    print "prs129:", response
    if fullResult:
        return response
    try:
        dom = minidom.parseString(response)
        calel = dom.getElementsByTagName("calibration")
        calurlel = dom.getElementsByTagName('url')[0].childNodes[0]
        calurlmd5 = dom.getElementsByTagName('md5')[0].childNodes[0]
    except exceptions.IndexError:
        print "No url for calibration in response, calibration not found"
        return (None,"Request Data:\n" + pformat(sequence)+"\nRepsonse:\n"+response)
    except:
        return (None,"Request Data:\n" + pformat(sequence)+"\nRepsonse:\n"+response)
    #print "prs70:", calurlel.data
    
    #@@TODO: test only 
    print "prspu124:", repr(calurlel.data)
    return (calurlel.data, calurlmd5.data)


def old_calibration_search(rq, fullResult = False):
    from astrodata.FitsStorageFeatures import FitsStorageSetup
    fss = FitsStorageSetup() # note: uses current working directory!!!
    print "ppu24: in here"
    if not fss.is_setup():
        return None
    print "prs38: the request",repr(rq)
    if 'caltype' not in rq:
        rq.update({"caltype":"processed_bias"})
    if 'datalabel' not in rq and "filename" not in rq:
        return None
        
    if "filename" in rq:
        import os
        token = os.path.basename(rq["filename"])
        tokenstr = "filename=%s" % token
    elif 'datalabel' in rq:
        token = rq["datalabel"]
        tokenstr = "datalabel=%s" % token
    
    if "source" not in rq:
        source = "central"
    else:
        source = rq["source"]
    
    print "ppu32:", repr(rq), source
    
    rqurl = None
    if source == "central" or source == "all":
        print "ppu52: CENTRAL SEARCH"
        rqurl = urljoin(CALMGR, CALTYPEDICT[rq['caltype']],token)
        print "ppu54: CENTRAL SEARCH: rqurl is "+ rqurl
        
    print "ppu52:", source
    if source == 'local' or (rqurl == None and source=="all"):
        return None
        rqurl = LOCALCALMGR % { "httpport": 8777,
                                "caltype":CALTYPEDICT[rq['caltype']],
                                "tokenstr":tokenstr}
        print "ppu57: LOCAL SEARCH: rqurl is "+ rqurl

    sequence = [("descriptors",rq.descriptors),('types',rq.types)]
    postdata = urllib.urlencode(sequence)
    
    # response = urllib.urlopen(rqurl).read()
    
    webrq = urllib2.Request(rqurl)
    u = urllib2.urlopen(webrq,postdata)
    
    # note: todo: read the request in case of error by except HTTPError
    response = u.read()
    
    
    if fullResult:
        return response
    dom = minidom.parseString(response)
    calel = dom.getElementsByTagName("calibration")
    try:
        calurlel = dom.getElementsByTagName('url')[0].childNodes[0]
    except exceptions.IndexError:
        return None
    #print "prs70:", calurlel.data
    
    #@@TODO: test only 
    return calurlel.data
