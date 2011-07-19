import urllib, urllib2


from xml.dom import minidom

CALMGR = "http://hbffits2.hi.gemini.edu/calmgr"
LOCALCALMGR = "http://localhost:%(httpport)d/calsearch.xml?caltype=%(caltype)s&%(tokenstr)s"
#"None # needs to have adcc http port in
CALTYPEDICT = { "bias": "bias",
                "flat": "flat",
                "processed_bias": "processed_bias",
                "processed_flat": "processed_flat"}

def urljoin(*args):
    for arg in args:
        if arg[-1] == '/':
            arg = arg[-1]
    ret = "/".join(args)
    print "prs31:", repr(args), ret
    return ret

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

def calibration_search(rq, fullResult = False):
    from astrodata.FitsStorageFeatures import FitsStorageSetup
    fss = FitsStorageSetup() # note: uses current working directory!!!
    
    print "ppu24: in here"
    #if not fss.is_setup():
    #    return None
    
    if "source" not in rq:
        source = "central"
    else:
        source = rq["source"]
    
    # print "ppu32:", repr(rq), source
    
    token = "" # used for GETs, we're using the post method
    rqurl = None
    if source == "central" or source == "all":
        # print "ppu107: CENTRAL", token
        # print "ppu108: ", rq['caltype']
        
        rqurl = urljoin(CALMGR, CALTYPEDICT[rq['caltype']])
        print "ppu109: CENTRAL SEARCH: rqurl is "+ rqurl
        
    print "ppu52:", source
    if source == 'local' or (rqurl == None and source=="all"):
        rqurl = LOCALCALMGR % { "httpport": 8777,
                                "caltype":CALTYPEDICT[rq['caltype']],
                                "tokenstr":tokenstr}
        print "ppu57: LOCAL SEARCH: rqurl is "+ rqurl

    # print "prs52:", rqurl
    
    ### send request
    sequence = [("descriptors", rq["descriptors"]), ("types", rq["types"])]
    postdata = urllib.urlencode(sequence)
    calRQ = urllib2.Request(rqurl)
    u = urllib2.urlopen(calRQ, postdata)
    response = u.read()
    #response = urllib.urlopen(rqurl).read()
    # print "prs66:", response
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
