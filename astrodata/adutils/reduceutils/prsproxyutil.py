import urllib

from xml.dom import minidom

CALMGR = "http://hbffits2.hi.gemini.edu/calmgr"
LOCALCALMGR = "http://localhost:%(httpport)d/calsearch.xml?caltype=%(caltype)s&%(tokenstr)s"
#"None # needs to have adcc http port in
CALTYPEDICT = { "bias": "processed_bias",
                "flat": "processed_flat",
                "processed_bias": "processed_bias",
                "processed_flat": "processed_flat"}

def urljoin(*args):
    for arg in args:
        if arg[-1] == '/':
            arg = arg[-1]
    ret = "/".join(args)
    print "prs31:", repr(args), ret
    return ret

def calibrationSearch(rq, fullResult = False):
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
    
    print "ppu32:", repr(rq)
    if rq["source"]=="central":
        rqurl = urljoin(CALMGR, CALTYPEDICT[rq['caltype']],token)
    if rq["source"]=='local':
        return None
        rqurl = LOCALCALMGR % { "httpport": 8777,
                                "caltype":CALTYPEDICT[rq['caltype']],
                                "tokenstr":tokenstr}
    print "prs52:", rqurl
    response = urllib.urlopen(rqurl).read()
    print "prs66:", response
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
