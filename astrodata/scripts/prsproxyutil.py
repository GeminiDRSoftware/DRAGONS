import urllib

from xml.dom import minidom

CALMGR = "http://hbffits1.hi.gemini.edu/calmgr"
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
        
    if 'datalabel' in rq:
        token = rq["datalabel"]
    else:
        token = rq["filename"]
    
    rqurl = urljoin(CALMGR, CALTYPEDICT[rq['caltype']],token)
    print "prs52:", rqurl
    response = urllib.urlopen(rqurl).read()
    #print "prs66:", response
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
