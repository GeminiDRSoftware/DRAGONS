#
#                                                                  gemini_python
#
#                                                                astrodata.utils
#                                                                     netutil.py
# ------------------------------------------------------------------------------
# $Id: netutil.py 5274 2015-06-11 14:39:37Z kanderson $
# ------------------------------------------------------------------------------
__version__      = '$Revision: 5274 $'[11:-2]
__version_date__ = '$Date: 2015-06-11 11:39:37 -0300 (Thu, 11 Jun 2015) $'[7:-2]
# ------------------------------------------------------------------------------
import os
import urllib2
import urlparse
import tempfile

import logutils
        
def urlfetch(url, store = None, clobber = False):
    log = logutils.get_logger(__name__)
    purl = urlparse.urlparse(url)
    host = "fits" 
    npurl = urlparse.ParseResult(purl.scheme,
                                 purl.hostname,
                                 purl.path,
                                 purl.params,
                                 purl.query,
                                 purl.fragment)
    
    url = npurl.geturl()
    
    try:
        res = urllib2.urlopen(url)
    except urllib2.HTTPError, error:
        print "ERROR"
        print error.read()
        raise 

    CHUNK = 65536
    if store:
        outname = os.path.join(store, os.path.basename(url))
    else:
        outf = tempfile.NamedTemporaryFile("w", prefix = "uftmp", 
                                           suffix=".fits", dir =".")
        outname = outf.name
        outf.close()

    if os.path.exists(outname):
        if not clobber:
            raise "File Already Exists:" + outname
        else:
            log.debug("File exists, clobber == True, will overwrite/update")
    f = open(outname,"w")

    log.debug("netutil: downloading +", url)
    while True:
        chunk = res.read(CHUNK)
        if chunk == "":
            break
        f.write(chunk)
    
    f.close()
    res.close()
    return outname
