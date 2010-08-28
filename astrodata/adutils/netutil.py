import urllib2
import urllib
import os
import tempfile

import cookielib 
import urlparse

def urlfetch(url, store = None, clobber = False):
    purl = urlparse.urlparse(url)
    host = "hbffits1.hi.gemini.edu"
    npurl = urlparse.ParseResult(purl.scheme,
                                 host,
                                 purl.path,
                                 purl.params,
                                 purl.query,
                                 purl.fragment)
    
    url = npurl.geturl()
    # print "nu20:", url

    jar = cookielib.CookieJar()


 
    ck = cookielib.Cookie(  version=0, name='gemini_fits_authorization', 
                            value='good_to_go', port=None, 
                            port_specified=False, domain=host, 
                            domain_specified=True, 
                            domain_initial_dot=False, 
                            path='/', path_specified=True, 
                            secure=False, expires=None, 
                            discard=True, comment=None, comment_url=None, 
                            rest={'HttpOnly': None}, rfc2109=False)
    jar.set_cookie(ck)
    
    handler = urllib2.HTTPCookieProcessor(jar)
    opener = urllib2.build_opener(handler)
    # for index, cookie in enumerate(jar):
        # print "nu40:",index, ":", cookie
       
    #res = urllib2.urlopen(req)
    #c = urllib.urlencode( {'gemini_fits_authorization':'value'})
    c = urllib.urlencode( {'gemini_fits_authorization':'value'})
    res = opener.open(url, c)

    #for index, cookie in enumerate(jar):
    #    print index, ":", cookie

    CHUNK = 65536
    if store:
        outname = os.path.join(store, os.path.basename(url))
    else:
        outf = tempfile.NamedTemporaryFile("w", prefix = "uftmp", suffix=".fits", dir =".")
        outname = outf.name
        outf.close()

    if os.path.exists(outname) and not clobber:
        raise "File Already Exists:" + outname
    f = open(outname,"w")
    print "${BOLD}netutil${NORMAL}: downloading",url
    while True:
        chunk = res.read(CHUNK)
        if chunk == "":
            break
        f.write(chunk)
    print "${BOLD}netutil${NORMAL}: retrieved", url
    print "${BOLD}netutil${NORMAL}:     to",outname
    
    f.close()
    res.close()
    
    return outname
