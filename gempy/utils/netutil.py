#
#                                                                  gemini_python
#
#                                                                     netutil.py
# ------------------------------------------------------------------------------
import os
import urllib2
import urlparse

from tempfile import NamedTemporaryFile

from . import logutils

CHUNK = 65536
def urlfetch(url, store=None, clobber=False):
    """
    Get and write data from a url.

    Parameters
    ----------
    url : <str>
        The url pointing to a downloadable file

    store : <str>
        A path to write and store the file.
        E.g., 
            store=/path/to/cache

    clobber : <bool>
        overwrite an existing file

    """
    log = logutils.get_logger(__name__)
    purl = urlparse.urlparse(url)
    host = "fits" 
    npurl = urlparse.ParseResult(purl.scheme,
                                 purl.hostname,
                                 purl.path, 
                                 purl.params,
                                 purl.query,
                                 purl.fragment)

    if store:
        outname = os.path.join(store, os.path.basename(url))
    else:
        outf = NamedTemporaryFile("w", prefix="uftmp", suffix=".fits", dir=".")
        outname = outf.name
        outf.close()

    if os.path.exists(outname):
        if not clobber:
            err = "File Already Exists: {}".format(outname)
            raise IOError(err)
        else:
            log.debug("File exists but clobber requested; will overwrite/update")

    url = npurl.geturl()
    try:
        res = urllib2.urlopen(url)
    except urllib2.HTTPError as error:
        log.error(str(error))
        raise error

    log.debug("netutil: downloading +", url)
    with open(outname, "w") as f:
        while True:
            chunk = res.read(CHUNK)
            if not chunk:
                break
            f.write(chunk)
    
    res.close()
    return outname
