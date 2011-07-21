import httplib, mimetypes

def post_multipart(host, selector, fields, files):
    """
    Post fields and files to an http host as multipart/form-data.
    fields is a sequence of (name, value) elements for regular form fields.
    files is a sequence of (name, filename, value) elements for data to be uploaded as files
    Return the server's response page.
    """
    content_type, body = encode_multipart_formdata(fields, files)
    h = httplib.HTTP(host)
    h.putrequest('POST', selector)
    h.putheader('content-type', content_type)
    h.putheader('content-length', str(len(body)))
    h.endheaders()
    h.send(body)
    errcode, errmsg, headers = h.getreply()
    return h.file.read()

def encode_multipart_formdata(fields, files):
    """
    fields is a sequence of (name, value) elements for regular form fields.
    files is a sequence of (name, filename, value) elements for data to be uploaded as files
    Return (content_type, body) ready for httplib.HTTP instance
    """
    BOUNDARY = '----------ThIs_Is_tHe_bouNdaRY_$'
    CRLF = '\r\n'
    L = []

    for (key, value) in fields:
        L.append('--' + BOUNDARY)
        L.append('Content-Disposition: form-data; name="%s"' % key)
        L.append('')
        L.append(value)
    for (key, filename, value) in files:
        L.append('--' + BOUNDARY)
        L.append('Content-Disposition: form-data; name="%s"; filename="%s"' % (key, filename))
        L.append('Content-Type: %s' % get_content_type(filename))
        L.append('')
        L.append(value)
    L.append('--' + BOUNDARY + '--')
    L.append('')
    body = CRLF.join(L)
    content_type = 'multipart/form-data; boundary=%s' % BOUNDARY
    return content_type, body

def get_content_type(filename):
    return mimetypes.guess_type(filename)[0] or 'application/octet-stream'

import os

fpath = "/home/callen/SVN-AD/gemini_python/test_data/test_cal_files/processed_biases/N20100221S0516_preparedBias.fits"
fn = os.path.basename(fpath)
fd = open(fpath)
d = fd.read()
fd.close()

#response = post_multipart("hbffits2.hi.gemini.edu", "upload_processed_cal/", [], 
#                [(fn, fn, d)])

import sys
import urllib, urllib2

url = "http://hbffits3.hi.gemini.edu/upload_processed_cal/"+fn

postdata = d # urllib.urlencode(d)

try:
    rq = urllib2.Request(url)
    u = urllib2.urlopen(rq, postdata)
except urllib2.HTTPError, error:
    contents = error.read()
    print "ERROR:"
    print contents
    sys.exit()

response = u.read()
print "RESPONSE"
print response
