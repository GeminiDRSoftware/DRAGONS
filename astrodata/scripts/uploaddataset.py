import httplib, mimetypes

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
