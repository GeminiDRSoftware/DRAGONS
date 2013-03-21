import urllib
from pprint import pprint,pformat
import json
import optparse
import os
import sys


class MetricManager:
    metrics = None
    def __init__(self, metrics):
        self.metrics = metrics
    
    def get_record(self, filename = None):
        for event in self.metrics:
            if "metadata" in event and "raw_filename" in event["metadata"]:
                if filename in event["metadata"]["raw_filename"]:
                    return event
        return None   
        
    def all_files(self):
        af = []
        for event in self.metrics:
            if "metadata" in event and "raw_filename" in event["metadata"]:
                af.append(event["metadata"]["raw_filename"])    
        af.sort() 
        return af      
# idea for source mkopipe1
parser = optparse.OptionParser(description="Get Metrics from ADCC")
parser.add_option("--save", dest="save", help="save to local file")
parser.add_option("--find", dest="find", help="fragmets of a filename")
parser.add_option("--list_files", dest="listfiles",action="store_true", help="list all files mentioned")
parms, args = parser.parse_args()

if len(args)>0:
    source = args[0]
else:
    source = "mkopipe1"

if os.path.exists(source):
    mfile = open(source)
    strmetrics = mfile.read()
    mfile.close()
else:
    url = "http://%s:8777/cmdqueue.json" % source
    msock = urllib.urlopen(url)
    strmetrics = msock.read()

if parms.save:
    saved = open(parms.save, "w")
    saved.write(strmetrics)
    saved.close()
   
metrics = json.loads(strmetrics)

metman = MetricManager(metrics)

if parms.find:
    pprint(metman.get_record(filename = parms.find))

if parms.listfiles:
    print "\n".join(metman.all_files())




