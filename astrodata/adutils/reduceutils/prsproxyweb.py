#
#                                                                     QAP Gemini
#
#                                                                 prsproxyweb.py
#                                                                        07-2013
# ------------------------------------------------------------------------------
# $Id$
# ------------------------------------------------------------------------------
__version__      = '$Revision$'[11:-2]
__version_date__ = '$Date$'[7:-2]
# ------------------------------------------------------------------------------
#
# This has been modified to make queries on fitstore qaforgui urls.
# 
# Updated parsepath() w/ urlparse.

import os
import json
import time
import select
import pprint
import urllib2
import urlparse
import datetime
import subprocess
from copy import copy

from xml.dom import minidom
from SocketServer import ThreadingMixIn
from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer

from astrodata import AstroData
from astrodata.Lookups import get_lookup_table
from astrodata.RecipeManager import RecipeLibrary

# try:
#     from fitsstore.GeminiMetadataUtils import *
# except:
#     print "Cannot import GeminiMetadataUtils from FITSSTORE"

# ------------------------------------------------------------------------------

class PRec():
    _buff = ""
    uri = "localhost:8777"
    method = "GET"
    def __init__(self, pdict=None, method="GET"):
        self._pdict = pdict
        self.method = method

    def write(self, string):
        self._buff += string
        return

    def read(self):
        return self._pdict

    def log_error(self, msg):
        print "PRec:log_error:", msg
        return

# ------------------------------------------------------------------------------

def flattenParms(parms):
    for parmkey in parms:
        if (hasattr(parms[parmkey], "__getitem__") 
            and not type(parms[parmkey]) == str
            and not parmkey == "orderby"):
            parms.update({parmkey:parms[parmkey][0]})
    return

# ------------------------------------------------------------------------------

def parsepath(path):
    """A better parsepath w/ urlparse.

    parameters: <string>
    return:     <dict>
    """
    rparms = {}
    parsed_url = urlparse.urlparse(path)
    rparms.update({"path": parsed_url.path})
    rparms.update({"query": parsed_url.query})
    rparms.update(urlparse.parse_qs(parsed_url.query))
    return rparms

# ------------------------------------------------------------------------------
#                                Timing functions

def server_time():
    """Return a dictionary of server timing quantities related to current time.
    This dict will be returned to a call on the server, /rqsite.json (See
    do_GET() method of ADCCHandler class.

    parameters: <void>
    return:     <dict>, dictionary of time now values.
    """
    lt_now  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    utc_now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")
    utc_offset = datetime.datetime.utcnow() - datetime.datetime.now()

    if utc_offset.days != 0:
        utc_offset = -utc_offset
        utc_offset = -int(round(utc_offset.seconds/3600.))
    else:
        utc_offset = int(round(utc_offset.seconds/3600.))
        
    timezone = time.timezone / 3600
    if timezone == 10:
        local_site = 'gemini-north'
    elif timezone == 4:
        local_site = 'gemini-south'
    else:
        local_site = 'remote'
                    
    time_dict = {"local_site": local_site,
                 "tzname"    : time.tzname[0],
                 "lt_now"    : lt_now,
                 "utc_now"   : utc_now,
                 "utc_offset": utc_offset}
    return time_dict


def stamp_to_ymd(timestamp):
    """Caller sends a timestamp in seconds of epoch. Return string for
    year month day of that time as YYYYMMDD' as used by url requests, as in
    http://<fitsstore_server>/qaforgui/20130616

    parameters: <float>,  seconds of epochs.
    return:     <string>, YYYYMMDD of passed time.
    """
    return time.strftime("%Y%m%d", time.localtime(timestamp))


def stamp_to_opday(timestamp):
    """Converts a passed time stamp (sec) into the corresponding operational
    day. I.e. timestamps >= 14.00h are the next operational day.

    parameters: <float>, time in epoch seconds
    return:     <string>, YYYYMMDD
    """
    dt_object = datetime.datetime.fromtimestamp(timestamp)
    if dt_object.hour >= 14:
        timestamp = timestamp + 86400
    return  stamp_to_ymd(timestamp)


def ymd_to_stamp(yy, mm, dd, hh=0):
    """Caller passes integers for year, month, and day. Return is
    the epoch time (sec). Year is 4 digit, eg., 2013

    parameters: <int>, <int>, <int> [, <int>] Year, Month, Day [,Hour]
    return:     <float>, epoch time in seconds.
    """
    return time.mktime(time.strptime("%s %s %s %s" % (yy, mm, dd, hh), 
                                     "%Y %m %d %H"))


def current_op_timestamp():
    """Return the epoch time (sec) of the start of current operational day,
    where turnover occurs @ 14.00h localtime. I.e. if the hour >= 14.00,
    then the current operational day is tomorrow.

    Eg., 2013-08-02 17.00h is 20130803

    parameters: <void>
    return:     <float>
    """
    hh = 14
    tnow = datetime.datetime.now()
    t_epoch = time.time()

    if tnow.hour >= 14.0:
        op_day = stamp_to_ymd(t_epoch)
    else:
        op_day = stamp_to_ymd(t_epoch - 86400)

    yy, mm, dd = op_day[:4], op_day[4:6], op_day[6:]
    timestamp = ymd_to_stamp(yy, mm, dd, hh)
    return timestamp


#                            End Timing functions
# ------------------------------------------------------------------------------
# FITS Store query.

def fstore_get(timestamp):
    """Open a url on fitsstore/qaforgui/ with the passed timestamp.
    timestamp is in epoch seconds, which is converted here to a 
    YMD string for the URL.  Return a list of dicts of qa metrics data.

    N.B. A timestamp that evaluates to False (0, None) will request everything 
    from fitsstore. This could be huge. Be careful passing no timestamp!

    parameters: <float>, time in epoch seconds
    return:     <list>,  list of dicts (json) of qametrics
    """
    # Get the fitsstore query url from calurl_dict
    qurlPath     = "Gemini/calurl_dict"
    fitsstore_qa = get_lookup_table(qurlPath, "calurl_dict")['QAQUERYURL']
    local_site   = server_time()["local_site"]

    if not timestamp:
        furl         = os.path.join(fitsstore_qa)
        store_handle = urllib2.urlopen(furl)
        qa_data      = json.loads(store_handle.read())   
    else:
        #if local_site == 'gemini-south':
        #    timestamp = timestamp - 86400  # push query time back a day for Pachon.
        date_query    = stamp_to_opday(timestamp)
        furl          = os.path.join(fitsstore_qa, date_query)
        store_handle  = urllib2.urlopen(furl)
        qa_data       = json.loads(store_handle.read())
    return qa_data

# ------------------------------------------------------------------------------
rl = RecipeLibrary()
# ------------------------------------------------------------------------------
class PPWState(object):
    dirdict = None
    dataSpider = None
    displayCmdHistory = None

# ------------------------------------------------------------------------------
ppwstate = PPWState()
webserverdone = False
# ------------------------------------------------------------------------------
class ADCCHandler(BaseHTTPRequestHandler):
    informers  = None
    dataSpider = None
    dirdict    = None
    state      = None
    counter    = 0
    stamp_register = []
    
    def address_string(self):
        host, port = self.client_address[:2]
        return host

    def getDirdict(self):
        if self.state.dirdict == None:
            from astrodata.DataSpider import DataSpider
            ds = self.state.dataSpider = DataSpider(".")
            dirdict = self.state.dirdict = ds.datasetwalk()
            dirdict.dataSpider = ds
        else:
            ds = self.state.dataSpider
            dirdict = self.state.dirdict
            dirdict.dataSpider = ds
        return dirdict

    def log_request(self, code='-', size='-'):
        """Log an accepted request.

        This is called by send_response().

        This is an override of BaseHTTPRequestHandler.log_request method.
        See that class for what the method does normally.
        """
        try:
            assert (self.informers["verbose"])
            self.log_message('"%s" %s %s',
                             self.requestline, str(code), str(size))
        except AssertionError:
            if "cmdqueue.json" in self.requestline:
                pass
            else:
                self.log_message('"%s" %s %s',
                             self.requestline, str(code), str(size))                
        return

    def do_GET(self):
        global webserverdone
        self.state = ppwstate
        rim = self.informers["rim"]
        parms = parsepath(self.path)

        # Older revisions of adcc may not supply 'verbose' key
        try: 
            self.informers["verbose"]
        except KeyError: 
            self.informers["verbose"] = True

        try:
            if self.path == "/":
                page = """
                <html>
                <head>
                </head>
                <body>
                <h4>prsproxy engineering interface</h4>
                <ul>
                <li><a href="/engineering">Engeering Interface</a></li>
                <li><a href="qap/engineering.html">Engeering AJAX App</a></li>
                <li><a href="datadir">Data Directory View</a></li>
                <li><a href="killprs">Kill this server</a> (%(numinsts)d """ +\
                    """copies ofreduce registered)</li>
                </ul>
                <body>
                </html>"""
                page % {"numinsts":rim.numinsts}
                self.send_response(200)
                self.send_header("content-type", "text-html")
                self.end_headers()
                self.wfile.write(page)
                return
                
            if parms["path"].startswith("/rqlog.json"):
                self.send_response(200)
                self.send_header('Content-type', "application/json")
                self.end_headers()
                
                if "file" in parms:
                    logfile = parms["file"][0]
                    print logfile
                    if not os.path.exists(logfile):
                        msg = "Log file not available"
                    else:
                        f = open(logfile, "r")      
                        msg = f.read()
                        f.close()
                else:
                    msg = "No log file available"

                tdic = {"log":msg}

                self.wfile.write(json.dumps(tdic, sort_keys=True, indent=4))
                return
 
           # ------------------------------------------------------------------
           # Server time

            if parms["path"].startswith("/rqsite.json"):
                self.send_response(200)
                self.send_header('Content-type', "application/json")
                self.end_headers()
                tdic = server_time()
                self.wfile.write(json.dumps(tdic, sort_keys=True, indent=4))
                return

            # ------------------------------------------------------------------
            # Metrics query employing fitsstore

            if parms["path"].startswith("/cmdqueue.json"):
                self._handle_cmdqueue_json(rim, parms)
                return

            # ------------------------------------------------------------------
            
            if parms["path"].startswith("/cmdqueue.xml"):
                self.send_response(200)
                self.send_header('Content-type','text/xml')
                self.end_headers()
                
                if "lastcmd" in parms:
                    start = int(parms["lastcmd"][0])+1
                else:
                    start = 0   
                elist = self.state.rim.displayCmdHistory.peekSince(cmdNum=start)
                print "prsw 200:", repr(elist)
                xml = '<commandQueue lastCmd="%d">' % (start-1)
                for cmd in elist:
                    # this is because there should be only one top key
                    #   in the cmd dict
                    cmdname = cmd.keys()[0] 
                                            
                    cmdbody = cmd[cmdname]
                    xml += '<command name="%s">' % cmdname
                    
                    if "files" in cmdbody:
                    
                        basenames = cmdbody["files"].keys()
                        for basename in basenames:
                            fileitem = cmdbody["files"][basename]
                            if "url" not in fileitem or fileitem["url"] == None:
                                url = "None"
                            else:
                                url = fileitem["url"]
                            xml += """<file basename="%(basename)s"
                            url = "%(url)s"
                            cmdnum = "%(cn)d"/>""" % {
                                "basename": basename,
                                "url": "" if "file" not in fileitem else fileitem["url"],
                                "cn":int(cmdbody["cmdNum"])}
                                            
                            # now any extension in the extdict
                            if "extdict" in fileitem:
                                extdict = fileitem["extdict"]
                                for name in extdict.keys():
                                    xml += """\n<file basename="%(basename)s"
                                             url="%(url)s"
                                             ext="%(ext)s"
                                             cmdnum="%(cn)d"/>""" % {
                                            "basename": basename,
                                            "ext": name,
                                            "url": extdict[name],
                                            "cn":int(cmdbody["cmdNum"])}
 
                    xml += '</command>'
                xml += "</commandQueue>"
                self.wfile.write(xml)
                return 
                
            if parms["path"] == "/recipeindex.xml":
                self.send_response(200)
                self.send_header('Content-type', 'text/xml')
                self.end_headers()
                
                self.wfile.write(rl.getRecipeIndex(as_xml=True))
                return
             
            if parms["path"].startswith("/summary"):
                from fitsstore.FitsStorageWebSummary.Summary import summary
                from fitsstore.FitsStorageWebSummary.Selection import getselection
                
                selection = getselection({})
                
                rec =  PRec()
                summary(rec, "summary", selection, [], links=False)
                buff = rec._buff
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(buff)
                return
                
            if parms["path"].startswith("/calmgr"):
                from FitsStorageWebSummary.Selection import getselection
                from FitsStorageWebSummary.CalMGR import calmgr
                things = parms["path"].split("/")[2:]
                # print "ppw457:"+ repr(things)
                self.send_response(200)
                self.send_header('Content-type', 'text/xml')
                self.end_headers()
                
                # Parse the rest of the URL.
                selection=getselection(things)
            
                # If we want other arguments like order by
                # we should parse them here
                req = PRec()
                retval = calmgr(req, selection)
                print "-------\n"*3,"ppw469:", req._buff
                self.wfile.write(req._buff)
                return 
                
            if parms["path"] == "/calsearch.xml":
                import searcher
                cparms = {}
                cparms.update(parms)
                print "pproxy466:"+repr(cparms)
                if "datalab" in parms:
                    cparms.update({"datalab":parms["datalab"][0]})
                if "filename" in parms:
                    print "ppw481:", repr(parms["filename"])
                    cparms.update({"filename":parms["filename"][0]})
                if "caltype" in parms:
                    cparms.update({"caltype":parms["caltype"][0]})
                else:
                    cparms.update({"caltype":"processed_bias"})
                    
                buff = searcher.search(cparms)
                self.send_response(200)
                self.send_header('Content-type', 'text/xml')
                self.end_headers()
                
                self.wfile.write(buff)
                return 
                
            if parms["path"].startswith("/globalcalsearch.xml"):
                from prsproxyutil import calibration_search
                flattenParms(parms)
                resultb = None
                resultf = None
                
                if "caltype" in parms:
                    caltype = parms["caltype"]
                    if caltype == "processed_bias" or caltype == "all":
                        parms.update({"caltype":"processed_bias"})
                        resultb = calibration_search(parms, fullResult=True)
                    if caltype == "processed_flat" or caltype == "all":
                        parms.update({"caltype":"processed_flat"})
                        resultf = calibration_search(parms, fullResult = True)
                
                if caltype == "all":
                    try:
                        domb = minidom.parseString(resultb)
                        domf = minidom.parseString(resultf)
                    except:
                        return None # can't parse input... no calibration
                    calnodefs = domf.getElementsByTagName("calibration")
                    if len(calnodefs) > 0:
                        calnodef = calnodefs[0]
                    else:
                        calnodef = None
                    calnodebs = domb.getElementsByTagName("dataset")
                    if len(calnodebs) > 0:
                        calnodeb = calnodebs[0]
                    
                    #print calnodef.toxml()
                    #print calnodeb.toxml()
                    # domb.importNode(calnodef, True)
                    if calnodef and calnodeb:
                        calnodeb.appendChild(calnodef)
                    elif calnodef:
                        result=domb.toxml()                        
                    else:
                        result=domb.toxml()
                    result = domb.toxml()
                
                print "prsw207:", result
                self.send_response(200)
                self.send_response(200)
                self.send_header('Content-type', 'text/xml')
                self.end_headers()
                
                self.wfile.write(result)
                return
                
            if parms["path"] == "/recipecontent":
                if "recipe" in parms:
                    recipe = parms["recipe"][0]
                    content = rl.retrieve_recipe(recipe)
                    self.send_response(200)
                    self.send_header('Content-type', 'text/plain')
                    self.end_headers()

                    self.wfile.write(content)
                    return

            if parms["path"] == "/adinfo":
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()

                if "filename" not in parms:
                    return "Error: Need Filename Parameter"
                if "filename" in parms:
                    try:
                        ad = AstroData(parms["filename"][0])
                    except:
                        self.wfile.write("Can't use AstroData to open %s" % parms["filename"])
                        return
                    if "fullpage" in parms:
                        self.wfile.write("<html><body>")
                    if "fullpage" not in parms:
                    # defaults to false
                        self.wfile.write("<b>Name</b>: %s \n" % os.path.basename(ad.filename))
                        self.wfile.write("<br/><b>Path</b>: %s \n" % os.path.abspath(ad.filename))
                        self.wfile.write("<br/><b>Types</b>: %s\n" % ", ".join(ad.types))
                        recdict = rl.get_applicable_recipes(ad, collate=True)
                        keys = recdict.keys()
                        keys.sort()
                        for key in keys:
                            recname = recdict[key]                        
                            self.wfile.write("<br/><b>Default Recipe(s)</b>:%s "+\
                                             "(<i>due to type</i>: %s)" % (recname, key))
                        alldesc = ad.all_descriptors()
                        self.wfile.write("<br/><b>Descriptors</b>:\n")
                        self.wfile.write('<table style="margin-left:4em">\n')
                        adkeys = alldesc.keys()
                        adkeys.sort()
                        self.wfile.flush()
                        for desc in adkeys:
                            value = str(alldesc[desc])
                            if "ERROR" in value:
                                value = '<span style="color:red">' + value + '</span>'
                            self.wfile.write("<tr><td>%s</td><td>%s</td></tr>\n" % (desc, value))
                            self.wfile.flush()
                        self.wfile.write("</table>")
                    if "fullpage" in parms:
                        self.wfile.write("</body></html>")
                return
                
            if parms["path"] == "/recipes.xml":
                self.send_response(200)
                self.send_header('Content-type', 'text/xml')
                self.end_headers()
                self.wfile.write(rl.list_recipes(as_xml = True) )
                return

            if parms["path"] == "/reduceconfigs.xml":
                import glob
                rcfgs = glob.glob("./*.rcfg")
                self.send_response(200)
                self.send_header('Content-type', 'text/xml')
                self.end_headers()
                retxml = '<?xml version="1.0" encoding="UTF-8" ?>\n'
                retxml += "<reduceconfigs>\n"
                for rcfg in rcfgs:
                    retxml += """\t<reduceconfig name="%s"/>\n""" % rcfg
                retxml += "</reduceconfigs>\n"
                self.wfile.write(retxml)
                return

            if parms["path"].startswith("/datadir.xml"):
                dirdict = self.getDirdict()
                ds = dirdict.dataSpider
                xml = dirdict.as_xml()
                
                self.send_response(200)
                self.send_header('Content-type', 'text/xml')
                self.end_headers()
                
                self.wfile.write('<?xml version="1.0" encoding="UTF-8" ?>\n')
                self.wfile.write("<datasetDict>\n")
                self.wfile.write(xml)
                self.wfile.write("</datasetDict>")
                self.wfile.flush()
                return

            if parms["path"] == "/runreduce":
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write("<html><head></head><body>\n")
                from StringIO import StringIO
                rout = StringIO()
                cmdlist = ["reduce", "--invoked", "--verbose=6"]
                cmdlist.extend(parms["p"])
                
                logdir = ".autologs"
                if not os.path.exists(logdir):
                    os.mkdir(logdir)

                reducelog = os.path.join(logdir, 
                                         "reduce-addcinvokedlog-%d%s" % (
                                             os.getpid(), str(time.time())
                                         ))
                f = open(reducelog, "w")
                
                loglink = "reducelog-latest"
                if os.path.exists(loglink):
                    os.remove(loglink)
                os.symlink(reducelog, loglink)
                            
                # WARNING, this call had used Popen and selected on the 
                # subprocess.PIPE... now uses call there is kruft remaining 
                # (may move it back to old style soon but there was a bug)

                print "adcc running: \n\t" + " ".join(cmdlist)
                pid = subprocess.call( cmdlist,
                                        stdout = f,
                                        stderr = f)
                
                self.wfile.write('<b style="font-size=150%">REDUCTION STARTED</b>')
                self.wfile.write("<pre>")
                # self.wfile.flush()
                f.close()
                f = open(reducelog, "r")      
                txt = f.read()
                # pretty the text
                ptxt = txt
                if (True): # make pretty
                    ptxt = re.sub("STARTING RECIPE:(.*)\n", 
                                  '<b>STARTING RECIPE:</b><span style="color:blue">\g<1></span>\n', ptxt)
                    ptxt = re.sub("STARTING PRIMITIVE:(.*)\n", 
                                  '<i>STARTING PRIMITIVE:</i><span style="color:green">\g<1></span>\n', ptxt)
                    ptxt = re.sub("ENDING PRIMITIVE:(.*)\n", 
                                  '<i>ENDING PRIMITIVE:</i>  <span style="color:green">\g<1></span>\n', ptxt)
                    ptxt = re.sub("ENDING RECIPE:(.*)\n", 
                                  '<b>ENDING RECIPE:</b>  <span style="color:blue">\g<1></span>\n', ptxt)
                    ptxt = re.sub("(STATUS|INFO|FULLINFO|WARNING|CRITICAL|ERROR)(.*?)-(.*?)-", 
                                  '<span style="font-size:70%">\g<1>\g<2>-\g<3>- </span>', ptxt)

                self.wfile.write(ptxt) # f.read())
                f.close()
                try:
                    while False:
                        error = False
                        while(True):
                            stdout = None
                            stderr = None
                            r,v,w = select.select([pid.stdout],[],[],.1)
                            print "prsw112:", repr(r)
                            if len(r):
                                stdout = r[0].read()
                                print "prsw487:", stdout
                                break;
                            else:
                                r,v,w = select.select([pid.stderr],[],[],.1)
                                if len(r):
                                    stderr = pid.stderr.read()
                                    print "prsw494:", stderr
                                    break;

                        if stdout:
                            self.wfile.write(str(stdout))
                        if stderr:
                            self.wfile.write("{"+stderr+"}")

                        self.wfile.flush()
                        if pid.poll()!= None:
                            self.wfile.flush()
                            break
                except:
                    print "PRSW516 EMERGENCY:"
                    
                self.wfile.write("</pre>")
                
                if False:
                    r,v,x = select.select([pid.stderr], [], [], .1)
                    if len(r):
                        stderr = pid.stderr.read()
                    else:
                        stderr = None
                # stderr = pid.stderr.read(100)
                    if stderr != None:
                        self.wfile.write("<b><pre>\n")
                        self.wfile.write(str(stderr))
                        self.wfile.write("</pre></b>")
                self.wfile.write('<b style="font-size=150%">REDUCTION ENDED</b>')
                self.wfile.write("\n</body></html>")
                self.wfile.flush()

                return
            
            if self.path == "/reducelist": #our dynamic content
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()

                # this is the tag in head that autopolls if wanted
                front = """
                <html>
                <head>
                    <meta http-equiv="refresh" content="2" />
                </head>
                <body>"""
                page = front + """
                %(body)s
                <body>
                </html>"""

                self.wfile.write(page)
                if True:
                    body = ""
                    body += "<b>date</b>: %s<br/>\n" \
                            % datetime.datetime.now().strftime("%A, %Y-%m-%d %H:%M:%S")
                    body += "<u>Reduce Instances</u><br/>\n"
                    body += "n.o. instances: %d\n" % rim.numinsts 
                    body += "<ul>"
                    rdict = copy(rim.reducedict)
                    rpids = rim.reducedict.keys()
                    for rpid in rpids:
                        body += "<li>client pid = %d at port %d</li>\n" \
                                % (rpid, rdict[rpid]["port"])
                    body += "</ul>"
                    self.wfile.write(page % {"body":body})
                    self.wfile.flush()
                return 

            if self.path == "/killprs":
                import datetime
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write("Killed this prsproxy instance, pid = %d at %s" \
                                 %(os.getpid(), str(datetime.datetime.now())))
                webserverdone = True
                return
            
            if self.path.startswith("/displaycache"):
                from CacheManager import get_cache_dir, get_cache_file
                
                path = os.path.split(self.path)
                print "prsw 569:", self.path
                if len (path)>1:
                    slot = path[-1]
                    tfile = get_cache_file(slot)
                    
                    try:
                        f = open(tfile)
                    except:
                        return
                    self.send_response(200)
                    self.send_header('Content-type', 'image/png')
                    self.end_headers()

                    while True:
                        t = f.read(102400)
                        if t == "":
                            self.wfile.flush()
                            break
                        self.wfile.write(t)
                return

            if self.path.startswith("/fullheader"):
                realpath = self.path.split('/')
                realpath = realpath[1:]
                
                dirdict = self.getDirdict()
                print "prsw514:", repr(realpath)
                
                name = realpath[-1]
                fname = dirdict.get_full_path(name)
                ad = AstroData(fname)

                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
        
                self.wfile.write("<html><body>\n")
                self.wfile.write('<h2>%s</h2>\n' % name)
                self.wfile.write(ad.infostr(as_html=True))
                alld = ad.all_descriptors()
                self.wfile.write(
                        """
                        <table cellspacing="2px">
                        <COLGROUP align="right" />
                        <COLGROUP align="left" />
                        <thead>
                        <tr>
                        <td style="background-color:grey">Descriptor</td>
                        <td style="background-color:grey">Value</td>
                        </tr>
                        </thead>
                        """)
                alldkeys = alld.keys()
                alldkeys.sort()
                for dname in alldkeys:
                    
                    if type(alld[dname]) == str and "ERROR" in alld[dname]:
                        redval = '<span  style="color:red">'+str(alld[dname])+"</span>"
                        dval = redval
                    else:
                        # print "ppw864:",type(alld[dname])
                        if not alld[dname].collapse_value():
                            import pprint
                            dval = """<pre>%s</pre> """ \
                                   % pprint.pformat(alld[dname].dict_val, indent=4, width=80)
                        else:
                            dval = str(alld[dname])
                    self.wfile.write("""
                        <tr>
                        <td style="text-align:right;border-bottom:solid grey 1px">
                        %(dname)s =
                        </td>
                        <td style="border-bottom:solid grey 1px">
                        %(value)s
                        </td>
                        </tr>
                        """ % { "dname":dname,
                                "value":dval})
                self.wfile.write("</table>")
                self.wfile.write("</body></html>\n")
                                
                return
                
            if self.path.startswith("/htmldocs"):
                import FitsStorage
                realpath = self.path.split('/')
                realpath = realpath[1:]
                dirname = os.path.dirname(FitsStorage.__file__)
                fname = os.path.join(dirname, "htmldocroot", *realpath)
                #print "psrw456: %s\n" % repr(fname)*10
                fnamelocal = os.path.join(
                                os.path.dirname(fname),
                                "FS_LOCALMODE_"+os.path.basename(fname)
                                )
                if os.path.exists(fnamelocal):
                    fname = fnamelocal
                try:
                    f = open(fname, "r")
                    data = f.read()
                    print repr(data)
                    f.close()
                except IOError:
                    data = "<b>NO SUCH RESOURCE FOUND</b>"
                self.send_response(200)
                if fname.endswith(".css"):
                    self.send_header('Content-type', "text/css")
                elif fname.endswith(".png"):
                    self.send_header('Content-type', "image/png")
                else:
                    self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(data)
                return
                
            if self.path.startswith("/cmd_queue"):
                self.counter += 1
                data = str(self.counter)
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(data)
                return 
                
            if self.path.startswith("/engineering"):
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                
                if "rim" in ADCCHandler.informers:
                    rim = ADCCHandler.informers["rim"]
                    evman = rim.events_manager
                    import pprint
                    data = "<u>Events</u><br/><pre>"
                    data += "num events: %d\n" % len(evman.event_list)
                    for mevent in evman.event_list:
                        data += pprint.pformat(mevent)
                        data += "------------------------------\n"
                    data += "</pre>"
                    self.wfile.write(data)
                return
                
            if self.path.startswith("/qap"):
                if ".." in self.path:
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    data = "<b>bad path error</b>"
                    self.wfile.write(data)
                dirname = os.path.dirname(__file__)
                joinlist = [dirname, "../../scripts/adcc_faceplate/"]
                
                # Split out any parameters in the URL
                self.path = self.path.split("?")[0]

                #append any further directory info.
                joinlist.append(self.path[5:])
                fname = os.path.join(*joinlist)
                self.log_message('"%s" %s %s', "Loading " + \
                                 joinlist[1] + os.path.basename(fname), 
                                 203, '-')
                try:
                    f = open(fname, "r")
                    data = f.read()
                    f.close()
                except IOError:
                    data = "<b>NO SUCH RESOURCE AVAILABLE</b>"
                self.send_response(200)
                if  self.path.endswith(".js"):
                    self.send_header('Content-type', 'text/javascript')
                elif self.path.endswith(".css"):
                    self.send_header("Content-type", "text/css")
                elif fname.endswith(".png"):
                    self.send_header('Content-type', "image/png")
                else:
                    self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(data)
                return 
            else:
                print "not qap"    
            if self.path == "/":
                self.path = "/KitchenSink.html"
                
            dirname = os.path.dirname(__file__)
            fname = os.path.join(dirname, "pyjamaface/prsproxygui/output", *(self.path[1:]))
            
            try:
                f = open(fname, "r")
                data = f.read()
                f.close()
            except IOError:
                data = "<b>NO SUCH RESOURCE FOUND</b>"
                
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(data)
            return 
        except IOError:
            raise
            print "handling IOError"
            self.send_error(404,'File Not Found: %s' % self.path)
     

    def do_POST(self):
        global webserverdone
        parms = parsepath(self.path)
        vlen = int(self.headers["Content-Length"])
        head = self.rfile.read(vlen)
        pdict = head
        
        if parms["path"].startswith("/runreduce"):
            import time
            import json

            # Get events manager
            evman = None
            if "rim" in ADCCHandler.informers:
                rim = ADCCHandler.informers["rim"]
                evman = rim.events_manager

            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            
            reduce_params = json.loads(pdict)
            if reduce_params.has_key("filepath"):
                fp = reduce_params["filepath"]
            else:
                fp = None
            if reduce_params.has_key("options"):
                opt = reduce_params["options"]
            else:
                opt = None
            if reduce_params.has_key("parameters"):
                prm = reduce_params["parameters"]
            else:
                prm = None

            cmdlist = ["reduce", "--invoked"]
            if opt is not None:
                for key in opt:
                    cmdlist.extend(["--"+str(key),str(opt[key])])
            if prm is not None:
                prm_str = ""
                for key in prm:
                    prm_str += str(key)+"="+str(prm[key])+","
                if prm_str!="":
                    prm_str = prm_str.rstrip(",")
                cmdlist.extend(["-p",prm_str])
            if fp is not None:
                cmdlist.append(str(fp))

                # Check that file can be opened
                try:
                    ad = AstroData(fp)
                except:
                    self.wfile.write("Can't use AstroData to open %s"% fp)
                    return

                # Report reduction status
                self.wfile.write("Reducing %s\n" % fp)
                self.wfile.write("Command: %s\n" % " ".join(cmdlist))
                evman.append_event(ad,"status",{"current":"reducing",
                                                "logfile":None},
                                   msgtype="reduce_status")

            # Send reduce log to hidden directory
            logdir = ".autologs"
            if not os.path.exists(logdir):
                os.mkdir(logdir)
            reducelog = os.path.join(
                logdir, "reduce-addcinvokedlog-%d%s" % (
                    os.getpid(), str(time.time())))
            f = open(reducelog, "w")
            loglink = "reducelog-latest"
            if os.path.exists(loglink):
                os.remove(loglink)
            os.symlink(reducelog, loglink)

            # Call reduce
            pid = subprocess.call( cmdlist,
                                   stdout = f,
                                   stderr = f)
            f.close()

            # Report finished status
            if fp is not None:
                if pid==0:
                    evman.append_event(ad,"status",
                                       {"current":"reduction finished",
                                        "logfile":reducelog},
                                       msgtype="reduce_status")
                else:
                    evman.append_event(ad,"status",
                                       {"current":"reduction ERROR",
                                        "logfile":reducelog},
                                       msgtype="reduce_status")

            # Get text from log
            f = open(reducelog, "r")      
            txt = f.read()
            f.close()

            self.wfile.write(txt)
            self.wfile.flush()
            
            return

        if parms["path"].startswith("/calmgr"):
            from FitsStorageWebSummary.Selection import getselection
            from FitsStorageWebSummary.CalMGR import calmgr
            things = parms["path"].split("/")[2:-1]
            print "ppwDOPOST:"+ repr(things)
            self.send_response(200)
            self.send_header('Content-type', 'text/xml')
            self.end_headers()
                
            # Parse the rest of the URL.
            selection=getselection(things)
            
            # If we want other arguments like order by
            # we should parse them here
            # print "PPW1125:"+repr(pdict)
            req = PRec(pdict=pdict, method="POST")
            retval = calmgr(req, selection)
            print "ppw1128::::"*3, req._buff                
            self.wfile.write(req._buff)
            return 

        global rootnode
        try:
            ctype, pdict = cgi.parse_header(self.headers.getheader('content-type'))
            if ctype == 'multipart/form-data':
                query=cgi.parse_multipart(self.rfile, pdict)
            self.send_response(301)
            
            self.end_headers()
            upfilecontent = query.get('upfile')
            print "filecontent", upfilecontent[0]
            self.wfile.write("<HTML>POST OK.<BR><BR>");
            self.wfile.write(upfilecontent[0]);
            
        except :
            pass

    # ------------------------------------------------------------------
    # privitized handling cmdqueue.json requests
    
    def _handle_cmdqueue_json(self, rim, parms):
        msg_form  = '"%s" %s %s'
        info_code = 203
        fail_code = 416
        size = "-"
        verbosity = self.informers["verbose"]

        self.send_response(200)
        self.send_header('Content-type', "application/json")
        self.end_headers()

        # N.B. A timestamp of zero will request everything from fitsstore
        # This could be huge. Be careful passing GET request on cmdqueue.json
        # with no timestamp.

        if "timestamp" in parms:
            fromtime = float(parms["timestamp"][0])
        else:
            fromtime = 0

        # event_list = [] implies a new adcc. Request current op day 
        # metrics from fitsstore.

        if not rim.events_manager.event_list:
            self.log_message(msg_form, "No extant RIM events.",
                             info_code, size)
            self.log_message(msg_form, 
                             "Requesting current OP day events @FITS store",
                             info_code, size)

            rim.events_manager.event_list = fstore_get(current_op_timestamp())

            self.log_message(msg_form, "Received " + 
                             str(len(rim.events_manager.event_list)) + 
                             " events.", info_code, size)
            self.log_message(msg_form, "On QA metrics request: " +
                             stamp_to_opday(current_op_timestamp()),
                             info_code, size)

            tdic = rim.events_manager.get_list()

            tdic.insert(0, {"msgtype"  : "cmdqueue.request",
                            "timestamp": time.time()})
            tdic.append({"msgtype"  : "cmdqueue.request",
                         "timestamp": time.time()})

            self.wfile.write(json.dumps(tdic, sort_keys=True, indent=4))

        # Handle current nighttime requests ...
        elif stamp_to_opday(fromtime) == stamp_to_opday(current_op_timestamp()):
            if verbosity:
                self.log_message(msg_form,"Request metrics on current OP day: "+
                                 stamp_to_opday(fromtime), 
                                 info_code, size)

            tdic = rim.events_manager.get_list(fromtime=fromtime)
            tdic.insert(0, {"msgtype"  : "cmdqueue.request",
                            "timestamp": time.time()})

            self.wfile.write(json.dumps(tdic, sort_keys=True, indent=4))

        # Handle previous day requests
        elif fromtime < current_op_timestamp():
            if verbosity:
                self.log_message(msg_form, 
                                 "Requested metrics on ... " +
                                 stamp_to_opday(fromtime), info_code, size)
                                
            tdic = fstore_get(fromtime)

            if verbosity:
                self.log_message(msg_form, "Received " + str(len(tdic)) + 
                                 " events from fitsstore.", info_code, size)
            
            # Append the last timestamp from the event_list. This is done
            # to trigger the client to pinging the adcc from the last 
            # recorded event.

            if tdic:
                tdic.append({"msgtype": "cmdqueue.request",
                             "timestamp": rim.events_manager.event_list[-1]["timestamp"]})
            else:
                tdic.append({"msgtype"  : "cmdqueue.request",
                             "timestamp": time.time()})

            self.wfile.write(json.dumps(tdic, sort_keys=True, indent=4))

        # Cannot handle the future ...
        else:
            self.log_message(msg_form,"Invalid timestamp received.", 
                             fail_code, size)
            self.log_message(msg_form,"Future events not known.",
                             fail_code, size)
        return
            


class MTHTTPServer(ThreadingMixIn, HTTPServer):
    """Handles requests using threads"""


def startInterfaceServer(port=8777, **informers):
    import socket

    try:
        # important to set prior to any instantiations of ADCCHandler
        ADCCHandler.informers = informers
        
        if "dirdict" in informers:
            ppwstate.dirdict = informers["dirdict"]

        if "dataSpider" in informers:
            ppwstate.dataSpider = informers["dataSpider"]

        if "rim" in informers:
            ppwstate.rim = informers["rim"]

        # e.g. below by the HTTPServer class
        findingPort = True
        while findingPort:
            try:
                print "starting httpserver on port ...", port,
                server = MTHTTPServer(('', port), ADCCHandler)
                findingPort = False
            except socket.error:
                print "failed, port taken"
                port += 1
                
        print "started"
        #server.serve_forever()
        while True:
            r, w, x = select.select([server.socket], [], [], .5)
            if r:
                server.handle_request()

            if webserverdone == True:
                print "shutting down http interface"
                break

    except KeyboardInterrupt:
        print '^C received, shutting down server'
        server.socket.close()

main = startInterfaceServer
