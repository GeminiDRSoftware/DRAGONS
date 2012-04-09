

#Copyright Jon Berg , turtlemeat.com

import string,cgi,time
from os import curdir, sep
from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
from astrodata.RecipeManager import RecipeLibrary
#import pri
import select
from copy import copy
import datetime
import os
import subprocess
import cgi
from astrodata import AstroData
from SocketServer import ThreadingMixIn
from xml.dom import minidom

def flattenParms(parms):
    for parmkey in parms:
        if (        hasattr(parms[parmkey],"__getitem__") 
            and not type(parms[parmkey]) == str
            and not parmkey == "orderby"):
            parms.update({parmkey:parms[parmkey][0]})
            
rl = RecipeLibrary()
def parsepath(path):
    rpath = None
    rquery = None
    rparms = {}
    
    if "?" in path:
        parts = path.split("?")
        rpath = parts[0]
        rquery = parts[1]
        rparms.update({"path":rpath})
        rparms.update({"query":rquery})
        qd = cgi.parse_qs(rquery)
        rparms.update(qd)
        if False:
            parms = rquery.split("&")
            for parm in parms:
                if "=" in parm:
                    parts = parm.split("=")
                    pkey = parts[0]
                    pval = parts[1]
                else:
                    pkey = parm
                    pval = True
                rparms.update({pkey: pval})
    else:
        rparms.update({"path":path})
        rparms.update({"query":""})
    
    return rparms

try:
    from fitsstore.GeminiMetadataUtils import *
except:
    print "Cannot import GeminiMetadataUtils from FITSSTORE"
  
def getselection(things):
  import apachehandler
  return apachehandler.getselection(things)
  
  # this takes a list of things from the URL, and returns a
  # selection hash that is used by the html generators
  selection = {}
  while(len(things)):
    thing = things.pop(0)
    recognised=False
    if(gemini_date(thing)):
      selection['date']=gemini_date(thing)
      recognised=True
    if(gemini_daterange(thing)):
      selection['daterange']=gemini_daterange(thing)
      recognised=True
    gp=GeminiProject(thing)
    if(gp.progid):
      selection['progid']=thing
      recognised=True
    go=GeminiObservation(thing)
    if(go.obsid):
      selection['obsid']=thing
      recognised=True
    gdl=GeminiDataLabel(thing)
    if(gdl.datalabel):
      selection['datalab']=thing
      recognised=True
    if(gemini_instrument(thing, gmos=True)):
      selection['inst']=gemini_instrument(thing, gmos=True)
      recognised=True
    if(gemini_fitsfilename(thing)):
      selection['filename'] = gemini_fitsfilename(thing)
      recognised=True
    if(gemini_obstype(thing)):
      selection['obstype']=gemini_obstype(thing)
      recognised=True
    if(gemini_obsclass(thing)):
      selection['obsclass']=gemini_obsclass(thing)
      recognised=True
    if(gemini_caltype(thing)):
      selection['caltype']=gemini_caltype(thing)
      recognised=True
    if(gmos_gratingname(thing)):
      selection['gmos_grating']=gmos_gratingname(thing)
      recognised=True
    if(gmos_fpmask(thing)):
      selection['gmos_fpmask']=gmos_fpmask(thing)
      recognised=True
    if(thing=='warnings' or thing=='missing' or thing=='requires' or thing=='takenow'):
      selection['caloption']=thing
      recognised=True
    if(thing=='imaging' or thing=='Imaging'):
      selection['spectroscopy']=False
      recognised=True
    if(thing=='spectroscopy' or thing=='Spectroscopy'):
      selection['spectroscopy']=True
      recognised=True
    if(thing=='Pass' or thing=='Usable' or thing=='Fail' or thing=='Win'):
      selection['qastate']=thing
      recognised=True
    if(thing=='AO' or thing=='NOTAO'):
      selection['ao']=thing
      recognised=True

    if(not recognised):
      if('notrecognised' in selection):
        selection['notrecognised'] += " "+thing
      else:
        selection['notrecognised'] = thing
  return selection

class PPWState(object):
    dataSpider = None
    dirdict = None
    displayCmdHistory = None
    
ppwstate = PPWState()
    
webserverdone = False
class MyHandler(BaseHTTPRequestHandler):
    informers = None
    dataSpider = None
    dirdict = None

    state = None
    counter = 0
    
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


        
    def do_GET(self):
        self.state = ppwstate
        global webserverdone
        parms = parsepath(self.path)
        # print "prsw147:", repr(parms)
        qd = cgi.parse_qs(self.path)
        # print "prsw149:", repr(qd)
        rim = self.informers["rim"]
        # print "prsw151: path=",self.path
        # print "prsw152: parms=",repr(parms)

        try:
            if False: # old root self.path == "/":
                page = """
                <html>
                <head>
                </head>
                <body>
                <h4>prsproxy engineering interface</h4>
                <ul>
                <li><a href="reducelist">"Stream" prsproxy status</a></li>
                <li><a href="datadir">Data Directory View</a></li>
                <li><a href="killprs">Kill this server</a> (%(numinsts)d copies of reduce registered)</li>
                
                </ul>
                <body>
                </html>""" % {"numinsts":rim.numinsts}
                self.wfile.write(page)
                return
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
                # qwe would use peekHistory to do this safely
                # which will be obivous if you uncomment and find cmdHistoy is no longer a 
                # member of this class, it is instead the displayCmdHistory, a TSCmdQueue
                # instance.
                #self.wfile.write("\n".join(repr(self.state.rim.cmd_history[start:]).split(" ")))        
                return 
                
            if parms["path"] == "/recipeindex.xml":
                self.send_response(200)
                self.send_header('Content-type',	'text/xml')
                self.end_headers()
                
                self.wfile.write(rl.getRecipeIndex(as_xml=True))
                return
             
            if parms["path"].startswith("/summary"):
                import searcher
                DEBSUM= False
                
                #break down path
                things = parms["path"].split("/")
                if DEBSUM:
                    print "prsq185:", repr(things)
                things = things[2:]
                if DEBSUM:
                    print "prsq187:",repr(things)
                selection = getselection(things)
                if DEBSUM:
                    print "psrw188: %s\n" % repr(selection)*20
               
                flattenParms(parms)
                if DEBSUM:
                    print "psrw194: %s" % repr(parms)
                parms.update(selection)
                if DEBSUM:
                    print "psrw196: %s" % repr(parms)
                
                buff = searcher.summary(parms)
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(buff)
                return
                
            if parms["path"] == "/calsearch.xml":
                from fitsstore import searcher
                cparms = {}
                cparms.update(parms)
                print "pproxy298:"+repr(cparms)
                if "datalab" in parms:
                    cparms.update({"datalab":parms["datalab"][0]})
                if "filename" in parms:
                    print "ppw302:", repr(parms["filename"])
                    cparms.update({"filename":parms["filename"][0]})
                if "caltype" in parms:
                    cparms.update({"caltype":parms["caltype"][0]})
                else:
                    cparms.update({"caltype":"processed_bias"})
                    
                buff = searcher.search(cparms)
                self.send_response(200)
                self.send_header('Content-type',	'text/xml')
                self.end_headers()
                
                self.wfile.write(buff)
                return 
                
            if parms["path"].startswith("/globalcalsearch.xml"):
                from prsproxyutil import calibration_search  as calibrationSearch
                flattenParms(parms)
                resultb = None
                resultf = None
                
                if "caltype" in parms:
                    caltype = parms["caltype"]
                    if caltype == "processed_bias" or caltype == "all":
                        parms.update({"caltype":"processed_bias"})
                        resultb = calibrationSearch(parms, fullResult=True)
                    if caltype == "processed_flat" or caltype == "all":
                        parms.update({"caltype":"processed_flat"})
                        resultf = calibrationSearch(parms, fullResult = True)
                
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
                self.send_header('Content-type',	'text/xml')
                self.end_headers()
                
                self.wfile.write(result)
                return
                

            
                
            if parms["path"] == "/recipecontent":
                if "recipe" in parms:
                    recipe = parms["recipe"][0]
                    content = rl.retrieve_recipe(recipe)
                    self.send_response(200)
                    self.send_header('Content-type',	'text/plain')
                    self.end_headers()

                    self.wfile.write(content)
                    return
            if parms["path"] == "/adinfo":
                self.send_response(200)
                self.send_header('Content-type',	'text/html')
                self.end_headers()
                from astrodata.RecipeManager import RecipeLibrary
                if "filename" not in parms:
                    return "Error: Need Filename Parameter"
                if "filename" in parms:
                    try:
                        ad = AstroData(parms["filename"][0])
                    except:
                        self.wfile.write("Can't use AstroData to open %s"% parms["filename"])
                        return
                    if "fullpage" in parms:
                        self.wfile.write("<html><body>")
                    if "fullpage" not in parms:
                    # defaults to false
                        self.wfile.write("<b>Name</b>: %s \n" % os.path.basename(ad.filename))
                        self.wfile.write("<br/><b>Path</b>: %s \n" % os.path.abspath(ad.filename))
                        self.wfile.write("<br/><b>Types</b>: %s\n" % ", ".join(ad.types))
                        recdict = rl.get_applicable_recipes(ad, collate = True)
                        keys = recdict.keys()
                        keys.sort()
                        for key in keys:
                            recname = recdict[key]                        
                            self.wfile.write("<br/><b>Default Recipe(s)</b>:%s (<i>due to type</i>: %s)"
                                                % (recname, key))
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
                self.send_header('Content-type',	'text/xml')
                self.end_headers()
                # returned in xml  self.wfile.write('<?xml version="1.0" encoding="UTF-8" ?>\n')

                #self.wfile.write("<html><body>")
                self.wfile.write(rl.list_recipes(as_xml = True) )
                #self.wfile.write("</body></html>")
                return
            
            if parms["path"] == "/reduceconfigs.xml":
                import glob
                rcfgs = glob.glob("./*.rcfg")
                
                self.send_response(200)
                self.send_header('Content-type',	'text/xml')
                self.end_headers()
                
                retxml  = '<?xml version="1.0" encoding="UTF-8" ?>\n'
                retxml += "<reduceconfigs>\n"
                for rcfg in rcfgs:
                    retxml += """\t<reduceconfig name="%s"/>\n""" % rcfg
                retxml += "</reduceconfigs>\n"
                self.wfile.write(retxml)
                return
            
            if parms["path"].startswith("/datadir.xml"):
                #print "*"*300
                #print "prsw168 in datadir.xml generation dirdict=", repr(self.dirdict)
                dirdict = self.getDirdict()
                ds = dirdict.dataSpider
                
                #print "prsw181: before as_xml"
                xml = dirdict.as_xml()
                #print "prsw185: after as_xml"
                
                self.send_response(200)
                self.send_header('Content-type',	'text/xml')
                self.end_headers()
                
                self.wfile.write('<?xml version="1.0" encoding="UTF-8" ?>\n')
                self.wfile.write("<datasetDict>\n")
                self.wfile.write(xml)
                self.wfile.write("</datasetDict>")
                self.wfile.flush()
                return
                
            rrpath = "/runreduce"
            if parms["path"] == "/runreduce":
                
                self.send_response(200)
                self.send_header('Content-type',	'text/html')
                self.end_headers()
                self.wfile.write("<html><head></head><body>\n")
                from StringIO import StringIO
                rout = StringIO()
                cmdlist = ["reduce", "--invoked", "--verbose=6"]
                cmdlist.extend(parms["p"])
                
                #print "prs97 executing: ", " ".join(cmdlist)
                # make a convienience link to the log
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
                            
                # WARNING, this call had used Popen and 
                # selected on the subprocess.PIPE... now uses call
                # there is kruft remaining (may move it back to old style
                # soon but there was a bug)
                print "adcc running: \n\t" + " ".join(cmdlist)
                pid = subprocess.call( cmdlist,
                                        stdout = f,
                                        stderr = f)
                #f.write("hello there")
                
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
                                # print "prsw116: reading"
                                # stdout = pid.stdout.read()
                                stdout = r[0].read()
                                print "prsw487:", stdout
                                break;
                            else:
                                r,v,w = select.select([pid.stderr],[],[],.1)
                                # print "prsw:ERR:112:", repr(r)
                                if len(r):
                                    stderr = pid.stderr.read()
                                    print "prsw494:", stderr
                                    break;



                        # stderr = pid.stderr.read(100)
                        #stdout, stderr  = pid.communicate()
                        #print "51: pid.poll()", str(pid.poll())
                        if stdout:
                            self.wfile.write(str(stdout))
                        if stderr:
                            #self.wfile.write("</pre><b><pre>\n")
                            self.wfile.write("{"+stderr+"}")
                            #self.wfile.write("</pre></b><pre>\n")
                            #self.wfile.flush()

                        #self.wfile.write("ERROR:"+str(stderr)+"\n")
                        #+"\nerror:\n"+str(stderr))
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
                self.send_header('Content-type',	'text/html')
                self.end_headers()
                # self.wfile.write(str(self.path))
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
                # body onload="javascript:setTimeout(\u201clocation.reload(true);\u201d,50000);">
                self.wfile.write(page)
                if True:
                    body = ""
                    body += "<b>date</b>: %s<br/>\n" % datetime.datetime.now().strftime("%A, %Y-%m-%d %H:%M:%S")
                    body += "<u>Reduce Instances</u><br/>\n"
                    body += "n.o. instances: %d\n" % rim.numinsts 
                    body += "<ul>"
                    rdict = copy(rim.reducedict)
                    rpids = rim.reducedict.keys()
                    for rpid in rpids:
                        body += "<li>client pid = %d at port %d</li>\n" % (rpid, rdict[rpid]["port"])
                    body += "</ul>"
                    self.wfile.write(page % {"body":body})
                    self.wfile.flush()
                                    
                return 
            if self.path == "/killprs":
                self.send_response(200)
                self.send_header('Content-type',	'text/html')
                self.end_headers()
                self.wfile.write("Killed this prsproxy instance, pid = %d at %s" %(os.getpid(), str(datetime.datetime.now())))
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
                    self.send_header('Content-type',	'image/png')
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
                self.send_header('Content-type',	'text/html')
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
                        # print "ppw7--:",type(alld[dname])
                        if not alld[dname].collapse_value():
                            import pprint
                            dval = """<pre>%s</pre> """ % pprint.pformat(alld[dname].dict_val, indent=4, width=80)
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
                else:
                    self.send_header('Content-type',	'text/html')
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
                
            if self.path.startswith("/qap"):
                dirname = os.path.dirname(__file__)
                print repr(os.path.split(self.path))
                joinlist = [dirname, "../../scripts/adcc_faceplate/"]
                for elem in os.path.split(self.path)[1:]:
                    joinlist.append(elem)
                
                fname = os.path.join(*joinlist)
                print "trying to open %s" % fname
                try:
                    f = open(fname, "r")
                    data = f.read()
                    f.close()
                except IOError:
                    data = "<b>NO SUCH RESOURCE AVAILABLE</b>"
                self.send_response(200)
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
            self.send_header('Content-type',	'text/html')
            self.end_headers()
            self.wfile.write(data)
            return 

                
        except IOError:
            raise
            print "handling IOError"
            self.send_error(404,'File Not Found: %s' % self.path)
     

    def do_POST(self):
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
            
class MTHTTPServer(ThreadingMixIn, HTTPServer):
    """Handles requests using threads"""

def startInterfaceServer(port = 8777, **informers):
    import socket
    try:
        # important to set prior to any instantiations of MyHandler
        MyHandler.informers = informers
        if "dirdict" in informers:
            ppwstate.dirdict    = informers["dirdict"]
        if "dataSpider" in informers:
            ppwstate.dataSpider = informers["dataSpider"]
        if "rim" in informers:
            ppwstate.rim = informers["rim"]
        # e.g. below by the HTTPServer class
        findingPort = True
        while findingPort:
            try:
                print "starting httpserver on port ...", port,
                server = MTHTTPServer(('', port), MyHandler)
                findingPort = False
            except socket.error:
                print "failed, port taken"
                port += 1
                
        print "started"
        #server.serve_forever()
        while True:
            r,w,x = select.select([server.socket], [],[],.5)
            if r:
                server.handle_request()
            # print "prsw: ",webserverdone
            if webserverdone == True:
                print "shutting down http interface"
                break
    except KeyboardInterrupt:
        print '^C received, shutting down server'
        server.socket.close()

main = startInterfaceServer

if __name__ == '__main__':
    main()

