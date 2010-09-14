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
    
class PPWState(object):
    dataSpider = None
    dirdict = None
    
ppwstate = PPWState()
    
webserverdone = False
class MyHandler(BaseHTTPRequestHandler):
    informers = None
    dataSpider = None
    dirdict = None

    state = None
    
    def address_string(self):
        host, port = self.client_address[:2]
        return host
        
    def do_GET(self):
        self.state = ppwstate
        global webserverdone
        parms = parsepath(self.path)
        print "prsw45:", repr(parms)
        qd = cgi.parse_qs(self.path)
        print "prsw52:", repr(qd)
        rim = self.informers["rim"]
        print "prsw70: path=",self.path
        print "prsw71: parms=",repr(parms)

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
                
            if parms["path"] == "/recipeindex.xml":
                self.send_response(200)
                self.send_header('Content-type',	'text/xml')
                self.end_headers()
                
                self.wfile.write(rl.getRecipeIndex(asXML=True))
                return
                
            if parms["path"] == "/adinfo":
                self.send_response(200)
                self.send_header('Content-type',	'text/html')
                self.end_headers()
                
                if "filename" not in parms:
                    parms.update({"filename" : "set001/N20090902S0099.fits" })
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
                        alldesc = ad.allDescriptors()
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
                self.wfile.write(rl.listRecipes(asXML = True) )
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
            
            if parms["path"] == "/datadir.xml":
                #print "*"*300
                print "prsw168 in datadir.xml generation dirdict=", repr(self.dirdict)
                if self.state.dirdict == None:
                    from astrodata.DataSpider import DataSpider
                    ds = self.state.dataSpider = DataSpider(".")
                    dirdict = self.state.dirdict = ds.datasetwalk()
                else:
                    ds = self.state.dataSpider
                    dirdict = self.state.dirdict
                    
                
                print "prsw181: before asXML"
                xml = dirdict.asXML()
                print "prsw185: after asXML"
                
                self.send_response(200)
                self.send_header('Content-type',	'text/xml')
                self.end_headers()
                
                self.wfile.write('<?xml version="1.0" encoding="UTF-8" ?>\n')
                self.wfile.write("<datasetDict>\n")
                self.wfile.write(xml)
                self.wfile.write("</datasetDict>")
                self.wfile.flush()
                oldway = False
                if oldway:
                    self.send_response(200)
                    self.send_header('Content-type',	'text/html')
                    self.end_headers()
                    self.wfile.write("<html><head></head><body>\n")
                    dirlistall = os.listdir("recipedata")
                    dirlist= []
                    for f in dirlistall:
                        if f[-5:].lower() == ".fits":
                            dirlist.append(f)
                    a = "</li><li>".join(dirlist)
                    a = "<ul><li>"+a+"</li></ul>"
                    self.wfile.write(a)
                    self.wfile.write("</body></html>")
                return
                
            rrpath = "/runreduce"
            if parms["path"] == "/runreduce":
                
                self.send_response(200)
                self.send_header('Content-type',	'text/html')
                self.end_headers()
                self.wfile.write("<html><head></head><body>\n")
                from StringIO import StringIO
                rout = StringIO()
                cmdlist = ["reduce", "--invoked"]
                cmdlist.extend(parms["p"])
                print "prs97 executing: ", " ".join(cmdlist)
                if os.path.exists(".prsproxy"):
                    os.path.mkdir(".prsproxy")
                
                pid = subprocess.Popen( cmdlist, #["reduce", "-c", "ctest.cfg"],
                                        stdout = subprocess.PIPE, 
                                        stderr = subprocess.PIPE)
                
                self.wfile.write('<b style="font-size=150%">REDUCTION STARTED</b>')
                self.wfile.write("<pre>")
                self.wfile.flush()
                while True:
                    error = False
                    while(True):
                        stdout = None
                        stderr = None
                        r,v,w = select.select([pid.stdout],[],[],.1)
                        #print "prsw112:", repr(r)
                        if len(r):
                            # print "prsw116: reading"
                            # stdout = pid.stdout.read()
                            stdout = r[0].read()
                            # print "prsw118:", stdout
                            break;
                        else:
                            r,v,w = select.select([pid.stderr],[],[],.1)
                            # print "prsw:ERR:112:", repr(r)
                            if len(r):
                                stderr = pid.stderr.read()
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
                self.wfile.write("</pre>")
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
            
            if self.path == "/":
                self.path = "/KitchenSink.html"
                
            print "JEN: chances"
            dirname = os.path.dirname(__file__)
            print "JEN: dirname =",dirname
            print "JEN: path = ", self.path 
            fname = os.path.join(dirname, "pyjamaface/prsproxygui/output", self.path[1:])
            print "JEN: fname =",fname
            
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
    try:
        print "starting httpserver on port ...", port,
        # important to set prior to any instantiations of MyHandler
        MyHandler.informers = informers
        if "dirdict" in informers:
            ppwstate.dirdict    = informers["dirdict"]
        if "dataSpider" in informers:
            ppwstate.dataSpider = informers["dataSpider"]
        # e.g. below by the HTTPServer class
        server = MTHTTPServer(('', port), MyHandler)
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

