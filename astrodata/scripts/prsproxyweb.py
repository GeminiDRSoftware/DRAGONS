#Copyright Jon Berg , turtlemeat.com

import string,cgi,time
from os import curdir, sep
from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
#import pri
import select
from copy import copy
import datetime
import os
import subprocess
import cgi

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
webserverdone = False
class MyHandler(BaseHTTPRequestHandler):
    informers = None

    def do_GET(self):
        global webserverdone
        print "path=",self.path
        
        parms = parsepath(self.path)
        print "prs45:", repr(parms)
        qd = cgi.parse_qs(self.path)
        print "prs52:", repr(qd)
        rim = self.informers["rim"]
        try:
            if self.path == "/":
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
            if parms["path"] == "/datadir":
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
                        print "prsw112:", repr(r)
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
            if self.path.endswith(".html"):
                f = open(curdir + sep + self.path) #self.path has /test.html
#note that this potentially makes every file on your computer readable by the internet

                self.send_response(200)
                self.send_header('Content-type',	'text/html')
                self.end_headers()
                self.wfile.write(f.read())
                f.close()
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
                
                
        except IOError:
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
def startInterfaceServer(port = 8777, **informers):
    try:
        print "starting httpserver on port ...", port,
        # important to set prior to any instantiations of MyHandler
        MyHandler.informers = informers 
        # e.g. below by the HTTPServer class
        server = HTTPServer(('', port), MyHandler)
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

