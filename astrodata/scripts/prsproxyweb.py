#Copyright Jon Berg , turtlemeat.com

import string,cgi,time
from os import curdir, sep
from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
#import pri
import select
from copy import copy
import datetime
import os

webserverdone = False
class MyHandler(BaseHTTPRequestHandler):
    informers = None

    def do_GET(self):
        global webserverdone
        
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
                <li><a href="killprs">Kill this server</a> (%(numinsts)d copies of reduce registered)</li>
                </ul>
                <body>
                </html>""" % {"numinsts":rim.numinsts}
                self.wfile.write(page)
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
                page = """
                <html>
                <head>
                    <meta http-equiv="refresh" content="2" />
                </head>
                <body>
                %(body)s
                <body>
                </html>"""
                # body onload="javascript:setTimeout(\u201clocation.reload(true);\u201d,50000);">
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
                
                return 
            if self.path == "/killprs":
                self.send_response(200)
                self.send_header('Content-type',	'text/html')
                self.end_headers()
                self.wfile.write("Killed this prsproxy instance, pid = %d at %s" %(os.getpid(), str(datetime.datetime.now())))
                webserverdone = True
            return
                
        except IOError:
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

