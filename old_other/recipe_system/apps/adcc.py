#!/usr/bin/env python
#
#
#                                                                     QAP Gemini
#
#                                                                        adcc.py
#                                                                        07-2013
# ------------------------------------------------------------------------------
# $Id$
# ------------------------------------------------------------------------------
__version__      = '$Revision$'[11:-2]
__version_date__ = '$Date$'[7:-2]
# ------------------------------------------------------------------------------
# gemini_python X1 (GPX1) version of adcc:
#  -- Updated to argparse*
#  -- Updated to externalized ReduceInstanceManager
#  -- Updated PRSProxy version to 'GPX1 v1.0 in get_version()
# ------------------------------------------------------------------------------
"""Automated Dataflow Coordination Center"""

import os
import sys
import select
import socket

from threading import Thread
from SimpleXMLRPCServer import SimpleXMLRPCServer

from recipe_system.adcc.servers import http_proxy
from recipe_system import cal_service
from recipe_system.config import globalConf, STANDARD_REDUCTION_CONF
from recipe_system.reduction.reduceInstanceManager import ReduceInstanceManager

# ------------------------------------------------------------------------------
def buildArgParser():
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Automated Data Communication Center "
                            "(ADCC). This is the proxy to PRS functionality, "
                            "also invoked locally, e.g. calibration requests.")

    parser.add_argument("-v", "--verbose", 
                        dest="verbosity", action="store_true",
                        help="increase HTTP client messaging on GET requests.")

    parser.add_argument("-i", "--invoked", 
                        dest = "invoked", action="store_true",
                        help = "Used by processes that invoke adcc, so "
                        "that PRS proxy knows when to exit. If not present, "
                        "the adcc registers itself and will only exit by "
                        "user control (or by os-level signal).")

    parser.add_argument("--startup-report", 
                        dest = "adccsrn", default=None, 
                        help = "file name for adcc startup report")

    parser.add_argument("-r", "--reduce-port", 
                        dest = "reduceport", default=54530, type=int,
                        help="Option informs adcc of the port on which "
                        "reduce listens for xmlrpc commands.")

    parser.add_argument("-p", "--reduce-pid", 
                        dest ="reducepid", default=None, type=int,
                        help = "Option informs adcc of the reduce "
                        "application's PID.")

    parser.add_argument("-l", "--listen-port", 
                        dest = "listenport", default=53530, type=int,
                        help="adcc listener port for the xmlrpc server.")

    parser.add_argument("-w", "--http-port", 
                        dest = "httpport", default=8777, type=int,
                        help="Response port for the web interface. "
                        "i.e. http://localhost:<http-port>/")

    args = parser.parse_args()
    return args

# ------------------------------- UTILITY FUNCS --------------------------------
def getPersistDir(dirtitle = "adcc"):
    dirs = {"adcc":".adcc"}
    for dirt in dirs.keys():
        if not os.path.exists(dirs[dirt]):
            os.mkdir(dirs[dirt])
    return dirs[dirtitle]

def writeADCCSR(filename, vals=None):
    if filename is None:
        print "adcc.writeADCCSR(): no filename for sr"
        filename = ".adcc/adccReport"

    print "adcc.writeADCCSR(): startup report in {}".format(filename)
    with open(filename, "w+") as sr:
        if vals is None:
            sr.write("ADCC ALREADY RUNNING\n")
        else:
            sr.write(repr(vals))
    return

def get_args():
    return buildArgParser()

def get_version():
    version = [("ADCC","GPX1 v1.0")]
    print "adcc version:", repr(version)
    return version

# ----------------------------- END UTILITY FUNCS ------------------------------
# begin negotiated startup... we won't run if another adcc owns this directory
# could be done later or in lazy manner, but for now ensure this is present
def main(args):
    racefile = ".adcc/adccinfo.py"
    # caller lock file name

    globalConf.load(STANDARD_REDUCTION_CONF, env_override=True)
    clfn = args.adccsrn
    adccdir = getPersistDir()
    if os.path.exists(racefile):
        print "adcc.main(): adcc lockfile present."
        from recipe_system.adcc.servers.xmlrpc_proxy import PRSProxy
        adcc = PRSProxy.get_adcc(check_once=True)
        if adcc == None:
            print "adcc.main(): no adcc running, clearing lockfile."
            os.remove(racefile)
        else:
            adcc.unregister()
            writeADCCSR(clfn)
            sys.exit("adcc instance already running. Halt.")

    # note: try to get a unique port starting at the standard port
    findingPort = True
    while findingPort:
        try:
            server = SimpleXMLRPCServer(("localhost", args.listenport), 
                                        allow_none=True)
            print "ADCC listening on port %d..." % args.listenport
            findingPort = False
        except socket.error:
            args.listenport += 1

    # write out XMLRPC and HTTP port   
    vals = {"xmlrpc_port": args.listenport,
            "http_port"  : args.httpport,
            "pid"        : os.getpid() }

    # Write racefile and ADCC Startup Report
    with open(racefile, "w") as ports:
        ports.write(repr(vals))

    writeADCCSR(clfn, vals=vals)
    server.register_function(get_version, "get_version")
    server.register_function(cal_service.cal_search_factory(),
                             "calibration_search")

    # store the port
    rim = ReduceInstanceManager(args.reduceport)
    server.register_instance(rim)

    # start webinterface
    web = Thread(None, http_proxy.main, "webface", 
                 kwargs = {"port":args.httpport,
                           "rim":rim,
                           "verbose": args.verbosity})
    web.start()

    outerloopdone = False
    while True:
        if outerloopdone:
            break
        try:
            while True:
                r, w, x = select.select([server.socket], [], [], 0.5)
                if r:
                    server.handle_request() 

                if http_proxy.webserverdone:
                    print "\nadcc: exiting due to command vie http interface"
                    print "adcc: %d reduce instances abandoned:" % rim.numinsts
                    outerloopdone = True
                    break

                if (args.invoked and rim.finished):
                    print "adcc exiting, no reduce instances to serve."
                    outerloopdone = True
                    http_proxy.webserverdone = True
                    break

        except KeyboardInterrupt:
            print "\nadcc: exiting due to Ctrl-C"
            print "adcc: %d instance(s) abandoned" % rim.numinsts
            outerloopdone = True
            http_proxy.webserverdone = True
            break

    if os.path.exists(racefile):
        os.remove(racefile)

# ------------------------------------------------------------------------------
if __name__ == '__main__':
    args = get_args()
    sys.exit(main(args))
