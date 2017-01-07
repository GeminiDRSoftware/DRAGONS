from __future__ import print_function
#                                                                     QAP Gemini
#
#                                                                     adcclib.py
# ------------------------------------------------------------------------------
from builtins import object
from future.utils import with_metaclass
__version__ = 'beta (new hope)'
# ------------------------------------------------------------------------------
"""
Automated Dataflow Coordination Center

"""
import os
import sys
import time
from threading import Event
from threading import Thread

from recipe_system.adcc.servers import http_proxy
from recipe_system.adcc.servers import eventsManager

from recipe_system.config import globalConf, STANDARD_REDUCTION_CONF
# ------------------------------------------------------------------------------
def getPersistDir(dirtitle="adcc"):
    dotadcc = {"adcc": ".adcc"}
    if not os.path.exists(dotadcc[dirtitle]):
        os.mkdir(dotadcc[dirtitle])
    return dotadcc[dirtitle]

def writeADCCSR(filename, vals=None):
    if filename is None:
        print("adcc.writeADCCSR(): no filename for sr")
        filename = ".adcc/adccReport"

    print("adcc.writeADCCSR(): startup report in {}".format(filename))
    with open(filename, "w+") as sr:
        if vals is None:
            sr.write("ADCC ALREADY RUNNING\n")
        else:
            sr.write(repr(vals))
    return
# ------------------------------------------------------------------------------
class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class ADCC(object):
    __metaclass__ = Singleton

    def __init__(self, args=None):
        if args is None:
            pass
        else:
            self.clfn      = args.adccsrn
            self.events    = eventsManager.EventsManager()
            self.http_port = args.httpport
            self.sreport   = args.adccsrn
            self.racefile  = ".adcc/adccinfo.py"
            self.verbose   = args.verbosity
            self.web       = None

    def _http_interface(self, run_event):
      # establish HTTP server and proxy.
        self.web = Thread(group=None, target=http_proxy.main, name="webface",
                          args=(run_event,),
                          kwargs={"port": self.http_port, 'events': self.events,
                                  "verbose": self.verbose})
        return

    def _handle_locks(self):
        adccdir = getPersistDir()
        if os.path.exists(self.racefile):
            print("adcclib: adcc lockfile present.")
            try:
                if not self.web.is_alive():
                    print("adcc.main(): no adcc running, clearing lockfile.")
                    os.remove(racefile)
                else:
                    writeADCCSR(clfn)
                    sys.exit("adcc instance already running. No-Op.")
            except AttributeError:
                print("Web Interface thread is not alive.")
                pass
        return

    def _write_locks(self):
        """
        Write racefile and ADCC Startup Report

        """
        # write HTTP port
        vals = {"http_port": self.http_port, "pid": os.getpid()}
        with open(self.racefile, "w") as ports:
            ports.write(repr(vals))

        writeADCCSR(self.clfn, vals=vals)
        return

    def main(self):
        globalConf.load(STANDARD_REDUCTION_CONF, env_override=True)
        self._handle_locks()
        self._write_locks()
        # start webinterface
        run_event = Event()
        run_event.set()
        self._http_interface(run_event)
        self.web.start()
        try:
            while True:
                time.sleep(.1)
        except KeyboardInterrupt:
            print("\nadcc: exiting due to Ctrl-C")
            run_event.clear()
            self.web.join()
            #if os.path.exists(self.racefile):
                #os.remove(self.racefile)
        return
