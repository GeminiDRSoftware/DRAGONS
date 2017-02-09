from __future__ import print_function
#                                                                     QAP Gemini
#
#                                                                     adcclib.py
# ------------------------------------------------------------------------------
from builtins import object
from future.utils import with_metaclass
__version__ = '2.0 (beta)'
# ------------------------------------------------------------------------------
"""
Automated Dataflow Coordination Center

"""
import os
import sys
import signal
import time
from threading import Event
from threading import Thread

from recipe_system.adcc.servers import http_proxy
from recipe_system.adcc.servers import eventsManager

from recipe_system.config import globalConf, STANDARD_REDUCTION_CONF
# ------------------------------------------------------------------------------
def get_adcc_dir(dirtitle="adcc"):
    dotadcc = {"adcc": ".adcc"}
    if not os.path.exists(dotadcc[dirtitle]):
        os.mkdir(dotadcc[dirtitle])
    return dotadcc[dirtitle]

def write_adcc_sr(srname, vals):
    print("adcclib: adcc startup report in {}".format(srname))
    with open(srname, "w+") as sr:
        sr.write(repr(vals))
    return
# ------------------------------------------------------------------------------
class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class ADCC(with_metaclass(Singleton, object)):
    def __init__(self, args=None):
        if args is None:
            pass
        else:
            self.dark      = args.dark
            self.events    = eventsManager.EventsManager()
            self.http_port = args.httpport
            self.sreport   = args.adccsrn
            self.racefile  = "adccinfo.py"
            self.verbose   = args.verbosity
            self.web       = None

    def _check_adcc(self):
        curpid = os.getpid()
        running = []
        for line in os.popen("ps ax | grep adcc | grep -v grep"):
            fields = line.split()
            pid = fields[0]
            if int(pid) != int(curpid):
                print("adcclib: adcc process {} running.".format(pid))
                running.append(pid)
        return running

    def _check_kill_adcc(self, pids):
        for pid in pids:
            os.kill(int(pid), signal.SIGKILL)
        return

    def _http_interface(self, run_event):
        # establish HTTP server and proxy.
        self.web = Thread(group=None, target=http_proxy.main, name="webface",
                          args=(run_event,),
                          kwargs={'port': self.http_port, 'dark': self.dark,
                                  'events': self.events, 'verbose': self.verbose})
        return

    def _handle_locks(self):
        adccdir = get_adcc_dir()
        lockf = os.path.join(adccdir, self.racefile)
        lfile = True if os.path.exists(lockf) else False
        pids = self._check_adcc()
        if pids and lfile:
            sys.exit("adcclib: adcc running and lockfile detected.")
        elif pids and not lfile:
            sys.exit("adcclib: adcc running on port {}".format(self.http_port))
        elif lfile and not pids:
            print("adcclib: No adcc running but lockfile found.")
            print("adcclib: adcc configuration appears to be corrupted. Clearing ...")
            os.unlink(lockf)
        return

    def _write_locks(self):
        """
        Write racefile and ADCC Startup Report

        """
        dotadcc = get_adcc_dir()
        vals = {"http_port": self.http_port, "pid": os.getpid()}
        rfile = os.path.join(dotadcc, self.racefile)
        with open(rfile, "w") as ports:
            ports.write(repr(vals))

        sr = os.path.join(dotadcc, self.sreport)
        write_adcc_sr(sr, vals)
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
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nadcc: exiting due to Ctrl-C")
            run_event.clear()
            self.web.join()

        if os.path.exists(self.racefile):
            os.remove(self.racefile)

        return
