#                                                                     QAP Gemini
#
#                                                                     adcclib.py
# ------------------------------------------------------------------------------
import os
import sys
import signal
import time

from copy import copy
from threading import Event
from threading import Thread

from recipe_system.adcc.servers import http_proxy
from recipe_system.adcc.servers import eventsManager

from recipe_system.config import globalConf
from recipe_system.config import STANDARD_REDUCTION_CONF
from recipe_system.utils.findexe import findexe

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


class Singleton(type):

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class ADCC(metaclass=Singleton):

    def __init__(self, args=None):
        if args is None:
            pass
        else:
            self.dark = args.dark
            self.events = eventsManager.EventsManager()
            self.spec_events = eventsManager.EventsManager()
            self.http_port = args.httpport
            self.sreport = args.adccsrn
            self.racefile = "adccinfo.py"
            self.verbose = args.verbosity
            self.web = None

    def _check_adcc(self, cpid):
        adccproc = findexe('adcc')
        xprocx = copy(adccproc)
        msg = "adcclib: adcc process {} running."
        try:
            xprocx.pop(adccproc.index(cpid))
        except ValueError as err:
            pass

        x = [print(msg.format(p)) for p in xprocx]
        return xprocx

    def _check_kill_adcc(self, pids):
        for pid in pids:
            os.kill(int(pid), signal.SIGKILL)
        return

    def _http_interface(self, run_event):
        # establish HTTP server and proxy.
        self.web = Thread(group=None, target=http_proxy.main, name="webface",
                          args=(run_event,),
                          kwargs={
                              'port': self.http_port,
                              'dark': self.dark,
                              'events': self.events,
                              'spec_events': self.spec_events,
                              'verbose': self.verbose
                          }
        )
        return

    def _handle_locks(self):
        curpid = os.getpid()
        adccdir = get_adcc_dir()
        lockf = os.path.join(adccdir, self.racefile)
        lfile = True if os.path.exists(lockf) else False
        pids = self._check_adcc(curpid)
        msgs = {
            'lockrun': "adcclib: adcc running and lockfile detected.",
            'portrun': "adcclib: adcc running on port {}",
            'norun': "adcclib: No adcc running but lockfile found.",
            'rupted': "adcclib: adcc config appears corrupted. Clearing ..."
        }

        if pids and lfile:
            sys.exit(msgs['lockrun'])
        elif pids and not lfile:
            sys.exit(msgs['portrun'].format(self.http_port))
        elif lfile and not pids:
            print(msgs['norun'])
            print(msgs['rupted'])
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
