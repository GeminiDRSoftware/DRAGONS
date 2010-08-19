import xmlrpclib
import subprocess
from time import sleep
import os

class PRSProxy(object):
    prs = None
    found = False
    version = None

    @classmethod
    def getPRSProxy(cls, start = True):
        newProxy = None
        print "P12: getting PRSProxy"
        xmlrpcproxy = xmlrpclib.ServerProxy("http://localhost:8777/", allow_none=True)
        found = False
        try:
            newProxy = PRSProxy()
            
            newProxy.version = xmlrpcproxy.version()
            print "P18: version = ", repr(newProxy.version)
            newProxy.prs = xmlrpcproxy
            print "P21: setting found equal true"
            newProxy.found = True
            print "P22: about to register"
            newProxy.prs.register()
            print "P22: Proxy found"
        except:
            print "P24: Proxy Not Found"
            # not running, try starting one...
            if start == False:
                raise
            if start:
                del(newProxy)
                print "P25: starting prsproxy.py"
                prsout = open("prsproxylog-reduce%d" % os.getpid(), "w")
                pid = subprocess.Popen(["prsproxy.py", "--invoked"], stdout = prsout, stderr=prsout).pid
                print "P23: pid =", pid
                notfound = True
                print "P29: waiting"
                sleep(2)
                prs = cls.getPRSProxy(start=False)
                print "P35: prs=",repr(prs), prs.found
                if prs.found:
                    notfound = False
                    return prs
            else:
                newProxy.found = False
                

        return newProxy
        
    def __del__(self):
        print "P54:",repr(self.prs)
        if hasattr(self.prs,"found"):
            print "about to unregister from found prs"
            self.prs.unregister()
            print "unregistered from the prs"
        
            
    def calibrationSearch(self, calRq):
        if self.found == False:
            return None
        else:
            cal = self.prs.calibrationSearch(calRq.asDict())
            print "P28:", cal
            return cal
