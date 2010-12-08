from Sink import Sink, SinkInfo
from pyjamas.ui.HTML import HTML

class Info(Sink):
    def __init__(self, parent=None):

        Sink.__init__(self, parent)

        text="""
            <div class='infoProse'>
             This is the AstroData Control Center Engineering Interface. 
             <p>
             The "astrodata control center" is an application which 
             acts as a common point of contact for the "reduce" command
             which executes "recipes", providing coordinated sevices such as
             calibration search, image display, stacking-list maintainance,
             and image quality statics.
             </p>
             <p>
             It is
             written in Pyjamas which is a system that compiles python 
             programs into HTML+Javascript. It is entirely separate from 
             the "adcc" 
             application, except that the adcc HTTP interface will serve the 
             HTML which is required for the interface to make use of HTTPRequest.
             </p>
             <p>
             When invoked from the command line, "reduce", if there is no adcc
             running, reduce will start one. This mode could be considered the
             baseline, or normal case. The reduce instance communicates with the
             adcc using XMLRPC, and the adcc essentially acts as a library
             which happens to have its own process.  This interface is only useful
             when adcc is started ahead of time, since otherwise when reduce is
             finished running, the adcc will shut down as well.
             </p>
            </div>"""
        self.initWidget(HTML(text, True))

    def onShow(self):
        pass


def init():
    return SinkInfo("About", "About the AstroData Control Center", Info)
