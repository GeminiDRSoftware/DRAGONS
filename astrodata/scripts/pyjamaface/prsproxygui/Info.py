from Sink import Sink, SinkInfo
from pyjamas.ui.HTML import HTML

class Info(Sink):
    def __init__(self):

        Sink.__init__(self)

        text="""
            <div class='infoProse'>This is the PRSProxy Interface.  
             <p>
             The prsproxy is an application which communicates with Pipeline 
             Resource Services for the recipe system within the Gemini astrodata
             packages. The recipe system is generally invoked using the 'reduce'
             command.  When a PRS is required, such as an official calibration
             search, reduce attempts to contact a prsproxy via xmlrpc on
             at localhost:53530. If none is found then reduce will spawn a
             prsproxy instance. The prsproxy instance will notify the spawning
             reduce application when it is ready to operate, and reduce will
             continue with it's request via the xmlrps interface.
                          
             The prsproxy also exposes an HTTP interface at (by default) 
             localhost:8777. However, a prsproxy instance invoked by reduce will
             shutdown as soon as reduce "unregisters". If some other reduce
             instance has registered, prsproxy will also wait for it to
             unregister, but again, when done, it will quit.  This makes the human
             interface at http://localhost:8777/ much less usable, as the server
             will go away, and if it comes back, it will be with a fresh environment.
             
             Thus, to reliably interact with prsproxy's web interface, it is better
             to invoke it ahead of time.  Whichever directory prsproxy is invoked in
             will be it's root data directory.  Then, if reduce instances are started
             from the command line they will appear in the prsproxygui, but the 
             prsproxy will not shut down when they finish. Also, it's possible to
             use this interface to have prsproxy launch the reduce commands,
             using the data browser and related AJAX interfaces.
             </p>
            </div>"""
        self.initWidget(HTML(text, True))

    def onShow(self):
        pass


def init():
    return SinkInfo("Info", "Introduction to the Kitchen Sink.", Info)
