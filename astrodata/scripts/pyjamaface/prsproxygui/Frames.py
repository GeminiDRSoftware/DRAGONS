from Sink import Sink, SinkInfo
from pyjamas.ui.Frame import Frame

class Frames(Sink):
    def __init__(self):
        Sink.__init__(self)
        self.frame=Frame(self.baseURL() + "rembrandt/LaMarcheNocturne.html")
        
        self.frame.setWidth("100%")
        self.frame.setHeight("48em")
        self.initWidget(self.frame)
    
def init():
    text="If you need to include multiple pages of good ol' static HTML, it's easy to do using the <code>Frame</code> class."
    return SinkInfo("Frames", text, Frames)

