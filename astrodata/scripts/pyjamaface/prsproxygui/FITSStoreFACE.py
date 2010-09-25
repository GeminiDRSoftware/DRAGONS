from Sink import Sink, SinkInfo
from pyjamas import Window
from pyjamas.ui.HTML import HTML
from pyjamas.ui.Frame import Frame
import rccutil

class FITSStore(Sink):
    def __init__(self, parent=None):

        Sink.__init__(self, parent)

        self.frame = Frame("/summary", Size=("100%",rccutil.getHeight()))
        self.initWidget(self.frame)
        Window.addWindowResizeListener(self)
        
    def onWindowResized(self, width, height):
        self.frame.setSize("100%", rccutil.getHeight())
        
    def onShow(self):
        pass


def init():
    return SinkInfo("Local FITS Store", 
                    "Introduction to the Kitchen Sink.", 
                    FITSStore)
