from Sink import Sink, SinkInfo
from pyjamas.ui.HTML import HTML
from pyjamas.ui.Frame import Frame
class FITSStore(Sink):
    def __init__(self, parent=None):

        Sink.__init__(self, parent)

        self.frame = Frame("/summary", Size=("100%","100%"))
        self.initWidget(self.frame)

    def onShow(self):
        pass


def init():
    return SinkInfo("Local FITS Store", 
                    "Introduction to the Kitchen Sink.", 
                    FITSStore)
