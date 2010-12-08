from Sink import Sink, SinkInfo
from pyjamas.ui.Grid import Grid
from pyjamas.ui.FlexTable import FlexTable
from pyjamas.ui import HasHorizontalAlignment
from pyjamas.ui.Image import Image

class Tables(Sink):
    def __init__(self):
        Sink.__init__(self)
        inner = Grid(10, 5, Width="100%", BorderWidth="1")
        outer = FlexTable(Width="100%", BorderWidth="1")

        outer.setWidget(0, 0, Image(self.baseURL() + "rembrandt/LaMarcheNocturne.jpg"))
        outer.getFlexCellFormatter().setColSpan(0, 0, 2)
        outer.getFlexCellFormatter().setHorizontalAlignment(0, 0, HasHorizontalAlignment.ALIGN_CENTER)

        outer.setHTML(1, 0, "Look to the right...<br>That's a nested table component ->")
        outer.setWidget(1, 1, inner)
        outer.getCellFormatter().setColSpan(1, 1, 2)
        
        for i in range(10):
            for j in range(5):
                inner.setText(i, j, "%d" % i + ",%d" % j)

        self.initWidget(outer)
        
    def onShow(self):
        pass

def init():
    text="The <code>FlexTable</code> widget doubles as a tabular data formatter and a panel.  In this example, you'll see that there is an outer table with four cells, two of which contain nested components."
    return SinkInfo("Tables", text, Tables)

