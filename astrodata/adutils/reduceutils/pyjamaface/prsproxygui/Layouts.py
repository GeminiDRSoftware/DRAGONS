from Sink import Sink, SinkInfo
from pyjamas.ui.Button import Button
from pyjamas.ui.CheckBox import CheckBox
from pyjamas.ui.VerticalPanel import VerticalPanel
from pyjamas.ui.HorizontalPanel import HorizontalPanel
from pyjamas.ui.HTML import HTML
from pyjamas.ui.DockPanel import DockPanel
from pyjamas.ui import HasAlignment
from pyjamas.ui.FlowPanel import FlowPanel
from pyjamas.ui.HTMLPanel import HTMLPanel
from pyjamas.ui.MenuBar import MenuBar
from pyjamas.ui.MenuItem import MenuItem
from pyjamas.ui.ScrollPanel import ScrollPanel
from pyjamas.ui.DisclosurePanel import DisclosurePanel 
from Logger import Logger
from pyjamas import DOM

class Layouts(Sink):
    def __init__(self):
        Sink.__init__(self)

        text="""This is a <code>ScrollPanel</code> contained at 
        the center of a <code>DockPanel</code>. 
        By putting some fairly large contents 
        in the middle and setting its size explicitly, it becomes a 
        scrollable area within the page, but without requiring the use of 
        an IFRAME.
        Here's quite a bit more meaningless text that will serve primarily 
        to make this thing scroll off the bottom of its visible area.  
        Otherwise, you might have to make it really, really small in order 
        to see the nifty scroll bars!"""
        
        contents = HTML(text)
        scroller = ScrollPanel(contents, StyleName="ks-layouts-Scroller")
        
        dock = DockPanel(HorizontalAlignment=HasAlignment.ALIGN_CENTER,
                         Spacing=10)
        north0 = HTML("This is the <i>first</i> north component", True)
        east = HTML("<center>This<br>is<br>the<br>east<br>component</center>", True)
        south = HTML("This is the south component")
        west = HTML("<center>This<br>is<br>the<br>west<br>component</center>", True)
        north1 = HTML("This is the <b>second</b> north component", True)
        dock.add(north0, DockPanel.NORTH)
        dock.add(east, DockPanel.EAST)
        dock.add(south, DockPanel.SOUTH)
        dock.add(west, DockPanel.WEST)
        dock.add(north1, DockPanel.NORTH)
        dock.add(scroller, DockPanel.CENTER)
        
        #Logger.write("Layouts", "TODO: flowpanel")
        flow = FlowPanel()
        for i in range(8):
            flow.add(CheckBox("Flow %d" % i))

        horz = HorizontalPanel(VerticalAlignment=HasAlignment.ALIGN_MIDDLE)
        horz.add(Button("Button"))
        horz.add(HTML("<center>This is a<br>very<br>tall thing</center>", True))
        horz.add(Button("Button"))

        vert = VerticalPanel(HorizontalAlignment=HasAlignment.ALIGN_CENTER)
        vert.add(Button("Small"))
        vert.add(Button("--- BigBigBigBig ---"))
        vert.add(Button("tiny"))

        menu = MenuBar()
        menu0 = MenuBar(True)
        menu1 = MenuBar(True)
        menu.addItem("menu0", menu0)
        menu.addItem("menu1", menu1)
        menu0.addItem("child00")
        menu0.addItem("child01")
        menu0.addItem("child02")
        menu1.addItem("child10")
        menu1.addItem("child11")
        menu1.addItem("child12")

        #Logger.write("Layouts", "TODO: htmlpanel")
        id = HTMLPanel.createUniqueId()
        text="""This is an <code>HTMLPanel</code>.  It allows you to add 
            components inside existing HTML, like this: <span id='%s' />
            Notice how the menu just fits snugly in there?  Cute.""" % id
        html = HTMLPanel(text)
        
        DOM.setStyleAttribute(menu.getElement(), "display", "inline")
        html.add(menu, id)

        disclose = DisclosurePanel("Click to disclose")
        disclose.add(HTML("""<b>Ta-daaaaa!</b><br />Ok - it could have
                             been<br />more of a surprise."""))

        panel = VerticalPanel(Spacing=8,
                              HorizontalAlignment=HasAlignment.ALIGN_CENTER)
        
        panel.add(self.makeLabel("Dock Panel"))
        panel.add(dock)
        panel.add(self.makeLabel("Flow Panel"))
        panel.add(flow)
        panel.add(self.makeLabel("Horizontal Panel"))
        panel.add(horz)
        panel.add(self.makeLabel("Vertical Panel"))
        panel.add(vert)
        panel.add(self.makeLabel("HTML Panel"))
        panel.add(html)
        panel.add(self.makeLabel("Disclosure Panel"))
        panel.add(disclose)
        
        self.initWidget(panel)
        self.setStyleName("ks-layouts")

    def onShow(self):
        pass

    def makeLabel(self, caption):
        html = HTML(caption)
        html.setStyleName("ks-layouts-Label")
        return html


def init():
    text="""This page demonstrates some of the basic GWT panels, each of which
        arranges its contained widgets differently.  
        These panels are designed to take advantage of the browser's 
        built-in layout mechanics, which keeps the user interface snappy 
        and helps your AJAX code play nicely with existing HTML.  
        On the other hand, if you need pixel-perfect control, 
        you can tweak things at a low level using the 
        <code>DOM</code> class."""
    return SinkInfo("Layouts", text, Layouts)
