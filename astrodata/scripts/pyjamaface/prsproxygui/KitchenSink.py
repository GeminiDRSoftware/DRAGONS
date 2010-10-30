import pyjd # this is dummy in pyjs

from pyjamas.ui.Button import Button
from pyjamas.ui.RootPanel import RootPanel
from pyjamas.ui.HTML import HTML
from pyjamas.ui.DockPanel import DockPanel
from pyjamas.ui import HasAlignment
from pyjamas.ui.Hyperlink import Hyperlink
from pyjamas.ui.VerticalPanel import VerticalPanel
from pyjamas import Window
from SinkList import SinkList
from pyjamas import History
import Info
import Buttons
import Layouts
import Images
import Menus
import Lists
import Popups
import Tables
import Text
import Trees
import Frames
import Tabs
import DataTree
import RecipeSystemIFACE
import RecipeViewer
import ADViewerIFACE
import FITSStoreFACE
import DisplayIFACE
from Logger import Logger
from pyjamas import log

import rccutil

class KitchenSink:

    filexml = None

    def onHistoryChanged(self, token):
        log.writebr("onHistoryChanged: %s" % token)
        info = self.sink_list.find(token)
        if info:
            self.show(info, False)
        else:
            self.showInfo()

    def onModuleLoad(self):
        self.curInfo=''
        self.curSink=None
        self.description=HTML()
        self.sink_list=SinkList()
        self.panel=DockPanel()
        
        self.loadSinks()
        self.sinkContainer = DockPanel()
        self.sinkContainer.setStyleName("ks-Sink")

        vp=VerticalPanel()
        vp.setWidth("100%")
        vp.add(self.description)
        vp.add(self.sinkContainer)

        self.description.setStyleName("ks-Info")

        self.panel.add(self.sink_list, DockPanel.WEST)
        self.panel.add(vp, DockPanel.CENTER)

        self.panel.setCellVerticalAlignment(self.sink_list, HasAlignment.ALIGN_TOP)
        self.panel.setCellWidth(vp, "100%")

        History.addHistoryListener(self)
        RootPanel().add(self.panel)
        RootPanel().add(Logger())

        #Show the initial screen.
        initToken = History.getToken()
        if len(initToken):
            self.onHistoryChanged(initToken)
        else:
            self.showInfo()
            
    def getHeight(self):
        return rccutil.getHeight()
        
    def show(self, info, affectHistory):
        if info == self.curInfo: return
        self.curInfo = info

        #Logger.write("showing " + info.getName())
        if self.curSink <> None:
            self.curSink.onHide()
            #Logger.write("removing " + self.curSink)
            self.sinkContainer.remove(self.curSink)

        self.curSink = info.getInstance()
        self.sink_list.setSinkSelection(info.getName())
        self.description.setHTML(info.getDescription())

        if (affectHistory):
            History.newItem(info.getName())

        self.sinkContainer.add(self.curSink, DockPanel.CENTER)
        self.sinkContainer.setCellWidth(self.curSink, "100%")
        self.sinkContainer.setCellHeight(self.curSink, "100%")
        self.sinkContainer.setCellVerticalAlignment(self.curSink, HasAlignment.ALIGN_TOP)
        self.curSink.onShow()
        
    def loadSinks(self):
        #self.sink_list.addSink(DataTree.init())
        #self.sink_list.addSink(RecipeSystemIFACE.init())
        self.sink_list.addSink(ADViewerIFACE.init())
        self.sink_list.addSink(RecipeViewer.init())
        self.sink_list.addSink(FITSStoreFACE.init())
        self.sink_list.addSink(DisplayIFACE.init())
        self.sink_list.addSink(Info.init())
        if False:
            self.sink_list.addSink(Buttons.init())
            self.sink_list.addSink(Menus.init())
            self.sink_list.addSink(Images.init())
            self.sink_list.addSink(Layouts.init())
            self.sink_list.addSink(Lists.init())
            self.sink_list.addSink(Popups.init())
            self.sink_list.addSink(Tables.init())
            self.sink_list.addSink(Text.init())
        if False: #preserving originaly order
            self.sink_list.addSink(Frames.init())
            self.sink_list.addSink(Tabs.init())

    def showInfo(self):
        self.show(self.sink_list.find("AstroData Viewer"), False)




if __name__ == '__main__':
    pyjd.setup("public/AstroDataCenter.html")
    app = KitchenSink()
    app.onModuleLoad()
    pyjd.run()
