from Sink import Sink, SinkInfo
from pyjamas import Window
from pyjamas.ui.Tree import Tree
from pyjamas.ui.TreeItem import TreeItem
from pyjamas.HTTPRequest import HTTPRequest
from pyjamas.ui.HTML import HTML
from __pyjamas__ import JS
from pyjamas.ui.DockPanel import DockPanel
from pyjamas.ui import HasAlignment
from pyjamas.ui.HTMLPanel import HTMLPanel
from pyjamas.ui.ListBox import ListBox
from pyjamas.ui.HorizontalPanel import HorizontalPanel
from pyjamas.ui.VerticalPanel import VerticalPanel
from pyjamas.ui.Label import Label
from pyjamas.ui.Button import Button

from urllib import quote
from DataDictTree import DataDictTree, PanelIFACE, AdinfoIFACE
from rccutil import create_xml_doc
import rccutil

class DirDictLoader:
    panel = None
    def __init__(self, panel):
        self.panel = panel
        
    def onCompletion(self, txt):
        #JS('alert("This is the thing\n"+txt)')
        self.panel.fromXML(txt)
        
class RecipeListLoader:
    def __init__(self, panel):
        self.panel = panel
        
    def onCompletion(self, txt):
        doc = minidom.parseString(txt)
        #node = doc.firstChild
        
        #node = doc.getElementById("topDirectory")
        nodes = doc.getElementsByTagName("recipe")
        if len(nodes)>0:
            rlist = []
            for i in range(0, len(nodes)):
                node = nodes.item(i)
                rlist.append(node.getAttribute("name"))
                
            rlist.sort()
            for r in rlist:
                self.panel.recipeList.addItem(r)

class ADInfoLoader:
    panel = None
    
    def __init__(self, panel):
        self.panel = panel
    
    def onCompletion(self, txt):
        self.panel.adInfo.setHTML(txt)
      
class CalibrationLoader:
    panel = None
    
    def __init__(self, panel, local=True):
        self.panel = panel
        self.local = local
        
        
    def onCompletion(self, txt):
        local = self.local
        try:
            try:
                doc = create_xml_doc(txt)
                rhtml = ""
                sci = doc.getElementsByTagName("dataset")
                scinode = sci.item(0)
                if scinode:
                    filenamenode= scinode.getElementsByTagName("filename")
                    sciname = filenamenode.item(0).textContent
                    datalabelnode = scinode.getElementsByTagName("datalabel").item(0)
                    datalab = datalabelnode.textContent
                else:
                    if local:
                        self.panel.localcalreport.setHTML(
                            """ 
                            <h2>Local Calibration Search Results</h2>
                            <span style="color:red">
                                Error From Local Calibration Result
                            </span>
                            """)
                    else:
                        self.panel.globalcalreport.setHTML(
                            """
                            <h2>Global Calibration Search Results</h2>
                            <span style="color:red">
                                    Error From Global Calibration Result
                            </span>
                            """)
                    return


                # title of results
                if local:
                    rhtml += "<h2>Local Calibration Search Results</h2>"
                else:
                    rhtml += "<h2>Global Calibration Search Results</h2>"
                rhtml += "For fileset: <b>%s</b><br/>\n" % str(sciname)
                
                # add data label                
                rhtml += "data label: %s<br/>" % str(datalab)
                
                # 
                nodes = doc.getElementsByTagName("calibration")
                rhtml += '<table cellspacing="2px">\n'
                rhtml += """
                            <COLGROUP align="right" />
                            <COLGROUP align="left" />
                         """
                rhtml += "<thead>\n"
                rhtml += "<tr>\n"
                rhtml += '<td style="background-color:grey">%s</td>\n' % 'Calibration Type'
                rhtml += '<td style="background-color:grey">%s</td>\n' % 'Canonical Filename'
                rhtml += "</tr>\n"
                rhtml += "</thead>\n"
                for i in range(0, len(nodes)):
                    node = nodes.item(i)
                    ctnode = node.getElementsByTagName("caltype").item(0)
                    fnnode = node.getElementsByTagName("filename").item(0)
                    caltype = ctnode.textContent
                    fname = fnnode.textContent
                    rhtml += "<tr>\n"
                    rhtml += "<td>%s:</td>\n"%caltype
                    rhtml += "<td><b>%s</b></td>\n"%fname
                    rhtml += "</tr>\n"
                rhtml += "</table>\n"


                if local:
                    self.panel.localcalreport.setHTML(rhtml)
                else:
                    self.panel.globalcalreport.setHTML(rhtml)
            except:

                if local:
                    self.panel.localcalreport.setHTML(
                        """ 
                        <h2>Local Calibration Search Results</h2>
                        <span style="color:red">
                            Error From Local Calibration Result
                        </span>
                        """)
                else:
                    self.panel.globalcalreport.setHTML(
                        """
                        <h2>Global Calibration Search Results</h2>
                        <span style="color:red">
                                Error From Global Calibration Result
                        </span>
                        """)
        except:
            pass
                    
from pyjamas.ui.SimplePanel import SimplePanel
from pyjamas.ui.TabPanel import TabPanel
import RecipeSystemIFACE
class CalsIFACE(RecipeSystemIFACE.PanelIFACE):
    def __init__(self):
        self.calsPanel = VerticalPanel(Size=("50%", ""))
        self.info = HTML("", Width="100%")
        self.calsPanel.add(self.info)

        self.localcalreport = HTML("", True, )
        self.globalcalreport = HTML("", True, )
        self.calsPanel.add(self.localcalreport)
        self.calsPanel.add(self.globalcalreport)
        self.panel = self.calsPanel
    
    def onTreeItemSelected(self, item):
        pathdict = self.pathdict
        tfile = item.getText()
        msg = repr(pathdict)
        #JS("alert(msg)")
        
        #check if already in
        
        self.info.setHTML("""
                            <h1>Filename: %(tfile)s</h1>
                          """ % {"tfile":tfile})
        parms = {"caltype":"all",
                 "filename":tfile}
        self.localcalreport.setHTML('<span style="text-color:red">Waiting for Local Calibration Result</span>')
        self.globalcalreport.setHTML('<span style="text-color:red">Waiting for Global Calibration Result</span>')
        HTTPRequest().asyncGet(
                "calsearch.xml?caltype=%(caltype)s&filename=%(filename)s" 
                    % parms, 
                CalibrationLoader(self, local=True))
        HTTPRequest().asyncGet(
                "globalcalsearch.xml?caltype=%(caltype)s&filename=%(filename)s" 
                    % parms, 
                CalibrationLoader(self, local=False))
        return
        
class ADViewerIFACE(DataDictTree):
    pathdict = {}
    reduceFiles = None
    curTabIFACE = None
    tabIFACEs = None
    tabIFACEdict = None
    def createRightPanel(self):
        span = self.stabPanel = SimplePanel(Height="100%")
        
        tabs = self.tabPanel = TabPanel(Width="100%", Border=1,Height="100%")
        
        
        adgui = self.adinfoPanel = AdinfoIFACE(self)
        tabs.add(adgui.panel, "AstroData Viewer")
        
        calsgui = self.calsPanel = CalsIFACE(self)
        tabs.add(calsgui.panel, "Calibrations")
        
        
        rsgui = RecipeSystemIFACE.ReducePanelIFACE(self)
        tabs.add(rsgui.panel, "Execute Reduce")

        rogui = RecipeSystemIFACE.ReduceOutputIFACE(self)
        tabs.add(rogui.panel, "Reduce Output")
        Window.addWindowResizeListener(rogui)
        
        tifs = self.tabIFACEs = [adgui, calsgui, rsgui, rogui]
        self.tabIFACEdict = {"adgui": tifs.index(adgui),
                            "calsgui": tifs.index(calsgui),
                            "rsgui": tifs.index(rsgui),
                            "rogui": tifs.index(rogui)}
        self.curTabIFACE = tifs[0]
        tabs.selectTab(0)
        
        span.add(tabs)
        
        
        tabs.addTabListener(self, getattr(self, "onTabSelected"))
        
        return span
    def fromXML(self, text):
        DataDictTree.fromXML(self, text)
        for tabIFACE in self.tabIFACEs:
            tabIFACE.pathdict = self.pathdict
        
    def getHeight(self):
        
        return rccutil.getHeight()
    def onRunReduce(self):
        recipe = self.recipeList.getItemText(self.recipeList.getSelectedIndex())
        
        if recipe=="None":
            rstr = ""
        else:
            rstr = "p=-r"+recipe

        rfiles = []            
        for i in range(0, self.reduceFiles.getItemCount()):
            fname = self.reduceFiles.getItemText(i)
            rfiles.append(quote(self.pathdict[fname]["path"]))
        filesstr = "&p=".join(rfiles)
                
        cl = "/runreduce?%s&p=%s" % (rstr, filesstr)
        JS("window.open(cl)")

    def onTabSelected(self, sender, tabIndex):
        self.curTabIFACE = self.tabIFACEs[tabIndex]
    def onBeforeTabSelected(self, sender, tabIndex):
        return True
        
    def onRecipeSelected(self, event):
        self.updateReduceCL()
    
    def onClearReduceFiles(self, event):
        self.reduceFiles.clear() 
        self.adInfo.setHTML("file info...") 
        self.updateReduceCL()
        

        
        # self.prepareReduce.setHTML('<a href="runreduce?p=-r&p=callen&p=%(fname)s">reduce -r callen %(fname)s</a>' %
        #                            {"fname":item.getText()})
        pass
    def onTreeItemSelected(self, item):
        pathdict = self.pathdict
        tfile = item.getText()
        msg = repr(pathdict)
        # JS("alert(msg)")
        if tfile in pathdict and "filetype" in pathdict[tfile]:
            ftype = pathdict[tfile]["filetype"]
        else:
            ftype = "unknown"
        if ftype != "fileEntry":
            
            state = item.getState()
            if state:
                item.setState(False)
            else:
                item.setState(True)
            return
        if hasattr(self.curTabIFACE, "onTreeItemSelected"):
            self.curTabIFACE.onTreeItemSelected(item)

    def onTreeItemStateChanged(self, item):
        child = item.getChild(0)
        if hasattr(child, "isPendingItem"):
            item.removeItem(child)
        
            proto = item.getUserObject()
            for i in range(len(proto.children)):
                self.createItem(proto.children[i])
                index = self.getSortIndex(item, proto.children[i].text)
                # demonstrate insertItem.  addItem is easy.
                item.insertItem(proto.children[i].item, index)
                item.setState(True)

 
def OLDcreate_xml_doc(text):
    JS("""
    var xmlDoc;
    try { //Internet Explorer
        xmlDoc=new ActiveXObject("Microsoft.XMLDOM");
        xmlDoc.async="false";
        xmlDoc.loadXML(text);
    } catch(e) {
        try { //Firefox, Mozilla, Opera, etc.
            parser=new DOMParser();
            xmlDoc=parser.parseFromString(text,"text/xml");
        } catch(e) {
            return null;
        }
    }
    return xmlDoc;
  """)

import os
class Proto:
    def __init__(self, text, children=None):
        self.children = []
        self.item = None
        if text[-1] == "/":
            text = text[:-1]
        # self.text = os.path.basename(text)
        self.text = text.split("/")[-1]
        if children is not None:
            self.children = children


class PendingItem(TreeItem):
    def __init__(self):
        TreeItem.__init__(self, "Please wait...")

    def isPendingItem(self):
        return True


def init():
    text="""The Recipe System Engineering Interface allows execution of the recipe
    system."""
    return SinkInfo("AstroData Viewer", "Allows Actions the loaded DataStore", 
                    ADViewerIFACE)
