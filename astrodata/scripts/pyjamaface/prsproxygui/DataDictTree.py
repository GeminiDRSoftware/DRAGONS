from Sink import Sink, SinkInfo
from pyjamas import Window
from pyjamas.ui.Tree import Tree
from pyjamas.ui.TreeItem import TreeItem
from pyjamas.ui.ScrollPanel import ScrollPanel
from pyjamas.HTTPRequest import HTTPRequest
from pyjamas.ui.HTML import HTML
from pyjamas.ui.SimplePanel import SimplePanel
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

from rccutil import create_xml_doc

class PanelIFACE(Sink):
    panel = None
    pathdict = None # set by owner object

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
        doc = create_xml_doc(txt)
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
        sp = self.panel.parent.tabIFACEs[self.panel.parent.tabIFACEdict["adgui"]]
        sp.scroll(0)
        
class AdinfoIFACE(PanelIFACE):
    def __init__(self, parent = None):
        PanelIFACE.__init__(self, parent)
        
        self.panel = ScrollPanel()
        adinfo = HTML("", Size=("100%", parent.getHeight()))
        self.panel.setSize("100%", parent.getHeight())
        
        self.panel.add(adinfo)
        self.adInfo = parent.adInfo = adinfo
        Window.addWindowResizeListener(self)
        return
        
    def onTreeItemSelected(self, item):
        pathdict = self.pathdict
        
        filename = item.getText()
        #check if already in
        if filename in pathdict:
            if pathdict[filename]["filetype"] == "fileEntry":
                url = "adinfo?filename=%s" % self.pathdict[item.getText()]["path"]
                HTTPRequest().asyncGet(url, 
                               ADInfoLoader(self),
                              )
            else:
                self.adInfo.setHTML("""
                    <b style="font-size:200%%">%s</b>""" % pathdict[filename]["filetype"])
        
    def scroll(self, where):
        self.panel.setScrollPosition(where)
        
    def onWindowResized(self, width, height):
        self.panel.setSize("100%", self.parent.getHeight())
        self.adInfo.setSize("100%", self.parent.getHeight())
                
class DataDictTree(Sink):
    pathdict = {}
    reduceFiles = None
    fTree = None
    
    def __init__(self, parent = None):
        Sink.__init__(self, parent)
        self.reduceFiles = []
        if True:
            HTTPRequest().asyncGet("datadir.xml", 
                                    DirDictLoader(self),
                                )
        dock = DockPanel(HorizontalAlignment=HasAlignment.ALIGN_LEFT, 
                            Spacing=10,
                             Size=("100%","100%"))
        self.dock = dock
        self.fProto = []

        self.fTree = Tree()
        self.treePanel = ScrollPanel()
        self.treePanel.setSize("100%", 
                                str(
                                 int(
                                  Window.getClientHeight()*.75
                                 )
                                )+"px")
        Window.addWindowResizeListener(self)
        
        self.treePanel.add(self.fTree)
        dock.add(self.treePanel, DockPanel.WEST)
        
        
        #self.treePanel.setBorderWidth(1)
        #self.treePanel.setWidth("100%")
        
        prPanel = self.createRightPanel()
        dock.add(prPanel,DockPanel.EAST)
        
        dock.setCellWidth(self.treePanel, "50%")
        dock.setCellWidth(prPanel, "50%")
        for i in range(len(self.fProto)):
            self.createItem(self.fProto[i])
            self.fTree.addItem(self.fProto[i].item)

        self.fTree.addTreeListener(self)
        self.initWidget(self.dock)
        
        if False: #self.parent.filexml != None:
            DirDictLoader(self).onCompletion(self.parent.filexml)

    def onWindowResized(self, width, height):
        self.treePanel.setSize("100%", 
                                str(
                                 int(
                                  height *.75
                                 )
                                )+"px")
    def onRecipeSelected(self, event):
        self.updateReduceCL()
    
    def onClearReduceFiles(self, event):
        self.reduceFiles.clear() 
        self.adInfo.setHTML("file info...") 
        self.updateReduceCL()
        
    def updateReduceCL(self):
        recipe = self.recipeList.getItemText(self.recipeList.getSelectedIndex())
        
        if recipe=="None":
            rstr = ""
        else:
            rstr = "-r "+recipe

        rfiles = []            
        for i in range(0, self.reduceFiles.getItemCount()):
            fname = self.reduceFiles.getItemText(i)
            rfiles.append(fname)
        filesstr = " ".join(rfiles)
        
                
        self.prepareReduce.setHTML('<b>reduce</b> %(recipe)s %(files)s' % 
                                        { "recipe":rstr, 
                                          "files":filesstr})

    def onTreeItemSelected(self, item):
        pathdict = self.pathdict
        
        tfile = item.getText()
        #check if already in
        if tfile in pathdict:
            ftype = pathdict[tfile]["filetype"]
            if ftype != "fileEntry":
                item.setState(True)
                return
        else:
            return
        for i in range(0, self.reduceFiles.getItemCount()):
            fname = self.reduceFiles.getItemText(i)
            if fname == tfile:
                return
        self.reduceFiles.addItem(tfile)
        self.updateReduceCL()
        
        filename = tfile
        if filename in pathdict:
            if pathdict[filename]["filetype"] == "fileEntry":
                HTTPRequest().asyncGet("adinfo?filename=%s" % self.pathdict[item.getText()]["path"], 
                               ADInfoLoader(self),
                              )
            else:
                self.adInfo.setHTML("""
                    <b style="font-size:200%%">%s</b>""" % pathdict[filename]["filetype"])
        else:
            self.adInfo.setHTML("unknown node")
        return
        
        # self.prepareReduce.setHTML('<a href="runreduce?p=-r&p=callen&p=%(fname)s">reduce -r callen %(fname)s</a>' %
        #                            {"fname":item.getText()})
        pass
    
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

    def getSortIndex(self, parent, text):
        nodes = parent.getChildCount()
        node = 0
        text = text.lower()

        while node < nodes:
            item = parent.getChild(node)
            if cmp(text, item.getText().lower()) < 0:
                break;
            else:
                node += 1
        
        return node
    
    def createProto(self, node, parent=None):
            #if node.nodeType != node.ELEMENT_NODE:
            #    return
            pathdict = self.pathdict
            if not node.hasChildNodes():
                if node.nodeType != 1:
                    return None
                nname = node.getAttribute("name")
                newproto = None

                newproto = Proto(str(node.getAttribute("name")))
                if node.tagName == "fileEntry":
                    pathdict.update({node.getAttribute("name"):
                                        { "path":node.getAttribute("fullpath"),
                                          "filetype": node.tagName }})
                elif node.tagName == "dirEntry":
                    pathdict.update({node.getAttribute("name"):
                                        { "path":node.getAttribute("name"),
                                          "filetype": node.tagName}})
                else:
                    pathdict.update({node.getAttribute("name"):
                                        { "path": "NOPATH",
                                          "filetype": node.tagName}})
                self.createItem(newproto)
                return newproto
            else:
                cprotos = []
                for i in range(0, node.childNodes.length):
                    childnode = node.childNodes.item(i)
                    if hasattr(childnode,"getAttribute") and childnode.getAttribute("name") == "files":
                        for j in range(0, childnode.childNodes.length):
                            childnodej = childnode.childNodes.item(j)
                            ncproto = self.createProto(childnodej)
                            if ncproto != None:
                                ncitem  = self.createItem(ncproto)
                                cprotos.append(ncproto)
                    else:
                        ncproto = self.createProto(childnode)
    
                    if ncproto != None:
                        ncitem  = self.createItem(ncproto)
                        cprotos.append(ncproto)
                  
                        
                        
                if len(cprotos)>0:
                    newproto = Proto(str(node.getAttribute("name")),cprotos)
                else:
                    newproto = Proto(str(node.getAttribute("name")))
                if node.tagName == "fileEntry":
                    pathdict.update({node.getAttribute("name"):
                                        { "path":node.getAttribute("fullpath"),
                                          "filetype": node.tagName }})
                elif node.tagName == "dirEntry":
                    pathdict.update({node.getAttribute("name"):
                                        { "path":node.getAttribute("name"),
                                          "filetype": node.tagName}})
                else:
                    pathdict.update({node.getAttribute("name"):
                                        { "path": "NOPATH",
                                          "filetype": node.tagName}})

                self.createItem(newproto)
            return newproto
    
    def fromXML(self,text):
        doc = create_xml_doc(text)
        #node = doc.firstChild
        
        #node = doc.getElementById("topDirectory")
        nodes = doc.getElementsByTagName("dirEntry")
        node = nodes.item(0)
        s = repr(node)
        
        newproto = self.createProto(node)    
        plist = [newproto]
        #plist = [Proto(node.tagName)]

        for i in range(len(plist)):
            num = str(len(plist))
            # self.createItem(plist[i])
            self.fTree.addItem(plist[i].item)
            plist[i].item.setState(True)

        
    def onShow(self):
        if False:
            for item in self.fTree.treeItemIterator():
                key = repr(item.tree)
                if key != "null":
                    item.setState(True)
                else:
                    JS("alert(item.getText())")

        pass

    def createItem(self, proto):
        proto.item = TreeItem(proto.text)
        proto.item.setUserObject(proto)
        if len(proto.children) > 0:
            proto.item.addItem(PendingItem())

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
    return SinkInfo("AstroData Viewer", text, Trees)
