from Sink import Sink, SinkInfo
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

class Trees(Sink):
    pathdict = {}
    reduceFiles = None
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
        self.prPanel = VerticalPanel(Size=("50%", ""))
        self.treePanel = HorizontalPanel(Size=("50%", "100%"))
        self.treePanel.add(self.fTree)
        dock.add(self.treePanel, DockPanel.WEST)
        
        self.treePanel.setBorderWidth(1)
        self.treePanel.setWidth("100%")
        self.prPanel.setBorderWidth(1)
        self.prPanel.setWidth("100%")
        # prepare panel
        self.prepareReduce = HTML("<tt> .. none yet .. </tt>", True, )
        
        self.recipeList = ListBox()
        self.recipeList.addChangeListener(getattr(self, "onRecipeSelected"))
        self.recipeList.addItem("None")
        HTTPRequest().asyncGet("recipes.xml",
                                RecipeListLoader(self))

        #EO prepare panel
        self.reduceCLPanel = DockPanel(Spacing = 5)
        self.reduceCLPanel.add(HTML("<i>Reduce Command Line</i>:"), DockPanel.NORTH)                        
        self.reduceCLPanel.add(self.prepareReduce, DockPanel.NORTH)

        self.reduceFilesPanel = DockPanel(Spacing = 5)
        self.reduceFilesPanel.add(HTML("<b>Datasets</b>:"), DockPanel.WEST)
        
        self.reduceFiles = ListBox()
        self.reduceFiles.setVisibleItemCount(5)
        self.reduceFilesPanel.add(self.reduceFiles, DockPanel.WEST)
        self.clearReduceFilesButton = Button("<b>Clear List</b>", listener = getattr(self, "onClearReduceFiles"))
        self.reduceFilesPanel.add(self.clearReduceFilesButton, DockPanel.SOUTH)

        self.recipeListPanel = DockPanel(Spacing = 5)
        self.recipeListPanel.add(HTML("<b>Recipes List</b>:"),DockPanel.WEST)
        self.recipeListPanel.add(self.recipeList, DockPanel.WEST)
        
        self.runReduceButton = Button("<b>RUN REDUCE</b>", listener = getattr(self, "onRunReduce"))
        
        # major sub panels
        self.prPanel.add(self.reduceCLPanel)
        self.prPanel.add(self.reduceFilesPanel)
        self.prPanel.add(self.recipeListPanel)
        self.prPanel.add(self.runReduceButton)
                
       
        
        dock.add(self.prPanel,DockPanel.EAST)
        
        dock.setCellWidth(self.treePanel, "50%")
        dock.setCellWidth(self.prPanel, "50%")
        for i in range(len(self.fProto)):
            self.createItem(self.fProto[i])
            self.fTree.addItem(self.fProto[i].item)

        self.fTree.addTreeListener(self)
        self.initWidget(self.dock)
        
        if False: #self.parent.filexml != None:
            DirDictLoader(self).onCompletion(self.parent.filexml)

    def onRecipeSelected(self, event):
        self.updateReduceCL()
    
    def onClearReduceFiles(self, event):
        self.reduceFiles.clear()  
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

        
    def onTreeItemSelected(self, item):
        pathdict = self.pathdict
        
        tfile = item.getText()
        #check if already in
        if tfile in pathdict:
            ftype = pathdict[tfile]["filetype"]
            if ftype != "fileEntry":
                return
        else:
            return
        for i in range(0, self.reduceFiles.getItemCount()):
            fname = self.reduceFiles.getItemText(i)
            if fname == tfile:
                return
        self.reduceFiles.addItem(tfile)
        self.updateReduceCL()
        
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
    
    def createProto(self, node):
            #if node.nodeType != node.ELEMENT_NODE:
            #    return
            pathdict = self.pathdict
            if not node.hasChildNodes():
                if node.nodeType != 1:
                    return None
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

def create_xml_doc(text):
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


class Proto:
    def __init__(self, text, children=None):
        self.children = []
        self.item = None
        self.text = text
        
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
    return SinkInfo("Recipe System", text, Trees)
