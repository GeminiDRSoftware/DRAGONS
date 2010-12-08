from Sink import Sink, SinkInfo
from pyjamas.ui.HTML import HTML
from pyjamas.ui.HorizontalPanel import HorizontalPanel
from pyjamas.ui.ListBox import ListBox
from __pyjamas__ import JS
from pyjamas.HTTPRequest import HTTPRequest

from rccutil import create_xml_doc

class RecipeListLoader:
    def __init__(self, panel):
        self.panel = panel
        
    def onCompletion(self, txt):
        doc = create_xml_doc(txt)
        #node = doc.firstChild
        
        #node = doc.getElementById("topDirectory")
        nodes = doc.getElementsByTagName("recipe")
        rdict = {}
        if len(nodes)>0:
            rlist = []
            for i in range(0, len(nodes)):
                node = nodes.item(i)
                rlist.append(node.getAttribute("name"))
                rdict.update({node.getAttribute("name"):node.getAttribute("path")})
                
            rlist.sort()
            for r in rlist:
                self.panel.RList.addItem(r)
            c = self.panel.RList.getItemCount()
            self.panel.recipeDict = rdict
            self.panel.RList.setVisibleItemCount(c)

class RecipeViewLoader:
    def __init__(self, panel):
        self.panel = panel
        
    def onCompletion(self, txt):
        content = '<pre style="font-size:150%%">%s</pre>' % txt
        self.panel.RView.setHTML(content)
        
class RecipeViewer(Sink):
    recipeDict = None
    def __init__(self, parent=None):
        Sink.__init__(self, parent)
        self.RVDock = HorizontalPanel(Spacing=5)
        self.RList = ListBox() 
        self.RList.addClickListener(getattr(self, "onRecipeSelected"))
        
        self.RView = HTML()
        HTTPRequest().asyncGet("recipes.xml", RecipeListLoader(self))
        self.RVDock.add(self.RList)
        self.RVDock.add(self.RView)
        self.initWidget(self.RVDock)
        
    def onRecipeSelected(self, item):
        recipe = self.RList.getItemText(self.RList.getSelectedIndex())
        HTTPRequest().asyncGet("/recipecontent?recipe=%s" % recipe, RecipeViewLoader(self))
        
        
    def onShow(self):
        pass


def init():
    return SinkInfo("Recipe Viewer", "Recipe Viewer", RecipeViewer)
