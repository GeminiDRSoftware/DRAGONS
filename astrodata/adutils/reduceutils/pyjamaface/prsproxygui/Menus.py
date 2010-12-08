from Sink import Sink, SinkInfo
from pyjamas.ui.MenuBar import MenuBar
from pyjamas.ui.MenuItem import MenuItem
from pyjamas import Window
from pyjamas.ui.HTML import HTML
from pyjamas.ui.VerticalPanel import VerticalPanel
from pyjamas import DOM
from pyjamas.ui.ContextMenuPopupPanel import ContextMenuPopupPanel

class Menus(Sink):
    def __init__(self):
        Sink.__init__(self)
        self.menu = MenuBar()
        
        subMenu = MenuBar(True)
        subMenu.addItem("<code>Code</code>", True, self)
        subMenu.addItem("<strike>Strikethrough</strike>", True, self)
        subMenu.addItem("<u>Underlined</u>", True, self)
        
        menu0 = MenuBar(True)
        menu0.addItem("<b>Bold</b>", True, self)
        menu0.addItem("<i>Italicized</i>", True, self)
        menu0.addItem("More &#187;", True, subMenu)
        menu1 = MenuBar(True)
        menu1.addItem("<font color='#FF0000'><b>Apple</b></font>", True, self)
        menu1.addItem("<font color='#FFFF00'><b>Banana</b></font>", True, self)
        menu1.addItem("<font color='#FFFFFF'><b>Coconut</b></font>", True, self)
        menu1.addItem("<font color='#8B4513'><b>Donut</b></font>", True, self)
        menu2 = MenuBar(True)
        menu2.addItem("Bling", self)
        menu2.addItem("Ginormous", self)
        menu2.addItem("<code>w00t!</code>", True, self)
        
        self.menu.addItem(MenuItem("Style", menu0))
        self.menu.addItem(MenuItem("Fruit", menu1))
        self.menu.addItem(MenuItem("Term", menu2))
        
        self.menu.setWidth("100%")
        
        self.panel = VerticalPanel()
        self.context = HTML("""Right-Click me<br/>to show a context menu.<br />
                               Left-click me<br />to do absolutely nothing.""")

        self.panel.add(self.menu)
        self.panel.add(self.context)

        self.initWidget(self.panel)

        self.context.setContextMenu(self)

    def onContextMenu(self, sender):

        event = DOM.eventGetCurrentEvent()

        subMenu = MenuBar(True)
        subMenu.addItem("<code>Code</code>", True, self)
        subMenu.addItem("<strike>Strikethrough</strike>", True, self)
        subMenu.addItem("<u>Underlined</u>", True, self)

        x = DOM.eventGetClientX(event) + 2
        y = DOM.eventGetClientY(event) + 2
        
        popup = ContextMenuPopupPanel(subMenu)
        popup.showAt(x, y)


    def execute(self):
        Window.alert("Thank you for selecting a menu item.")
        
    def onShow(self):
        pass


def init():
    return SinkInfo("Menus", "The GWT <code>MenuBar</code> class makes it easy to build menus, including cascading sub-menus.", Menus)
