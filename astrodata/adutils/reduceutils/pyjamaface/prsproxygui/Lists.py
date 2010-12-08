from Sink import Sink, SinkInfo
from pyjamas.ui.ListBox import ListBox
from pyjamas.ui.HorizontalPanel import HorizontalPanel
from pyjamas.ui.VerticalPanel import VerticalPanel
from pyjamas.ui import HasAlignment
from pyjamas.ui.Label import Label
from pyjamas.ui.Widget import Widget

class Lists(Sink):
    def __init__(self):
        Sink.__init__(self)

        self.sStrings=[["foo0", "bar0", "baz0", "toto0", "tintin0"],
            ["foo1", "bar1", "baz1", "toto1", "tintin1"],
            ["foo2", "bar2", "baz2", "toto2", "tintin2"],
            ["foo3", "bar3", "baz3", "toto3", "tintin3"],
            ["foo4", "bar4", "baz4", "toto4", "tintin4"]]

        self.combo=ListBox(VisibleItemCount=1)
        self.list=ListBox(MultipleSelect=True, VisibleItemCount=10)
        self.echo=Label()

        self.combo.addChangeListener(self)
        
        for i in range(len(self.sStrings)):
            txt = "List %d" % i
            self.combo.addItem(txt)
            # test setItemText
            self.combo.setItemText(i, txt + " using set text")
        self.combo.setSelectedIndex(0)
        self.fillList(0)
        
        self.list.addChangeListener(self)
        
        horz = HorizontalPanel(VerticalAlignment=HasAlignment.ALIGN_TOP,
                               Spacing=8)
        horz.add(self.combo)
        horz.add(self.list)
        
        panel = VerticalPanel(HorizontalAlignment=HasAlignment.ALIGN_LEFT)
        panel.add(horz)
        panel.add(self.echo)
        self.initWidget(panel)
        
        self.echoSelection()

    def onChange(self, sender):
        if sender == self.combo:
            self.fillList(self.combo.getSelectedIndex())
        elif sender == self.list:
            self.echoSelection()

    def onShow(self):
        pass
    
    def fillList(self, idx):
        self.list.clear()
        strings = self.sStrings[idx]
        for i in range(len(strings)):
            self.list.addItem(strings[i])

        self.echoSelection()

    def echoSelection(self):
        msg = "Selected items: "
        for i in range(self.list.getItemCount()):
            if self.list.isItemSelected(i):
                msg += self.list.getItemText(i) + " "
        self.echo.setText(msg)


def init():
    text="Here is the ListBox widget in its two major forms."
    return SinkInfo("Lists", text, Lists)

