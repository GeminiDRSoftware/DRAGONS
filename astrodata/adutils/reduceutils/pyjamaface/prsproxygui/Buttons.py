from Sink import Sink, SinkInfo
from pyjamas.ui.Button import Button
from pyjamas.ui.CheckBox import CheckBox
from pyjamas.ui.VerticalPanel import VerticalPanel
from pyjamas.ui.HorizontalPanel import HorizontalPanel
from pyjamas.ui.RadioButton import RadioButton

class Buttons(Sink):
    def __init__(self):

        Sink.__init__(self)

        disabledButton = Button("Disabled Button")
        disabledCheck = CheckBox("Disabled Check")
        normalButton = Button("Normal Button")
        normalCheck = CheckBox("Normal Check")
        panel = VerticalPanel()
        radio0 = RadioButton("group0", "Choice 0")
        radio1 = RadioButton("group0", "Choice 1")
        radio2 = RadioButton("group0", "Choice 2 (Disabled)")
        radio3 = RadioButton("group0", "Choice 3")

        hp=HorizontalPanel()
        panel.add(hp)
        hp.setSpacing(8)
        hp.add(normalButton)
        hp.add(disabledButton)
        
        hp=HorizontalPanel()
        panel.add(hp)
        hp.setSpacing(8)
        hp.add(normalCheck)
        hp.add(disabledCheck)
        
        hp=HorizontalPanel()
        panel.add(hp)
        hp.setSpacing(8)
        hp.add(radio0)
        hp.add(radio1)
        hp.add(radio2)
        hp.add(radio3)
        
        disabledButton.setEnabled(False)
        disabledCheck.setEnabled(False)
        radio2.setEnabled(False)
        
        panel.setSpacing(8)
        self.initWidget(panel)

    def onShow(self):
        pass


def init():
    text="GWT supports all the myriad types of buttons that exist in HTML.  Here are a few for your viewing pleasure."
    return SinkInfo("Buttons", text, Buttons)
