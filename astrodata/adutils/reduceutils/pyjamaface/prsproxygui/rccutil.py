from __pyjamas__ import JS
from pyjamas import Window
def getHeight():
    useheight = int(Window.getClientHeight()*.75)
    return useheight
    
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
