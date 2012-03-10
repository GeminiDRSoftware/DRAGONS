#!/usr/bin/env python

import xml
from httplib import *
from xml.dom.minidom import parse, parseString
from html2rest import html2rest
import re
from copy import copy,deepcopy
import os

WIKIHOST = "gdpsg.wikis-internal.gemini.edu"
#WIKIHOST = 'nihal.hi.gemini.edu'
WIKIROOT = "/index.php/"

def grab(name, subdir = "" ):
    print "grabWiki: getting",name,"...",
    h1 = HTTPConnection(WIKIHOST)
    fullname = WIKIROOT + name
    print "gW19: getting ... ",fullname
    h1.request("GET", fullname)

    r1 = h1.getresponse()
    d1 = r1.read()
    # kludgy way to remove edit links
    d1= re.sub(r"\[.*?action=edit.*?\]", "", d1)
    #print repr(d1)
    try:
        dom1 = parseString(d1)
    except:
        i = 1
        for line in d1.split("\n"):
            print i,":",line
            i +=1
        raise

    divs = dom1.getElementsByTagName("div")
    # print "gW29:", dom1.toxml()
    div = None
    for divcand in divs:
        # print "gW32:", divcand.getAttribute("class")
        if divcand.getAttribute("class") == "WIKIDOC":
            div = divcand
            break
            
    try:
        tocdivs= div.getElementsByTagName("table")
        for divcand in tocdivs:
            if divcand.getAttribute("class") == "toc":

                div.removeChild(divcand)
                break
    except AttributeError:
        pass # page has no TOC
                    
    if div == None:
        print ('ERROR: Article %s has no div with class="WIKIDOC"' % name +
               "\nUse {{START-WIKIDOC}} and {{END-WIKIDOC}} templates for text "
               "to be included in document.")
        return 
    def killLink(node):
        child = node.firstChild
        while child is not None:
            if child:
                    killLink(child)
            nextchild = child.nextSibling
            if child.nodeType == xml.dom.minidom.Node.ELEMENT_NODE:
                if child.tagName == "a":
                    node.removeChild(child)
            
            child = nextchild
            
    # print "before",html   
    # not needed now killLink(div)
    html = div.toxml()
    outfname = os.path.join("source",subdir, "gen."+name+".rst")
    outf = file(outfname, "w+")
    html2rest(html, outf)
    outf.close()
    print "to", outfname

if False:    
    # for the introduction
    grab("ADMANUAL_RevisionHistory")
    grab("ADMANUAL_Purpose")
    grab("ADMANUAL_ADConcepts")
    grab("AstroDataPackageOverview")
    grab("AstroDataLexicon")

    grab("GATREF-RevisionHistory", subdir = "gatref")
    grab("GATREF-Purpose",subdir = "gatref")
    grab("GATREF-Audience",subdir = "gatref")

    # for the AstroData Class Reference
    grab("ADMANUAL_SingleHDUAD")
    grab("ADMANUAL-ADSubdata")
    grab("ADMANUAL-AccessingPyfitsObjects")

print "="*80
print "-"*80
print "="*80
print "GRABBING FROM WIKI TURNED OFF, MODIFY rst SOURCE, will not regenerate"
print "    from WIKI"
