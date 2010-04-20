#!/usr/bin/env python

from astrodata.AstroDataType import *
from optparse import OptionParser
from astrodata import Descriptors as ds
import os
try:
    import pydot
    from pydot import *
except:
    print "couldn't import pydot"
    pass
    
cl = getClassificationLibrary()
# print repr(cl.typesDict)
#FROM COMMANDLINE WHEN READY
parser = OptionParser()
(options, args) = parser.parse_args()

if len(args)>0:
    astrotype = args[0]
else:
    astrotype = None
    
import astrodata
import astrodata.RecipeManager as rm

def createEdges(typ, parents=False, children=False):
    ces = {}
    if parents and typ.parentDCO :
        pen = typ.parentDCO.name+"_"+typ.name
        pe = Edge(typ.parentDCO.name, typ.name)
        ces.update({pen:pe})
    if children and typ.childDCOs:
        for child in typ.childDCOs:
            ce = Edge(typ.name, child.name)
            cen = typ.name+"_"+child.name
            ces.update({cen:ce})
    return ces
    
psdict = rm.centralPrimitivesIndex
descdict = ds.centralCalculatorIndex
ndict = {}
edict = {}
adict = {} # contains node list keyed by type
ddict = {} # contains node list keyed by type
gmos = cl.getTypeObj(astrotype)

lasttyp = None
lastnode = None
#nodes and annotations
for typ in gmos.walk():
    node = Node(typ.name, shape="house")
    ndict.update({typ.name: node})
    
    if typ.name in psdict:
        anodes = []
        labels = []
        i = 0
        for cfg in psdict[typ.name]:
            labels.append("<f%d>" % i + cfg[0])
            
            i += 1
            #anode = Node(cfg[0], shape="box")
            #anodes.append(anode)
        label = "{"+"|".join(labels)+"}"
        tnode = Node(typ.name+"_PrimSet", 
                        
                        shape="record", 
                        label = label)
        anodes.append(tnode)
        adict.update({typ:anodes})
        
    if typ.name in descdict:
        anodes = []
        cfg = descdict[typ.name]
        nam = typ.name+"Descriptor"
        anode = Node(nam, shape="box", 
                    label= nam)
        anodes.append(anode)
        ddict.update({typ:anodes})

# edges
for typ in gmos.walk(style="children"):
    es = createEdges(typ, children=True)
    edict.update(es)
for typ in gmos.walk(style="parent"):
    es = createEdges(typ, parents = True)
    print "tl59:",repr(es)
    edict.update(es)

# create graph
graph = Dot(graph_type="digraph")
for n in ndict:
    graph.add_node(ndict[n])
for e in edict:
    graph.add_edge(edict[e])
# do primitive set dict
for typ in adict:
    subg = Cluster(typ.name,
                     
                    bgcolor = "#40a0ff",
                    label=typ.name+" PrimSet")
    
    for an in adict[typ]:
        subg.add_node(an)
    graph.add_subgraph(subg)
    print "tl98:", repr(adict)
    anedge = Edge(adict[typ][0], ndict[typ.name], lhead=subg.get_name())
    graph.add_edge(anedge)
# do descriptor set dict
for typ in ddict:
    subg = Cluster(typ.name,bgcolor = "#a02070",label="",
                fontsize="3")
    
    for an in ddict[typ]:
        subg.add_node(an)
    graph.add_subgraph(subg)
    anedge = Edge(ddict[typ][0], ndict[typ.name], lhead=subg.get_name())
    graph.add_edge(anedge)
        
graph.write_svg("output.svg")
graph.write_png("output.png")

graph.write_dot("output.dot")    
# a = cl.gvizDoc(astrotype= astrotype, writeout = True, assDict = assdict)
import webbrowser
# url = "file://"+os.path.join(os.path.abspath("."), "gemdtype.viz.svg")
url = "file://"+os.path.join(os.path.abspath("."), "output.svg")
webbrowser.open(url);
