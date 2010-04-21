#!/usr/bin/env python

from astrodata.AstroDataType import *
from optparse import OptionParser
from astrodata import Descriptors as ds
import os
import re
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
if (astrotype):
    typeobjs = [astrotype]
    displaytype = astrotype
else:
    displaytype = "GEMINI"
    typeobjs = cl.getAvailableTypes();

lasttyp = None
lastnode = None
#nodes and annotations
for typename in typeobjs:
    typeobj = cl.getTypeObj(typename)

    for typ in typeobj.walk():
        node = Node(typ.name, shape="house",
                    URL = typ.name+"-tree.svg")
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
                            label = label,
                            fontsize = "10")
            anodes.append(tnode)
            adict.update({typ:anodes})

        if typ.name in descdict:
            anodes = []
            cfg = descdict[typ.name]
            nam = typ.name+"_Descriptor"
            anode = Node(nam, shape="box", 
                        label= nam,
                        fontsize="10",
                        fillcolor = "#ffd0d0",
                        style="filled"
                        )
            anodes.append(anode)
            ddict.update({typ:anodes})

    # edges
    for typ in typeobj.walk(style="children"):
        es = createEdges(typ, children=True)
        edict.update(es)
    for typ in typeobj.walk(style="parent"):
        es = createEdges(typ, parents = True)
        print "tl59:",repr(es)
        edict.update(es)

    # create graph
    graph = Dot(graph_type="digraph", )
    for n in ndict:
        graph.add_node(ndict[n])
    for e in edict:
        graph.add_edge(edict[e])
    # do primitive set dict
    for typ in adict:
        subg = Cluster(typ.name,

                        bgcolor = "#e0e0ff",
                        label=typ.name+" PrimSet",
                        fontsize="9")

        for an in adict[typ]:
            subg.add_node(an)
        graph.add_subgraph(subg)
        print "tl98:", repr(adict)
        anedge = Edge(adict[typ][0], ndict[typ.name], 
                        lhead=subg.get_name(),
                        style="dotted")
        graph.add_edge(anedge)
    # do descriptor set dict
    for typ in ddict:
        # subg = Cluster(typ.name,bgcolor = "#a02070",label="")
        print repr(ddict)
        for an in ddict[typ]:
            graph.add_node(an)
        #    subg.add_node(an)
        #graph.add_subgraph(subg)

        anedge = Edge(ddict[typ][0], ndict[typ.name],
             style="dashed",)
        graph.add_edge(anedge)

    outbase = typename+"-tree"
    graph.write_svg(outbase+".broke.svg")
    graph.write_png(outbase+".png")
    graph.write_dot(outbase+".dot")   

    # fix svg
    svg = file(outbase+".broke.svg")
    nsvg = file(outbase+".svg", mode="w")
    for line in svg.readlines():
        nline = re.sub(r'font-size:(.*?);', r'font-size:\1px;', line)
        print "tl137:", line, "--->", line
        nsvg.write(nline)

    svg.close()
    nsvg.close()


# a = cl.gvizDoc(astrotype= astrotype, writeout = True, assDict = assdict)
import webbrowser
# url = "file://"+os.path.join(os.path.abspath("."), "gemdtype.viz.svg")
url = "file://"+os.path.join(os.path.abspath("."), displaytype +"-tree.svg")
webbrowser.open(url);
