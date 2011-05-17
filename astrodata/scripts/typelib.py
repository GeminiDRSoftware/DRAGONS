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
    
cl = get_classification_library()
# print repr(cl.typesDict)
#FROM COMMANDLINE WHEN READY
parser = OptionParser()
parser.add_option("-a", "--assignments", dest = "showAssignments", 
    action="store_true",
    default = False, help="Show Primitive and Descriptor Assignments")
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
if (astrotype):
    typeobjs = [astrotype]
    displaytype = astrotype
else:
    displaytype = "GEMINI"
    typeobjs = cl.get_available_types();

lasttyp = None
lastnode = None
#nodes and annotations
for typename in args:
    typeobj = cl.get_type_obj(typename)
    ndict = {}
    edict = {}
    adict = {} # contains node list keyed by type
    ddict = {} # contains node list keyed by type
    postfix = ""

    if options.showAssignments:
        print "Creating type tree graph with assignments for ...", typename
    else:
        print "Creating type tree graph for ...", typename

    for typ in typeobj.walk():
        if options.showAssignments:
            postfix = "-pd"
        # just always link to the assignment charts    
        root4url = typ.name.split("_")[0]

        tip = re.sub(r".*?ADCONFIG_", "ADCONFIG_", typ.fullpath)
        node = Node(typ.name, shape="house",
                    URL = root4url + "-tree%s.svg"%"-pd", # hard coded to link to assignment graphs
                    tooltip = tip
                    )
        ndict.update({typ.name: node})

        if options.showAssignments:
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
                            fillcolor = "#f0a0a0",
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
        edict.update(es)

    # create graph
    graph = Dot(typ.name+"_Type_Graph", graph_type="digraph", )
    
    
    
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
        anedge = Edge(adict[typ][0], ndict[typ.name], 
                        lhead=subg.get_name(),
                        style="dotted")
        graph.add_edge(anedge)
    # do descriptor set dict
    for typ in ddict:
        # subg = Cluster(typ.name,bgcolor = "#a02070",label="")
        for an in ddict[typ]:
            graph.add_node(an)
        #    subg.add_node(an)
        #graph.add_subgraph(subg)

        anedge = Edge(ddict[typ][0], ndict[typ.name],
             style="dashed",)
        graph.add_edge(anedge)

    outbase = typename+"-tree"+postfix
    graph.write_svg(outbase+".broke.svg")
    graph.write_png(outbase+".png")
    graph.write_dot(outbase+".dot")   

    # fix svg
    svg = file(outbase+".broke.svg")
    nsvg = file(outbase+".svg", mode="w")
    for line in svg.readlines():
        nline = re.sub(r'font-size:(.*?);', r'font-size:\1px;', line)
        # print "tl137:", line, "--->", line
        nsvg.write(nline)

    svg.close()
    nsvg.close()


if False:# a = cl.gviz_doc(astrotype= astrotype, writeout = True, assign_dict = assdict)
    import webbrowser
    # url = "file://"+os.path.join(os.path.abspath("."), "gemdtype.viz.svg")
    url = "file://"+os.path.join(os.path.abspath("."), displaytype +"-tree.svg")
    webbrowser.open(url);
