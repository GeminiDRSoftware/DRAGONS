#!/usr/bin/env python

import os
import shutil
import re

# this script generates sphinx source... it's output is a source dir

outdir="source/gen.typedefs"
restoutdir = "gen.typedefs"
if os.path.exists(outdir):
    shutil.rmtree(outdir)  
    
os.mkdir(outdir)
    
from astrodata import AstroDataType as ADT

cl = ADT.getClassificationLibrary()

typesdict = cl.typesDict
# source files with types covered
sourcedict = {}
sourceindex = [] # for ordering

typkeys = typesdict.keys()
print "gtd26", repr(typkeys)
typkeys.sort()

# prepare the info loop (handles fact that types can be in same file)
for typ in typkeys:
    typo = typesdict[typ]
    if typo.fullpath in sourcedict:
        sourcedict[typo.fullpath].append(typ)
    else:
        sourcedict.update({typo.fullpath:[typ]})
        sourceindex.append(typo.fullpath)
print "gtd36:", repr(sourcedict), repr(sourceindex)

outchapname = os.path.join(outdir, "gen.TypeSourceAppendix.rst")
print "name of appendix rst:",outchapname

outchap = open(outchapname, "w")
outchap.write("Gemini Type Source Reference\n")
outchap.write("----------------------------\n")
outchap.write("""
.. toctree::
    :numbered:
    :maxdepth: 5

""")
for fil in sourceindex:
    print "getTypeDefs.py processing: ",fil, repr(sourcedict[fil])
    # outchap = the reSt file for the type reference chapter
    # outfile is the reST file for the individual type(s) from a sourcefile
    outname = "gen."+os.path.basename(fil)[:-3]+".rst"
    outfile = os.path.join(outdir, outname)
    restoutfile = outname
    tsf = open(fil)
    typsrclines = tsf.readlines()
    tsf.close()
    
    typsrc = ""
    for line in typsrclines:
        line = "    "+ line
        typsrc += line
    
    outf = open(outfile,"w")

    
    types = ", ".join(sourcedict[fil])
    typesunderline = "~"*(len(types)+3+len(" Type Source "))
    relsrcpath = re.sub(".*?ADCONFIG_", "ADCONFIG_", fil)        
    outf.write("""
%(types)s Type Source
%(typesunderline)s

.. toctree::
    :numbered:
     
:Types:
    %(types)s

:Source: 
    %(relsrcpath)s

.. code-block:: python

%(src)s
""" % { "types": types,
        "typesunderline": typesunderline,
        "relsrcpath":relsrcpath,
        "src":typsrc
        })
    outf.write("\n\n")
    outf.close()
    outchap.write("    "+restoutfile+"\n")
    
outchap.write("""
The following documentation lists current Gemini types and shows their source
files in raw python form. When types are defined in the same source, their 
section is combined.  Entries are alphebatized.

""")

outchap.close()
# output loop
