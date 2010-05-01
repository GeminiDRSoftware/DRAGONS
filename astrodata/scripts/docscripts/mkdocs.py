#!/usr/bin/env python

from optparse import OptionParser
import subprocess as sp

parser = OptionParser()
parser.add_option("-a", "--all", dest = "doAll",
            action="store_true",
            default = False,
            help = "Do full build and display PDF when done")

parser.add_option("-f", "--full-build", dest = "fullBuild",
            action="store_true",
            default = False,
            help = "Do all build steps and retrievals from scratch.")
parser.add_option("-t", "--type-graphs", dest = "buildTypeGraphs",
            action="store_true",
            default = False,
            help = "Build type graphs (requires graphviz, pydot)")
parser.add_option("-w", "--grab-wiki", dest = "grabWiki",
            action="store_true",
            default = False,
            help = "grab articles from wiki")
parser.add_option("-d", "--display-PDF", dest = "displayPDF",
            action="store_true",
            default = False,
            help = "Display PDF after build (requires acroread")
parser.add_option("-b","--sphinx-build", dest = "doSphinx",
            action = "store_true",
            default = False,
            help = "Do the Sphinx Build, defaults on if no other flags set")
(options, args) = parser.parse_args()

if options.doAll:
    options.displayPDF = True
    options.fullBuild = True
    
if options.fullBuild:
    options.buildTypeGraphs = True
    options.grabWiki = True
    options.doSphinx = True
    
if not (options.buildTypeGraphs or options.grabWiki or options.displayPDF):
    options.doSphinx = True

if options.buildTypeGraphs:
    sp.call(["./createTypeTrees.sh"])

if options.grabWiki:
    sp.call(["./grabWiki.py"])
    
if options.doSphinx:
    sp.call(["./sphbuild.sh"])

if options.displayPDF:
    sp.call(["acroread", "build/_latex_build/astrodatadocumentation.pdf"])
    

