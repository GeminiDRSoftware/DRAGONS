#
#                                                                     QAP Gemini
#
#                                                                  parseUtils.py
#                                                               Kenneth Anderson
#                                                                        06-2013
#                                                            kanderso@gemini.edu
# ------------------------------------------------------------------------------

# $Id$
# ------------------------------------------------------------------------------
__version__      = '$Revision$'[11:-3]
__version_date__ = '$Date$'[7:-3]
__author__       = "k.r. anderson, <kanderso@gemini.edu>"
# ------------------------------------------------------------------------------
# This cli parser employs argparse instead of depracated optparse.
#
# kra 04-06-13
#
#
# N.B. argparse chokes when type="string" is used to specify variable type.
# raising a ValueError:
#
# raise ValueError('%r is not callable' % (type_func,))
# ValueError: 'string' is not callable
# 
# Since all values default as strings, especially through the command line,
# specifying type="string" is redundant. These type specifiers are removed.
# ------------------------------------------------------------------------------
import os
import sys
import re
from argparse import ArgumentParser

from astrodata import RecipeManager
from astrodata.adutils import gemLog, strutil
from astrodata.AstroDataType import get_classification_library

# ------------------------------------------------------------------------------
def buildNewParser(version):
    parser = ArgumentParser(description="_"*11 + " Gemini Observatory Recipe System Processor "
                            " (v1.0 2011) " + "_"*10 + "\n" + "_"*19 +\
                            " Written by GDPSG (klabrie@gemini.edu) " + "_"*19,
                            prog="reduce2")
    parser.add_argument("-v", "--version", action='version', version='%(prog)s v'+ version)
    parser.add_argument("-d", "--displayflags", dest='displayflags', action="store_true", 
                        default=False, help="display all parsed option flags.")
    parser.add_argument('files', metavar='fitsfile', nargs = "*",
                        help="fitsfile [fitsfile ...] or @file containing a list of fits files.")
    parser.add_argument("-c", "--paramfile", dest="paramfile", default=None,
                        help="specify parameter file")
    parser.add_argument("-i", "--intelligence", dest='intelligence', default=False,
                        action="store_true", help="Endow recipe system with "
                        "intelligence to perform operations faster and smoother")
    parser.add_argument("-m", "--monitor", dest="bMonitor", action="store_true",
                        default=False,
                        help="Open TkInter window to monitor progress of "
                        "execution (NOTE: One window will open per recipe run)")
    parser.add_argument("-p", "--param", dest="userparam", default=None,
                        help="Set a parameter from the command line. The form '-p' "
                        "paramname=val' sets the parameter in the reduction context "
                        "such that all primitives will 'see' it.  The form: '-p "
                        "ASTROTYPE:primitivename:paramname=val', sets the parameter "
                        "such that it applies only when the current reduction type "
                        "(type of current reference image) is 'ASTROTYPE' and the "
                        "primitive is 'primitivename'. Multiple settings can appear "
                        "separated by commas, but no whitespace in the setting (i.e."
                        "'param=val,param2=val2', not 'param=val, param2=val2')")
    parser.add_argument("--context", dest="running_contexts", default=None,
                        help="provides general 'context name' for primitives"
                        "sensitive to context.")
    parser.add_argument("-r", "--recipe", dest="recipename", default=None,
                        help="specify which recipe to run by name")
    parser.add_argument("-t", "--astrotype", dest="astrotype", default=None,
                        help="Run a recipe based on astrotype (either overrides the"
                        " default type, or begins without initial input, eg. "
                        "recipes that begin with primitives that acquire data)")
    ##@@FIXME: This next option should not be put into the package
    parser.add_argument("-x", "--rtf-mode", dest="rtf", default=False, 
                        action="store_true", help="only used for rtf")
    parser.add_argument("--throw_descriptor_exceptions", dest = "throwDescriptorExceptions", 
                        default=False, action = "store_true", 
                        help="debug mode, throws exceptions when Descriptors fail")
    parser.add_argument("--addprimset", dest="primsetname", default = None,
                        help="add user supplied primitives to reduction")
    parser.add_argument("--calmgr", dest="cal_mgr", default=None,
                        help="calibration manager url overides lookup table")
    parser.add_argument("--force-height", dest="forceHeight", default=None,
                        help="force height of terminal output")
    parser.add_argument("--force-width", dest="forceWidth", default=None,
                        help="force width of terminal output")
    parser.add_argument("--invoked", dest="invoked", default=False, 
                        action="store_true", help="tell user reduce invoked by adcc")
    parser.add_argument("--logmode", dest="logmode", default="standard", 
                        help="Set logging mode (standard, "
                        "console, debug, null)")
    parser.add_argument("--logfile", dest="logfile", default="reduce.log",
                        help="name of log (default = 'reduce.log')") 
    parser.add_argument("--loglevel", dest="loglevel", default="stdinfo", 
                        help="Set the verbose level for console "
                        "logging; (critical, error, warning, status, stdinfo, "
                        "fullinfo, debug)")
    parser.add_argument("--showcolors", dest="show_colors", default=False, 
                        action="store_true", help="Shows available colors based "
                        "on terminal setting (used for debugging color issues)")
    parser.add_argument("--override_cal", dest="user_cals", default=None,
                        help="Add calibration to User Calibration Service of this format:"
                        "'-usercal=CALTYPE_1:CALFILEPATH_1,...,CALTYPE_N:CALFILEPATH_N'")
    parser.add_argument("--writeInt", dest='writeInt', default=False, 
                        action="store_true", help="write intermediate outputs"
                        " (UNDER CONSTRUCTION)")   
    parser.add_argument("--suffix", dest='suffix', default=None,
                        help="Suffix to add to filenames written at end of reduction.")    
    return parser

# --------------------------- Emulation functions -------------------------------
# The functions below encapsulate ArgumentParser access to option strings and 
# matches them to 'dest' attributes and attribute values. There is no public 
# interface as with OptionParser.has_option() and OptionParser.get_option() for 
# testing and getting option flags.

# The functions
#
#     parser_has_option()
#     get_option_flags()
#
# emulate those methods.
#
#     insert_option_value()    -- assigns an option value to matching 'dest' attr 
#     show_parser_options()    -- pretty print options, 'dest' attrs, values.
# -------------------------------------------------------------------------------

def parser_has_option(parser, option):
    return parser.__dict__['_option_string_actions'].has_key(option)

def get_option_flags(parser, option):
    return parser.__dict__['_option_string_actions'][option].__dict__['option_strings']

def insert_option_value(parser, args, option, value):
    exec("args.%s=value" % \
         str(parser.__dict__['_option_string_actions'][option].__dict__['dest']))
    return

def show_parser_options(parser, args):
    all_opts = parser.__dict__['_option_string_actions'].keys()
    handled_flag_set = []
    print "\n\t"+"-"*25+" switches, vars, vals "+"-"*20+"\n"
    print "\t  Literals\t\t\tvar 'dest'\t\tValue"
    print "\t","-"*60
    for opt in all_opts:
        all_option_flags = get_option_flags(parser, opt)
        if opt in handled_flag_set:
            continue
        elif "--help" in all_option_flags:
            continue
        elif "--version" in all_option_flags:
            continue
        else:
            handled_flag_set.extend(all_option_flags)
            dvar = parser.__dict__['_option_string_actions'][opt].__dict__['dest']
            val = args.__dict__[dvar]
            if len(all_option_flags) == 1 and len(all_option_flags[0]) > 24:
                print "\t",all_option_flags,"::",dvar,"\t::",val
            elif len(all_option_flags) == 1 and len(all_option_flags[0]) < 11:
                print "\t",all_option_flags,"\t"*3+"::",dvar,"\t\t::",val
            elif len(all_option_flags) == 2 and len(all_option_flags[1]) > 12:
                print "\t",all_option_flags,"\t"+"::",dvar,"\t::",val
            elif len(all_option_flags) == 2:
                print "\t",all_option_flags,"\t"*2+"::",dvar,"\t\t::",val
            else: print "\t",all_option_flags,"\t"*2+"::",dvar,"\t\t::",val
    print "\t"+"-"*60+"\n"
    return
# ----------------------------------------------------------------------------------

def checkImageParam(image, logBadlist=False):
    """
    Tries to accomplish the same thing as most IRAF tasks in regards to how they
    handle the input file parameter.
    
    @param image: 
    What is supported:
    
    Strings:
    1) If the string starts with '@' then it is expected that after the '@' is 
       a filename with the input image filenames in it.
    2) Input image filename.
    
    List:
    1) Must be a list of strings. It is assumed the each string refers an 
    input image filename.
    
    @type image: String or List of Strings
    
    @param logBadlist: Controls if lists of images that cannot be accessed are
        written to the log or not.
        
    @return: A tuple of lists of filenames of images, runnable and "bad".

    @rtype: 2-tuple of lists, list,list
    
    """
    outList = []
    badList = []
    inList  = []
    log     = gemLog.getGeminiLog()
    root, imageName = os.path.split(image)

    if type(image) == list:
        for img in image:
            if type(img) == str:
                inList.append(img)
            else:
                log.warning('Type '+str(type(image))+ 
                    ' is not supported. The only supported types are String'+
                    ' and List of Strings.')

    elif imageName and type(imageName) == str:
        if imageName[0] == '@':
            imageName = imageName[1:]
            try:
                image    = os.path.join(root, imageName)
                readList = open(image).readlines()

                for i in range(len(readList)):
                    readList[i] = readList[i].strip()
                    if not readList[i] or readList[i][0] == '#':
                        continue
                    if os.path.dirname(readList[i]) == '':
                        readList[i] = os.path.join(root, readList[i])
                    nospace_str = readList[i].replace(' ','')
                    inList.append(strutil.appendFits(nospace_str))

            except IOError:
                log.critical('An error occurred when opening and reading '+
                'from the image '+ image)
        else:
            inList.append(image)            
            inList[0] = strutil.appendFits(inList[0])
    else:
        log.warning('Type'+ str(type(image))+ 
                    'is not supported. The only supported types are String '+
                    'and List of Strings.')
    if inList:
        for img in inList:
            if not os.access(img,os.R_OK):
                log.error('Cannot read file: '+str(img))   
                badList.append(img)
            else:
                outList.append(img)

            if badList:
                if logBadlist:
                    err = "\n\t".join(badList)
                    log.warning("Some files not found or cannot be opened:\n\t"+err)

    return outList, badList

#--------------------------------------------------------------------------------

def abortBadParamfile(log, lines):
    for i in range(len(lines)):
        log.error("  %03d:%s" % (i, lines[i]))
    log.error("  %03d:<<stopped parsing due to error>>" % (i+1))
    sys.exit(1)
    return


def command_line(parser, args, log):
    """
    command line oriented parsing in one common location.
    """
    # this is done first because command line options can be set in config file
    if args.paramfile:
        ups       = []
        gparms    = {}
        lines     = []
        astrotype = None
        primname  = None
        cl        = get_classification_library()

        lines = [re.sub("#.*?$", "", line).strip() \
                 for line in open(args.paramfile).readlines()]

        # see if they are command options
        plines = []
        i = 0

        for line in lines:
            i += 1
            pline = line
            plines.append(pline)

            if not line:
                continue

            # '--xoption' or '--xoption=value' long switch
            elif len(line)>2 and line[:2] == "--":
                #then it's an option
                if "=" not in line:
                    try:
                        assert(parser_has_option(parser, line.strip()))
                        opt = line.strip()
                        val = True
                    except AssertionError:
                        log.error("Badly formatted parameter file (%s)\n" \
                                  "  Line #%d: %s""" % (args.paramfile, i, pline))
                        log.error("Unrecognized argument: %s " % line)
                        abortBadParamfile(log, plines)
                else:
                    opt,val = line.split("=")
                    opt = opt.strip()
                    val = val.strip()

                # --files may have multiple values ...
                # eg., --files=file1 file2 file3
                if opt == "--files":
                    log.info("found --files option in param file.")
                    files = val.split()
                    log.info("Extending args.files by %s" % files)
                    args.files.extend(files)
                    continue

                # Emulator funcs perform has_option(), get_option() tasks.
                # insert_option_value() wraps get_option() and exec().
                try:
                    assert(parser_has_option(parser, opt))
                    log.info("Got option in paramfile: %s" % opt)
                    insert_option_value(parser, args, opt, val)
                    continue
                except AssertionError:
                    log.error("Badly formatted parameter file (%s)\n" \
                              "  Line #%d: %s""" % (args.paramfile, i, pline))
                    log.error("Unrecognized argument: %s " % opt)
                    abortBadParamfile(log, plines)

            # '-x xvalue' short option & value
            elif len(line.split()) == 2 and line[0] == '-':
                opt, val = line.split()
                try:
                    assert(parser_has_option(parser, opt))
                    log.info("Got option, value in paramfile: %s, %s" % (opt, val))
                    insert_option_value(parser, args, opt, val)
                    continue
                except AssertionError:
                    log.error("Badly formatted parameter file (%s)\n" \
                              "  Line #%d: %s""" % (args.paramfile, i, pline))
                    log.error("Unrecognized argument: %s " % line)
                    abortBadParamfile(log, plines) 

            # '-x' short switch
            elif len(line) == 2 and line[0] == '-':
                opt = line.strip()
                val = True
                try:
                    assert(parser_has_option(parser, opt))
                    log.info("Got option in paramfile: %s" % opt)
                    insert_option_value(parser, args, opt, val)
                    continue
                except AssertionError:
                    log.error("Badly formatted parameter file (%s)\n" \
                              "  Line #%d: %s""" % (args.paramfile, i, pline))
                    log.error("Unrecognized argument: %s " % line)
                    abortBadParamfile(log, plines)

            elif len(line)>0:
                if "]" in line:                 # section line
                    name = re.sub("[\[\]]", "", line).strip()
                    if len(name)== 0:
                        astrotype = None
                        primname = None
                    elif cl.is_name_of_type(name):
                        astrotype = name
                    else:
                        primname = name
                else:                            # par=val line
                    keyval = line.split("=")
                    if len(keyval)<2:
                        log.error("Badly formatted parameter file (%s)" \
                            "\n  Line #%d: %s""" % (args.paramfile, i, pline))
                        abortBadParamfile(log, plines)
                        # sys.exit(1) # abortBAdParamfile calls sys.exit(1)
                    key = keyval[0].strip()
                    val = keyval[1].strip()
                    if val[0] == "'" or val[0] == '"':
                        val = val[1:]
                    if val[-1] == "'" or val[-1] == '"':
                        val = val[0:-1]
                    if primname and not astrotype:
                        log.error("Badly formatted parameter file (%s)" \
                            '\n  The primitive name is set to "%s", but the astrotype is not set' \
                            "\n  Line #%d: %s" % (args.paramfile, primname, i, pline[:-1]))
                        abortBadParamfile(log, plines)
                    if not primname and astrotype:
                        log.error("Badly formatted parameter file (%s)" \
                            '\n  The astrotype is set to "%s", but the primitive name is not set' \
                            "\n  Line #%d: %s" % (args.paramfile, astrotype, i, pline))
                        abortBadParamfile(log, plines)
                    if not primname and not astrotype:
                        gparms.update({key:val})
                    else:
                        up = RecipeManager.UserParam(astrotype, primname, key, val)
                        ups.append(up)

        # parameter file ups and gparms
        pfups = ups
        pfgparms = gparms

    if args.show_colors:
        print dir(filteredstdout.term)
        sys.exit(0)

    try:
        assert(args.files or args.astrotype)
    except AssertionError:
        log.info("Either file(s) OR an astrotype is required;"
                 "-t or --astrotype.")
        log.error("NO INPUT FILE or ASTROTYPE specified")
        log.info("type 'reduce -h' for usage information")
        sys.exit(1)

    input_files = []
    badList     = []

    for inf in args.files:
        olist,blist = checkImageParam(inf)      # checkImageParam return 2-tuple.
        input_files.extend(olist)
        badList.extend(blist)
    try:
        assert(badList)
        print "Got a badList ... ", badList
        print "I.e. File not found or unreadable."
        err = "\n\t".join(badList)
        log.error("Some files not found or can't be loaded:\n\t"+err)
        log.error("Exiting due to missing datasets.")
        try:
            assert(input_files)
            found = "\n\t".join(input_files)
            log.info("These datasets were found and loaded:\n\t"+found)
        except AssertionError:
            print "Got no input files"
            pass
        sys.exit(1)
    except AssertionError: pass

    # parameters from command line and/or parameter file
    clups = []
    pfups = []
    clgparms = {}
    pfgparms = {}

    if args.userparam:
        # print "r451: user params", args.userparam
        ups = []
        gparms = {}
        allupstr = args.userparam
        allparams = allupstr.split(",")
        # print "r456:", repr(allparams)
        for upstr in allparams:
            # print "r458:", upstr
            tmp = upstr.split("=")
            spec = tmp[0].strip()
            # @@TODO: check and convert to correct type
            val = tmp[1].strip()

            if ":" in spec:
                typ,prim,param = spec.split(":")
                up = RecipeManager.UserParam(typ, prim, param, val)
                ups.append(up)
            else:
                up = RecipeManager.UserParam(None, None, spec, val)
                ups.append(up)
                gparms.update({spec:val})
        # command line ups and gparms
        clups = ups
        clgparms = gparms

    # print "r473:", repr(clgparms)
    fups = RecipeManager.UserParams()
    for up in clups:
        #print "r473:", up
        fups.add_user_param(up)
    for up in pfups:
        #print "r476:", up
        fups.add_user_param(up)

    args.user_params = fups
    args.globalParams = {}
    args.globalParams.update(clgparms)
    args.globalParams.update(pfgparms)

    return input_files
