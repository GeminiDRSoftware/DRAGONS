#!/usr/bin/env python
import os
import sys
import pwd
import string
import signal
import commands
import subprocess

from getpass import getuser
from optparse import OptionParser

from astrodata.utils import Errors

# parsing the command line
parser = OptionParser()
parser.set_description("""'superclean' is a script used to clean the current \
working directory of files, hidden directories, and processes that may have \
occured from running reduce.py. An option must be selected from the list \
below. 
""")
parser.add_option("-a", "--adcc", dest="adcc_process", action="store_true", 
                  default=False, 
                  help="stop all user adcc processes currently running")
parser.add_option("-c", "--cache", dest="cache", action="store_true", 
                  default=False, help="clear the cache directory (.reducecache)")
parser.add_option("-d", dest="dirs", action="store_true", 
                  default=False, help="remove reduce-spawned directories"
                  " (backups, calibrations, locks)")
parser.add_option("-f", dest="fits", action="store_true", default=False,
                  help="remove *.fits files")
parser.add_option("-l", dest="log", action="store_true", default=False,
                  help="remove *.log files")
parser.add_option("-p", dest="pkl", action="store_true", 
                  default=False, 
                  help="remove *.pkl files in .reducecache")
parser.add_option("-o", "--hiddendirs", dest="hdirs", 
                  action="store_true", default=False, 
                  help="Clear all the hidden directories (.adcc, .autologs,"
                  ".fitsstore, .reducecache) and adcclog-latest, which "
                  "is a symbolic link to the latest log in the .adcc "
                  "directory")
parser.add_option("-r", "--reduce", dest="reduce_process", action="store_true",
                  default=False, 
                  help="stop all user reduce processes currently runnings")
parser.add_option("-s", "--showall", dest="show_all_processes", 
                  action="store_true", default=False,
                  help="show all reduce and adcc running instances")
parser.add_option("-t", dest="tmps", action="store_true", 
                  default=False, help="remove tmp* files")
parser.add_option("-u", "--showuser", dest="show_user_processes", 
                  action="store_true",
                  default = False, help="show user reduce and adcc running "
                  "instances")
parser.add_option("-v", dest="verbose", action="store_true", 
                  default=False, help="produce superclean report")
parser.add_option("--all", dest="ALL", action="store_true", default=False,
                  help="rm *.fits, *.log, tmp*, and adcclog-latest in the "
                  "currrent working directory. Remove hidden directories"
                  "(.reducecache, .adcc, .fitsstore, .autologs). Remove "
                  "reduce-spawned directories (backups, calibrations,locks)"
                  ". Stop user reduce and adcc processes")
parser.add_option("--ra", dest="reduce_adcc", action="store_true", 
                  default=False, help="stop all reduce and adcc processes.")
parser.add_option("--safe", dest="safe", action="store_true", 
                  default=False, help="-radot, and keeps calibrations")
parser.add_option("--silent", dest="silent", action="store_true", 
                  default=False, help="no standard output")
parser.add_option("--txt", dest="txt", action="store_true", 
                  default=False, help="remove text files (.txt)")
(options, args) = parser.parse_args()

# helper functions
def process_str(processinfo=None):
    rstr = ""
    rstr += "%-20s:%s %s\n" % ("Command", processinfo[7], processinfo[8])
    rstr += "%-20s:%s\n" % ("Process ID", processinfo[1])
    rstr += "%-20s:%s\n" % ("Owner", processinfo[0])
    rstr += "%--20s:%s\n\n" % ("Time Started", processinfo[4])
    return rstr

def kill_message(processinfo=None):
    rstr = ""
    rstr += "stopped process id %s (user =  %s), command: " % \
        (processinfo[1], processinfo[0])
    rstr += "\n    %s %s\n" % (processinfo[7], processinfo[8])
    return rstr

def rmfiles(dir_=None, ext=None, path_file="", pre=None):
    flist = ""
    fstr = ""
    if path_file == "":
        # creates list of cache content, 
        # if .<ext> files are in the list, rm them, and kick them off list
        # compare length of old to new list to determine if .<ext> removed
        # Note** (must append / to dir_)
        flist = os.listdir(dir_)
        for f in flist:
            path = dir_ + f
            if os.path.exists(path):
                if f[-4:] == ext or f[-5:] == ext:
                        pass_ = subprocess.call(["rm", "-r", path])
                        if pass_ == 0:
                            fstr += "\nremoved %s" % f
                if pre != None:
                    if f[:len(pre)] == pre:
                        pass_ = subprocess.call(["rm", "-r", dir_ + f])
                        if pass_ == 0:
                                fstr += "\nremoved %s" % f
    else:
        if os.path.exists(path_file):
            pass_ = subprocess.call(["rm", "-r", path_file])
            if pass_ == 0:
                fstr += "\nremoved %s" % path_filei
    return fstr

# Show options if none selected
if len(sys.argv) == 1:
    parser.print_help()
    sys.exit()

# sort out options
optka = options.adcc_process
optkr = options.reduce_process
optsap = options.show_all_processes
optsup = options.show_user_processes
if optsup and optsap:
    optsup = False
if options.ALL:
    options.local_files = True
    options.hdirs = True
    optka = True
    optkr = True
    options.dirs = True
    options.tmps = True
    options.log = True
    options.fits = True
    options.txt = True
if options.reduce_adcc:
    optka = True
    optkr = True
if options.safe:
    options.hdirs = True
    optka = True
    optkr = True
    options.dirs = True
    options.tmps = True

    
rstr = ""

# remove hidden directories
if options.cache or options.hdirs:
    hidden_dirs = []
    if options.hdirs:
        hdirs = ["./.reducecache", "./.adcc", "./.autologs","./.fitsstore"] 
    else:
        hdirs = ["./.reducecache"]
    hid = False
    for hd in hdirs:
        if os.path.exists(hd):
            hid = True
    if hid:
        rstr += "\n\nHidden Directories\n------------------"
    for hd in hdirs:
        if os.path.exists(hd):
            pass_ = subprocess.call(["rm", "-r", hd] )
            if pass_ == 0:
                rstr += "\nremoved %s directory" % hd
    if os.path.lexists("adcclog-latest"):
        pass_ = subprocess.call(["rm", "adcclog-latest"])
        if pass_ == 0:
            rstr += "\nremoved adcclog-latest (symlink to log in .autologs)"
    if os.path.lexists("reducelog-latest"):
        pass_ = subprocess.call(["rm", "reducelog-latest"])
        if pass_ == 0:
            rstr += "\nremoved reducelog-latest (symlink to log in .autologs)"
if options.dirs:
    dirstr = "\n\nDirectories\n-----------"
    pass_ = 1
    if os.path.exists("backups"):
        pass_ = subprocess.call(["rm", "-r", "backups"])
        if pass_ == 0:
            dirstr += "\nremoved 'backups'"
    if os.path.exists("calibrations") and not options.safe:
        pass_ = subprocess.call(["rm", "-r", "calibrations"])
        if pass_ == 0:
            dirstr += "\nremoved 'calibrations'"
    if os.path.exists("locks"):
        pass_ = subprocess.call(["rm", "-r", "locks"])
        if pass_ == 0:
            dirstr += "\nremoved 'locks'"
    if pass_ == 0:
        rstr += dirstr
    
# remove files (hidden or current working dir)
if options.pkl or options.log or options.tmps or options.fits:
    fhead = "\n\nFiles\n-----"
    fstr = ""
    if options.pkl:
        fstr += rmfiles(dir_="./.reducecache/", ext=".pkl")
    if options.log:
        fstr += rmfiles(dir_="./", ext=".log")
    if options.fits:
        fstr += rmfiles(dir_="./", ext=".fits")
    if options.tmps:
        fstr += rmfiles(dir_="./", pre="tmp")
    if options.txt:
        fstr += rmfiles(dir_="./", ext=".txt")
    if len(fstr) > 0:
        rstr += fhead + fstr

# stop or show processes
prep = ""
if optka or optkr or optsap or optsup:
    outputall = commands.getoutput("ps -ef | grep python")
    lines = outputall.split("\n")
    # must use username and userid to avoid mac conflict
    username = getuser()
    pw = pwd.getpwnam(username)
    userid = pw.pw_uid
    showstr = ""
    for line in lines:
        processinfo = string.split(line)
        
        # skips over grep and superclean commands
        if (line.rfind("grep") > -1) or (line.rfind("superclean") > -1):
            continue
        puserid = None
        pusername = None 
        if processinfo[0].isdigit():
            puserid = int(processinfo[0])
        else:
            pusername = processinfo[0]

        # option selections
        if optsap:
            showstr += "\n" + process_str(processinfo)
        if optsup:
            if userid == puserid or username == pusername:
                showstr += "\n" + process_str(processinfo)
        if optkr:
            if (line.rfind("/reduce") > -1) or (line.rfind("recipe") > -1):
                if userid == puserid or username == pusername:
                    os.kill(int(processinfo[1]), signal.SIGKILL)
                    prep += "\n" + kill_message(processinfo)
        if optka:
            if line.rfind("/adcc") > -1:
                if userid == puserid or username == pusername:
                    os.kill(int(processinfo[1]), signal.SIGKILL)
                    prep += "\n" + kill_message(processinfo)
    if prep != "":
        rstr += "\n\nProcesses\n" + "-"*9 + prep + "\n"

hstr = "\n" + "="*78 + "\nSuperclean Report\n" + "="*78
if optsup or optsap:
    if showstr == "":
        instr = ""
        if optsup:
            instr = "user"
        print("%s\nno %s processes running" % (hstr,instr))
    else:
        print("%s\n\nProcesses\n%s%s" % (hstr, "-"*9, showstr))
    

if options.verbose:
    if rstr != "":
        print hstr + rstr
    else:
        print "no files, dirs, or processes to clean"
elif options.silent:
    pass
else:
    print "."
