#!/usr/bin/env python
# changed executable flag, file must change to check in again
import commands, os, string
import signal

program = "python" #raw_input("Enter the name of the program to check: ")

try:
    #perform a ps command and assign results to a list
    outputall = commands.getoutput("ps -ef|grep " + program)
    lines = outputall.split("\n")
    for output in lines:
        proginfo = string.split(output)
        #display results
        if os.getpid() == int(proginfo[1]):
            continue
        if "reduce" in proginfo[8]:
            print "\n\
cmd:", proginfo[8], "\n\
                Owner: ", proginfo[0], "\n\
           Process ID: ", proginfo[1], "\n\
    Parent process ID: ", proginfo[2], "\n\
         Time started: ", proginfo[4]
            os.kill(int(proginfo[1]), signal.SIGHUP)
        if "adcc" in proginfo[8]:
            print "\n\
cmd", proginfo[8], "\n\
                Owner: ", proginfo[0], "\n\
           Process ID: ", proginfo[1], "\n\
    Parent process ID: ", proginfo[2], "\n\
         Time started: ", proginfo[4]
            os.kill(int(proginfo[1]), signal.SIGHUP)
        
except:
    print "There was a problem with the program."
    raise
