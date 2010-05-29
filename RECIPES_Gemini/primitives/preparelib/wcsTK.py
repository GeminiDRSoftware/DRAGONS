#!/usr/bin/env python
import os

from astrodata.AstroData import AstroData

class WCSTK():
    
    def __init__(self,rc):
        #print 'wcsTK: a WCSTK object has been instantiated'
        print 'dog'
        
    def wcsCheck(self,arg):
        print arg
        wcsError = False
        if wcsError:
            print "WCSTK: seems there was an error in the WCS"