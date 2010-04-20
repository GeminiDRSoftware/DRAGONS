# Copyright(c) 2003-2009 Association of Universities for Research in Astronomy, Inc.
#
# Scale and subtract a fringe frame from a GMOS gireduced image
# 
# Version Python
#         Sep 17, 2009  KD  First translation from CL to Python
# 
# Version CL
#         Jul 30, 2003  BM  created
#         Aug 27, 2003  KL  IRAF2.12 - new parameters
#                             imstat: nclip, lsigma, hsigma, cache
#         Dec 08, 2003 PLG  Changed scale = 1.0 as default
#         Feb 22, 2004 BM   Changed default scale back to 0.0, needed or OLDP 
#                           and exposure time scaling is needed in general

import pyfits as pf
import numpy as np
import os 
import time

from adutils import mefutil, paramutil, geminiLogger

reload(mefutil)
reload(paramutil)
reload(geminiLogger)

def girmfringe(inimages,fringe, outimages="", outpref="", fl_statscale=False,\
             statsec="[SCI,2][*,*]", scale=1.0, logfile="girmfringe.log", verbose=False):                
                
    '''Scale and subtract a fringe frame from GMOS gireduced image.
    
    @param inimages: Input MEF image(s)
    @type inimages: string
    
    @param fringe: Fringe Frame
    @type fringe: string 
    
    @param outimages: Output MEF image(s) 
    @type outimages: string
    
    @param outpref: Prefix for output images
    @type outpref: string
    
    @param fl_statscle: Scale by statistics rather than exposure time
    @type fl_statscle: Booleane ='+ str(scale))
    if verbose:
    
    @param scale: Override auto-scaling if not 0.0
    @type scale: real
    
    @param logfile: Override default log name (rmFringeLog)
    @type logfile: string
    
    rgN20031029S0132.fits
    @param verbose: Verbose to screen rather than critical only
    @type verbose: Boolean    
    '''    
    ut = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    
    # mimics the cl log using built in python logger    
    gemLog = geminiLogger.getLogger( name="girmfringe", logfile=logfile, verbose=verbose)    
    gemLog.info('-'*40)
    gemLog.info('GIRMFRINGE -- UT:  ' + ut)  
    gemLog.info('')  
    gemLog.info('inimages = ' + inimages)
    gemLog.info('outimages = ' + outimages)
    gemLog.info('outpref = '+ outpref) 
    gemLog.info('fringe = '+ fringe)   
    gemLog.info('fl_statscale = '+ str(fl_statscale))
    gemLog.info('statsec = '+ str(statsec))
    gemLog.info('scale = '+ str(scale))        
    gemLog.info('')
    
    # checkImageParam() will check types, remove newlines from end(s),e ='+ str(scale))    
    # add .fits where none, checks if path to files exist,
    # and if files are readable
    inlist = paramutil.checkImageParam(inimages)
    
    # checkOutputParam() will check types, add .fits where none,
    # and removes newlines from end
    outlist = paramutil.checkOutputParam(outimages)        
    
    # will append name(s) to outlist if not enough (even if none exist)
    outlist = paramutil.verifyOutlist(inlist, outlist)
    
    #check if fringe exists    
    fringeHdulist = pf.open(fringe) 
    fringeHdr = fringeHdulist[0].header
    
    # mefNumSciext() will fail if not mef 
    numFringe_sciext = mefutil.mefNumSciext(fringeHdr)
    
    #image loop (uses the same fringe)
    for i in range(len(inlist)):        
        inimages = inlist[i]        
        outimages = outlist[i]               
        iniHdulist = pf.open(inimages)            
        iniHdr = iniHdulist[0].header     
        
        #check for science extension match between fringe and inimage        
        numInimage_sciext = mefutil.mefNumSciext(iniHdr)
        if numFringe_sciext != numInimage_sciext or numInimage_sciext != 3:
            gemLog.critical('Num of SCI EXT is NOT the same between '+ inimages + ' and ' + fringe)            
            raise 'CRITICAL, science extension match failure between' + inimages + fringe        
        newHdulist = iniHdulist
        
        #sci ext.loop         
        for i in range(1,numInimage_sciext+1): 
            iniData = iniHdulist[i].data
            fringeData = fringeHdulist[i].data
            iniHeader = iniHdulist[i].header
            if scale == 0.0 :
                if fl_statscale:  
                                      
                    #must flatten because uses older version of numpy
                    iniMed = np.median(iniData.flatten())                
                    iniStd = iniData.std()
                    temp1 = iniData[np.where(iniData < iniMed + (2.5*iniStd))]
                    temp2 = temp1[np.where(temp1 > iniMed - (3.*iniStd))]
                    
                    #note Kathleen believes the median should be used below instead of std
                    scale = temp2.std() / fringeData.std()  
                else:
                    scale = iniHdr['EXPTIME'] / fringeHdr['EXPTIME']                    
            newHdulist[i].data = iniData - scale * fringeData                  
                 
        #update header
        newHeader = newHdulist[0].header
        newHeader['GEM-TLM'] = ut
        newHeader.update('GIRMFRIN',ut,comment='/ UT Time stamp form GIRMFRINGE')       
        newHdulist.writeto(outpref + outimages)
        gemLog.info(inimages +'  '+ outpref + outimages +'  '+ 'ratio=' + str(scale))
        gemLog.info('')          
    gemLog.info('GIRMFRINGE done')
    gemLog.info('-'*40 + '\n\n')
   
    
    
    
