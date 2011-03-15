#Author: Kyle Mede, May 2010
#This module provides functions to be used by all GMOS primitives.
import os

import pyfits as pf
import numpy as np
from copy import deepcopy
import time
from astrodata.adutils import gemLog
from astrodata.AstroData import AstroData
from astrodata.Errors import ToolboxError

def stdInstHdrs(ad, logLevel=1):  
    """ 
    A function used by StandardizeInstrumentHeaders in primitives_GMOS.
    It currently just adds the DISPAXIS header key to the SCI extensions.
    
    :param ad: input astrodata instance to have its headers standardized
    :type ad: a single astrodata instance
    
    :param logLevel: Verbosity setting for the log messages to screen,
                     default is 'critical' messages only.
                     Note: independent of logLevel setting, all messages always go 
                     to the logfile if noLogFile=False.
    :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to screen.
                    OR the message level as a string (ie. 'critical', 'status', 
                    'fullinfo'...)
    """
    # Adding the missing/needed keywords into the headers
    if not ad.isType('GMOS_IMAGE'):
    # Do the stuff to the headers that is for the MOS, those for IMAGE are 
    # taken care of with stdObsHdrs all ready 
    
        log=gemLog.getGeminiLog(logLevel=logLevel)
    
        # Formatting so logger looks organized for these messages
        log.fullinfo('*'*50, 
                     'header') 
        log.fullinfo('file = '+ad.filename, 'header')
        log.fullinfo('~'*50, 
                     'header')
        for ext in ad['SCI']:
            ext.header.update(('SCI',ext.extver()),'DISPAXIS', \
                            ext.dispersion_axis() , 'Dispersion axis')
            # Updating logger with new header key values
            log.fullinfo('SCI extension number '+str(ext.header['EXTVER'])+
                         ' keywords updated/added:\n', 'header')       
            log.fullinfo('DISPAXIS = '+str(ext.header['DISPAXIS']), 'header' )
            log.fullinfo('-'*50,
                         'header')

def valInstData(ad, logLevel=1):  
    """
    A function used by validateInstrumentData in primitives_GMOS.
    
    It currently just checks if there are 1, 3, 6 or 12 SCI extensions 
    in the input. 
    
    :param ad: input astrodata instance to validate
    :type ad: a single astrodata instance
    
    :param logLevel: Verbosity setting for the log messages to screen,
                     default is 'critical' messages only.
                     Note: independent of logLevel setting, all messages always go 
                     to the logfile if noLogFile=False.
    :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to screen.
                    OR the message level as a string (ie. 'critical', 'status', 
                    'fullinfo'...)
    """
    log=gemLog.getGeminiLog(logLevel=logLevel)
    length=ad.countExts('SCI')
    # If there are 1, 3, 6, or 12 extensions, all good, if not log a critical 
    # message and raise an exception
    if length==1 or length==3 or length==6 or length==12:
        pass
    else: 
        log.critical('There are NOT 1, 3, 6 or 12 extensions in file = '+
                     ad.filename)
        raise ToolboxError('Error occurred in valInstData for input '+
                           ad.filename)

#------------- GMOS_IMAGE fringe removal funcs ---------------------------  

# There was talk about generalizing this module to work on all imaging data
# rather than just GMOS images, this is left as a task to look into later.
def rmImgFringe(inimage, fringe, fl_statscale=False, statsec='', 
               scale=0.0, logLevel=1):                         
    """
    Scale and subtract a fringe frame from GMOS gireduced image.
    
    :param inimage: Input image
    :type inimage: AstroData instance
    
    :param fringe: Fringe Frame
    :type fringe: AstroData instance
    
    :param fl_statscale: Scale by statistics rather than exposure time
    :type fl_statscale: Boolean
    
    :param statsec: image section used to determine the scale factor 
                    if fl_statsec=True
    :type statsec: string of format '[EXTNAME,EXTVER][x1:x2,y1:y2]'
    :default statsec: If CCDSUM = '1 1' :[SCI,2][100:1900,100:4500]'
                      If CCDSUM = '2 2' : [SCI,2][100:950,100:2250]'
                      
    :param scale: Override auto-scaling if not 0.0
    :type scale: real 
    
    :param logLevel: Verbosity setting for the log messages to screen,
                     default is 'critical' messages only.
                     Note: independent of logLevel setting, all messages always go 
                     to the logfile if noLogFile=False.
    :type logLevel: integer from 0-6, 0=nothing to screen, 6=everything to screen.
                    OR the message level as a string (ie. 'critical', 'status', 
                    'fullinfo'...)
    
    """    
    log=gemLog.getGeminiLog(logLevel=logLevel)
    
    ut = time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime())
    
    # Mimics the cl log using the new Gemini Logger gemlog.py 
    log.fullinfo('-'*15+'Input Parameters'+'-'*15)
    log.fullinfo('GIRMFRINGE -- UT:  ' + ut)  
    log.fullinfo('', category='format')  
    log.fullinfo('inimage = ' + inimage.filename)
    log.fullinfo('fringe = '+ fringe.filename)   
    log.fullinfo('fl_statscale = '+ str(fl_statscale))
    log.fullinfo('statsec = '+ str(statsec))
    log.fullinfo('scale = '+ str(scale))        
    log.fullinfo('', category='format')
    
    # Ensuring the input image and fringe have the same number of SCI extensions
    if (fringe.countExts('SCI') != inimage.countExts('SCI')):
        log.critical('Number of SCI EXT is NOT the same between '+ 
                     inimage.filename + ' and ' + fringe.filename)            
        raise ToolboxError('CRITICAL, science extension match '+
                                  'failure between' + inimage.filename + 
                                  fringe.filename)
                
    # Setting statsec to the correct default value if needed (assumes square binning)
    if (statsec == '') and fl_statscale:
        imagexbin = inimage[('SCI',2)].detector_x_bin(pretty=True, asDict=False)
        fringexbin = fringe[('SCI',2)].detector_x_bin(pretty=True, asDict=False)
        imageybin = inimage[('SCI',2)].detector_y_bin(pretty=True, asDict=False)
        fringeybin = fringe[('SCI',2)].detector_y_bin(pretty=True, asDict=False)
        # Ensuring images are square binned and image binning = fringe binning
        if imagexbin == fringexbin == imageybin == fringeybin:
            # Setting to default value for 1x1 images and logging value
            if imagexbin == 1:
                statsec = '[SCI,2][100:1900,100:4500]'
                log.fullinfo('Using statsec = '+statsec)
            # Setting to default value for 2x2 images and logging value
            elif imagexbin == 2:
                statsec = '[SCI,2][100:950,100:2250]'
                log.fullinfo('Using statsec = '+statsec)
            else:
                log.critical('The CCD X binning '+imagexbin+
                             ' is not 1 or 2 for the input image '+
                             inimage.filename)
        # Logging critical message and raising exception if ccd x binning doesn't match 
        else:
            log.critical('The CCD X and Y binning for the input image '+inimage.filename+
                         ' and the input fringe '+fringe.filename+ 
                         ' do not match.')
            raise ToolboxError('The CCD X and Y binning for the input image '+inimage.filename+
                         ' and the input fringe '+fringe.filename+ 
                         ' do not match.')
    
    # Converting the statsec string into its useful components if needed                
    if fl_statscale:    
        (extname, extver, x1, x2, y1, y2) = statsecConverter(statsec) 
        
    # Getting data arrays from the input image and fringe for this extension
    iniData = inimage[(extname, extver)].data
    fringeData = fringe[(extname, extver)].data
    # Extracting statsec defined sections of the data arrays 
    # for scale calculation
    iniDataSec = iniData[x1:x2,y1:y2]
    fringeDataSec = fringeData[x1:x2,y1:y2]
    
    # Making sure arrays are same size/shape
    if iniData.shape == fringeData.shape:
        # Calculating the variable scale if needed
        if scale == 0.0 :
            # Use statistics to calculate scale if requested
            if fl_statscale:        
                log.status('Using statistics to calculate the scale value')
                # Must flatten because uses older version of numpy
                iniMed = np.median(iniDataSec.flatten())        
                # Calculated the standard deviation of the input data        
                iniStd = iniDataSec.std()
                # Make an array of all the points where the pixel value is 
                # less than the median value + 2.5 x the standard deviation.
                temp1 = iniDataSec[np.where(iniDataSec < iniMed + (2.5*iniStd))]
                # Make an array from the previous one where all the pixels 
                # in it that have a value greater than the median - 3x the 
                # standard deviation.  Thus a final array of all the pixels
                # with values between (median + 2.5xstd) and (median - 3xstd).
                temp2 = temp1[np.where(temp1 > iniMed - (3.*iniStd))]
                
                #note Kathleen believes the median should be used below instead of std
                scale = temp2.std() / fringeDataSec.std()  
                
            else:
                # Calculate the scale value from the exposure times of the 
                # image and fringe PHUs
                scale = inimage[(extname, extver)].getKeyValue('EXPTIME') / \
                            fringe[(extname, extver)].getKeyValue('EXPTIME')  
            
        # Logging the scale value being used                
        log.stdinfo('The scale value being applied to the fringe '+
                        'frames is '+str(scale))    
        # format line in logger to indicate end of parameters being used
        log.fullinfo('-'*45)
           
        # Calculating the output based on 
        # output image frame = input image frame - (scale * fringe frame)
        log.debug('calling fringe.mult(scale)')
        # Using the mult function from the arith toolbox as it takes care of VAR and DQs        
        scaledFringe = fringe.mult(scale)  
        log.debug('calling inimage.sub(scaledFringe)')  
        # Using the sub function from the arith toolbox as it takes care of VAR and DQs 
        outImage = inimage.sub(scaledFringe)
        
    else:
        log.critical('The input image '+inimage.filename+' and fringe '+
                     fringe.filename+' had arrays of different sizes')                  
        raise ToolboxError('CRITICAL, The input image '+inimage.filename+' and fringe '+
                     fringe.filename+' had arrays of different sizes')  
    
    log.fullinfo('', category='format')          
    log.fullinfo('GIRMFRINGE done')
    log.fullinfo('-'*45 + '\n', category='format')
   
    return outImage
    
def statsecConverter(statsecStr):
    try:
        (left,right)=statsecStr.split('][')
        (extname,extver)=left.split('[')[1].split(',')
        (X,Y)=right.split(']')[0].split(',')
        (x1,x2)=X.split(':')
        (y1,y2)=Y.split(':')
        
        return extname.upper(), int(extver), int(x1), int(x2), int(y1), int(y2)
    except: 
        raise ToolboxError('An problem occured trying to convert the statsecStr '+
                     statsecStr+' into its useful components')         
#----------------------------------------------------------------------------       