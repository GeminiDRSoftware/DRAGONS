import os
import pyfits as pf
import numpy as np
#from astrodata.adutils import mefutil, paramutil
from astrodata.adutils import gemLog
from astrodata.AstroData import AstroData

log=gemLog.getGeminiLog() 




def div(Numerator, Denominator):
    '''
    A function to divide a input science image by a another image(or flat) or an floating point integer.
    If the Denominator is a AstroData MEF then this function will loop through the SCI, VAR and DQ frames
    to divide each SCI of the Numerator by the Denominator SCI of the same EXTVER. It will apply a 
    bitwise-or to the DQ frames to preserve their binary formats.
    If the Denominator is a float integer then only the SCI frames of the Numerator are each divided by the int.
    
    $$$$$$$$$$$$$$$$$  WARNING, THIS DOES NOT CALCULATE THE VAR FRAMES CORRECTLY YET. JUST ADDS THEM $$$$$
    
    @param Numerator: input image to be divided by the Denominator
    @type Numerator: a MEF or single extension fits file in the form of an AstroData instance
    
    @param Denominator: denominator to divide the numerator by
    @type Denominator: a MEF of SCI, VAR and DQ frames in the form of a AstroData instance or an float int  
    '''
    Num=Numerator
    Den=Den
    Out=Numerator.__deepcopy__('COPY') #why does this require a msg?? silly
    if type(Den)==AstroData:
        for i in range(0,len(Num['SCI'])-1):
            try:
                if Num['SCI'][i].data.shape==Den['SCI'][i].data.shape: #making sure arrays are same size/shape
                
                    # dividing the SCI frames
                    np.divide(Num['SCI'][i].data,Den['SCI'][i].data,out['SCI'][i].data)
                    
                    # simply adding the VAR frames WARNING THIS IS ONLY TEMP, MUST CORRECT LATER!!!!!!!!!!!!!!
                    np.add(Num['VAR'][i].data,Den['VAR'][i].data,out['VAR'][i].data)
                    
                    # bitwise-or 'adding' DQ frames 
                    np.bitwise_or(Num['DQ'][i].data,Den['DQ'][i].data,out['DQ'][i].data)
                else:
                    log.critical('arrays are different sizes for SCI extension '+i+' of the input '\
                                 +Num.filename+' and '+Den.filename,'critical')
                    raise ArithError('An error occurred while performing an arith task')
            except:
                raise ArithError('An error occurred while performing an arith task')
    elif type(Den)==float:
        for i in range(0,len(Num['SCI'])-1):
            try:
                # dividing the SCI frames
                np.divide(Num['SCI'][i].data,Den,out['SCI'][i].data)
                
            except:
                raise ArithError('An error occurred while performing an arith task')
                
                