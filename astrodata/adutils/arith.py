import os
import pyfits as pf
import numpy as np
#from astrodata.adutils import mefutil, paramutil
from astrodata.adutils import gemLog
from astrodata.AstroData import AstroData
import astrodata
log=gemLog.getGeminiLog() 

def div(numerator, denominator):
    '''
    A function to divide a input science image by a another image(or flat) or an floating point integer.
    If the denominator is a AstroData MEF then this function will loop through the SCI, VAR and DQ frames
    to divide each SCI of the numerator by the denominator SCI of the same EXTVER. It will apply a 
    bitwise-or to the DQ frames to preserve their binary formats.
    If the denominator is a float integer then only the SCI frames of the numerator are each divided by the int.
    
    $$$$$$$$$$$$$$$$$  WARNING, THIS DOES NOT CALCULATE THE VAR FRAMES CORRECTLY YET. JUST ADDS THEM $$$$$
    
    @param numerator: input image to be divided by the denominator
    @type numerator: a MEF or single extension fits file in the form of an AstroData instance
    
    @param denominator: denominator to divide the numerator by
    @type denominator: a MEF of SCI, VAR and DQ frames in the form of an AstroData instance or a float int  
    '''
   
    num=numerator
    den=denominator 
    from copy import deepcopy
    #out=AstroData.prepOutput(inputAry = num, clobber = False)
    out=deepcopy(num)
    #print out.info()
    #print 'a30: type(den) ',type(den), den.filename,(type(den)==astrodata.AstroData)
    if type(den)==astrodata.AstroData:
        #print 'a30: den is type AstroData'
        for sci in num['SCI']:
            extver = sci.extver()
            try:
                if num[('SCI',extver)].data.shape==den[('SCI',extver)].data.shape: #making sure arrays are same size/shape
                    #print 'a35: deviding SCI frames'
                    # dividing the SCI frames
                    np.divide(num[('SCI',extver)].data,den[('SCI',extver)].data,out[('SCI',extver)].data)
              
                    #print 'a38: adding the VAR frames'
                    # simply adding the VAR frames WARNING THIS IS ONLY TEMP, MUST CORRECT LATER!!!!!!!!!!!!!!
                    np.add(num[('VAR',extver)].data,den[('VAR',extver)].data,out[('VAR',extver)].data)
                   
                    #print 'a41: bitwise_or on DQ frames'
                    # bitwise-or 'adding' DQ frames 
                    np.bitwise_or(num[('DQ',extver)].data,den[('DQ',extver)].data,out[('DQ',extver)].data)
                    
                else:
                    log.critical('arrays are different sizes for SCI extension '+i+' of the input '\
                                 +num.filename+' and '+den.filename,'critical')
                    raise ArithError('An error occurred while performing an arith task')
            except:
                raise ArithError('An error occurred while performing an arith task')
    elif type(den)==float:
        for i in range(0,len(num['SCI'])-1):
            try:
                print 'a53: simple division of SCI frames by the float '+den
                # dividing the SCI frames
                np.divide(num[('SCI',extver)].data,den,out[('SCI',extver)].data)
                
            except:
                raise ArithError('An error occurred while performing an arith task')
            
    return out       
                
def mult(inputA, inputB):
    '''
    A function to multiply a input science image by a another image(or flat) or an floating point integer.
    If the inputB is a AstroData MEF then this function will loop through the SCI, VAR and DQ frames
    to divide each SCI of the inputA by the inputB SCI of the same EXTVER. It will apply a 
    bitwise-or to the DQ frames to preserve their binary formats.
    If the inputB is a float integer then only the SCI frames of the inputA are each divided by the int.
    
    $$$$$$$$$$$$$$$$$  WARNING, THIS DOES NOT CALCULATE THE VAR FRAMES CORRECTLY YET. JUST ADDS THEM $$$$$
    
    @param inputA: input image to be multiplied by the inputB
    @type inputA: a MEF or single extension fits file in the form of an AstroData instance
    
    @param inputB: inputB to multiply the inputA by
    @type inputB: a MEF of SCI, VAR and DQ frames in the form of an AstroData instance or a float int  
    '''
   
    inA=inputA
    inB=inputB 
    from copy import deepcopy
    #out=AstroData.prepOutput(inputAry = inA, clobber = False)
    out=deepcopy(inA)
    #print out.info()
    #print 'a30: type(inB) ',type(inB), inB.filename,(type(inB)==astrodata.AstroData)
    if type(inB)==astrodata.AstroData:
        #print 'a30: inB is type AstroData'
        for sci in inA['SCI']:
            extver = sci.extver()
            try:
                if inA[('SCI',extver)].data.shape==inB[('SCI',extver)].data.shape: #making sure arrays are same size/shape
                    print 'a100: deviding SCI frames'
                    #  multipling the SCI frames
                    np.multiply(inA[('SCI',extver)].data,inB[('SCI',extver)].data,out[('SCI',extver)].data)
                    
                    print 'a104: starting the VAR frame calculations'
                    ## creating the output VAR frame following varOut=sciOut^2 * ( varA/(sciA^2) + varB/(sciB^2) )
                    # making empty sciOutSqured array and squaring the sciOut frame to load it up
                    sciOutSquared=np.zeros(out[('SCI',extver)].data.shape,dtype=np.float32) 
                    np.multiply(out[('SCI',extver)].data,out[('SCI',extver)].data,sciOutSquared)
                    # ditto for sciA and sciB 
                    sciASquared=np.zeros(inA[('SCI',extver)].data.shape,dtype=np.float32) # $$$ all zeros arrays are the same size, consider optimizing initializing these arrays
                    sciBSquared=np.zeros(inB[('SCI',extver)].data.shape,dtype=np.float32)
                    np.multiply(inA[('SCI',extver)].data,inA[('SCI',extver)].data,sciASquared)
                    np.multiply(inB[('SCI',extver)].data,inB[('SCI',extver)].data,sciBSquared)
                    # now var_A/sciASquared and var_B/sciBSquared
                    varAoverSciASquared=np.zeros(inA[('SCI',extver)].data.shape,dtype=np.float32)
                    varBoverSciBSquared=np.zeros(inB[('SCI',extver)].data.shape,dtype=np.float32)
                    np.divide(inA[('VAR',extver)].data,sciASquared,varAoverSciASquared)
                    np.divide(inB[('VAR',extver)].data,sciBSquared,varBoverSciBSquared)
                    # now varAoverSciASquared + varBoverSciBSquared
                    varOverAplusB=np.zeros(inA[('SCI',extver)].data.shape,dtype=np.float32)
                    np.add(varAoverSciASquared,varBoverSciBSquared,varOverAplusB)
                    # put it all together varOut=sciOut^2 * ( varA/(sciA^2) + varB/(sciB^2) )
                    np.multiply(sciOutSquared,varOverAplusB,out[('VAR',extver)].data)
                   
                    print 'a41: bitwise_or on DQ frames'
                    # bitwise-or 'adding' DQ frames 
                    np.bitwise_or(inA[('DQ',extver)].data,inB[('DQ',extver)].data,out[('DQ',extver)].data) #$$$$$$$$$$$$$$$$$$$$$$$$$
                    
                else:
                    log.critical('arrays are different sizes for SCI extension '+i+' of the input '\
                                 +inA.filename+' and '+inB.filename,'critical')
                    raise ArithError('An error occurred while performing an arith task')
            except:
                raise ArithError('An error occurred while performing an arith task')
    elif type(inB)==float:
        for i in range(0,len(inA['SCI'])-1):
            try:
                print 'a53: simple division of SCI frames by the float '+inB
                # multipling the SCI frames by the constant
                np.multiply(inA[('SCI',extver)].data,inB,out[('SCI',extver)].data)
                
                #multipling the VAR frames by the constant^2
                np.multiply(inA[('VAR',extver)].data,inB*inB,out[('VAR',extver)].data)
                
            except:
                raise ArithError('An error occurred while performing an arith task')
            
    return out       
                