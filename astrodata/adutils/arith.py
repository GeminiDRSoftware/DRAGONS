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
                    #print 'a35: deviding SCI frames '+str(extver)
                    # dividing the SCI frames
                    out[('SCI',extver)].data=np.divide(num[('SCI',extver)].data,den[('SCI',extver)].data)
                    
                    try:
                        #print 'a44: starting the VAR frame calculations '+str(extver)
                        ## creating the output VAR frame following varOut=sciOut^2 * ( varA/(sciA^2) + varB/(sciB^2) )
                        # making empty sciOutSqured array and squaring the sciOut frame to load it up 
                        sciOutSquared=np.multiply(out[('SCI',extver)].data,out[('SCI',extver)].data)
                        # ditto for sciA and sciB 
                        sciASquared=np.multiply(num[('SCI',extver)].data,num[('SCI',extver)].data)
                        sciBSquared=np.multiply(den[('SCI',extver)].data,den[('SCI',extver)].data)
                        # now var_A/sciASquared and var_B/sciBSquared
                        varAoverSciASquared=np.divide(num[('VAR',extver)].data,sciASquared)
                        varBoverSciBSquared=np.divide(den[('VAR',extver)].data,sciBSquared)
                        # now varAoverSciASquared + varBoverSciBSquared
                        varOverAplusB=np.add(varAoverSciASquared,varBoverSciBSquared)
                        # put it all together varOut=sciOut^2 * ( varA/(sciA^2) + varB/(sciB^2) )
                        out[('VAR',extver)].data=np.multiply(sciOutSquared,varOverAplusB)
                    except:   
                        log.error('no new VAR frames being calculated in arith.div','error')                                           
                    #print 'a41: bitwise_or on DQ frames '+str(extver)
                    # bitwise-or 'adding' DQ frames 
                    out[('DQ',extver)].data=np.bitwise_or(num[('DQ',extver)].data,den[('DQ',extver)].data)
                    
                else:
                    log.critical('arrays are different sizes for SCI extension '+str(extver)+' of the input '\
                                 +num.filename+' and '+den.filename,'critical')
                    raise 'An error occurred while performing an arith task'
            except:
                raise 'An error occurred while performing an arith task'
    elif type(den)==float:
        for sci in num['SCI']:
            extver = sci.extver()
            try:
                #print 'a53: simple division of SCI frames by the float '+str(den)
                # dividing the SCI frames by the constant
                out[('SCI',extver)].data=np.divide(num[('SCI',extver)].data,den)
                
                # multiplying the VAR frames by the constant^2
                out[('VAR',extver)].data=np.multiply(num[('VAR',extver)].data,den*den)
                
            except:
                raise 'An error occurred while performing an arith task'
    elif type(den)==list:
        for sci in num['SCI']:
            extver = sci.extver()
            try:
                int = den[extver-1]
                #print 'a53: simple division of SCI frames by the float '+str(int)
                # dividing the SCI frames by the constant
                out[('SCI',extver)].data=np.divide(num[('SCI',extver)].data,int)
                
                # multiplying the VAR frames by the constant^2
                out[('VAR',extver)].data=np.multiply(num[('VAR',extver)].data,int*int)
                
            except:
                raise 'An error occurred while performing an arith task'
    else:
        log.critical('arith.div() only accepts inputB of types AstroData, list and float, '+str(type(den))+' passed in', 'critical')    
        raise 'An error occurred while performing an arith task'            
    return out       
                
def mult(inputA, inputB):
    '''
    A function to multiply a input science image by a another image(or flat) or an floating point integer.
    If the inputB is a AstroData MEF then this function will loop through the SCI, VAR and DQ frames
    to divide each SCI of the inputA by the inputB SCI of the same EXTVER. It will apply a 
    bitwise-or to the DQ frames to preserve their binary formats.
    If the inputB is a float integer then only the SCI frames of the inputA are each divided by the int.
    
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
                    #print 'a100: multiplying SCI frames '+str(extver)
                    #  multipling the SCI frames
                    out[('SCI',extver)].data=np.multiply(inA[('SCI',extver)].data,inB[('SCI',extver)].data)
                    
                    #print 'a104: starting the VAR frame calculations '+str(extver)
                    ## creating the output VAR frame following varOut=sciOut^2 * ( varA/(sciA^2) + varB/(sciB^2) )
                    # squaring the sciOut frame 
                    sciOutSquared=np.multiply(out[('SCI',extver)].data,out[('SCI',extver)].data)
                    # ditto for sciA and sciB 
                    sciASquared=np.multiply(inA[('SCI',extver)].data,inA[('SCI',extver)].data)
                    sciBSquared=np.multiply(inB[('SCI',extver)].data,inB[('SCI',extver)].data)
                    # now var_A/sciASquared and var_B/sciBSquared
                    varAoverSciASquared=np.divide(inA[('VAR',extver)].data,sciASquared)
                    varBoverSciBSquared=np.divide(inB[('VAR',extver)].data,sciBSquared)
                    # now varAoverSciASquared + varBoverSciBSquared
                    varOverAplusB=np.add(varAoverSciASquared,varBoverSciBSquared)
                    # put it all together varOut=sciOut^2 * ( varA/(sciA^2) + varB/(sciB^2) )
                    out[('VAR',extver)].data=np.multiply(sciOutSquared,varOverAplusB)
                   
                    #print 'a41: bitwise_or on DQ frames '+str(extver)
                    # bitwise-or 'adding' DQ frames 
                    out[('DQ',extver)].data=np.bitwise_or(inA[('DQ',extver)].data,inB[('DQ',extver)].data) 
                    
                else:
                    log.critical('arrays are different sizes for SCI extension '+i+' of the input '\
                                 +inA.filename+' and '+inB.filename,'critical')
                    raise 'An error occurred while performing an arith task'
            except:
                raise 'An error occurred while performing an arith task'
    elif type(inB)==float:
        for sci in inA['SCI']:
            extver = sci.extver()
            try:
                print 'a53: simple multiplication of SCI frames by the float '+str(inB)
                # multiplying the SCI frames by the constant
                out[('SCI',extver)].data=np.multiply(inA[('SCI',extver)].data,inB)
                
                #multiplying the VAR frames by the constant^2
                out[('VAR',extver)].data=np.multiply(inA[('VAR',extver)].data,inB*inB)
            except:
                raise 'An error occurred while performing an arith task'
              
    elif type(inB)==list:
        for sci in inA['SCI']:
            extver = sci.extver()
            try:
                int = inB[extver-1]
                #print 'a53: simple multiplication of SCI frames by the float '+str(int)
                # multiplying the SCI frames by the constant
                out[('SCI',extver)].data=np.multiply(inA[('SCI',extver)].data,int)
                
                #multiplying the VAR frames by the constant^2
                out[('VAR',extver)].data=np.multiply(inA[('VAR',extver)].data,int*int)
                
            except:
                raise 'An error occurred while performing an arith task'
    else:
        log.critical('arith.mult() only accepts inputB of types AstroData, list and float, '+str(type(inB))+' passed in', 'critical')    
        raise 'An error occurred while performing an arith task'      
    return out   

def add(inputA, inputB):
    '''
    A function to add a input science image to another image or an floating point integer.
    If the inputB is a AstroData MEF then this function will loop through the SCI, VAR and DQ frames
    to add each SCI of the inputA to the inputB SCI of the same EXTVER. It will apply a 
    bitwise-or to the DQ frames to preserve their binary formats. The VAR frames will be added.
    
    If the inputB is a float integer then only the SCI frames of inputA will each have the float value added 
    (ie. VAR and DQ frames of inputA are left alone).
    
    
    @param inputA: input image to have inputB added to it
    @type inputA: a MEF or single extension fits file in the form of an AstroData instance
    
    @param inputB: inputB to add to the inputA 
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
                    #print 'a100: multiplying SCI frames '+str(extver)
                    #  adding the SCI frames
                    out[('SCI',extver)].data=np.add(inA[('SCI',extver)].data,inB[('SCI',extver)].data)
                    
                    #print 'a104: starting the VAR frame calculations '+str(extver)
                    # creating the output VAR frame following varOut= varA + varB
                    out[('VAR',extver)].data=np.add(inA[('VAR',extver)].data,inB[('VAR',extver)].data)
                   
                    #print 'a41: bitwise_or on DQ frames '+str(extver)
                    # bitwise-or 'adding' DQ frames 
                    out[('DQ',extver)].data=np.bitwise_or(inA[('DQ',extver)].data,inB[('DQ',extver)].data) 
                    
                else:
                    log.critical('arrays are different sizes for SCI extension '+i+' of the input '\
                                 +inA.filename+' and '+inB.filename,'critical')
                    raise 'An error occurred while performing an arith task'
            except:
                raise 'An error occurred while performing an arith task'
    elif type(inB)==float:
        for sci in inA['SCI']:
            extver = sci.extver()
            try:
                #print 'a53: simple addition of SCI frames by the float '+str(inB)
                # adding the SCI frames by the constant
                out[('SCI',extver)].data=np.add(inA[('SCI',extver)].data,inB)
                
            except:
                raise 'An error occurred while performing an arith task'
    else:
        log.critical('arith.add() only accepts inputB of types AstroData and float, '+str(type(inB))+' passed in', 'critical')    
        raise 'An error occurred while performing an arith task'            
    return out     
        
def sub(inputA, inputB):
    '''
    A function to subtract a input science image from another image or a floating point integer.
    If the inputB is a AstroData MEF then this function will loop through the SCI, VAR and DQ frames
    to subtract each SCI of the inputA from the inputB SCI of the same EXTVER. It will apply a 
    bitwise-or to the DQ frames to preserve their binary formats. The VAR frames will be added.
    
    If the inputB is a float integer then only the SCI frames of inputA will each have the float value subtracted 
    (ie. VAR and DQ frames of inputA are left alone).
    
    
    @param inputA: input image to be subtracted by inputB
    @type inputA: a MEF or single extension fits file in the form of an AstroData instance
    
    @param inputB: inputB to subtracted from inputA 
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

                    #print 'a100: subtracting SCI frames '+str(extver)
                    #  subtracting the SCI frames
                    out[('SCI',extver)].data=np.subtract(inA[('SCI',extver)].data,inB[('SCI',extver)].data)
                    
                    #print 'a104: adding the VAR frames '+str(extver)
                    # creating the output VAR frame following varOut= varA + varB
                    out[('VAR',extver)].data=np.add(inA[('VAR',extver)].data,inB[('VAR',extver)].data)
                   
                    #print 'a41: bitwise_or on DQ frames '+str(extver)
                    # bitwise-or 'adding' DQ frames 
                    out[('DQ',extver)].data=np.bitwise_or(inA[('DQ',extver)].data,inB[('DQ',extver)].data) 
                    
                else:
                    log.critical('arrays are different sizes for SCI extension '+i+' of the input '\
                                 +inA.filename+' and '+inB.filename,'critical')
                    raise 'An error occurred while performing an arith task'
            except:
                raise 'An error occurred while performing an arith task'
    elif type(inB)==float:
        for sci in inA['SCI']:
            extver = sci.extver()
            try:
                #print 'a53: simple subtraction of SCI frames by the float '+str(inB)
                # subtracting the SCI frames by the constant
                out[('SCI',extver)].data=np.subtract(inA[('SCI',extver)].data,inB)
                
            except:
                raise 'An error occurred while performing an arith task'
    else:
        log.critical('arith.sub() only accepts inputB of types AstroData and float, '+str(type(inB))+' passed in', 'critical')    
        raise 'An error occurred while performing an arith task'            
    return out 