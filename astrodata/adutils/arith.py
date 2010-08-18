import os
import pyfits as pf
import numpy as np
#from astrodata.adutils import mefutil, paramutil
from astrodata.adutils import gemLog
from astrodata.AstroData import AstroData
import astrodata
log=gemLog.getGeminiLog() 

class ArithExcept:
    def __init__(self, msg="Exception Raised in arith toolbox"):
        self.message = msg
    def __str__(self):
        return self.message

def div(numerator, denominator):
    '''
    A function to divide a input science image by a another image(or flat) or an floating point integer.
    If the denominator is a AstroData MEF then this function will loop through the SCI, VAR and DQ frames
    to divide each SCI of the numerator by the denominator SCI of the same EXTVER. It will apply a 
    bitwise-or to the DQ frames to preserve their binary formats.
    If the denominator is a float integer then only the SCI frames of the numerator are each divided by the float.
    
    @param numerator: input image to be divided by the denominator
    @type numerator: a MEF or single extension fits file in the form of an AstroData instance
    
    @param denominator: denominator to divide the numerator by
    @type denominator: a MEF of SCI, VAR and DQ frames in the form of an AstroData instance, a float 
                        list, a single float (list must be in order of the SCI extension EXTVERs) OR 
                        a dictionary of the format {('SCI',#):##,('SCI',#):##...} where # are the EXTVERs 
                        of the SCI extensions and ## are the corresponding float values to multiply that extension by.
    '''
    num=numerator
    den=denominator 
    from copy import deepcopy
    out=AstroData.prepOutput(inputAry = num, clobber = False)      
    if type(den)==astrodata.AstroData:
        #print 'a30: den is type AstroData'
        for sci in num['SCI']:
            extver = sci.extver()
            outsci = deepcopy(num[('SCI',extver)]) # we assume there are at least SCI extensions in the input
        
            try:
                if num[('SCI',extver)].data.shape==den[('SCI',extver)].data.shape: #making sure arrays are same size/shape
                    #print 'a35: deviding SCI frames '+str(extver)
                    # dividing the SCI frames
                    outsci.data=np.divide(num[('SCI',extver)].data,den[('SCI',extver)].data)
                    out.append(outsci)
                    
                    if num.countExts('VAR')==den.countExts('VAR')==num.countExts('SCI'): # ie there are an equal numbers of VAR frames to operate on
                        outvar = deepcopy(num[('VAR',extver)])
                        
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
                        outvar.data=np.multiply(sciOutSquared,varOverAplusB)
                        out.append(outvar)
                    if num.countExts('DQ')==den.countExts('DQ')==num.countExts('SCI'):  # ie there are an equal number of DQ frames to operate on 
                        outdq = deepcopy(num[('DQ',extver)])
                                                                  
                        #print 'a41: bitwise_or on DQ frames '+str(extver)
                        # bitwise-or 'adding' DQ frames 
                        outdq.data=np.bitwise_or(num[('DQ',extver)].data,den[('DQ',extver)].data)
                        out.append(outdq)
                        
                    else:
                        log.critical('arrays are different sizes for SCI extension '+str(extver)+' of the input '\
                                     +num.filename+' and '+den.filename,'critical')
                        raise ArithExcept('An error occurred while performing an arith task')
                
            except:
                raise ArithExcept('An error occurred while performing an arith task')

    elif type(den)==dict or type(den)==list or type(den)==float:
        # creating the dict if input is a float or float list
        if type(den)==float: 
            denDict={}
            for ext in num['SCI']:
                extver=ext.extver()
                denDict[('SCI',extver)]=den
                print repr(denDict)
        if type(den)==list:    
            denDict={}
            for ext in num['SCI']:
                extver=ext.extver()
                denDict[('SCI',extver)]=den[extver-1]
                print repr(denDict)
        if type(den)==dict:
            denDict=den
        
        for extver in range(1,num.countExts("SCI")+1):
            int=denDict[('SCI',extver)]
            outsci=deepcopy(num[('SCI',extver)]) #$$$ since the dict has the extname we could make this more general??
            try:
                outsci.data=np.divide(num[('SCI',extver)].data,int)  
                out.append(outsci)
                if num.countExts('VAR')==num.countExts('SCI'): # ie there are VAR frames to operate on
                    outvar = deepcopy(num[('VAR',extver)])
                    # multiplying the VAR frames by the constant^2
                    outvar.data=np.multiply(num[('VAR',extver)].data,int*int)
                    out.append(outvar)
                if num.countExts('DQ')==num.countExts('SCI'):  # ie there are DQ frames to operate on 
                    outdq = deepcopy(num[('DQ',extver)])
                    out.append(outdq)
            except:
                raise ArithExcept('An error occurred while performing an arith task')
        
    else:
        log.critical('arith.div() only accepts inputB of types AstroData, list, float or dict, '+str(type(den))+' passed in', 'critical')    
        raise ArithExcept('An error occurred while performing an arith task')            
    return out       
                
def mult(inputA, inputB):
    '''
    A function to multiply a input science image by a another image(or flat) or an floating point integer.
    If the inputB is a AstroData MEF then this function will loop through the SCI, VAR and DQ frames
    to divide each SCI of the inputA by the inputB SCI of the same EXTVER. It will apply a 
    bitwise-or to the DQ frames to preserve their binary formats.
    If the inputB is a float integer then only the SCI frames of the inputA are each divided by the float.
    
    @param inputA: input image to be multiplied by the inputB
    @type inputA: a MEF or single extension fits file in the form of an AstroData instance
    
    @param inputB: inputB to multiply the inputA by
    @type inputB: a MEF of SCI, VAR and DQ frames in the form of an AstroData instance or a float 
                    list or a single float (list must be in order of the SCI extension EXTVERs) OR 
                    a dictionary of the format {('SCI',#):##,('SCI',#):##...} where # are the EXTVERs 
                    of the SCI extensions and ## are the corresponding float values to multiply that extension by.   
    '''
   
    inA=inputA
    inB=inputB 
    from copy import deepcopy
    out=AstroData.prepOutput(inputAry = inA, clobber = False)
    if type(inB)==astrodata.AstroData:
        for sci in inA['SCI']:
            extver = sci.extver()
            outsci = deepcopy(inA[('SCI',extver)]) # we assume there are at least SCI extensions in the input
            try:
                if inA[('SCI',extver)].data.shape==inB[('SCI',extver)].data.shape: #making sure arrays are same size/shape
                    #print 'a100: multiplying SCI frames '+str(extver)
                    #  multipling the SCI frames
                    outsci.data=np.multiply(inA[('SCI',extver)].data,inB[('SCI',extver)].data)
                    out.append(outsci)
                    
                    if inA.countExts('VAR')==inB.countExts('VAR')==inA.countExts('SCI'): # ie there are an equal numbers of VAR frames to operate on
                        outvar = deepcopy(inA[('VAR',extver)])
                        
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
                        outvar.data=np.multiply(sciOutSquared,varOverAplusB)
                        out.append(outvar)
                    
                    if inA.countExts('DQ')==inB.countExts('DQ')==inA.countExts('SCI'):  # ie there are an equal number of DQ frames to operate on 
                        outdq = deepcopy(inA[('DQ',extver)])    
                        #print 'a41: bitwise_or on DQ frames '+str(extver)
                        # bitwise-or 'adding' DQ frames 
                        outdq.data=np.bitwise_or(inA[('DQ',extver)].data,inB[('DQ',extver)].data) 
                        out.append(outdq)
                else:
                    log.critical('arrays are different sizes for SCI extension '+i+' of the input '\
                                 +inA.filename+' and '+inB.filename,'critical')
                    raise ArithExcept('An error occurred while performing an arith task')
            except:
                raise ArithExcept('An error occurred while performing an arith task')

    elif type(inB)==dict or type(inB)==list or type(inB)==float:
        # creating the dict if input is a float or float list
        if type(inB)==float: 
            inBDict={}
            for ext in inA['SCI']:
                extver=ext.extver()
                inBDict[('SCI',extver)]=inB
                #print repr(inBDict)
        if type(inB)==list:    
            inBDict={}
            for ext in inA['SCI']:
                extver=ext.extver()
                inBDict[('SCI',extver)]=inB[extver-1]
                #print repr(inBDict)
        if type(inB)==dict:
            inBDict=inB
        
        for extver in range(1,inA.countExts("SCI")+1):
            int=inBDict[('SCI',extver)]
            outsci=deepcopy(inA[('SCI',extver)]) #$$$ since the dict has the extname we could make this more general??
            try:
                outsci.data=np.multiply(inA[('SCI',extver)].data,int)  
                out.append(outsci)
                if inA.countExts('VAR')==inA.countExts('SCI'): # ie there are VAR frames to operate on
                    outvar = deepcopy(inA[('VAR',extver)])
                    # multiplying the VAR frames by the constant^2
                    outvar.data=np.multiply(inA[('VAR',extver)].data,int*int)
                    out.append(outvar)
                if inA.countExts('DQ')==inA.countExts('SCI'):  # ie there are DQ frames to operate on 
                    outdq = deepcopy(inA[('DQ',extver)])
                    out.append(outdq)
            except:
                raise ArithExcept('An error occurred while performing an arith task')
    
    else:
        log.critical('arith.mult() only accepts inputB of types AstroData, list and float, '+str(type(inB))+' passed in', 'critical')    
        raise ArithExcept('An error occurred while performing an arith task')      
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
    out=AstroData.prepOutput(inputAry = inA, clobber = False)
    if type(inB)==astrodata.AstroData:
        for sci in inA['SCI']:
            extver = sci.extver()
            outsci = deepcopy(inA[('SCI',extver)]) # we assume there are at least SCI extensions in the input
            try:
                if inA[('SCI',extver)].data.shape==inB[('SCI',extver)].data.shape: #making sure arrays are same size/shape
                    #print 'a100: multiplying SCI frames '+str(extver)
                    #  adding the SCI frames
                    outsci.data=np.add(inA[('SCI',extver)].data,inB[('SCI',extver)].data)
                    out.append(outsci)
                    if inA.countExts('VAR')==inB.countExts('VAR')==inA.countExts('SCI'): # ie there are an equal numbers of VAR frames to operate on
                        outvar = deepcopy(inA[('VAR',extver)])
                        #print 'a104: starting the VAR frame calculations '+str(extver)
                        # creating the output VAR frame following varOut= varA + varB
                        outvar.data=np.add(inA[('VAR',extver)].data,inB[('VAR',extver)].data)
                        out.append(outvar)
                    if inA.countExts('DQ')==inB.countExts('DQ')==inA.countExts('SCI'):  # ie there are an equal number of DQ frames to operate on
                        outdq = deepcopy(inA[('DQ',extver)])   
                        #print 'a41: bitwise_or on DQ frames '+str(extver)
                        # bitwise-or 'adding' DQ frames 
                        outdq.data=np.bitwise_or(inA[('DQ',extver)].data,inB[('DQ',extver)].data) 
                        out.append(outdq)
                else:
                    log.critical('arrays are different sizes for SCI extension '+i+' of the input '\
                                 +inA.filename+' and '+inB.filename,'critical')
                    raise ArithExcept('An error occurred while performing an arith task')
            except:
                raise ArithExcept('An error occurred while performing an arith task')
    elif type(inB)==float:
        for sci in inA['SCI']:
            extver = sci.extver()
            outsci = deepcopy(inA[('SCI',extver)]) # we assume there are at least SCI extensions in the input
            try:
                #print 'a53: simple addition of SCI frames by the float '+str(inB)
                # adding the SCI frames by the constant
                outsci.data=np.add(inA[('SCI',extver)].data,inB)
                out.append(outsci)
                # adding the inputA VAR and DQ frames un-edited to the outputs
                if inA.countExts('VAR')==inA.countExts('SCI'): # ie there are VAR frames to operate on
                    outvar = deepcopy(inA[('VAR',extver)])
                    out.append(outvar)
                if inA.countExts('DQ')==inA.countExts('SCI'):  # ie there are DQ frames to operate on 
                    outdq = deepcopy(inA[('DQ',extver)])   
                    out.append(outdq)
            except:
                raise ArithExcept('An error occurred while performing an arith task')
    else:
        log.critical('arith.add() only accepts inputB of types AstroData and float, '+str(type(inB))+' passed in', 'critical')    
        raise ArithExcept('An error occurred while performing an arith task')            
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
    out=AstroData.prepOutput(inputAry = inA, clobber = False)
    if type(inB)==astrodata.AstroData:
        for sci in inA['SCI']:
            extver = sci.extver()  
            outsci = deepcopy(inA[('SCI',extver)]) # we assume there are at least SCI extensions in the input          
            try:
                if inA[('SCI',extver)].data.shape==inB[('SCI',extver)].data.shape: #making sure arrays are same size/shape

                    #print 'a100: subtracting SCI frames '+str(extver)
                    #  subtracting the SCI frames
                    outsci.data=np.subtract(inA[('SCI',extver)].data,inB[('SCI',extver)].data)
                    out.append(outsci)
                    
                    if inA.countExts('VAR')==inB.countExts('VAR')==inA.countExts('SCI'): # ie there are an equal numbers of VAR frames to operate on
                        outvar = deepcopy(inA[('VAR',extver)])
                        #print 'a104: adding the VAR frames '+str(extver)
                        # creating the output VAR frame following varOut= varA + varB
                        outvar.data=np.add(inA[('VAR',extver)].data,inB[('VAR',extver)].data)
                        out.append(outvar)
                    if inA.countExts('DQ')==inB.countExts('DQ')==inA.countExts('SCI'):  # ie there are an equal number of DQ frames to operate on
                        outdq = deepcopy(inA[('DQ',extver)])       
                        #print 'a41: bitwise_or on DQ frames '+str(extver)
                        # bitwise-or 'adding' DQ frames 
                        outdq.data=np.bitwise_or(inA[('DQ',extver)].data,inB[('DQ',extver)].data) 
                        out.append(outdq)
                else:
                    log.critical('arrays are different sizes for SCI extension '+i+' of the input '\
                                 +inA.filename+' and '+inB.filename,'critical')
                    raise ArithExcept('An error occurred while performing an arith task')
            except:
                raise ArithExcept('An error occurred while performing an arith task')
    elif type(inB)==float:
        for sci in inA['SCI']:
            extver = sci.extver()
            outsci = deepcopy(inA[('SCI',extver)]) # we assume there are at least SCI extensions in the input
            try:
                #print 'a53: simple subtraction of SCI frames by the float '+str(inB)
                # subtracting the SCI frames by the constant
                outsci.data=np.subtract(inA[('SCI',extver)].data,inB)
                out.append(outsci)
                
                # adding the inputA VAR and DQ frames un-edited to the outputs
                if inA.countExts('VAR')==inA.countExts('SCI'): # ie there are VAR frames to operate on
                    outvar = deepcopy(inA[('VAR',extver)])
                    out.append(outvar)
                if inA.countExts('DQ')==inA.countExts('SCI'):  # ie there are DQ frames to operate on 
                    outdq = deepcopy(inA[('DQ',extver)])   
                    out.append(outdq)
            except:
                raise ArithExcept('An error occurred while performing an arith task')
    else:
        log.critical('arith.sub() only accepts inputB of types AstroData and float, '+str(type(inB))+' passed in', 'critical')    
        raise ArithExcept('An error occurred while performing an arith task')            
    return out 