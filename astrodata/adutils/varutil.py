#Author: Kyle Mede, Aug 2010
# This module provides functions that perform numpy operations on astrodata
# objects

import os

import pyfits as pf
import numpy as np
import astrodata
from astrodata.AstroData import AstroData
from astrodata.Errors import ArithError

def varianceArrayCalculator(sciExtA=None, sciExtB=None, constB=None, 
                            varExtA=None, varExtB=None, div=False, mult=False,
                                                        sub=False, add=False):
    """
    ####### make this handle float value for the sciExtB argument and follow
    ####### outvar = varA*int*int
    
    The A input MUST have a varExtA defined!
    If both A and B inputs are AstroData's then they must BOTH have varExt's.
    #######
    
    For multiplications and division operations it boils down to:
    if mult: sciOut = sciA * sciB 
    for div: sciOut = sciA / sciB 
    # varOut=sciOut^2 * ( varA/(sciA^2) + varB/(sciB^2) )
    
    for addition and subtraction operations:
    # varOut= varA + varB
    
    Only ONE mathematical operation can be performed at a time!!
    
    :param sciExtA: science extension of an AstroData instance being multiplied
                    or divided by sciExtB
    :type sciExtA: AstroData single extension. ex. sciExtA = adA['SCI',#]
    
    :param sciExtB: science extension of an AstroData instance being multiplied
                    by sciExtA OR dividing sciExtA by
    :type sciExtB: AstroData single extension.ex. sciExtB = adB['SCI',#]
    
    :param constB: constant multiplying sciExtA by
    :tyep constB: float
    
    :param varExtA: variance extension of an AstroData instance being multiplied
                    or divided by extB
    :type varExtA: AstroData single extension. ex. varExtA = adA['VAR',#]
    
    :param varExtB: variance extension of an AstroData instance being multiplied
                    or divided by extB
    :type varExtB: AstroData single extension. ex. varExtB = adA['VAR',#]
    """
    try:
        # checking if it more than one math operation is set True
        yup = False
        ops = [mult,div,sub,add]
        for op in ops:
            if op:
                if yup:
                    raise ArithError('only ONE math operation can be True')
                else:
                    yup = True
            
        # Checking all the inputs are AstroData's then grabbing their data arrays
        if isinstance(sciExtA, astrodata.AstroData) or \
                            isinstance(sciExtA, astrodata.AstroData.AstroData):
            sciA = sciExtA.data
            if isinstance(varExtA, astrodata.AstroData) or \
                            isinstance(varExtA, astrodata.AstroData.AstroData):
                varA = varExtA.data
            else:
                raise ArithError('varExtA must be an AstroData instances')
        if isinstance(sciExtB, astrodata.AstroData) or \
                            isinstance(sciExtB, astrodata.AstroData.AstroData):
            sciB = sciExtB.data
            if isinstance(varExtB, astrodata.AstroData) or \
                            isinstance(varExtB, astrodata.AstroData.AstroData):
                varB = varExtB.data
            else:
                raise ArithError('varExtB must be an AstroData instances')
        elif isinstance(constB,float):
            pass
        else:
            raise ArithError('Either sciExtB must be an AstroData instances '+
                             'OR constB is a float, neither case satisfied.')
        
        # preparing the sciOut array if performing a mult or div
        if mult:
            sciOut = np.multiply(sciA,sciB)
        if div:
            sciOut = np.divide(sciA,sciB)
        if mult or div:
            sciOutSquared = np.multiply(sciOut, sciOut)
            sciAsquared = np.multiply(sciA, sciA)
            sciBsquared = np.multiply(sciB, sciB)
            # Now varA/sciAsquared and varB/sciBsquared
            varAoverSciASquared = np.divide(varA, sciAsquared)
            varBoverSciBSquared = np.divide(varB, sciBsquared)
            # Now varAoverSciASquared + varBoverSciBSquared
            varOverAplusB = np.add(varAoverSciASquared, varBoverSciBSquared) 
            # Put it all together 
            # varOut=sciOut^2 * ( varA/(sciA^2) + varB/(sciB^2) )
            varOut = np.multiply(sciOutSquared,varOverAplusB)
        if sub or add:
            varOut = np.add(varA, varB)
       
        # return final variance data array
        return varOut
    except:
        
        raise 

def calculateInitialVarianceArray(sciExt=None):
    """
    This function uses numpy to calculate the variance of a SCI extension
    of an AstroData instance.
    
    The calculation will follow the formula:
    variance = (read noise/gain)2 + max(data,0.0)/gain
    
    returns variance as a numpy array.
    
    This array should be put into the .data part of the variance extension 
    of the astrodata instance this array is being calculated with matching
    EXTVER of the SCI extension it was calculated for.
    
    :param sciExt: science extension of an astrodata instance
    :type sciExt: Astrodata single extension
    """
    try:
        # var = (read noise/gain)**2 + max(data,0.0)/gain
                        
        # Retrieving necessary values (read noise, gain)
        readNoise=sciExt.read_noise()
        gain = sciExt.gain().asPytype()
        # Creating (read noise/gain) constant
        rnOverG=readNoise/gain
        # Convert negative numbers (if they exist) to zeros
        maxArray=np.where(sciExt.data>0.0,0,sciExt.data)
        #maxArray=sciExt.data[np.where(sciExt.data>0.0,0)]
        # Creating max(data,0.0)/gain array
        maxOverGain=np.divide(maxArray,gain)
        # Putting it all together
        varArray=np.add(maxOverGain,rnOverG*rnOverG)
    
        # return calculated variance numpy array
        return varArray
    except:
        raise

def createInitialVarianceHeader(extver):
    """
    This function creates a variance pyfits header object and loads it up with 
    the standard header keys.
    Since these are basically the same for all initial variance frames, 
    excluding the EXTVER key, this func is very simple :-)
    
    :param extver: extension version to put into the EXTVER header key matching
                   the SCI extension's EXTVER
    :type extver: int
    """    
    # Creating the variance frame's header with pyfits and updating it     
    varheader = pf.Header()
    varheader.update('NAXIS', 2)
    varheader.update('PCOUNT', 0, 'required keyword; must = 0')
    varheader.update('GCOUNT', 1, 'required keyword; must = 1')
    varheader.update('EXTNAME', 'VAR', 'Extension Name')
    varheader.update('EXTVER', extver, 
                     'Extension Version')
    varheader.update('BITPIX', -32,
                     'number of bits per data pixel')
    
    return varheader