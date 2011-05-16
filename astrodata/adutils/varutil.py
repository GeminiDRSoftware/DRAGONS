# Author: Kyle Mede, April 2011
# This module provides functions that perform variance frame calculations
# and the creation of their headers on AstroData objects

import os

import pyfits as pf
import numpy as np
import astrodata
from astrodata.AstroData import AstroData
from astrodata.Errors import ArithError

def varianceArrayCalculator(sciExtA=None, sciExtB=None, constB=None,  
                            sciOut=None, varExtA=None, varExtB=None, div=False, 
                                            mult=False, sub=False, add=False):
    """    
    This function will update a currently existing variance plane due to a 
    mathematical operation performed on the science data.    
    
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    For multiplications and division operations:
    --------------------------------------------
    If sciExtB is an AstroData instance:
    varOut=sciOut^2 * ( varA/(sciA^2) + varB/(sciB^2) )
    
    Else:
    varOut=varA * constB^2
    
    For addition and subtraction operations:
    ----------------------------------------
    If sciExtB is an AstroData instance:
    varOut= varA + varB
    
    Else:
    varOut=varA 
    variance is not affected if the science data is just consistently raised
    or lowered by a constant.
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    The A input MUST have a varExtA defined!
    
    If both A and B inputs are AstroData's then they must BOTH have varExt's.
    
    Only ONE mathematical operation can be performed at a time!!
    
    :param sciExtA: science extension of an AstroData instance being multiplied
                    or divided by sciExtB
    :type sciExtA: AstroData single extension. ex. sciExtA = adA['SCI',#]
    
    :param sciExtB: science extension of an AstroData instance being multiplied
                    by sciExtA OR dividing sciExtA by
    :type sciExtB: AstroData single extension.ex. sciExtB = adB['SCI',#]
    
    :param sciOut: science extension of an output AstroData instance
                   ie. the science frame resulting from the operation.
    :type sciOut: AstroData single extension.ex. sciOut = adOut['SCI',#]
    
    :param constB: constant multiplying sciExtA by
    :tyep constB: float
    
    :param varExtA: variance extension of an AstroData instance being multiplied
                    or divided by extB
    :type varExtA: AstroData single extension. ex. varExtA = adA['VAR',#]
    
    :param varExtB: variance extension of an AstroData instance being multiplied
                    or divided by extB. 
    :type varExtB: AstroData single extension. ex. varExtB = adA['VAR',#]
    
    :param mult: was a multiplication performed between A and B?
    :type mult: Python boolean (True/False)
    
    :param div: was a division performed between A and B?
    :type div: Python boolean (True/False)
    
    :param add: was an addition performed between A and B?
    :type add: Python boolean (True/False)
    
    :param sub: was a subtraction performed between A and B?
    :type sub: Python boolean (True/False)
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
        if (sciExtB is not None) and (constB is not None):
            raise ArithError('sciExtB and constB cannot both be defined')
        
        ## Perform all checks and math needed for mult and div cases
        if mult or div:
            # Checking all the inputs are AstroData's then grabbing their 
            # data arrays
            if sciExtA is not None:
                if isinstance(sciExtA, astrodata.AstroData) or \
                            isinstance(sciExtA, astrodata.AstroData.AstroData):
                    sciA = sciExtA.data
                    if isinstance(varExtA, astrodata.AstroData) or \
                            isinstance(varExtA, astrodata.AstroData.AstroData):
                        varA = varExtA.data
                    else:
                        raise ArithError('varExtA must be an AstroData '+
                                         'instances')
            if sciExtB is not None:
                if isinstance(sciExtB, astrodata.AstroData) or \
                            isinstance(sciExtB, astrodata.AstroData.AstroData):
                    sciB = sciExtB.data
                    if isinstance(varExtB, astrodata.AstroData) or \
                            isinstance(varExtB, astrodata.AstroData.AstroData):
                        varB = varExtB.data
                    else:
                        raise ArithError('varExtB must be an AstroData '+
                                         'instances')
            elif isinstance(constB,float):
                pass
            else:
                raise ArithError('Either sciExtB must be an AstroData '+
                                 'instances OR constB is a float, neither '+
                                 'case satisfied.')
            # calculate the output variance array if mult or div
            if constB is None:
                # Science was multiplied/divided by an array so follow:
                # varOut=sciOut^2 * ( varA/(sciA^2) + varB/(sciB^2) )
                
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
            else:
                # Science was multiplied/divided by constant so follow:
                # varOut = varA * constB^2
                varOut = np.multiply(varA,constB*constB)
                
        ## Perform all checks and math needed for add and sub cases        
        elif add or sub:
            if varExtA is not None:
                if isinstance(varExtA, astrodata.AstroData) or \
                            isinstance(varExtA, astrodata.AstroData.AstroData):
                            varA = varExtA.data
            else:
                raise ArithError('varExtA must be not be None')
            if varExtB is not None:
                if isinstance(varExtB, astrodata.AstroData) or \
                            isinstance(varExtB, astrodata.AstroData.AstroData):
                            varB = varExtB.data
                else:
                    raise ArithError('varExtB must be an AstroData instances')
                # calculate the output variance array varOut=varA+varB
                varOut = np.add(varA, varB)
            else:
                # No second variance array defined so just pass the first 
                # through to the output
                varOut = varA

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
        readNoise=sciExt.read_noise().as_pytype()
        gain = sciExt.gain().as_pytype()
        # Creating (read noise/gain) constant
        rnOverG=readNoise/gain
        # Convert negative numbers (if they exist) to zeros
        maxArray=np.where(sciExt.data>0.0,sciExt.data,0.0)
        #maxArray=sciExt.data[np.where(sciExt.data>0.0,0)]
        # Creating max(data,0.0)/gain array
        maxOverGain=np.divide(maxArray,gain)
        # Putting it all together
        varArray=np.add(maxOverGain,rnOverG*rnOverG)
    
        # return calculated variance numpy array
        return varArray
    except:
        raise

def createInitialVarianceHeader(extver=None,shape=None):
    """
    This function creates a variance pyfits header object and loads it up with 
    the standard header keys.
    Since these are basically the same for all initial variance frames, 
    excluding the EXTVER key, this func is very simple :-)
    
    NOTE: maybe this function should be further generalized in the future to 
    handle variance frames with more than two axes? IFU VAR frames?
    
    :param extver: extension version to put into the EXTVER header key matching
                   the SCI extension's EXTVER
    :type extver: int
    
    :param shape: shape of data array for this VAR extension. 
                  Can be found using ad['VAR',extver].data.shape
    :type shape: tuple. ex. (2304,1024), , ie(number of rows, number of columns)
    """    
    # Creating the variance frame's header with pyfits and updating it     
    varheader = pf.Header()
    varheader.update('XTENSION','IMAGE','IMAGE extension')
    varheader.update('BITPIX', -32,'number of bits per data pixel')
    varheader.update('NAXIS', 2)
    varheader.update('NAXIS1',shape[1],'length of data axis 1')
    varheader.update('NAXIS2',shape[0],'length of data axis 2')
    varheader.update('PCOUNT', 0, 'required keyword; must = 0')
    varheader.update('GCOUNT', 1, 'required keyword; must = 1')
    varheader.update('EXTNAME', 'VAR', 'Extension Name')
    varheader.update('EXTVER', extver, 'Extension Version')
    
    return varheader