# This module contains functions that perform numpy operations on the input
# dataset

import numpy as np
from copy import deepcopy
import astrodata
from astrodata.AstroData import AstroData
from astrodata.Errors import ArithError

def div(numerator, denominator):
    """
    The div function divides an AstroData object (numerator) by another
    AstroData object or a single value (denominator). If inputB is an AstroData
    object, the mult function will multiple each science, variance and data
    quality extension in the inputA AstroData object with the corresponding
    science, variance and data quality extension in the inputB AstroData
    object.
    
    The VAR frames output will follow:
    
    If denominator is an AstroData instance:
    varOut=sciOut^2 * ( varA/(sciA^2) + varB/(sciB^2) ), 
    where A=numerator and B=denominator frames.
    Else:
    varOut=varA * float^2
            
    If the denominator is a float integer then only the SCI frames of the 
    numerator are each divided by the float.
    
    @param numerator: input image to be divided by the denominator
    @type numerator: a MEF or single extension fits file in the form of an 
                     AstroData instance
    
    @param denominator: denominator to divide the numerator by
    @type denominator: a MEF of SCI, VAR and DQ frames in the form of an 
                       AstroData instance, a float list, a single float 
                       (list must be in order of the SCI extension EXTVERs) OR
                       a dictionary of the format 
                       {('SCI',#):##,('SCI',#):##...} where # are the EXTVERs 
                       of the SCI extensions and ## are the corresponding 
                       float values to divide that extension by.
    """
    # Check to see if the denominator is a dictionary, list, float or int
    if isinstance(denominator, dict) or isinstance(denominator, list) or \
       isinstance(denominator, float) or isinstance(denominator, int):
        # Create a dictionary of identical values for each extension if the 
        # input is a single float
        if isinstance(denominator, float) or isinstance(denominator, int): 
            denominator_dict = {}
            for ext in numerator["SCI"]:
                # Retrieve the EXTVER for this extension
                extver = ext.extver()
                # Add element to the dictionary for this extension
                denominator_dict[("SCI", extver)] = denominator
        # Create a dictionary if the input is a list of values
        if isinstance(denominator, list):
            denominator_dict = {}
            for ext in numerator["SCI"]:
                extver = ext.extver()
                denominator_dict[("SCI", extver)] = denominator[extver-1]
        # Just rename the variable if denominator is already a dictionary
        if isinstance(denominator, dict):
            denominator_dict = denominator
        
        # The denominator is now stored in a dictionary
        for sci in numerator["SCI"]:
            # Retrieve the extension version for this extension
            extver = sci.extver()
            # Retrieving the denominator for this extension from the dictionary
            denominator = denominator_dict[("SCI", extver)]
            # Divide the science extension by the denominator
            sci.data = np.divide(numerator[("SCI", extver)].data, denominator)
            # Update the variance extension if it exists in the numerator
            if numerator["VAR", extver]:
                # Multiplying the variance extension by the denominator^2
                numerator["VAR", extver].data = np.multiply(
                    numerator[("VAR", extver)].data, denominator*denominator)
            
    # Check to see if the denominator is of type astrodata.AstroData.AstroData
    elif isinstance(denominator, astrodata.AstroData.AstroData):
        # Loop over each science extension in the input AstroData object
        for sci in numerator["SCI"]:
            # Retrieving the version of this extension
            extver = sci.extver()
            # Make sure arrays are same size/shape
            if numerator[("SCI", extver)].data.shape != \
               denominator[("SCI", extver)].data.shape:
                raise Errors.Error("The input science extensions %s and %s " \
                                   "are not the same size" \
                                   % (numerator[("SCI", extver)],
                                      denominator[("SCI", extver)]))
            # Divide the science extension in the numerator by the
            # science extension in the denominator
            numerator[("SCI", extver)].data = np.divide(
                numerator[("SCI", extver)].data,
                denominator[("SCI", extver)].data)
            # Update the variance extension in the numerator if a variance
            # extension exists in the denominator
            if numerator["VAR", extver] and denominator["VAR", extver]:
                # Creating the output VAR frame following 
                # varOut=sciOut^2 * ( varA/(sciA^2) + varB/(sciB^2) )
                # using the varianceArrayCalculator() function
                numerator["VAR", extver].data = \
                varutil.varianceArrayCalculator(
                    sciExtA=numerator["SCI",extver],
                    sciExtB=denominator["SCI",extver],
                    varExtA=numerator["VAR",extver],
                    varExtB=denominator["VAR",extver], div=True)
            # Update the data quality extension in the numeratory if a data
            # quality extension exists in the denominator
            if numerator["DQ", extver] and denominator["DQ", extver]:
                numerator["DQ", extver].data = np.bitwise_or(
                    numerator[("DQ", extver)].data,
                    denominator[("DQ", extver)].data)
    
    # If the input was not of type astrodata, float, float list or dictionary
    # then raise an exception
    else:
        raise 
    # Return the fully updated output astrodata object
    return numerator
                
def mult(input1, input2):
    """
    The mult function multiplies an AstroData object (inputA) by another
    AstroData object or a single value (input2). If input2 is an AstroData
    object, the mult function will multiple each science, variance and data
    quality extension in the input1 AstroData object with the corresponding
    science, variance and data quality extension in the input2 AstroData
    object.
    
    The VAR frames output will follow:
    
    If input2 is an AstroData instance:
    varOut=sciOut^2 * ( var1/(sci1^2) + var2/(sci2^2) ), 
    Else:
    varOut=var1 * float^2
    
    If input2 is a single value, only the science extensions of the input1
    AstroData object are multiplied by the single value.
    
    @param input1: input image to be multiplied by the input2
    @type input1: a MEF or single extension fits file in the form of an 
                  AstroData instance
    
    @param input2: input to multiply the input1 by
    @type input2: a MEF of SCI, VAR and DQ frames in the form of an AstroData
                  instance, a float list or a single float (list must be 
                  in order of the SCI extension EXTVERs) OR a dictionary 
                  of the format {('SCI',#):##,('SCI',#):##...} 
                  where # are the EXTVERs of the SCI extensions 
                  and ## are the corresponding float values 
                  to multiply that extension by.
    """
    # Check to see if input2 is a dictionary, list, float or int
    if isinstance(input2, dict) or isinstance(input2, list) or \
       isinstance(input2, float) or isinstance(input2, int):
        # Create a dictionary of identical values for each extension if the 
        # input is a single float
        if isinstance(input2, float) or isinstance(input2, int):
            input2_dict = {}
            for ext in input1["SCI"]:
                # Retrieve the EXTVER for this extension
                extver = ext.extver()
                # Add element to the dictionary for this extension
                input2_dict[("SCI", extver)] = input2
        # Create a dictionary if the input is a list of values
        if isinstance(input2, list):
            input2_dict = {}
            for ext in input1["SCI"]:
                extver = ext.extver()
                input2_dict[("SCI", extver)] = input2[extver-1]
        # Just rename the variable if input2 is already a dictionary
        if isinstance(input2, dict):
            input2_dict = input2
        
        # input2 is now stored in a dictionary
        for sci in input1["SCI"]:
            # Retrieve the extension version for this extension
            extver = sci.extver()
            # Retrieving the input2 value for this extension from the
            # dictionary
            input2 = input2_dict[("SCI", extver)]
            # Multiply the science extension by the input2 value
            sci.data = np.multiply(sci.data, input2)
            # Update the variance extension if it exists in the numerator
            if input1["VAR", extver]:
                # Multiplying the variance extension by the input2^2
                input1["VAR", extver].data = np.multiply(
                    input1[("VAR", extver)].data, input2*input2)
    
    # Check to see if input2 is of type astrodata.AstroData.AstroData
    elif isinstance(input2, astrodata.AstroData.AstroData):
        # Loop over each science extension in the input AstroData object
        for sci in input1["SCI"]:
            # Retrieving the version of this extension
            extver = sci.extver()
            # Make sure arrays are same size/shape
            if input1[("SCI", extver)].data.shape == \
               input2[("SCI", extver)].data.shape: 
                # Multiply the science extensions together
                sci.data = np.multiply(input1[("SCI", extver)].data,
                                       input2[("SCI", extver)].data)
                # Update the variance extension in input1 if a variance
                # extension exists in input2
                if input1["VAR", extver] and input2["VAR", extver]:
                    # Creating the output VAR frame following 
                    # varOut=sciOut^2 * ( var1/(sci1^2) + var2/(sci2^2) )
                    # using the varianceArrayCalculator() function
                    input1["VAR", extver].data = \
                                  varutil.varianceArrayCalculator(
                        sciExtA=input1["SCI",extver],
                        sciExtB=input2["SCI",extver], sciOut=outsci,
                        varExtA=input1["VAR",extver],
                        varExtB=input2["VAR",extver], mult=True)
                # Update the data quality extension in input1 if a data quality
                # extension exists in input2
                if input1["DQ", extver] and input2["DQ", extver]:
                    input1["DQ", extver].data = np.bitwise_or(
                        input1[("DQ", extver)].data,
                        input2[("DQ", extver)].data)
    
    # If the input was not of type astrodata, float, float list or dictionary
    # then raise an exception
    else:
        raise
    # Return the fully updated output astrodata object
    return input1

def add(inputA, inputB):
    """
    A function to add a input science image to another image or a floating 
    point integer. If inputB is an AstroData MEF then this function will 
    loop through the SCI, VAR and DQ frames to add each SCI of the inputA 
    to the inputB SCI of the same EXTVER. It will apply a bitwise-or to the DQ
    frames to preserve their binary formats. 
    The VAR frames output will follow:
    
    If inputB is an AstroData instance:
    varOut= varA + varB
    Else:
    varOut=varA 
    
    If the inputB is a float integer then only the SCI frames of inputA will 
    each have the float value added, while the VAR and DQ frames of inputA 
    are left alone.
    #$$$$$$$$$$ ARE WE SURE WE DON'T WANT TO OFFER THE ABILITY FOR inputB 
    TO BE A FLOAT LIST OR DICT???????
    
    @param inputA: input image to have inputB added to it
    @type inputA: a MEF or single extension fits file in the form of an 
                  AstroData instance
    
    @param inputB: input to add to the inputA 
    @type inputB: a MEF of SCI, VAR and DQ frames in the form of an AstroData 
                  instance or a float integer 
    #$$$$$ OR A SINGLE EXTENSION FITS FILE TOO???
     
    """
    # Rename inputs to shorter names to save typing
    inA = inputA
    inB = inputB 
    # Preparing the output astrodata instance
    out = AstroData.prep_output(input_ary=inA, clobber=False)
    
    # Check if inputB is of type float, if so, perform the float specific
    # addition calculations     
    if isinstance(inB, float):
        # Loop through the SCI extensions of InputA
        for sci in inA['SCI']:
            # Retrieve the EXTVER for this extension
            extver = sci.extver()
            # Start with the out SCI HDU being the current 
            # we assume there are at least SCI extensions in the input
            outsci = deepcopy(inA[('SCI', extver)]) 
            
            try:
                # Adding the SCI frames by the constant
                outsci.data = np.add(inA[('SCI', extver)].data,inB)
                # Append updated SCI extension to the output 
                out.append(outsci)
                # Appending the inputA VAR and DQ frames un-edited to the output
                # ie no change, just propagate the frames
                
                # Check there are VAR frames to propagate
                if inA.count_exts('VAR') == inA.count_exts('SCI'): 
                    # Start with the out VAR HDU being the current
                    outvar = deepcopy(inA[('VAR', extver)])
                    # Just propagate VAR frames to the output
                    out.append(outvar) 
                # Check there are DQ frames to propagate   
                if inA.count_exts('DQ') == inA.count_exts('SCI'): 
                    # Start with the out DQ HDU being the current 
                    outdq = deepcopy(inA[('DQ', extver)])   
                    # Just propagate DQ frames to the output
                    out.append(outdq) 
            except:
                raise
    # Check to see if the denominator is of type astrodata.AstroData.AstroData
    elif isinstance(inB, astrodata.AstroData.AstroData):
        # Loop through the SCI extensions
        for sci in inA['SCI']:
            # Retrieving the version of this extension
            extver = sci.extver()
            # Start with the out SCI HDU being the current, 
            # we assume there are at least SCI extensions in the input
            outsci = deepcopy(inA[('SCI', extver)]) 
            
            try:
                # Making sure arrays are same size/shape
                if inA[('SCI', extver)].data.shape == \
                inB[('SCI', extver)].data.shape: 
                    #  Adding the SCI frames of the inputs
                    outsci.data = np.add(inA[('SCI', extver)].data, 
                                         inB[('SCI', extver)].data)
                    # Appending the updated SCI frame to the output
                    out.append(outsci)
                    
                    # Check there are an equal numbers of VAR and SCI frames to 
                    # operate on in both the inputs
                    if inA.count_exts('VAR') == inB.count_exts('VAR') == \
                    inA.count_exts('SCI'): 
                        # Start with the out VAR HDU being the current 
                        outvar = deepcopy(inA[('VAR', extver)])
                        
                        # Creating the output VAR frame following 
                        # varOut= varA + varB
                        outvar.data = np.add(inA[('VAR', extver)].data, 
                                           inB[('VAR', extver)].data)
                        # Append the updated out VAR frame to the output
                        out.append(outvar)
                        
                    # Check there are an equal number of DQ frames to operate on
                    if inA.count_exts('DQ') == inB.count_exts('DQ') == \
                    inA.count_exts('SCI'):  
                        outdq = deepcopy(inA[('DQ', extver)])   
                        # Performing bitwise-or 'adding' DQ frames 
                        outdq.data = np.bitwise_or(inA[('DQ', extver)].data, 
                                                   inB[('DQ', extver)].data)
                        # Append the updated out DQ frame to the output  
                        out.append(outdq)
                
                # If arrays are different sizes then raise an exception
                else:
                    raise ArithError('different numbers of SCI, VAR extensions')
            except:
                raise 
     
    # If the input was not of type astrodata or float, raise an exception
    else:
        raise             
    # Return the fully updated output astrodata object 
    return out     
        
def sub(inputA, inputB):
    """
    A function to subtract a input science image from another image or a 
    floating point integer. If inputB is an AstroData MEF then this function 
    will loop through the SCI, VAR and DQ frames to subtract each SCI of the 
    inputB from the inputA SCI of the same EXTVER. It will apply a bitwise-or 
    to the DQ frames to preserve their binary formats.
    The VAR frames output will follow:
    
    If inputB is an AstroData instance:
    varOut= varA + varB
    Else:
    varOut=varA 
    
    If the inputB is a float integer then only the SCI frames of inputA will 
    each have the float value subtracted while the VAR and DQ frames of 
    inputA are left alone.
    #$$$$$$$$$$ ARE WE SURE WE DON'T WANT TO OFFER THE ABILITY FOR inputB 
    TO BE A FLOAT LIST OR DICT???????
    
    @param inputA: input image to be subtracted by inputB
    @type inputA: a MEF or single extension fits file in the form of an 
                  AstroData instance
    
    @param inputB: inputB to subtracted from inputA 
    @type inputB: a MEF of SCI, VAR and DQ frames in the form of an AstroData 
                  instance or a float int 
    #$$$$$ OR A SINGLE EXTENSION FITS FILE TOO???
    
    """
    # Rename inputs to shorter names to save typing
    inA = inputA
    inB = inputB 
    # Preparing the output astrodata instance
    out=AstroData.prep_output(input_ary = inA, clobber = False)
    
    # Check if inputB is of type float, if so, perform the float specific
    # addition calculations
    if isinstance(inB, float):
        # Loop through the SCI extensions of InputA
        for sci in inA['SCI']:
            # Retrieve the EXTVER for this extension
            extver = sci.extver()
            # Start with the out SCI HDU being the current 
            # we assume there are at least SCI extensions in the input
            outsci = deepcopy(inA[('SCI', extver)]) 
            
            try:
                # Subtracting the SCI frames by the constant
                outsci.data = np.subtract(inA[('SCI', extver)].data, inB)
                # Append updated SCI extension to the output 
                out.append(outsci)
                
                # Appending the inputA VAR and DQ frames un-edited to the output
                # ie no change, just propagate the frames
                
                # Check there are VAR frames to propagate
                if inA.count_exts('VAR') == inA.count_exts('SCI'): 
                    # Start with the out VAR HDU being the current
                    outvar = deepcopy(inA[('VAR', extver)])
                    # Just propagate VAR frames to the output
                    out.append(outvar) 
                    # ie there are DQ frames to operate on 
                if inA.count_exts('DQ') == inA.count_exts('SCI'): 
                    # Start with the out DQ HDU being the current 
                    outdq = deepcopy(inA[('DQ', extver)]) 
                    # Just propagate DQ frames to the output  
                    out.append(outdq) 
            except:
                raise
            
    # Check to see if the denominator is of type astrodata.AstroData.AstroData
    elif isinstance(inB, astrodata.AstroData):
        # Loop through the SCI extensions
        for sci in inA['SCI']:
            # Retrieving the version of this extension
            extver = sci.extver()  
            # Start with the out SCI HDU being the current,
            # we assume there are at least SCI extensions in the input    
            outsci = deepcopy(inA[('SCI', extver)])    
               
            try:
                # Making sure arrays are same size/shape
                if inA[('SCI', extver)].data.shape == \
                inB[('SCI', extver)].data.shape: 
                    #  Subtracting the SCI frames
                    outsci.data = np.subtract(inA[('SCI', extver)].data, 
                                              inB[('SCI', extver)].data)
                    out.append(outsci)
                    
                    # Check there are an equal numbers of VAR frames to 
                    # operate on
                    if inA.count_exts('VAR') == inB.count_exts('VAR') == \
                    inA.count_exts('SCI'): 
                        # Start with the out VAR HDU being the current 
                        outvar = deepcopy(inA[('VAR', extver)])
                        # Creating the output VAR frame following
                        # varOut= varA + varB
                        outvar.data = np.add(inA[('VAR', extver)].data, 
                                           inB[('VAR', extver)].data)
                        # Append the updated out VAR frame to the output
                        out.append(outvar)
                    # Check there are an equal number of DQ frames to operate on
                    if inA.count_exts('DQ') == inB.count_exts('DQ') == \
                    inA.count_exts('SCI'):  
                        # Start with the out DQ HDU being the current
                        outdq = deepcopy(inA[('DQ', extver)])       
                        # Performing bitwise-or 'adding' DQ frames 
                        outdq.data = np.bitwise_or(inA[('DQ', extver)].data, 
                                                 inB[('DQ', extver)].data)
                        # Append the updated out DQ frame to the output 
                        out.append(outdq)
                
                # If arrays are different sizes then raise an exception
                else:
                    raise ArithError('different numbers of SCI, VAR extensions')
            except:
                raise 
     
    # If the input was not of type astrodata or float, raise an exception
    else:
        raise            
    # Return the fully updated output astrodata object 
    return out 

