# This module contains functions that perform numpy operations on the input
# dataset

import numpy as np
import astrodata
from astrodata import Errors

def determine_inputs(input_a=None, input_b=None):
    """
    When either the add, div, mult or sub functions are called, input_a is
    always the input AstroData object that will have the operation applied to
    it:
    
      add  --> input_a + input_b
      div  --> input_a / input_b
      mult --> input_a * input_b
      sub  --> input_a - input_b
    
    This helper function determines what input_b is so that the operation can
    be executed appropriately. Possible options include another AstroData
    object, a dictionary, list, float or integer or a DescriptorValue object.
    
    In those cases where input_b is another AstroData object, the size of the
    data are checked to ensure they are the same so that the operation can be
    performed.
    
    In those cases where input_b is a dictionary, list, float, integer or a
    DescriptorValue object, create and return a dictionary.
    """
    # Check to see if input_b is a DescriptorValue object
    if isinstance(input_b, astrodata.Descriptors.DescriptorValue):
        input_b = input_b.get_value()
    
    # If input_b is a single float or a single integer, create a dictionary
    # where the key is the (EXTNAME, EXTVER) tuple for each science extension
    # and the value is the single float or single integer.
    if isinstance(input_b, float) or isinstance(input_b, int):
        return_value = {}
        for ext in input_a["SCI"]:
            # Retrieve the extension number for this extension
            extver = ext.extver()
            # Add the single float or single integer to the dictionary for
            # this extension
            return_value[("SCI", extver)] = input_b
    
    # If input_b is a list, create a dictionary where the key is the
    # (EXTNAME, EXTVER) tuple for each science extension and the value is
    # taken sequentially from the list.
    elif isinstance(input_b, list):
        return_value = {}
        for ext in input_a["SCI"]:
            # Retrieve the extension number for this extension
            extver = ext.extver()
            # Add the appropriate element of the list to the dictionary for
            # this extension
            return_value[("SCI", extver)] = input_b[extver-1]
    
    # If input_b is a dictionary, just rename the variable
    elif isinstance(input_b, dict):
        return_value = input_b
    
    # Check to see if input_b is an AstroData object
    elif isinstance(input_b, astrodata.AstroData):
        for ext in input_a["SCI"]:
            # Retrieve the extension number for this extension
            extver = ext.extver()
            # Make sure that the pixel data arrays are same size/shape
            if input_a[("SCI", extver)].data.shape != \
               input_b[("SCI", extver)].data.shape:
                raise Errors.Error("The input science extensions %s and %s " \
                                   "are not the same size" \
                                   % (input_a[("SCI", extver)],
                                      input_b[("SCI", extver)]))
            # Return the AstroData object
            return_value = input_b
    
    else:
        raise Errors.Error("Unknown type for input_b")
    
    return return_value

def add(input_a=None, input_b=None):
    """
    The add function adds input_b to an AstroData object (input_a), where
    input_b could be either another AstroData object, a single value or a
    DescriptorValue object.
    
    If input_b is an AstroData object, the add function will add each science
    extension in the input AstroData object (input_a) with the corresponding
    science extension in input_b and update the variance and data quality
    extensions accordingly.
    
    If input_b is a single value, the single value is added to each science
    extension in the input AstroData object (input_a).
    
    :param input_a: input AstroData object to be added to input_b
    :type input_a: AstroData
    
    :param input_b: input value to add to the input AstroData object (input_a).
                   This value can be an AstroData object, a dictionary, where
                   the key is the (EXTNAME, EXTVER) tuple for each science
                   extension and the value is a float or integer, a list of
                   floats or integers, where the values in the list correspond
                   to the science extensions in numerical order, a single float
                   or integer, or a DescriptorValue object.
    :type input_b: AstroData, dictionary, list, float, integer, DescriptorValue
    """
    # The output of determine_inputs is either an AstroData object or a
    # dictionary, where the key is the (EXTNAME, EXTVER) tuple for each science
    # extension and the value is the value that will be added to the input
    # AstroData object (input_a)
    new_input_b = determine_inputs(input_a=input_a, input_b=input_b)
    
    # Since the variance calculation may require access to the original pixel
    # values in the science extensions of the input AstroData object (input_a)
    # and new_input_b, the variance propagation is done before adding
    # new_input_b to the input AstroData object (input_a).
    if input_a["VAR"]:
        input_a = propagate_variance(input_a=input_a, input_b=new_input_b,
                                    operation="add")
    
    # Loop over each science extension in the input AstroData object (input_a)
    for sci in input_a["SCI"]:
        
        # Retrieve the extension number for this extension
        extver = sci.extver()
        
        # Add new_input_b to the science extension of the input AstroData
        # object (input_a)
        if isinstance(new_input_b, dict):
            sci.data = np.add(sci.data, new_input_b[("SCI", extver)])
        if isinstance(new_input_b, astrodata.AstroData):
            sci.data = np.add(sci.data, new_input_b["SCI", extver].data)
    
    # If any data quality extensions exist in new_input_b, combine the data
    # quality extension in the input AstroData object (input_a) with the data
    # quality extension in new_input_b using a bitwise or. Otherwise, make no
    # changes to any data quality extensions in the input AstroData object
    # (input_a).
    if input_a["DQ"] and isinstance(new_input_b, astrodata.AstroData) and \
       new_input_b["DQ"]:
        input_a = propagate_dq(input_a=input_a, input_b=new_input_b)
    
    # Return the updated AstroData object
    return input_a

def div(input_a=None, input_b=None):
    """
    The div function divides an AstroData object (input_a) by input_b, where
    input_b could be either another AstroData object, a single value or a
    DescriptorValue object.
    
    If input_b is an AstroData object, the div function will divide each
    science extension in the input AstroData object (input_a) with the
    corresponding science extension in input_b and update the variance and data
    quality extensions accordingly.
    
    If input_b is a single value, each science extension in the input AstroData
    object (input_a) is divided by the single value.
    
    :param input_a: input AstroData object to be divided by input_b
    :type input_a: AstroData
    
    :param input_b: input value to divide by the input AstroData object
                   (input_a). This value can be an AstroData object, a
                   dictionary, where the key is the (EXTNAME, EXTVER) tuple for
                   each science extension and the value is a float or integer,
                   a list of floats or integers, where the values in the list
                   correspond to the science extensions in numerical order, a
                   single float or integer, or a DescriptorValue object.
    :type input_b: AstroData, dictionary, list, float, integer, DescriptorValue
    """
    # The output of determine_inputs is either an AstroData object or a
    # dictionary, where the key is the (EXTNAME, EXTVER) tuple for each science
    # extension and the value is the value that will be used to divide the
    # input AstroData object (input_a)
    new_input_b = determine_inputs(input_a=input_a, input_b=input_b)
    
    # Since the variance calculation may require access to the original pixel
    # values in the science extensions of the input AstroData object (input_a)
    # and new_input_b, the variance propagation is done before dividing the
    # input AstroData object (input_a) by new_input_b.
    if input_a["VAR"]:
        input_a = propagate_variance(input_a=input_a, input_b=new_input_b,
                                    operation="div")
    
    # Loop over each science extension in the input AstroData object (input_a)
    for sci in input_a["SCI"]:
        
        # Retrieve the extension number for this extension
        extver = sci.extver()
        
        # Divide the science extension of the input AstroData object (input_a)
        # by new_input_b
        if isinstance(new_input_b, dict):
            sci.data = np.divide(sci.data, new_input_b[("SCI", extver)])
        if isinstance(new_input_b, astrodata.AstroData):
            sci.data = np.divide(sci.data, new_input_b["SCI", extver].data)
    
    # If any data quality extensions exist in new_input_b, combine the data
    # quality extension in the input AstroData object (input_a) with the data
    # quality extension in new_input_b using a bitwise or. Otherwise, make no
    # changes to any data quality extensions in the input AstroData object
    # (input_a).
    if input_a["DQ"] and isinstance(new_input_b, astrodata.AstroData) and \
       new_input_b["DQ"]:
        input_a = propagate_dq(input_a=input_a, input_b=new_input_b)
    
    # Return the updated AstroData object
    return input_a

def mult(input_a=None, input_b=None):
    """
    The mult function multiplies an AstroData object (input_a) by input_b,
    where input_b could be either another AstroData object, a single value or a
    DescriptorValue object.
    
    If input_b is an AstroData object, the mult function will multiple each
    science extension in the input AstroData object (input_a) with the
    corresponding science extension in input_b and update the variance and data
    quality extensions accordingly.
    
    If input_b is a single value, each science extension in the input AstroData
    object (input_a) is multiplied by the single value.
    
    :param input_a: input AstroData object to be multiplied by input_b
    :type input_a: AstroData
    
    :param input_b: input value to multiply by the input AstroData object
                   (input_a). This value can be an AstroData object, a
                   dictionary, where the key is the (EXTNAME, EXTVER) tuple for
                   each science extension and the value is a float or integer,
                   a list of floats or integers, where the values in the list
                   correspond to the science extensions in numerical order, a
                   single float or integer, or a DescriptorValue object.
    :type input_b: AstroData, dictionary, list, float, integer, DescriptorValue
    """
    # The output of determine_inputs is either an AstroData object or a
    # dictionary, where the key is the (EXTNAME, EXTVER) tuple for each science
    # extension and the value is the value that will be used to multiply the
    # input AstroData object (input_a)
    new_input_b = determine_inputs(input_a=input_a, input_b=input_b)
    
    # Since the variance calculation may require access to the original pixel
    # values in the science extensions of the input AstroData object (input_a)
    # and new_input_b, the variance propagation is done before multiplying the
    # input AstroData object (input_a) by new_input_b.
    if input_a["VAR"]:
        input_a = propagate_variance(input_a=input_a, input_b=new_input_b,
                                    operation="mult")
    
    # Loop over each science extension in the input AstroData object (input_a)
    for sci in input_a["SCI"]:
        
        # Retrieve the extension number for this extension
        extver = sci.extver()
        
        # Multiply the science extension of the input AstroData object
        # (input_a) by new_input_b
        if isinstance(new_input_b, dict):
            sci.data = np.multiply(sci.data, new_input_b[("SCI", extver)])
        if isinstance(new_input_b, astrodata.AstroData):
            sci.data = np.multiply(sci.data, new_input_b["SCI", extver].data)
    
    # If any data quality extensions exist in new_input_b, combine the data
    # quality extension in the input AstroData object (input_a) with the data
    # quality extension in new_input_b using a bitwise or. Otherwise, make no
    # changes to any data quality extensions in the input AstroData object
    # (input_a).
    if input_a["DQ"] and isinstance(new_input_b, astrodata.AstroData) and \
       new_input_b["DQ"]:
        input_a = propagate_dq(input_a=input_a, input_b=new_input_b)
    
    # Return the updated AstroData object
    return input_a

def sub(input_a=None, input_b=None):
    """
    The sub function subtracts input_b from an AstroData object (input_a),
    where input_b could be either another AstroData object, a single value or a
    DescriptorValue object.
    
    If input_b is an AstroData object, the sub function will subtract each
    science extension from input_b from the corresponding science extension in
    the input AstroData object (input_a) and update the variance and data
    quality extensions accordingly.
    
    If input_b is a single value, the single value is subtracted from each 
    science extension in the input AstroData object (input_a).
    
    :param input_a: input AstroData object to have input_b subtracted from it
    :type input_a: AstroData
    
    :param input_b: input value to subtract from the input AstroData object
                   (input_a). This value can be an AstroData object, a
                   dictionary, where the key is the (EXTNAME, EXTVER) tuple for
                   each science extension and the value is a float or integer,
                   a list of floats or integers, where the values in the list
                   correspond to the science extensions in numerical order, a
                   single float or integer, or a DescriptorValue object.
    :type input_b: AstroData, dictionary, list, float, integer, DescriptorValue
    """
    # The output of determine_inputs is either an AstroData object or a
    # dictionary, where the key is the (EXTNAME, EXTVER) tuple for each science
    # extension and the value is the value that will be subtracted from the
    # input AstroData object (input_a)
    new_input_b = determine_inputs(input_a=input_a, input_b=input_b)
    
    # Since the variance calculation may require access to the original pixel
    # values in the science extensions of the input AstroData object (input_a)
    # and new_input_b, the variance propagation is done before subtracting
    # new_input_b from the input AstroData object (input_a).
    if input_a["VAR"]:
        input_a = propagate_variance(input_a=input_a, input_b=new_input_b,
                                    operation="sub")
    
    # Loop over each science extension in the input AstroData object (input_a)
    for sci in input_a["SCI"]:
        
        # Retrieve the extension number for this extension
        extver = sci.extver()
        
        # Subtract new_input_b from the science extension of the input
        # AstroData object (input_a)
        if isinstance(new_input_b, dict):
            sci.data = np.subtract(sci.data, new_input_b[("SCI", extver)])
        if isinstance(new_input_b, astrodata.AstroData):
            sci.data = np.subtract(sci.data, new_input_b["SCI", extver].data)
    
    # If any data quality extensions exist in new_input_b, combine the data
    # quality extension in the input AstroData object (input_a) with the data
    # quality extension in new_input_b using a bitwise or. Otherwise, make no
    # changes to any data quality extensions in the input AstroData object
    # (input_a).
    if input_a["DQ"] and isinstance(new_input_b, astrodata.AstroData) and \
       new_input_b["DQ"]:
        input_a = propagate_dq(input_a=input_a, input_b=new_input_b)
    
    # Return the updated AstroData object
    return input_a

def propagate_variance(input_a=None, input_b=None, operation=None):
    """
    The propagate_variance function is used to propagate the variation after an
    operation has been performed on the input AstroData object (input_a) and
    should only be called if input_a contains a variance extension.
    
    At each reduction step, the variance extension is manipulated according to
    the operation being performed on the science extension. When applying
    calibrations, the statistical errors from the science frame and the
    statistical errors from the calibration are added in quadrature. Since the
    science frame and the calibration have a variance extension that describe
    the statistical, uncorrelated errors, the variance extensions can simply be
    added together. The general equation for this is:
    
    var(f(a,b)) = [var(a) * (df/da)^2] + [var(b) * (df/db)^2] + covariance term
    
    where df/da and df/db are partial derivatives. More specific equations are:
    
    var(a + b) = var(a) + var(b) + covariance term
    var(a - b) = var(a) + var(b) - covariance term
    var(a * b) = (var(a) * b^2) + (var(b) * a^2) + covariance term
    var(a / b) = (var(a) / b^2) + (var(b) * a^2 / b^4) - covariance term
    
    Since the variance extensions contain only uncorrelated noise, the
    covariance terms above are zero.
    
    When the variable b above is a single value that does not have its own
    variance, the equations can still be used (the variance for a is scaled
    accordingly).
    
    The uncertainty in a measurement that is equal for all pixels in a dataset
    is defined as correlated noise and should not be included when propagating
    the variance
    
    If either the calibration frame or the science frame does not have a
    variance extension, no variance is propagated
    
    At the end of the complete data reduction process, the variance propagation
    allows the final noise values to be estimated just by taking the square
    root of the final variance extension.
    """
    # Loop over each variance extension in the input AstroData object (input_a)
    for var in input_a["VAR"]:
        
        # Retrieve the extension number for this extension
        extver = var.extver()

        # The variable a is the data in the science extension of the input
        # AstroData object (input_a)
        a = input_a["SCI", extver].data

        # The variable var_a is the data in the variance extension of the input
        # AstroData object (input_a)
        var_a = var.data

        # If input_b is a dictionary, the variable b is the single value
        # specified by the key (EXTNAME, EXTVER)
        if isinstance(input_b, dict):
            b = input_b[("SCI", extver)]
            
            # The variable var_b is zero, since there is no variance associated
            # with a single value 
            var_b = 0

        # If input_b is an AstroData object, the variable b is the data in the
        # science extension of input_b
        if isinstance(input_b, astrodata.AstroData):
            b = input_b["SCI", extver].data
            
            # The variable var_b is the data in the variance extension of
            # input_b, if it exists
            if input_b[("VAR", extver)]:
                var_b = input_b["VAR", extver].data
            else:
                var_b = 0
        
        if operation == "add" and isinstance(var_b, numpy.ndarray):
            # The variance is propagated using:
            #     var(a + b) = var(a) + var(b)
            var.data = np.add(var_a, var_b)

        elif operation == "sub" and isinstance(var_b, numpy.ndarray):
            # The variance is propagated using:
            #     var(a - b) = var(a) + var(b)
            var.data = np.add(var_a, var_b)

        elif operation == "mult":
            # The variance is propagated using:
            #     var(a * b) = (var(a) * b^2) + (var(b) * a^2)
            var.data = np.add(
                np.multiply(var_a, b**2), np.multiply(var_b, a**2))
                            
        elif operation == "div":
            # The variance is propagated using:
            #     var(a / b) = (var(a) / b^2) + (var(b) * a^2 / b^4)
            var.data = np.add(
                np.divide(var_a, b**2),
                np.multiply(var_b, np.divide(a**2, b**4)))
        else:
            raise Errors.Error("The operation parameter must have a value " \
                               "of either add, sub, mult or div")
    
    return input_a

def propagate_dq(input_a=None, input_b=None):
    """
    When combining frames that have data quality extensions, the pixel data in
    the data quality extensions should be combined with a bitwise OR, to allow
    a pixel in the data quality extension to contain information about multiple
    problems with the associated pixel in the science extension. For example:

    0      0000
    1      0001
    2      0010
    3      0011
    4      0100
    5      0101
    6      0110
    7      0111
    8      1000
    16    10000

    0001 OR 0010  = 0011
    0001 OR 0001  = 0001
    """
    # Loop over each data quality extension in the input AstroData object
    # (input_a)
    for dq in input_a["DQ"]:
        
        # Retrieve the extension number for this extension
        extver = dq.extver()

        # Check whether input_b contains the equivalent data quality extension
        if input_b[("DQ", extver)]:
            # Propagate the data quality
            dq.data = np.bitwise_or(dq.data, input_b[("DQ", extver)].data)

    return input_a
