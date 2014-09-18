# SERVICE FUNCTIONS and FACTORIES
#
# /* N.B. The following functions are not used in astrodata or astrodata_Gemini
#
#    correlate()
#    prep_output()
#
# */ 

def correlate(*iary):
    """
    :param iary: A list of AstroData instances for which a correlation dictionary
        will be constructed.
    :type iary: list of AstroData instance
    :returns: a list of tuples containing correlated extensions from the arguments. 
    :rtype: list of tuples

    The ``correlate(..)`` function is a module-level helper function which returns
    a list of tuples of Single Extension ``AstroData`` instances which associate
    extensions from each listed AstroData object, to identically named
    extensions among the rest of the input array. The ``correlate(..)`` function
    accepts a variable number of arguments, all of which should be ``AstroData``
    instances.
    
    The function returns a structured dictionary of dictionaries of lists of
    ``AstroData`` objects. For example, given three inputs, *ad*, *bd* and *cd*, all
    with three "SCI", "VAR" and "DQ" extensions. Given *adlist = [ad, bd,
    cd]*, then *corstruct = correlate(adlist)* will return to *corstruct* a
    dictionary first keyed by the EXTNAME, then keyed by tuple. The contents
    (e.g. of *corstruct["SCI"][1]*) are just a list of AstroData instances each
    containing a header-data unit from *ad*, *bd*, and *cd* respectively.
        
    :info: to appear in the list, all the given arguments must have an extension
        with the given (EXTNAME,EXTVER) for that tuple.
    """
    numinputs = len(iary)
    if numinputs < 1:
        raise Errors.AstroDataError("Inputs for correlate method < 1")
    outlist = []
    outrow = []
    baseGD = iary[0]
    for extinbase in baseGD:
        try:
            extname = extinbase.header["EXTNAME"]
        except:
            extname = "NONE"
        try:
            extver  = extinbase.header["EXTVER"]
        except:
            extver  = 0
        outrow = [extinbase]

        for gd in iary[1:]:
            correlateExt = gd[(extname, extver)]
            if correlateExt == None:
                break
            else:
                outrow.append(correlateExt)
        if len(outrow) == numinputs:
            # if the outrow is short then some input didn't correlate with the
            # cooresponding extension, otherwise, add it to the table (list of lists)
            outlist.append(outrow)
    return outlist


def prep_output(input_ary=None, name=None, clobber=False):
    """
    :param input_ary: The input array from which propagated content (such as
        the  source PHU) will be taken. Note: the zero-th element in the list
        is  used as the reference dataset for PHU or other items which require
        a particular reference.
    :type input_ary: list of AstroData Instances
    
    :param name: File name to use for returned AstroData, optional.
    
    :param clobber: By default ``prep_output(..)`` checks to see if a file of the
        given name already exists, and will raise an exception if found.
        Set *clobber* to *True* to override this behavior and potentially
        overwrite the extant file.  The dataset on disk will not be overwritten
        as a direct result of prep_output, which only prepares the object
        in memory, but will occur when the AstroData object returned is 
        written (i.e. ``ad.write(..)``)). 
    :type clobber: bool
        
    :returns: an ``AstroData`` instance initialized with appropriate
        header-data units such as the PHU, Standard Gemini headers
        and with type-specific associated  data-header units such as
        binary table Mask Definition tables (MDF).        
    :rtype: AstroData

    ..info: File will not have been written to disk by ``prep_output(..)``.
    
    The ``prep_output(..)`` function creates a new ``AstroData`` object ready for
    appending output information (e.g. ``ad.append(..)``).  While you can also
    create an empty ``AstroData`` object by giving no arguments to the ``AstroData``
    constructor  (i.e. ``ad = AstroData()``), ``prep_output(..)`` exists for the
    common case where a new dataset object is intended as the output of
    some processing on a list of source datasets, and some information
    from the source inputs must be propagated. 
    
    The ``prep_output(..)`` function makes use of this knowledge to ensure the
    file meets standards in what is considered a complete output file given
    such a combination.  In the future this function can make use of dataset
    history and structure definitions in the ADCONFIG configuration space. As
    ``prep_output`` improves, scripts and primitives that use it
    will benefit in a forward compatible way, in that their output datasets will
    benefit from more automatic propagation, validations, and data flow control,
    such as the emergence of history database propagation.
    
    Presently, it already provides the following:
    
    +  Ensures that all standard headers are in place in the new file, using the
       configuration .
    +  Copy the PHU of the reference image (``input_ary[0]``). 
    +  Propagate associated information such as the MDF in the case of a MOS 
       observation, configurable by the Astrodata Structures system. 
    """ 
    if input_ary == None: 
        raise Errors.AstroDataError("prep_output input is None") 
        return None
    if type(input_ary) != list:
        iary = [input_ary]
    else:
        iary = input_ary
    
    #get PHU from input_ary[0].hdulist
    hdl = iary[0].hdulist
    outphu = copy(hdl[0])
    outphu.header = outphu.header.copy()
        
    # make outlist the complete hdulist
    outlist = [outphu]

    #perform extension propagation
    newhdulist = pyfits.HDUList(outlist)
    retgd = AstroData(newhdulist, mode = "update")
    
    # Ensuring the prepared output has the __origFilename private variable
    retgd._AstroData__origFilename = input_ary._AstroData__origFilename
    if name != None:
        if os.path.exists(name):
            if clobber == False:
                raise Errors.OutputExists(name)
            else:
                os.remove(name)
        retgd.filename = name
    return retgd
