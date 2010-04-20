extensions = [".fits", ".fit", ".pl", ".imh", ".hhh", ".tab", ".dat"]

#---------------------------------------------------------------------------
def appendFits (images):
    """Append ".fits" to each name in 'images' that lacks an extension.

    >>> print appendFits ('abc')
    abc.fits
    >>> print appendFits ('abc.fits')
    abc.fits
    >>> print appendFits (['abc', 'xyz.fits'])
    ['abc.fits', 'xyz.fits']

    @param images: a file name or a list of file names
    @type images: a string or a list of strings

    @return: the input file names with ".fits" appended to each, unless
        the name already ended in a recognized extension.
    @rtype: list of strings
    """

    if isinstance (images, str):
        is_a_list = False
        images = [images]
    else:
        is_a_list = True
    modified = []
    for image in images:
        found = False
        # extensions is a list of recognized filename extensions.
        for extn in extensions:
            if image.endswith (extn):
                found = True
                break
        if found:
            modified.append (image)
        else:
            modified.append (image + ".fits")

    if is_a_list:
        return modified
    else:
        return modified[0]
#---------------------------------------------------------------------------

def chomp(line):
    """
    Removes newline(s) from end of line if present.
    
    @param line: A possible corrupted line of code
    @type line: str
    
    @return: Line without any '\n' at the end.
    @rtype: str
    """
    if type( line ) != str:
        raise "Bad Argument - Passed parameter is not str", type(line)
    
    while len(line) >=1 and line[-1] == '\n':            
        line = line[:-1]                 
    return line