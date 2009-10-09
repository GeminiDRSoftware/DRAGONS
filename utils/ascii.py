

"""This file contains the following utilities:
    fieldsOfTable (input, fields=None)
    """
    
def fieldsOfTable (input, fields=None):
    """Extract the specified fields from a list or from a file.

    Blank lines (list elements) and lines that begin with '#' will be
    ignored.  Lines are assumed to contain at least as many words as
    specified in the 'fields' argument.
    Note:  currently does not ignore in-line comments
    Note:  This should return a record array.  xxx

    @param input: either the name of a text file or
        a list of whitespace-separated words
    @type input: either a string (file name) or a list of strings

    @param fields: zero-indexed column numbers, separated by commas or blanks
    @type fields: string, or None

    @return: list of the specified fields from the input
    @rtype: list of strings
    """

    # If the input is a single string, assume it's the name of a file
    # containing the table from which we should extract the columns.
    if isinstance (input, str):
        fd = open (input)
        input = fd.readlines()
        fd.close()

    if fields is None:
        return input

    # this is a string
    elements = fields.replace (",", " ")
    # this is a list of strings
    element_numbers_str = elements.split()
    # this is a list of integers
    element_numbers = [int (x) for x in element_numbers_str]

    result = []
    for line in input:
        words = line.split()
        nwords = len (words)
        # ignore blank lines and comments
        if nwords == 0 or words[0] == "#":
            continue
        output_line = []
        for i in element_numbers:
            if i >= nwords:
                break
            output_line.append (words[i])
        result.append ("   ".join (output_line)+"   ")

    return result
