

'''This file contains the following utilities:
    joinlines (input, delim=" ", missing="Missing", maxchars=161,
               shortest=True)
    joinlists (list1, list2, delim=" ", missing="Missing", shortest=True)
    atList (input, filenames)
    expandlist (input)
    '''

#---------------------------------------------------------------------------

def joinlines (input, delim=" ", missing="Missing",
               maxchars=161, shortest=True):
    """Join lines from the input list of files.

    This is an implementation of the iraf.proto.joinlines task, with
    the following differences:  The result is as a list of strings,
    returned as the function value, rather than writing to standard
    output.  There is no verbose mode.  No warnings will be printed.

    @param input: names of files, separated by commas (and optional
        whitespace)
    @type input: string

    @param delim: delimiter to separate joined lines
    @type delim: string

    @param missing: string to use for files with fewer lines,
        if shortest is False
    @type missing: string

    @param maxchars: the output strings will be truncated after this length
    @type maxchars: int

    @param shortest: if True, the number of elements in the function
        value will be the smallest number of lines in any input file;
        if False, the number of elements will be the largest number of
        lines in any input file
    @type shortest: Boolean

    @return: the contents of the input files
    @rtype: list of strings
    """

    filenames = input.split (",")
    if not filenames[0]:        # an empty string?
        return filenames

    for i in range (len (filenames)):
        filenames[i] = filenames[i].strip()

    # There will be one element of all_lines for each file in input;
    # all_lines[i] will be a list of the lines (with leading and
    # trailing whitespace removed) of file i from input.
    all_lines = []
    first = True
    for name in filenames:
        fd = open (name)
        lines = fd.readlines()
        fd.close()
        for i in range (len (lines)):
            lines[i] = lines[i].strip()
        all_lines.append (copy.deepcopy (lines))
        numlines = len (lines)
        if first:
            min_numlines = numlines
            max_numlines = numlines
            first = False
        else:
            min_numlines = min (numlines, min_numlines)
            max_numlines = max (numlines, max_numlines)

    if min_numlines < max_numlines:
        if shortest:
            numlines = min_numlines
        else:
            numlines = max_numlines

    if len (all_lines[0]) > numlines:
        result = all_lines[0][0:numlines]
    else:
        result = all_lines[0]
    for k in range (len (result), numlines):
        result.append (missing)

    for i in range (1, len (all_lines)):
        lines = all_lines[i]
        for j in range (len (lines)):
            if j >= numlines:
                break
            result[j] = result[j] + delim + lines[j]
        for j in range (len (lines), numlines):
            result[j] = result[j] + delim + missing

    for j in range (len (result)):
        result[j] = result[j][0:maxchars]

    return result

#---------------------------------------------------------------------------

def joinlists (list1, list2, delim=" ", missing="Missing", shortest=True):
    """Join corresponding elements from two input lists.

    This is similar to the iraf.proto.joinlines task, except that the
    input is a pair of lists rather than files (just two input lists),
    and the result is as a list of strings, returned as the function
    value, rather than writing to standard output.  There is no verbose
    mode.  No warnings will be printed.

    @param list1: a list of values
    @type list1: list

    @param list2: another list of values
    @type list2: list

    @param delim: delimiter to separate joined elements
    @type delim: string

    @param missing: string to use for lists with fewer lines,
        if shortest is False
    @type missing: string

    @param shortest: if True, the number of elements in the function
        value will be the smaller of the number of lines in either of
        the input lists;
        if False, the number of elements will be the larger of the
        number lines in either input list
    @type shortest: Boolean

    @return: the contents of the input lists
    @rtype: list of strings
    """

    len1 = len (list1)
    len2 = len (list2)
    min_numlines = min (len1, len2)
    max_numlines = max (len1, len2)

    if min_numlines < max_numlines:
        if shortest:
            numlines = min_numlines
        else:
            numlines = max_numlines
    else:
        numlines = len1

    result = []
    for i in range (numlines):
        if i > len1-1:
            result.append (missing + delim + str (list2[i]))
        elif i > len2-1:
            result.append (str (list1[i]) + delim + missing)
        else:
            result.append (str (list1[i]) + delim + str (list2[i]))

    return result

#---------------------------------------------------------------------------

def atList (input, filenames):
    """Either append the current name, or read contents if it's a file.

    @param input: one or more names (or @name) separated by commas
    @type input: string

    @param filenames: (modified in-place) a list of the names extracted
        from input, or from the contents of input if it's an '@file'
    @type filenames: list
    """

    input = input.strip()
    if not input:
        return

    if input[0] == '@' and input.find (',') < 0:
        # just a single word, and it begins with '@'
        line = irafutil.expandFileName (input[1:])  # expand environment var.
        fd = open (line)
        lines = fd.readlines()
        fd.close()
    else:
        # one or more words, and the first does not begin with '@'
        lines = input.split (',')

    for line in lines:
        line = line.strip()
        if line[0] == '@':
            atList (line, filenames)
        else:
            line = irafutil.expandFileName (line)
            filenames.append (line)
#---------------------------------------------------------------------------
def expandlist (input):
    """Convert a string of comma-separated names to a list of names.

    @param input: one or more names separated by commas;
        a name of the form '@filename' implies that 'filename' is
        the name of a file containing names
    @type input: string

    @return: list of the names in 'input'
    @rtype: list of strings
    """

    filenames = []
    atList (input, filenames)
    return filenames


#---------------------------------------------------------------------------

