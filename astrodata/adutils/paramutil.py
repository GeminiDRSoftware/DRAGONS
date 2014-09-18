import os
import strutil

from astrodata.adutils import logutils

log = None
 
"""This file contains the following utilities:

    checkImageParam( image )
    checkOutputParam( outfile, defaultValue='out.fits' )
    def verifyOutlist( inlist, outlist )
    checkParam( parameter, paramType, defaultValue, compareValue=0.0 )
    checkFileFitExtension( filename )
"""
def checkImageParam(image, logBadlist=False):
    """
    Tries to accomplish the same thing as most IRAF tasks in regards to how they
    handle the input file parameter.
    
    @param image: 
    What is supported:
    
    Strings:
    1) If the string starts with '@' then it is expected that after the '@' is 
       a filename with the input image filenames in it.
    2) Input image filename.
    
    List:
    1) Must be a list of strings. It is assumed the each string refers an 
    input image filename.
    
    @type image: String or List of Strings
    
    @param logBadlist: Controls if lists of images that cannot be accessed are
        written to the log or not.
        
    @return: The list of filenames of images to be run. If an error occurs, 
             None is returned.
    @rtype: list, None
    
    """
    global log
    if log==None:
        log = logutils.get_logger(__name__)

    root = os.path.dirname(image)
    imageName = os.path.basename(image)
    inList = []   
    
    if type(imageName) == str and len(imageName) > 0:
        if imageName[0] == '@':       
            imageName = imageName[1:]            
            try:
                image = os.path.join( root, imageName )         
                imageFile = open(image, 'r')                
                readList = imageFile.readlines() 
                # Removes any newline with strutil method 'chomp'                 
                for i in range(len(readList)):
                    readList[i] = strutil.chomp(readList[i])
                    if (readList[i] == '') or (readList[i] == '\n')\
                     or (readList[i][0] == '#'):
                        continue
                    if os.path.dirname(readList[i]) == '':
                        readList[i] = os.path.join(root, readList[i])
                    nospace_str = readList[i].replace(' ','')
                    # Adds .fits if there is none
                    inList.append(strutil.appendFits(nospace_str))
                imageFile.close()
            except:
                log.critical('An error occurred when opening and reading '+
                'from the image '+os.path.basename(image))
                return None
        else:
            inList.append(image)            
            inList[0] = strutil.appendFits(inList[0])
    # Exception for an image of type 'List'       
    elif type(image) == list:
        for img in image:
            if type(img) == str:
                inList.append(img)
            else:
                log.warning('Type '+str(type(image))+ 
                    ' is not supported. The only supported types are String'+
                    ' and List of Strings.')
                return None
    else:
        log.warning('Type'+ str(type(image))+ 
                    'is not supported. The only supported types are String '+
                    'and List of Strings.')
        return None
    outList = []
    badList = []
    for img in inList:
        if not os.access(img,os.R_OK):
            # log.error('Cannot read file: '+str(img))   
            badList.append(img)
        else:
            outList.append(img)
    
    if badList:
        if logBadlist:
            err = "\n\t".join(badList)
            log.warning("Some files not found or cannot be opened:\n\t"+err)
        return None
    
    return outList

#------------------------------------------------------------------------------------------      
        
def checkOutputParam(outfile, defaultValue='out.fits'):
    """
    Tries to accomplish the same thing as most IRAF tasks in regards to 
    how they handle the output file parameter.
        
    @param outfile: 
    What is supported:
    
    Strings:
    1) If the string starts with '@' then it is expected that after the '@' is a 
       filename with the output filenames in it.
    2) Output file name.
    
    List:
    1) Must be a list of strings. It is assumed the each string refers to a 
       desired output file name. 
       
    @type outfile: String or List of Strings 
    
    @param defaultValue: If the outfile is '', then the defaultValue will 
                         returned.
    @type defaultValue: str
    
    @return: A list with all the desired output names. If an error occurs, 
            None is returned.
    @rtype: list, None
    """
    global log
    if log==None:
        log = logutils.get_logger(__name__)
        
    root = os.path.dirname(outfile)
    outfileName = os.path.basename(outfile)
    outList = []    
    if type(outfileName) == str:
        outfile = checkParam(outfile, type(''), defaultValue)
        if outfileName[0] == '@':       
            outfileName = outfileName[1:]            
            try:
                outfile = os.path.join( root, outfileName )         
                outfileFile = open(outfile, 'r')                
                readList = outfileFile.readlines() 
                # Removes any newline with strutil method 'chomp'                 
                for i in range(len(readList)):
                    readList[i] = strutil.chomp(readList[i])
                    if (readList[i] == '') or (readList[i] == '\n')\
                     or (readList[i][0] == '#'):
                        continue
                    if os.path.dirname(readList[i]) == '':
                        readList[i] = os.path.join(root, readList[i])
                    # Adds .fits if there is none
                    outList.append(strutil.appendFits(readList[i]))
            except:
                log.critical('An error occurred when opening and reading '+
                'from the outfile '+os.path.basename(outfile))
                return None
            finally:
                outfileFile.close()
        else:
            outList.append(outfile)            
            outList[0] = strutil.appendFits(outList[0])
    # Exception for an image of type 'List'       
    elif type(outfile) == list:
        for img in outfile:
            if type(out) == str:
                outList.append(out)
            else:
                log.warning('Type '+str(type(outfile))+ 
                    ' is not supported. The only supported types are String'+
                    ' and List of Strings.')
                return None
    else:
        log.warning('Type'+ str(type(outfile))+ 
                    'is not supported. The only supported types are String '+
                    'and List of Strings.')
        return None
    for out in outList:
        if not os.access(out,os.R_OK):
            log.error('Cannot read file: '+str(out))   
    return outList

#------------------------------------------------------------------------------ 

def verifyOutlist( inlist, outlist ):
    """
    Verifies that for every file in the inList, there is a corresponding 
    output file.
    
    @param inList: A list of input file paths.
    @type inList: list
    
    @param outlist: A list of output file paths.
    @type outlist: list    
    
    """
    global log
    if log==None:
        log = logutils.get_logger(__name__)
    
    try:    
        if outlist == []:
            # Will append unique filenames if none exist in outlist             
            for i in range(len(inlist)):
                line = 'output' + str(i+1)+ '.fits'
                outlist.append(line)             
            return outlist
        elif len(outlist) < len(inlist):
            # Will append unique filenames if not enough in outlist
            l = len(inlist) - len(outlist)           
            for i in range(l):
                line = 'output' + str(l+i)+ '.fits'
                outlist.append(line)
            return outlist
        else:
            return outlist
    except:
        log.error('An error occured while trying to verify'+
                    ' the outputs existance for inlist '+repr(inlist))
        return None                

#------------------------------------------------------------------------------ 

def checkParam(parameter, paramType, defaultValue, compareValue=0.0):
    """
    This is a very basic and general parameter checking function. Basically, 
    pass a parameter, expected type 
    and a default value. If it is the same type then:
    1) If it is a list, return it.
    2) If it is a string, check if it is  '', if it is then return default value.
    3) If it is a number, check that it is greater than 'compareValue', if it 
    is not then return defaultValue.
    
    Example usage:
    
    -checkParam( mode, type(''), 'constant' )
    -checkParam( fwhm, type(0.0), 5.5, compareValue=1.5 ) 
    
    @param parameter: Parameter for testing.
    @type parameter: Any
    
    @param paramType: What the expected type of the parameter should be.
    @type paramType: type( Any )
    
    @param defaultValue: What the default value will be if failures occur. 
                        Hopefully, the type matches paramType.
    @type defaultValue: type( Any )
    
    @param compareValue: Value to compare against in the case the parameter 
                        is a float or int. This will check whether
    the parameter is greater than the compareValue.
    @type compareValue: float or int
    
    @return: Parameter if it is the correct type and compare, default if it 
            fails comparison, or None if errors.
    @rtype: paramType or None
    """
    global log
    if log==None:
        log = logutils.get_logger(__name__)
        
    if type(parameter) == paramType:
        if (paramType == type(0)) or (paramType == type(0.0)):
            if parameter > compareValue:
                return parameter
        elif paramType == str:
            if parameter != '': 
                return parameter
        else:
            return parameter
    else:
        log.warning('Type of parameter, '+str(type(parameter))+
                    ' is not the correct type:'+str(paramType))
        #$$$ There needs to be an exception class to properly handle this raise
        raise 'Incorrect Parameter Type', type(parameter)
    return defaultValue
            
#------------------------------------------------------------------------------ 

def checkFileFitExtension( filename ):
    """
    Determines if the fits file has a [X], [sci,X] or [SCI,X], to see if a  
    particular extension is to be opened.
    
    @param filename: Name of a fits file
    @type filename: str
    
    @return: Tuple with the root filename and extension specified. 
             In the case that no '[]' exists in the filename, (filename, None) is returned.
             In the case of an error parsing the input filename, (None,None) is returned.
    @rtype: tuple 
    """
    if filename.find('[') >= 0:
        try:
            # Getting the filename without extension specification
            file = filename.split('[')[0]
            # Getting extension for both possible [] cases
            if filename.find('sci')>(-1) or filename.find('SCI')>(-1):
                exten = int(filename.split( '[' )[1].split( ']' )[0].split(',')[1])
            else:
                exten = int(filename.split( '[' )[1].split( ']' )[0])
        except:
            return (None, None)
        return (file, exten)
    
    else:
        return (filename, None)
    
#------------------------------------------------------------------------------ 

#---------------------------------------------------------------------------
def appendFits (images):
    """Append '.fits' to each name in 'images' that lacks an extension.

    >>> print appendFits ('abc')
    abc.fits
    >>> print appendFits ('abc.fits')
    abc.fits
    >>> print appendFits (['abc', 'xyz.fits'])
    ['abc.fits', 'xyz.fits']

    @param images: a file name or a list of file names
    @type images: a string or a list of strings

    @return: the input file names with '.fits' appended to each, unless
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
            modified.append (image + '.fits')

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
    if type(line) != str:
        raise 'Bad Argument - Passed parameter is not str', type(line)
    
    while (len(line) >=1) and (line[-1] == '\n'):            
        line = line[:-1]                 
    return line


#------------------------------------------------------------------------------ 

def getDataFromInput(inputf, ext=None):
    """
    !!! NOT FINISHED !!!
    
    Retrieve the science data from a fits file, science data or AstroData.
     
    """
    
    try:
        import astrodata
        from astrodata.AstroData import AstroData
        astroPossible = True
    except:
        astroPossible = False
    
    try:
        import numpy as np
    except:
        raise
    
    
    exttype = type(ext)
    inputtype = type(inputf)

    if ext is not None:
        if exttype == int:
            pass
        else:
            raise RuntimeError('Bad argument type. Received %s, expecting int.' %(str(exttype)))
    else:
        ext = 1
    
    if inputtype == np.Array:
        pass
    elif astroPossible and inputtype == AstroData:
        pass
    elif inputtype == str:
        pass
    else:
        raise RuntimeError('Bad argument type.')
    
    
        
        
    
