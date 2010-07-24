import os
import strutil
#------------------------------------------------------------------------------ 
#import geminiLogger

# removed old logger, commented out old calls, added prints,
# need to incorporate the new logger
#------------------------------------------------------------------------------ 
"""This file contains the following utilities:

    checkImageParam( image )
    checkOutputParam( outfile, defaultValue="out.fits" )
    def verifyOutlist( inlist, outlist )
    checkParam( parameter, paramType, defaultValue, compareValue=0.0 )
    checkFileFitExtension( filename )
"""
def checkImageParam( image ):
    """
    Tries to accomplish the same thing as most IRAF tasks in regards to how they
    handle the input file parameter.
    
    @param image: 
    What is supported:
    
    Strings:
    1) If the string starts with '@' then it is expected that after the '@' is 
       a filename with the input image filenames in it.
    2) Input image filename.checkParam( parameter, paramType, 
       defaultValue, compareValue=0.0 ):
    
    List:
    1) Must be a list of strings. It is assumed the each string refers an 
    input image filename.
    
    @type image: String or List of Strings
    
    @return: The list of filenames of images to be run. If an error occurs, 
             None is returned.
    @rtype: list, None
    checkParam( parameter, paramType, defaultValue, compareValue=0.0 ):
    """
    #logger set up
    #glog = geminiLogger.getLogger( name="checkImageParam")
    root = os.path.dirname( image )
    imageName = os.path.basename( image )
    inList = []    
    if type(imageName) == str:
        if imageName[0] == "@":       
            imageName = imageName[1:]            
            try:
                image = os.path.join( root, imageName )         
                imageFile = open(image, 'r')                
                readList = imageFile.readlines() 
                #removes any newline with strutil method 'chomp'                 
                for i in range(len(readList)):
                    readList[i] = strutil.chomp(readList[i])
                    if readList[i] == "" or readList[i] == "\n" or readList[i][0] == "#":
                        continue
                    if os.path.dirname(readList[i]) == "":
                        readList[i] = os.path.join( root, readList[i] )
                #adds .fits if there is none
                    inList.append(strutil.appendFits(readList[i]))
            except:
                #glog.exception("An error occurred when opening and reading from the image.")
                print ("paramutil_65: An error occurred when opening and reading from the image.")
                return None
            finally:
                imageFile.close()
        else:
            inList.append( image )            
            inList[0] = strutil.appendFits(inList[0])
    #exception for an image of type 'List'       
    elif type(image) == list:
        for img in image:
            if type(img) == str:
                inList.append( img )
            else:
                #glog.warning('Type'+ str(type(image))+ \
                print ('paramutil_79: Type'+ str(type(image))+ \
                    'is not supported. The only supported types are String and List of Strings.')
                return None
    else:
        #glog.warning('Type'+ str(type(image))+ \
        print ('paramutil_84: Type'+ str(type(image))+ \
                    'is not supported. The only supported types are String and List of Strings.')
        return None
    for img in inList:
        if not os.access(img,os.R_OK):
            #glog.error('cannot read file: ' + str(img))
            raise 'Cannot read file', img    
    return inList

#------------------------------------------------------------------------------------------      
        
def checkOutputParam( outfile, defaultValue="out.fits" ):
    """
    Tries to accomplish the same thing as most IRAF tasks in regards to how they handle the output file parameter.
        
    @param outfile: 
    What is supported:
    
    Strings:
    1) If the string starts with '@' then it is expected that after the '@' is a filename with the output filenames in
    it.
    2) Output file name.
    
    List:
    1) Must be a list of strings. It is assumed the each string refers to a desired output file name. 
    @type outfile: String or List of Strings 
    
    @param defaultValue: If the outfile is "", then the defaultValue will returned.
    @type defaultValue: str
    
    @return: A list with all the desired output names. If an error occurs, None is returned.
    @rtype: list, None
    """
    #logger set up
    #glog = geminiLogger.getLogger( name="checkOutputParam") 
    ##@FIXME: This needs to be fixed to work like its input sibling.
    outList = []
    if type(outfile) == str:
        outfile = checkParam( outfile, type(""), defaultValue )
        if outfile[0] == "@": 
            outfile = outfile[1:]            
            try:
                outListFile = open(outfile,'r')                
                outList = outListFile.readlines() 
                if outList != []:              
                    if outList[0] == '\n':                                       
                        for i in range(len(outList)):                        
                            print i
                            line=outList[i]
                            outList[i] = line[0:len(line)-1]                                      
                    else:              
                        for i in range(len(outList)):  
                            #removes newlines from end of list                      
                            outList[i] = strutil.chomp(outList[i])
                        #adds .fits if there is none                            
                        outList = strutil.appendFits(outList)                
            except:
                #glog.exception("An error occurred when opening and reading from the outlist file.")
                print ("paramutil_142: An error occurred when opening and reading from the outlist file.")
                return None
            finally:
                outListFile.close()            
        else:
            outList.append( outfile )
            outList[0] = strutil.appendFits( outList[0] )
    elif type(outfile) == list:
        for file in outfile:
            if type( file ) == str:
                inList.append( file )
            else:
                #glog.warning('Type'+ str(type(image))+ \
                print ('paramutil_155: Type'+ str(type(image))+ \
                    'is not supported. The only supported types are String and List of Strings.')
                return None
    else:
         #glog.warning('Type'+ str(type(image))+ \
         print ('paramutil_160: Type'+ str(type(image))+ \
                    'is not supported. The only supported types are String and List of Strings.')
         return None
    return outList

#------------------------------------------------------------------------------ 

def verifyOutlist( inlist, outlist ):
    """
    Verifies that for every file in the inList, there is a corresponding output file.
    
    @param inList: A list of input file paths.
    @type inList: list
    
    @param outlist: A list of output file paths.
    @type outlist: list    
    
    """
    #logger set up
    #glog = geminiLogger.getLogger( name="verifyOutlist") 
    try:    
        if outlist == []:
            #will append unique filenames if none exist in outlist             
            for i in range(len(inlist)):
                line = 'output' + str(i+1)+ '.fits'
                outlist.append(line)             
            return outlist
        elif len(outlist) < len(inlist):
            #will append unique filenames if not enough in outlist
            l = len(inlist) - len(outlist)           
            for i in range(l):
                line = 'output' + str(l+i)+ '.fits'
                outlist.append(line)
            return outlist
        else:
            return outlist
    except:
        #glog.exception('verifyOutlist in paramUtil failure')
        print ('paramutil_198: verifyOutlist in paramUtil failure')
        return None                

#------------------------------------------------------------------------------ 

def checkParam( parameter, paramType, defaultValue, compareValue=0.0 ):
    """
    This is a very basic and general parameter checking function. Basically, pass a parameter, expected type 
    and a default value. If it is the same type then:
    1) If it is a list, return it.
    2) If it is a string, check if it is  "", if it is then return default value.
    3) If it is a number, check that it is greater than 'compareValue', if it is not then return defaultValue.
    
    Example usage:
    
    -checkParam( mode, type(""), "constant" )
    -checkParam( fwhm, type(0.0), 5.5, compareValue=1.5 ) 
    
    @param parameter: Parameter for testing.
    @type parameter: Any
    
    @param param_type: What the expected type of the parameter should be.
    @type param_type: type( Any )
    
    @param defaultValue: What the default value will be if failures occur. Hopefully, the type matches paramType.
    @type defaultValue: type( Any )
    
    @param compareValue: Value to compare against in the case the parameter is a float or int. This will check whether
    the parameter is greater than the compareValue.
    @type compareValue: float or int
    
    @return: Parameter if it is the correct type and compare, default if it fails comparison, or None if errors.
    @rtype: paramType or None
    """
    if type( parameter ) == paramType:
        if paramType == type( 0 ) or paramType == type( 0.0 ):
            if parameter > compareValue:
                return parameter
        elif paramType == str:
            if parameter != "": 
                return parameter
        else:
            return parameter
    else:
        print "Parameter: '", parameter, "' is not the correct type:", paramType
        raise "Incorrect Parameter Type", type(parameter)
    return defaultValue
            
#------------------------------------------------------------------------------ 

def checkFileFitExtension( filename ):
    """
    Determines if the fits file has a [X], to see if a particular extension is to be opened.
    
    @param filename: Name of a fits file
    @type filename: str
    
    @return: Tuple with the filename and extension specified. 0 means all extension > 0. In the case of any error, 0 
    is returned.
    @rtype: tuple 
    """
    file_and_extensions = filename.rsplit( "[" )
    if( len(file_and_extensions) > 1 ):
        try:
            extension_number = file_and_extensions[-1][0]
            extension_number = int( extension_number )
        except:
            extension_number = 0
        return (file_and_extensions[0], extension_number)
    else:
        return (filename, 0)
    
#------------------------------------------------------------------------------ 

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


#------------------------------------------------------------------------------ 

def getDataFromInput( inputf, ext=None ):
    '''
    !!! NOT FINISHED !!!
    
    Retrieve the science data from a fits file, science data or AstroData.
     
    
    '''
    
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
    
    
    exttype = type( ext )
    inputtype = type( inputf )

    if ext is not None:
        if exttype == int:
            pass
        else:
            raise RuntimeError( 'Bad argument type. Received %s, expecting int.' %(str(exttype)) )
    else:
        ext = 1
    
    if inputtype == np.Array:
        pass
    elif astroPossible and inputtype == AstroData:
        pass
    elif inputtype == str:
        pass
    else:
        raise RuntimeError( 'Bad argument type.' )
    
    
        
        
    
