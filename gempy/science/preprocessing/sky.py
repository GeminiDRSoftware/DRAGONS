# This module contains user level functions related to the preprocessing of
# the input dataset with a sky or fringe frame

import sys
import numpy as np
from astrodata import Errors
from astrodata.adutils import gemLog
from astrodata.adutils.gemutil import pyrafLoader
from gempy import geminiTools as gt
from gempy import managers as man
from gempy.geminiCLParDicts import CLDefaultParamsDict

def make_fringe_image_gmos(adinput=None, operation='median', 
                                 suffix='_fringe'):
    """
    This function will create and return a single fringe image from all the 
    inputs.  It uses the CL script gifringe to create the fringe image.
    
    NOTE: The inputs to this function MUST be prepared. 

    Either a 'main' type logger object, if it exists, or a null logger 
    (ie, no log file, no messages to screen) will be retrieved/created
    and used within this function.
    
    :param adinput: Astrodata inputs to be combined
    :type adinput: Astrodata objects, either a single or a list of objects
    
    :param operation: type of combining operation to use.
    :type operation: string, options: 'average', 'median'.

    :param suffix: string to add on the end of the first input filename 
                   to make the output filename.
    :type suffix: string
    """
    
    # instantiate log
    log = gemLog.getGeminiLog()

    # ensure that adinput is not None and make it into a list
    # if it is not one already
    adinput = gt.validate_input(adinput=adinput)

    # time stamp keyword
    keyword = 'FRINGE'
    
    # initialize output list
    adoutput_list = []    

    try:
        
        # Ensure there is more than one input to make a fringe frame from
        if (len(adinput)<2):
            raise Errors.InputError('Only one input was passed in for ' +
                                    'adinput. At least two frames are ' +
                                    'required to make a fringe' +
                                    'frame.')
            
        # load and bring the pyraf related modules into the name-space
        pyraf, gemini, yes, no = pyrafLoader() 

        # Clean up log and screen if multiple inputs
        log.fullinfo('+'*50, category='format')                                 
                
        # Determine whether VAR/DQ needs to be propagated 
        for ad in adinput:
            if (ad.count_exts('VAR') == 
                ad.count_exts('DQ') == 
                ad.count_exts('SCI')):
                fl_vardq=yes
            else:
                fl_vardq=no
                break
           
        # Prepare input files, lists, parameters... for input to 
        # the CL script
        clm = man.CLManager(imageIns=adinput, suffix=suffix, 
                            funcName='makeFringeFrame', 
                            combinedImages=True, log=log)
            
        # Check the status of the CLManager object, 
        # True=continue, False= issue warning
        if not clm.status:
            raise Errors.ScienceError('One of the inputs has not been ' +
                                      'prepared,the combine function ' + 
                                      'can only work on prepared data.')
            
        # Parameters set by the man.CLManager or the definition 
        # of the primitive 
        clPrimParams = {
            # Retrieve the inputs as a list from the CLManager
            'inimages'    :clm.imageInsFiles(type='listFile'),
            # Maybe allow the user to override this in the future. 
            'outimage'    :clm.imageOutsFiles(type='string'), 
            # This returns a unique/temp log file for IRAF  
            'logfile'     :clm.templog.name,
            'fl_vardq'    :fl_vardq,
            }
    
        # Create a dictionary of the parameters from the Parameter 
        # file adjustable by the user
        clSoftcodedParams = {
            'combine'       :operation,
            'reject'        :'none',
            }

        # Grab the default parameters dictionary and update 
        # it with the two above dictionaries
        clParamsDict = CLDefaultParamsDict('gifringe')
        clParamsDict.update(clPrimParams)
        clParamsDict.update(clSoftcodedParams)
                
        # Log the values in the soft and prim parameter dictionaries
        log.fullinfo('\nParameters set by the CLManager or  '+
                     'dictated by the definition of the primitive:\n', 
                     category='parameters')
        gt.logDictParams(clPrimParams)
        log.fullinfo('\nUser adjustable parameters in the '+
                     'parameters file:\n', category='parameters')
        gt.logDictParams(clSoftcodedParams)
                
        log.debug('Calling the gifringe CL script for input list '+
                  clm.imageInsFiles(type='listFile'))
                
        gemini.gifringe(**clParamsDict)
                
        if gemini.gifringe.status:
            raise Errors.ScienceError('gifringe failed for inputs '+
                                      rc.inputs_as_str())
        else:
            log.info('Exited the gifringe CL script successfully')
                    
        # Rename CL outputs and load them back into memory 
        # and clean up the intermediate temp files written to disk
        # refOuts and arrayOuts are None here
        imageOuts, refOuts, arrayOuts = clm.finishCL() 
                
        ad_out = imageOuts[0]
                
        # Update GEM-TLM (automatic) and COMBINE time stamps to the PHU
        # and update logger with updated/added time stamps
        gt.mark_history(adinput=ad_out, keyword=keyword)

        adoutput_list.append(ad_out)
        return adoutput_list
    except:
        # log the exact message from the actual exception that was raised
        # in the try block. Then raise a general ScienceError with message.
        log.critical(repr(sys.exc_info()[1]))
        raise 

def scale_fringe_to_science(adinput=None, science=None, 
                            stats_section=None, stats_scale=True):
    """
    $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    THIS FUNCTION WAS ORIGINALLY GOING TO BE A GENERIC SCALE_A_TO_B, BUT IT WAS
    REALIZED THAT IT PERFORMED VERY FRINGE SPECIFIC CLIPPING DURING THE SCALING,
    THUS IT WAS RENAMED SCALE_FRINGE_TO_SCIENCE.  A VERSION OF THIS FUNCTION 
    THAT PERFORMS SPECIFIC THINGS FOR SKIES NEEDS TO BE CREATED, OR THIS 
    FUNCTION NEEDS TO BE MODIFIED TO WORK FOR BOTH AND RENAMED.  IDEALLY A 
    FUNCTION THAT COULD SCALE A TO B WOULD BE GREAT, BUT HARD TO ACCOMPLISH 
    WITHOUT ADDING A LARGE NUMBER OF PARAMETERS (IE CLUTTER).
    TO MAKE FUTURE REFACTORING EASIER SCIENCE INPUTS = B AND FRINGE = A, SO JUST
    THROUGH AND CONVERT PHRASES FOR SCIENCE BACK TO B AND SIMILAR FOR FRINGES.
    $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    
    
    This function will take the SCI extensions of the fringes and scale them
    up/down to match those of science.  There are two ways to find the 
    value to scale fringes by:
    1. If stats_scale is set to True, the equation:
    (letting science data = b (or B), and fringe = a (or A))
    
    arrayB = where({where[SCIb < (SCIb.median+2.5*SCIb.std)]} 
                        > [SCIb.median-3*SCIb.std])
    scale = arrayB.std / SCIa.std
    
    A section of the SCI arrays to use for calculating these statistics can
    be defined with stats_section, or the default; the default is the
    original SCI data excluding the outer 100 pixels on all 4 sides (so 
    less 200 pixels in width and height).
    
    2. If stats_scale=False, then scale will be calculated using:
    exposure time of science / exposure time of fringe
    
    The outputs of adoutput_list will be the scaled version of adinput.
    
    NOTE: There MUST be a matching number of inputs for science and fringes, 
    AND every pair of inputs MUST have matching size SCI frames.
    
    NOTE: If you are looking to simply perform basic scaling by a predetermined 
    value, rather than calculating it from a second set of inputs inside this
    function, then the .div(), .mult(), .sub() and .add() functions of the 
    arith.py toolbox in astrodata are perfect to perform such opperations. 
    
    :param adinput: fringe inputs to be scaled to those of science
    :type adinput: Astrodata objects, either a single or a list of objects
                   Note: there must be an equal number of science as adinput
    
    :param science: Astrodata inputs to have those of adinput scaled to.
    :type science: AstroData objects in a list, or a single instance.
                     Note: there must be an equal number of science as adinput
                     Note: no changes will be made to the science images.
                     
    :param stats_section: sections of detectors to use for calculating
                          the statistics
    :type stats_section: Dictionary of the format:
                         {('SCI',1):[x1,x2,y1,y2], ('SCI',2):[x1,x2,y1,y2], ...}
                         with every SCI extension having a data section 
                         defined. Default is the inner region 100 pixels
                         from all 4 sides of SCI data.
    
    :param stats_scale: Use statistics to calculate the scale values?
    :type stats_scale: Python boolean (True/False). Default, True.               
    """

    # instantiate log
    log = gemLog.getGeminiLog()

    # ensure that adinput and science are not None and make 
    # them into lists if they are not already
    adinput = gt.validate_input(adinput=adinput)
    science = gt.validate_input(adinput=science)

    # time stamp keyword
    keyword = 'SCALEFRG'
    
    # initialize output list
    adoutput_list = []
    
    try:

        # check the inputs have matching filters, binning and SCI shapes.
        gt.checkInputsMatch(adInsA=adinput, adInsB=science)

        # Loop through the inputs to perform scaling of fringes to the science
        # NOTE: for clarity and simplicity, fringes objects are type 'A' and 
        #       science input objects are type 'B'.
        count=0
        for adA in adinput:  
            
            # Clean up log and screen if multiple inputs
            log.fullinfo('+'*50, category='format')

            # set up empty dict to hold scale vals for each extension
            scaleDict = {}
            # get matching B input
            adB = science[count]
            log.info('Scaling this fringe to input (%s):\n%s' %
                        (adB.filename,adA.filename))
            
            for sciExtA in adA['SCI']:
                # Grab the A and B SCI extensions to operate on
                curExtver = sciExtA.extver()
                sciExtB = adB[('SCI', curExtver)]
                
                log.fullinfo('Scaling SCI extension '+str(curExtver))
                
                if stats_scale:
                    # use statistics to calculate the scaling factor, following
                    # arrayB = where({where[sciExtB < 
                    #                       (sciExtB.median+2.5*sciExtB.std)]} 
                    #                 > [sciExtB.median-3*sciExtB.std])
                    # scale = arrayB.std / sciExtA.std
                    log.info('Using statistics to calculate the ' +
                               'scaling factor')
                    # Get current SCI's stats_section
                    if stats_section is None:
                        # use default inner region
                        
                        # Get the data section as a int list of form:
                        # [y1, y2, x1, x2] 0-based and non-inclusive
                        datsecAlist = sciExtA.data_section().as_pytype()
                        dAl = datsecAlist
                        # Take 100 pixels off each side
                        curStatsecList = [dAl[0]+100,dAl[1]-100,
                                          dAl[2]+100,dAl[3]-100]
                    else:
                        # pull value from stats_section dict provided
                        if isinstance(stats_section,dict):
                            curStatsecList = stats_section[('SCI',curExtver)]
                        else:
                            raise Errors.InputError('stats_section must be ' +
                                                    'a dictionary. '+
                                                    'It was found to be a '+
                                                    str(type(stats_section)))
               
                    cl = curStatsecList
                    log.info('Using section '+repr(cl)+' of data to '+
                             'calculate the scaling factor')      
                    # pull the data arrays from the extensions, 
                    # for the stats_section region
                    A = sciExtA.data[cl[0]:cl[1],cl[2]:cl[3]]
                    B = sciExtB.data[cl[0]:cl[1],cl[2]:cl[3]]
                    # Must flatten for compatibility with older
                    # versions of numpy    

                    # B's median
                    Bmed = np.median(B.flatten()) 
                    # B's standard deviation
                    Bstd = B.std()

                    # make an array of all the points where the pixel value is 
                    # less than the median value + 2.5 x the standard deviation.
                    Bbelow = B[np.where(B<(Bmed+(2.5*Bstd)))]  

                    # make an array from the previous one where all the pixels  
                    # in it have a value greater than the median -3 x the 
                    # standard deviation. Thus a final array of all the pixels 
                    # with values between (median + 2.5xstd) and (median -3xstd)
                    Bmiddle = Bbelow[np.where(Bbelow>(Bmed-(3.*Bstd)))]

                    ######## NOTE: kathleen believes the median should #########
                    ########       be used below instead of the std    #########
                    ### This needs real scientific review and discussion with ##
                    ### DA's to make a decision as to what is appropriate/works#
                    curScale = Bmiddle.std() / A.std() 
                
                else:
                    # use the exposure times to calculate the scale
                    log.info('Using exposure times to calculate the scaling'+
                             ' factor')
                    curScale = sciExtB.exposure_time() / sciExtA.exposure_time()
                
                log.info('Scale factor found = '+str(curScale))
                
                # load determined scale for this extension into scaleDict    
                scaleDict[('SCI',sciExtA.extver())] = curScale
                
            # Use mult from the arith toolbox to perform the scaling of 
            # A (fringe input) to B (science input), it does deepcopy
            # so none needed here.
            ad_out = adA.mult(input_b=scaleDict)          
            
            # Update GEM-TLM (automatic) and SUBDARK time stamps to the PHU
            # and update logger with updated/added time stamps
            gt.mark_history(adinput=ad_out, keyword=keyword)
        
            # Append to output list
            adoutput_list.append(ad_out)
    
            count=count+1
                
        # Return the output list
        # These are the scaled fringe ad's
        return adoutput_list
    except:
        # log the exact message from the actual exception that was raised
        # in the try block. Then raise a general ScienceError with message.
        log.critical(repr(sys.exc_info()[1]))
        raise 

def subtract_fringe(adinput=None, fringe=None):
    """
    This function will subtract the SCI of the input fringes from each SCI frame
    of the inputs and take care of the VAR and DQ frames if they exist.  
    
    This is all conducted in pure Python through the arith "toolbox" of 
    astrodata. 

    Either a 'main' type logger object, if it exists, or a null logger 
    (ie, no log file, no messages to screen) will be retrieved/created 
    and used within this function.

    :param adinput: Astrodata input science data
    :type adinput: Astrodata objects, either a single or a list of objects
    
    :param fringe: The fringe(s) to be added to the input(s).
    :type fringe: AstroData objects in a list, or a single instance.
                Note: If there are multiple inputs and one fringe provided, 
                then the same fringe will be applied to all inputs; else the 
                fringe list must match the length of the inputs.
    """

    # instantiate log
    log = gemLog.getGeminiLog()

    # ensure that adinput and fringe are not None and make 
    # them into lists if they are not already
    adinput = gt.validate_input(adinput=adinput)
    fringe = gt.validate_input(adinput=fringe)

    # time stamp keyword
    keyword = 'SUBFRING'
    
    # initialize output list
    adoutput_list = []  
    
    try:

        # check the inputs have matching filters, binning and SCI shapes.
        gt.checkInputsMatch(adInsA=fringe, adInsB=adinput)
            
        # Loop through the input
        count=0
        for ad in adinput:  

            # Clean up log and screen if multiple inputs
            log.fullinfo('+'*50, category='format')    

            # Get the right fringe for this input
            if len(fringe)>1:
                this_fringe = fringe[count]
            else:
                this_fringe = fringe[0]
           
            # sub each fringe SCI  from each input SCI and handle the updates to
            # the DQ and VAR frames.
            # the sub function of the arith toolbox performs a deepcopy so
            # it doesn't need to be done here. 
            ad_out = ad.sub(this_fringe)
            
            # Update GEM-TLM (automatic) and SUBFRINGE time stamps to the PHU
            # and update logger with updated/added time stamps
            gt.mark_history(adinput=ad_out, keyword=keyword)
        
            # Append to output list
            adoutput_list.append(ad_out)
    
            count=count+1
                
        # Return the outputs list
        return adoutput_list
    except:
        # log the exact message from the actual exception that was raised
        # in the try block. Then raise a general ScienceError with message.
        log.critical(repr(sys.exc_info()[1]))
        raise
