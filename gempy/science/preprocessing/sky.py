# This module contains user level functions related to the preprocessing of
# the input dataset with a sky or fringe frame

import sys
import numpy as np
from astrodata import Errors
from gempy import geminiTools as gt
from gempy import managers as man

def subtract_fringe(adInputs, fringes=None, outNames=None, suffix=None):
    """
    This function will subtract the SCI of the input fringes from each SCI frame 
    of the inputs and take care of the VAR and DQ frames if they exist.  
    
    This is all conducted in pure Python through the arith "toolbox" of 
    astrodata. 
       
    Either a 'main' type logger object, if it exists, or a null logger 
    (ie, no log file, no messages to screen) will be retrieved/created in the 
    ScienceFunctionManager and used within this function.
    
    :param adInputs: Astrodata input science data
    :type adInputs: Astrodata objects, either a single or a list of objects
    
    :param fringes: The fringe(s) to be added to the input(s).
    :type fringes: AstroData objects in a list, or a single instance.
                Note: If there are multiple inputs and one fringe provided, 
                then the same fringe will be applied to all inputs; else the 
                fringes list must match the length of the inputs.
    
    :param outNames: filenames of output(s)
    :type outNames: String, either a single or a list of strings of same length 
                    as adInputs.
    
    :param suffix:
            string to add on the end of the input filenames 
            (or outNames if not None) for the output filenames.
    :type suffix: string
    """
    # Instantiate ScienceFunctionManager object
    sfm = man.ScienceFunctionManager(adInputs, outNames, suffix,
                                                    funcName='subtract_fringe') 
    # Perform start up checks of the inputs, prep/check of outnames, and get log
    adInputs, outNames, log = sfm.startUp()
    
    # casting fringes into a list if not one all ready for later indexing
    if not isinstance(fringes, list):
        fringes = [fringes]
    
    # checking the inputs have matching filters, binning and SCI shapes.
    gt.checkInputsMatch(adInsA=fringes, adInsB=adInputs)
    
    try:
        # Set up counter for looping through outNames list
        count=0
        
        # Creating empty list of ad's to be returned that will be filled below
        adOutputs=[]
        
        # Loop through the inputs to perform the non-linear and saturated
        # pixel searches of the SCI frames to update the BPM frames into
        # full DQ frames. 
        for ad in adInputs:  
            # Getting the right fringe for this input
            if len(fringes)>1:
                fringe = fringes[count]
            else:
                fringe = fringes[0]
           
            # sub each fringe SCI  from each input SCI and handle the updates to 
            # the DQ and VAR frames.
            # the sub function of the arith toolbox performs a deepcopy so
            # it doesn't need to be done here. 
            adOut = ad.sub(fringe)
            
            # renaming the output ad filename
            adOut.filename = outNames[count]
                    
            log.status('File name updated to '+adOut.filename+'\n')
            
            # Updating GEM-TLM (automatic) and SUBFRINGE time stamps to the PHU
            # and updating logger with updated/added time stamps
            sfm.markHistory(adOutputs=adOut, historyMarkKey='SUBFRING')
        
            # Appending to output list
            adOutputs.append(adOut)
    
            count=count+1
                
        log.status('**FINISHED** the subtract_fringe function')
        # Return the outputs (list or single, matching adInputs)
        return adOutputs
    except:
        # logging the exact message from the actual exception that was raised
        # in the try block. Then raising a general ScienceError with message.
        log.critical(repr(sys.exc_info()[1]))
        raise 
    
def scale_fringe_to_science(fringes=None, sciInputs=None, statsec=None, 
                                    statScale=True, outNames=None, suffix=None):
    """
    $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    THIS FUNCTION WAS ORIGINALLY GOING TO BE A GENERIC SCALE_A_TO_B, BUT IT WAS
    REALIZED THAT IT PERFORMED VERY FRINGE SPECIFIC CLIPPING DURING THE SCALING,
    THUS IT WAS RENAMED SCALE_FRINGE_TO_SCIENCE.  A VERSION OF THIS FUNCTION 
    THAT PERFORMS SPECIFIC THINGS FOR SKY'S NEEDS TO BE CREATED, OR THIS 
    FUNCTION NEEDS TO BE MODIFIED TO WORK FOR BOTH AND RENAMED.  IDEALLY A 
    FUNCTION THAT COULD SCALE A TO B WOULD BE GREAT, BUT HARD TO ACCOMPLISH 
    WITHOUT ADDING A LARGE NUMBER OF PARAMETERS (IE CLUTTER).
    TO MAKE FUTURE REFACTORING EASIER SCIENCE INPUTS = B AND FRINGE = A, SO JUST
    THROUGH AND CONVERT PHRASES FOR SCIENCE BACK TO B AND SIMILAR FOR FRINGES.
    $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    
    
    This function will take the SCI extensions of the fringes and scale them
    up/down to match those of sciInputs.  There are two ways to find the 
    value to scale fringes by:
    1. If statScale is set to True, the equation:
    (letting science data = b (or B), and fringe = a (or A))
    
    arrayB = where({where[SCIb < (SCIb.median+2.5*SCIb.std)]} > [SCIb.median-3*SCIb.std])
    scale = arrayB.std / SCIa.std
    
    A section of the SCI arrays to use for calculating these statistics can
    be defined with statsec, or the default; the default is the original SCI
    data excluding the outer 100 pixels on all 4 sides (so less 200 pixels in  
    width and height).
    
    2. If statScale=False, then scale will be calculated using:
    exposure time of science / exposure time of fringe
    
    The outputs of adOutputs will be the scaled version of fringes.
    
    NOTE: There MUST be a matching number of inputs for sciInputs and fringes, 
    AND every pair of inputs MUST have matching size SCI frames.
    
    NOTE: If you are looking to simply perform basic scaling by a predetermined 
    value, rather than calculating it from a second set of inputs inside this
    function, then the .div(), .mult(), .sub() and .add() functions of the 
    arith.py toolbox in astrodata are perfect to perform such opperations. 
    
    :param fringes: fringe inputs to be scaled to those of sciInputs
    :type fringes: Astrodata objects, either a single or a list of objects
                   Note: there must be an equal number of sciInputs as fringes
    
    :param sciInputs: Astrodata inputs to have those of adInputsA scaled to.
    :type sciInputs: AstroData objects in a list, or a single instance.
                     Note: there must be an equal number of sciInputs as fringes
                     Note: no changes will be made to the sciInputs.
                     
    :param statsec: sections of detectors to use for calculating the statistics
    :type statsec: 
    Dictionary of the format:
    {(SCI,1):[x1:x2,y1:y2], (SCI,2):[x1:x2,y1:y2], ...} 
    with every SCI extension having a data section defined.
    Default is the inner region 100pixels from all 4 sides of SCI data.
    
    :param statScale: Use statistics to calculate the scale values?
    :type statScale: Python boolean (True/False). Default, True.               
    
    :param outNames: filenames of output(s)
    :type outNames: String, either a single or a list of strings of same length 
                    as adInputs.
    
    :param suffix: string to add on the end of the input filenames 
                   (or outNames if not None) for the output filenames.
    :type suffix: string
    
    """
    # Instantiate ScienceFunctionManager object
    sfm = man.ScienceFunctionManager(fringes, outNames, suffix,
                                            funcName='scale_fringe_to_science') 
    # Perform start up checks of the inputs, prep/check of outnames, and get log
    fringes, outNames, log = sfm.startUp()
    
    # casting sciInputs into a list if not one all ready for later indexing
    if not isinstance(sciInputs, list):
        sciInputs = [sciInputs]
    
    # checking the inputs have matching filters, binning and SCI shapes.
    gt.checkInputsMatch(adInsA=fringes, adInsB=sciInputs)
    
    try:
        # Set up counter for looping through outNames list
        count=0
        
        # Creating empty list of ad's to be returned that will be filled below
        adOutputs=[]
        
        # Loop through the inputs to perform scaling of fringes to the sciInputs
        # NOTE: for clarity and simplicity, fringes objects are type 'A' and 
        #       science input objects are type 'B'.
        for adA in fringes:  
            # set up empty dict to hold scale vals for each extension
            scaleDict = {}
            # get matching B input
            adB = sciInputs[count]
            
            log.fullinfo('\n'+'*'*50)
            log.status('Starting to scale '+adA.filename+' to match '+
                                                                adB.filename)
            
            for sciExtA in adA['SCI']:
                # Grab the A and B SCI extensions to opperate on
                SCIa = sciExtA
                curExtver = sciExtA.extver()
                SCIb = adB[('SCI', curExtver)]
                
                log.fullinfo('Scaling SCI extension '+str(curExtver))
                
                if statScale:
                    # use statistics to calculate the scaling factor, following
                    # arrayB = where({where[SCIb < (SCIb.median+2.5*SCIb.std)]} 
                    # > [SCIb.median-3*SCIb.std])
                    # scale = arrayB.std / SCIa.std
                    log.status('Using statistics to calculate the scaling'+
                                                                    ' factor')
                    # Get current SCI's statsec
                    if statsec is None:
                        # use default inner region
                        
                        # Getting the data section as a int list of form:
                        # [y1, y2, x1, x2] 0-based and non-inclusive
                        datsecAlist = sciExtA.data_section().as_pytype()
                        dAl = datsecAlist
                        # Take 100 pixels off each side
                        curStatsecList = [dAl[0]+100,dAl[1]-100,dAl[2]+100,
                                         dAl[3]-100]
                    else:
                        # pull value from statsec dict provided
                        if isinstance(statsec,dict):
                            curStatsecList = statsec[('SCI',curExtver)]
                        else:
                            log.critical('statsec must be a dictionary, it '+
                                         'was found to be a '+
                                         str(type(statsec)))
                            raise Errors.ScienceError()
               
                    cl = curStatsecList  
                    log.stdinfo('Using section '+repr(cl)+' of data to '+
                                'calculate the scaling factor')      
                    # pull the data arrays from the extensions, 
                    # for the statsec region
                    A = SCIa.data[cl[0]:cl[1],cl[2]:cl[3]]
                    B = SCIb.data[cl[0]:cl[1],cl[2]:cl[3]]
                    # Must flatten because incase using older verion of numpy    
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
                    log.status('Using exposure times to calculate the scaling'+
                               ' factor')
                    curScale = SCIb.exposure_time() / SCIa.exposure_time()
                
                log.stdinfo('Scale factor found = '+str(curScale))
                
                # load determined scale for this extension into scaleDict    
                scaleDict[('SCI',sciExtA.extver())] = curScale
                
            # Using mult from the arith toolbox to perform the scaling of 
            # A (fringe input) to B (science input), it does deepcopy
            # so none needed here.
            adOut = adA.mult(input_b=scaleDict)          
            
            # renaming the output ad filename
            adOut.filename = outNames[count]
                    
            log.status('File name updated to '+adOut.filename+'\n')
            
            # Updating GEM-TLM (automatic) and SUBDARK time stamps to the PHU
            # and updating logger with updated/added time stamps
            sfm.markHistory(adOutputs=adOut, historyMarkKey='SCALEA2B')
        
            # Appending to output list
            adOutputs.append(adOut)
    
            count=count+1
                
        log.status('**FINISHED** the scale_fringe_to_science function')
        # Return the outputs (list or single, matching adInputs)
        # These are the scaled fringe ad's
        return adOutputs
    except:
        # logging the exact message from the actual exception that was raised
        # in the try block. Then raising a general ScienceError with message.
        log.critical(repr(sys.exc_info()[1]))
        raise 
