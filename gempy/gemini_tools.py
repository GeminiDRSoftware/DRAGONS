import os, sys
import re
from copy import deepcopy
import pyfits as pf
import numpy as np
import tempfile
import astrodata
from astrodata import Lookups
from astrodata.adutils import gemLog
from astrodata.ConfigSpace import lookup_path
from astrodata.AstroData import AstroData
from astrodata import Errors

# Load the standard comments for header keywords that will be updated
# in these functions
keyword_comments = Lookups.get_lookup_table("Gemini/keyword_comments",
                                            "keyword_comments")

def add_objcat(adinput=None, extver=1, replace=False, columns=None):
    """
    Add OBJCAT table if it does not exist, update or replace it if it does.
    
    :param adinput: AD object(s) to add table to
    :type adinput: AstroData objects, either a single instance or a list
    
    :param extver: Extension number for the table (should match the science
                   extension).
    :type extver: int
    
    :param replace: Flag to determine if an existing OBJCAT should be
                    replaced or updated in place. If replace=False, the
                    length of all lists provided must match the number
                    of entries currently in OBJCAT.
    :type replace: boolean
    
    :param columns: Columns to add to table.  Columns named 'X_IMAGE',
                    'Y_IMAGE','X_WORLD','Y_WORLD' are required if making
                    new table.
    :type columns: dictionary of Pyfits Column objects with column names
                   as keys
    """
    
    # Instantiate the log. This needs to be done outside of the try block,
    # since the log object is used in the except block 
    log = gemLog.getGeminiLog()
    
    # The validate_input function ensures that adinput is not None and returns
    # a list containing one or more AstroData objects
    adinput = validate_input(adinput=adinput)
    
    # Initialize the list of output AstroData objects
    adoutput_list = []
    try:
        
        # Parse sextractor parameters for the list of expected columns
        expected_columns = parse_sextractor_param()

        # Append a few more that don't come from directly from detectSources
        expected_columns.extend(["REF_NUMBER","REF_MAG","REF_MAG_ERR"])
        
        # Loop over each input AstroData object in the input list
        for ad in adinput:
            
            # Check if OBJCAT already exists and just update if desired
            objcat = ad["OBJCAT",extver]
            if objcat and not replace:
                log.fullinfo("Table already exists; updating values.")
                for name in columns.keys():
                    objcat.data.field(name)[:] = columns[name].array
            else:
            
                # Make new table: x, y, ra, dec required
                x = columns.get("X_IMAGE",None)
                y = columns.get("Y_IMAGE",None)
                ra = columns.get("X_WORLD",None)
                dec = columns.get("Y_WORLD",None)
                if x is None or y is None or ra is None or dec is None:
                    raise Errors.InputError("Columns X_IMAGE, Y_IMAGE, "\
                                            "X_WORLD, Y_WORLD must be present.")

                # Append columns in order of definition in sextractor params
                table_columns = []
                nlines = len(x.array)
                for name in expected_columns:
                    if name in ["NUMBER"]:
                        default = range(1,nlines+1)
                        format = "J"
                    elif name in ["FLAGS","IMAFLAGS_ISO","REF_NUMBER"]:
                        default = [-999]*nlines
                        format = "J"
                    else:
                        default = [-999]*nlines
                        format = "E"

                    # Get column from input if present, otherwise
                    # define a new Pyfits column with sensible placeholders
                    data = columns.get(name,
                                       pf.Column(name=name,format=format,
                                                 array=default))
                    table_columns.append(data)

                # Make new pyfits table
                col_def = pf.ColDefs(table_columns)
                tb_hdu = pf.new_table(col_def)
                tb_ad = AstroData(tb_hdu)
                tb_ad.rename_ext("OBJCAT",extver)
            
                # Replace old version or append new table to AD object
                if objcat:
                    log.fullinfo("Replacing existing OBJCAT in %s" % 
                                 ad.filename)
                    ad.remove(("OBJCAT",extver))
                ad.append(tb_ad)
            
            # Append the output AstroData object to the list of output
            # AstroData objects
            adoutput_list.append(ad)
        
        # Return the list of output AstroData objects
        return adoutput_list
    except:
        # Log the message from the exception
        log.critical(repr(sys.exc_info()[1]))
        raise

def array_information(adinput=None):
    # Instantiate the log. This needs to be done outside of the try block,
    # since the log object is used in the except block 
    log = gemLog.getGeminiLog()
    
    # The validate_input function ensures that adinput is not None and returns
    # a list containing one or more AstroData objects
    adinput = validate_input(adinput=adinput)
    
    # Initialize the list of dictionaries of output array numbers
    # Keys will be (extname,extver)
    array_info_list = []
    try:
        # Loop over each input AstroData object in the input list
        for ad in adinput:

            arrayinfo = {}

            # Get the number of science extensions
            nsciext = ad.count_exts("SCI")

            # Get the correct order of the extensions by sorting on
            # the first element in detector section
            # (raw ordering is whichever amps read out first)
            detsecs = ad.detector_section().as_list()
            if not isinstance(detsecs[0],list):
                detsecs = [detsecs]
            detx1 = [sec[0] for sec in detsecs]
            ampsorder = range(1,nsciext+1)
            orderarray = np.array(
                zip(ampsorder,detx1),dtype=[('ext',np.int),('detx1',np.int)])
            orderarray.sort(order='detx1')
            if np.all(ampsorder==orderarray['ext']):
                in_order = True
            else:
                ampsorder = orderarray['ext']
                in_order = False
                
            # Get array sections for determining when
            # a new array is found
            arraysecs = ad.array_section().as_list()
            if not isinstance(arraysecs[0],list):
                arraysecs = [arraysecs]
            if len(arraysecs)!=nsciext:
                arraysecs*=nsciext
            arrayx1 = [sec[0] for sec in arraysecs]

            # Initialize these so that first extension will always
            # start a new array
            last_detx1 = detx1[ampsorder[0]-1]-1
            last_arrayx1 = arrayx1[ampsorder[0]-1]

            arraynum = {}
            amps_per_array = {}
            num_array = 0
            for i in ampsorder:
                sciext = ad["SCI",i]
                this_detx1 = detx1[i-1]
                this_arrayx1 = arrayx1[i-1]
                
                if (this_detx1>last_detx1 and this_arrayx1<=last_arrayx1):
                    # New array found
                    num_array += 1
                    amps_per_array[num_array] = 1
                else:
                    amps_per_array[num_array] += 1
                
                arraynum[(sciext.extname(),sciext.extver())] = num_array

            # Reference extension if tiling/mosaicing all data together
            try:
                refext = ampsorder[int((amps_per_array[2]+1)/2.0-1)
                                   + amps_per_array[1]]
            except KeyError:
                refext = None

            arrayinfo['array_number'] = arraynum
            arrayinfo['amps_order'] = ampsorder
            arrayinfo['amps_per_array'] = amps_per_array
            arrayinfo['reference_extension'] = refext

            # Append the output AstroData object to the list of output
            # AstroData objects
            array_info_list.append(arrayinfo)
        
        # Return the list of output AstroData objects
        return array_info_list
    except:
        # Log the message from the exception
        log.critical(repr(sys.exc_info()[1]))
        raise
  
def check_inputs_match(ad1=None, ad2=None, check_filter=True):
    """
    This function will check if the inputs match.  It will check the filter,
    binning and shape/size of the every SCI frames in the inputs.
    
    There must be a matching number of inputs for 1 and 2.
    
    :param ad1: input astrodata instance(s) to be check against ad2
    :type ad1: AstroData objects, either a single or a list of objects
                Note: inputs 1 and 2 must be matching length lists or single 
                objects
    
    :param ad2: input astrodata instance(s) to be check against ad1
    :type ad2: AstroData objects, either a single or a list of objects
                  Note: inputs 1 and 2 must be matching length lists or single 
                  objects
    """
    log = gemLog.getGeminiLog() 
    
    # Check inputs are both matching length lists or single objects
    if (ad1 is None) or (ad2 is None):
        log.error('Inputs ad1 and ad2 must not be None')
        raise Errors.ToolboxError('Either inputs ad1 or ad2 was None')
    if isinstance(ad1,list):
        if isinstance(ad2,list):
            if len(ad1)!=len(ad2):
                log.error('Both ad1 and ad2 inputs must be lists of MATCHING'+
                          ' lengths.')
                raise Errors.ToolboxError('There were mismatched numbers ' \
                                          'of ad1 and ad2 inputs.')
    if isinstance(ad1,AstroData):
        if isinstance(ad2,AstroData):
            # casting both ad1 and ad2 inputs to lists for looping later
            ad1 = [ad1]
            ad2 = [ad2]
        else:
            log.error('Both ad1 and ad2 inputs must be lists of MATCHING'+
                      ' lengths.')
            raise Errors.ToolboxError('There were mismatched numbers of '+
                               'ad1 and ad2 inputs.')
    
    for count in range(0,len(ad1)):
        A = ad1[count]
        B = ad2[count]
        log.fullinfo('Checking inputs '+A.filename+' and '+B.filename)
        
        if A.count_exts('SCI')!=B.count_exts('SCI'):
            log.error('Inputs have different numbers of SCI extensions.')
            raise Errors.ToolboxError('Mismatching number of SCI ' \
                                      'extensions in inputs')
        for sciA in A["SCI"]:
            # grab matching SCI extensions from A's and B's
            extCount = sciA.extver()
            sciB = B[('SCI',extCount)]
            
            log.fullinfo('Checking SCI extension '+str(extCount))
            
            # Check shape/size
            if sciA.data.shape!=sciB.data.shape:
                log.error('Extensions have different shapes')
                raise Errors.ToolboxError('Extensions have different shape')
            
            # Check binning
            aX = sciA.detector_x_bin()
            aY = sciA.detector_y_bin()
            bX = sciB.detector_x_bin()
            bY = sciB.detector_y_bin()
            if (aX!=bX) or (aY!=bY):
                log.error('Extensions have different binning')
                raise Errors.ToolboxError('Extensions have different binning')
        
            # Check filter if desired
            if check_filter:
                if (sciA.filter_name().as_pytype() != 
                    sciB.filter_name().as_pytype()):
                    log.error('Extensions have different filters')
                    raise Errors.ToolboxError('Extensions have different ' +
                                              'filters')
        
        log.fullinfo('Inputs match')    


def clip_auxiliary_data(adinput=None, aux=None, aux_type=None):
    """
    This function clips auxiliary data like calibration files or BPMs
    to the size of the data section in the science.  It will pad auxiliary
    data if required to match un-overscan-trimmed data, but otherwise
    requires that the auxiliary data contain the science data.
    """
    # Instantiate the log. This needs to be done outside of the try block,
    # since the log object is used in the except block 
    log = gemLog.getGeminiLog()
    
    # The validate_input function ensures that the input is not None and
    # returns a list containing one or more AstroData objects
    adinput = validate_input(adinput=adinput)
    aux = validate_input(adinput=aux)

    # Create a dictionary that has the AstroData objects specified by adinput
    # as the key and the AstroData objects specified by aux as the value
    aux_dict = make_dict(key_list=adinput, value_list=aux)
    
    # Initialize the list of output AstroData objects
    aux_output_list = []
 
    try:
        
        # Check aux_type parameter for valid value
        if aux_type is None:
            raise Errors.InputError("The aux_type parameter must not be None")

        # If dealing with BPMs, relevant extensions are DQ;
        # otherwise use SCI
        aux_type = aux_type.lower()
        if aux_type=="bpm":
            extname = "DQ"
        else:
            extname = "SCI"

        # Loop over each input AstroData object in the input list
        for ad in adinput:

            # Get the associated auxiliary file
            this_aux = aux_dict[ad]

            # Make a new blank auxiliary file for appending to
            new_aux = AstroData()
            new_aux.filename = this_aux.filename
            new_aux.phu = this_aux.phu

            # Get the necessary section information from descriptors
            # This should be done outside the loop over extensions
            # for efficiency

            # For the science file
            sci_detsec_dv = ad.detector_section()
            if sci_detsec_dv is None:
                raise Errors.InputError("Input file %s does " \
                                        "not have a detector section" %
                                        ad.filename)
            else:
                detsec_kw = sci_detsec_dv.keyword
                sci_detsec_dict = sci_detsec_dv.dict_val

            sci_datasec_dv = ad.data_section()
            if sci_datasec_dv is None:
                raise Errors.InputError("Input file %s does " \
                                        "not have a data section" %
                                        ad.filename)
            else:
                datasec_kw = sci_datasec_dv.keyword
                sci_datasec_dict = sci_datasec_dv.dict_val

            sci_arraysec_dv = ad.array_section()
            if sci_arraysec_dv is None:
                raise Errors.InputError("Input file %s does " \
                                        "not have an array section" %
                                        ad.filename)
            else:
                arraysec_kw = sci_arraysec_dv.keyword
                sci_arraysec_dict = sci_arraysec_dv.dict_val

            sci_xbin_dv = ad.detector_x_bin()
            if sci_xbin_dv is None:
                raise Errors.InputError("Input file %s does " \
                                        "not have an x-binning" %
                                        ad.filename)
            else:
                sci_xbin_dict = sci_xbin_dv.dict_val

            sci_ybin_dv = ad.detector_y_bin()
            if sci_ybin_dv is None:
                raise Errors.InputError("Input file %s does " \
                                        "not have a y-binning" %
                                        ad.filename)
            else:
                sci_ybin_dict = sci_ybin_dv.dict_val

            # For the auxiliary file
            aux_detsec_dv = this_aux.detector_section(extname=extname)
            if aux_detsec_dv is None:
                raise Errors.InputError("Input file %s does " \
                                        "not have a detector section" %
                                        ad.filename)
            else:
                aux_detsec_dict = aux_detsec_dv.dict_val

            aux_datasec_dv = this_aux.data_section(extname=extname)
            if aux_datasec_dv is None:
                raise Errors.InputError("Input file %s does " \
                                        "not have a data section" %
                                        ad.filename)
            else:
                aux_datasec_dict = aux_datasec_dv.dict_val

            aux_arraysec_dv = this_aux.array_section(extname=extname)
            if aux_arraysec_dv is None:
                raise Errors.InputError("Input file %s does " \
                                        "not have an array section" %
                                        ad.filename)
            else:
                aux_arraysec_dict = aux_arraysec_dv.dict_val
            
            for sciext in ad["SCI"]:

                # Get the section information for this extension
                # from the dictionary formed above
                dict_key = (sciext.extname(),sciext.extver())
                sci_detsec = sci_detsec_dict[dict_key]
                sci_datasec = sci_datasec_dict[dict_key]
                sci_arraysec = sci_arraysec_dict[dict_key]

                # Array section is unbinned; to use as indices for
                # extracting data, need to divide by the binning
                xbin = sci_xbin_dict[dict_key]
                ybin = sci_ybin_dict[dict_key]
                sci_arraysec = [sci_arraysec[0]/xbin,
                                sci_arraysec[1]/xbin,
                                sci_arraysec[2]/ybin,
                                sci_arraysec[3]/ybin]
                

                # Check whether science data has been overscan-trimmed
                sci_shape = sciext.data.shape
                if (sci_shape[1]==sci_datasec[1] and 
                    sci_shape[0]==sci_datasec[3] and
                    sci_datasec[0]==0 and
                    sci_datasec[2]==0):
                    sci_trimmed = True
                    sci_offsets = [0,0,0,0]
                else:
                    sci_trimmed = False

                    # Offsets give overscan regions on either side of data:
                    # [left offset, right offset, bottom offset, top offset]
                    sci_offsets = [sci_datasec[0],sci_shape[1]-sci_datasec[1],
                                   sci_datasec[2],sci_shape[0]-sci_datasec[3]]

                found = False
                for auxext in this_aux[extname]:
                    
                    # Get the section information for this extension
                    dict_key = (auxext.extname(),auxext.extver())
                    aux_detsec = aux_detsec_dict[dict_key]
                    aux_datasec = aux_datasec_dict[dict_key]
                    aux_arraysec = aux_arraysec_dict[dict_key]

                    # Array section is unbinned; to use as indices for
                    # extracting data, need to divide by the binning
                    aux_arraysec = [aux_arraysec[0]/xbin,
                                    aux_arraysec[1]/xbin,
                                    aux_arraysec[2]/ybin,
                                    aux_arraysec[3]/ybin]

                    # Check whether auxiliary detector section contains
                    # science detector section
                    if (aux_detsec[0] <= sci_detsec[0] and # x lower
                        aux_detsec[1] >= sci_detsec[1] and # x upper
                        aux_detsec[2] <= sci_detsec[2] and # y lower
                        aux_detsec[3] >= sci_detsec[3]):   # y upper

                        # Auxiliary data contains or is equal to science data
                        found=True
                    else:
                        continue

                    # Check whether auxiliary data has been overscan-trimmed
                    aux_shape = auxext.data.shape
                    if (aux_shape[1]==aux_datasec[1] and 
                        aux_shape[0]==aux_datasec[3] and
                        aux_datasec[0]==0 and
                        aux_datasec[2]==0):
                        aux_trimmed = True
                        aux_offsets = [0,0,0,0]
                    else:
                        aux_trimmed = False

                        # Offsets give overscan regions on either side of data:
                        # [left offset, right offset, bottom offset, top offset]
                        aux_offsets = [aux_datasec[0],
                                       aux_shape[1]-aux_datasec[1],
                                       aux_datasec[2],
                                       aux_shape[0]-aux_datasec[3]]

                    # Define data extraction region corresponding to science
                    # data section (not including overscan)
                    x_translation = sci_arraysec[0] - sci_datasec[0] \
                                    - aux_arraysec[0] + aux_datasec[0]
                    y_translation = sci_arraysec[2] - sci_datasec[2] \
                                    - aux_arraysec[2] + aux_datasec[2]
                    region = [sci_datasec[2] + y_translation,
                              sci_datasec[3] + y_translation,
                              sci_datasec[0] + x_translation,
                              sci_datasec[1] + x_translation]

                    # Deepcopy auxiliary SCI plane
                    # and auxiliary VAR/DQ planes if they exist
                    # (in the non-BPM case)
                    # This must be done here so that the same
                    # auxiliary extension can be used for a
                    # different science extension; without the
                    # deepcopy, the original auxiliary extension
                    # gets clipped
                    ext_to_clip = [deepcopy(auxext)]
                    if aux_type!="bpm":
                        varext = this_aux["VAR",auxext.extver()]
                        if varext is not None:
                            ext_to_clip.append(deepcopy(varext))
                            
                        dqext = this_aux["DQ",auxext.extver()]
                        if dqext is not None:
                            ext_to_clip.append(deepcopy(dqext))

                    # Clip all relevant extensions
                    for ext in ext_to_clip:

                        # Pull out specified region
                        clipped = ext.data[region[0]:region[1],
                                           region[2]:region[3]]
                        
                        # Stack with overscan region if needed
                        if aux_trimmed and not sci_trimmed:

                            # Pad DQ planes with zeros to match
                            # science shape
                            # Note: this only allows an overscan
                            # region at one edge of the data array.
                            # If there ends up being more
                            # than one for some instrument, this code
                            # will have to be revised.
                            if aux_type=="bpm":
                                if sci_offsets[0]>0:
                                    # Left-side overscan
                                    overscan = np.zeros((sci_shape[0],
                                                         sci_offsets[0]),
                                                        dtype=np.int16)
                                    ext.data = np.hstack([overscan,clipped])
                                elif sci_offsets[1]>0:
                                    # Right-side overscan
                                    overscan = np.zeros((sci_shape[0],
                                                         sci_offsets[1]),
                                                        dtype=np.int16)
                                    ext.data = np.hstack([clipped,overscan])
                                elif sci_offsets[2]>0:
                                    # Bottom-side overscan
                                    overscan = np.zeros((sci_offsets[2],
                                                         sci_shape[1]),
                                                        dtype=np.int16)
                                    ext.data = np.vstack([clipped,overscan])
                                elif sci_offsets[3]>0:
                                    # Top-side overscan
                                    overscan = np.zeros((sci_offsets[3],
                                                         sci_shape[1]),
                                                        dtype=np.int16)
                                    ext.data = np.vstack([overscan,clipped])
                            else:
                                # Science decision: trimmed calibrations
                                # can't be meaningfully matched to untrimmed
                                # science data
                                raise Errors.ScienceError(
                                    "Auxiliary data %s is trimmed, but " \
                                    "science data %s is untrimmed." %
                                    (auxext.filename,sciext.filename))

                        elif not sci_trimmed:
                            
                            # Pick out overscan region corresponding
                            # to data section from auxiliary data
                            if aux_offsets[0]>0:
                                if aux_offsets[0]!=sci_offsets[0]:
                                    raise Errors.ScienceError(
                                        "Overscan regions do not match in " \
                                        "%s, %s" % 
                                        (auxext.filename,sciext.filename))

                                # Left-side overscan: height is full ylength,
                                # width comes from 0 -> offset
                                overscan = ext.data[region[0]:region[1],
                                                    0:aux_offsets[0]]
                                ext.data = np.hstack([overscan,clipped])

                            elif aux_offsets[1]>0:
                                if aux_offsets[1]!=sci_offsets[1]:
                                    raise Errors.ScienceError(
                                        "Overscan regions do not match in " \
                                        "%s, %s" % 
                                        (auxext.filename,sciext.filename))

                                # Right-side overscan: height is full ylength,
                                # width comes from xlength-offset -> xlength
                                overscan = ext.data[region[0]:region[1],
                                    aux_shape[1]-aux_offsets[1]:aux_shape[1]]
                                ext.data = np.hstack([clipped,overscan])

                            elif aux_offsets[2]>0: 
                                if aux_offsets[2]!=sci_offsets[2]:
                                    raise Errors.ScienceError(
                                        "Overscan regions do not match in " \
                                        "%s, %s" % 
                                        (auxext.filename,sciext.filename))

                                # Bottom-side overscan: width is full xlength,
                                # height comes from 0 -> offset
                                overscan = ext.data[0:aux_offsets[2],
                                                    region[2]:region[3]]
                                ext.data = np.vstack([clipped,overscan])

                            elif aux_offsets[3]>0:
                                if aux_offsets[3]!=sci_offsets[3]:
                                    raise Errors.ScienceError(
                                        "Overscan regions do not match in " \
                                        "%s, %s" % 
                                        (auxext.filename,sciext.filename))

                                # Top-side overscan: width is full xlength,
                                # height comes from ylength-offset -> ylength
                                overscan = ext.data[
                                    aux_shape[0]-aux_offsets[3]:aux_shape[0],
                                    region[2]:region[3]]
                                ext.data = np.vstack([overscan,clipped])

                        else:
                            # No overscan needed, just use the clipped region
                            ext.data = clipped

                        # Set the section keywords as appropriate
                        if sciext.get_key_value(datasec_kw) is not None:
                            ext.set_key_value(datasec_kw,
                                              sciext.header[datasec_kw],
                                              keyword_comments[datasec_kw])
                        if sciext.get_key_value(detsec_kw) is not None:
                            ext.set_key_value(detsec_kw,
                                              sciext.header[detsec_kw],
                                              keyword_comments[detsec_kw])
                        if sciext.get_key_value(arraysec_kw) is not None:
                            ext.set_key_value(arraysec_kw,
                                              sciext.header[arraysec_kw],
                                              keyword_comments[arraysec_kw])
        
                        # Rename the auxext to the science extver
                        ext.rename_ext(name=ext.extname(),ver=sciext.extver())
                        new_aux.append(ext)

                if not found:
                    raise Errors.ScienceError("No auxiliary data in %s "\
                                              "matches the detector section "\
                                              "%s in %s[SCI,%d]" % 
                                              (this_aux.filename,
                                               sci_detsec,
                                               ad.filename,
                                               sciext.extver()))

            new_aux.refresh_types()
            aux_output_list.append(new_aux)

        return aux_output_list    

    except:
        # Log the message from the exception
        log.critical(repr(sys.exc_info()[1]))
        raise
                    
def clip_sources(ad):
    """
    This function takes the source data from the OBJCAT and returns the best
    sources for IQ measurement.
    
    :param ad: input image
    :type ad: AstroData instance with OBJCAT attached
    """

    good_source = {}
    for sciext in ad["SCI"]:
        extver = sciext.extver()

        objcat = ad["OBJCAT",extver]
        if objcat is None:
            continue
        if objcat.data is None:
            continue

        x = objcat.data.field("X_IMAGE")
        y = objcat.data.field("Y_IMAGE")
        fwhm_pix = objcat.data.field("FWHM_IMAGE")
        fwhm_arcsec = objcat.data.field("FWHM_WORLD")
        ellip = objcat.data.field("ELLIPTICITY")
        sxflag = objcat.data.field("FLAGS")
        dqflag = objcat.data.field("IMAFLAGS_ISO")
        class_star = objcat.data.field("CLASS_STAR")
        area = objcat.data.field("ISOAREA_IMAGE")

        # Source is good if ellipticity defined and <0.5
        eflag = np.where((ellip>0.5)|(ellip==-999),1,0)

        # Source is good if probability of being a star >0.6
        sflag = np.where(class_star<0.6,1,0)

        flags = sxflag | eflag | sflag

        # Source is good if greater than 10 connected pixels
        # Ignore criterion if all undefined (-999)
        if not np.all(area==-999):
            aflag = np.where(area<100,1,0)
            flags |= aflag

        # Source is good if not flagged in DQ plane
        # Ignore criterion if all undefined (-999)
        if not np.all(dqflag==-999):
            flags |= dqflag

        # Use flag=0 to find good data
        good = (flags==0)
        rec = np.rec.fromarrays(
            [x[good],y[good],fwhm_pix[good],fwhm_arcsec[good],ellip[good]],
            names=["x","y","fwhm","fwhm_arcsec","ellipticity"])

        # Clip outliers, in FWHM
        num_total = len(rec)
        if num_total>=3:

            data = rec["fwhm_arcsec"]
            mean = data.mean()
            sigma = data.std()

            num = num_total
            clipped_rec = rec
            clip = 0
            while (num>0.5*num_total):
                clipped_rec = rec[(data<mean+sigma) & (data>mean-3*sigma)]
                num = len(clipped_rec)

                if num>0:
                    mean = clipped_rec["fwhm_arcsec"].mean() 
                    sigma = clipped_rec["fwhm_arcsec"].std()
                    previous_rec = clipped_rec
                elif clip==0:
                    clipped_rec = rec
                    break
                else:
                    clipped_rec = previous_rec
                    break

                clip+=1
                if clip>10:
                    break

            rec = clipped_rec

        # Store data
        good_source[("SCI",extver)] = rec

    return good_source



def convert_to_cal_header(adinput=None, caltype=None):
    """
    This function replaces position, object, and program information 
    in the headers of processed calibration files that are generated
    from science frames, eg. fringe frames, maybe sky frames too.
    It is called, for example, from the storeProcessedFringe primitive.

    :param adinput: astrodata instance to perform header key updates on
    :type adinput: an AstroData instance

    :param caltype: type of calibration.  Accepted values are 'fringe',
                    'sky', or 'flat'
    :type caltype: string
    """

    # Instantiate the log. This needs to be done outside of the try block,
    # since the log object is used in the except block 
    log = gemLog.getGeminiLog()
    
    # The validate_input function ensures that the input is not None and
    # returns a list containing one or more AstroData objects
    adinput = validate_input(adinput=adinput)
    
    # Initialize the list of output AstroData objects
    adoutput_list = []
    
    try:

        if caltype is None:
            raise Errors.InputError("Caltype should not be None")

        fitsfilenamecre = re.compile("^([NS])(20\d\d)([01]\d[0123]\d)(S)"\
                                     "(?P<fileno>\d\d\d\d)(.*)$")

        for ad in adinput:

            log.fullinfo("Setting OBSCLASS, OBSTYPE, GEMPRGID, OBSID, " +
                         "DATALAB, RELEASE, OBJECT, RA, DEC, CRVAL1, " +
                         "and CRVAL2 to generic defaults")

            # Do some date manipulation to get release date and 
            # fake program number

            # Get date from day data was taken if possible
            date_taken = ad.ut_date()
            if date_taken.collapse_value() is None:
                # Otherwise use current time
                import datetime
                date_taken = datetime.date.today()
            else:
                date_taken = date_taken.as_pytype()
            site = str(ad.telescope()).lower()
            release = date_taken.strftime("%Y-%m-%d")

            # Fake ID is G(N/S)-CALYYYYMMDD-900-fileno
            if "north" in site:
                prefix = "GN-CAL"
            elif "south" in site:
                prefix = "GS-CAL"
            prgid = "%s%s" % (prefix,date_taken.strftime("%Y%m%d"))
            obsid = "%s-%d" % (prgid, 900)

            m = fitsfilenamecre.match(ad.filename)
            if m:
                fileno = m.group("fileno")
                try:
                    fileno = int(fileno)
                except:
                    fileno = None
            else:
                fileno = None

            # Use a random number if the file doesn't have a
            # Gemini filename
            if fileno is None:
                import random
                fileno = random.randint(1,999)
            datalabel = "%s-%03d" % (obsid,fileno)

            # Set class, type, object to generic defaults
            ad.phu_set_key_value("OBSCLASS","partnerCal",
                                 keyword_comments["OBSCLASS"])

            if "fringe" in caltype:
                ad.phu_set_key_value("OBSTYPE","FRINGE",
                                     keyword_comments["OBSTYPE"])
                ad.phu_set_key_value("OBJECT","Fringe Frame",
                                     keyword_comments["OBJECT"])
            elif "sky" in caltype:
                ad.phu_set_key_value("OBSTYPE","SKY",
                                     keyword_comments["OBSTYPE"])
                ad.phu_set_key_value("OBJECT","Sky Frame",
                                     keyword_comments["OBJECT"])
            elif "flat" in caltype:
                ad.phu_set_key_value("OBSTYPE","FLAT",
                                     keyword_comments["OBSTYPE"])
                ad.phu_set_key_value("OBJECT","Flat Frame",
                                     keyword_comments["OBJECT"])
            else:
                raise Errors.InputError("Caltype %s not supported" % caltype)
            
            # Blank out program information
            ad.phu_set_key_value("GEMPRGID",prgid,
                                 keyword_comments["GEMPRGID"])
            ad.phu_set_key_value("OBSID",obsid,
                                 keyword_comments["OBSID"])
            ad.phu_set_key_value("DATALAB",datalabel,
                                 keyword_comments["DATALAB"])

            # Set release date
            ad.phu_set_key_value("RELEASE",release,
                                 keyword_comments["RELEASE"])

            # Blank out positional information
            ad.phu_set_key_value("RA",0.0,keyword_comments["RA"])
            ad.phu_set_key_value("DEC",0.0,keyword_comments["DEC"])
            
            # Blank out RA/Dec in WCS information in PHU if present
            if ad.phu_get_key_value("CRVAL1") is not None:
                ad.phu_set_key_value("CRVAL1",0.0,keyword_comments["CRVAL1"])
            if ad.phu_get_key_value("CRVAL2") is not None:
                ad.phu_set_key_value("CRVAL2",0.0,keyword_comments["CRVAL2"])

            # Do the same for each SCI,VAR,DQ extension
            # as well as the object name
            for ext in ad:
                if ext.extname() not in ["SCI","VAR","DQ"]:
                    continue
                if ext.get_key_value("CRVAL1") is not None:
                    ext.set_key_value("CRVAL1",0.0,keyword_comments["CRVAL1"])
                if ext.get_key_value("CRVAL2") is not None:
                    ext.set_key_value("CRVAL2",0.0,keyword_comments["CRVAL2"])
                if ext.get_key_value("OBJECT") is not None:
                    if "fringe" in caltype:
                        ext.set_key_value("OBJECT","Fringe Frame",
                                          keyword_comments["OBJECT"])
                    elif "sky" in caltype:
                        ext.set_key_value("OBJECT","Sky Frame",
                                          keyword_comments["OBJECT"])
                    elif "flat" in caltype:
                        ext.set_key_value("OBJECT","Flat Frame",
                                          keyword_comments["OBJECT"])

            adoutput_list.append(ad)

        return adoutput_list    

    except:
        # Log the message from the exception
        log.critical(repr(sys.exc_info()[1]))
        raise

def filename_updater(adinput=None, infilename='', suffix='', prefix='',
                    strip=False):
    """
    This function is for updating the file names of astrodata objects.
    It can be used in a few different ways.  For simple post/pre pending of
    the infilename string, there is no need to define adinput or strip. The 
    current filename for adinput will be used if infilename is not defined. 
    The examples below should make the main uses clear.
        
    Note: 
    1.if the input filename has a path, the returned value will have
    path stripped off of it.
    2. if strip is set to True, then adinput must be defined.
          
    :param adinput: input astrodata instance having its filename being updated
    :type adinput: astrodata object
    
    :param infilename: filename to be updated
    :type infilename: string
    
    :param suffix: string to put between end of current filename and the 
                   extension 
    :type suffix: string
    
    :param prefix: string to put at the beginning of a filename
    :type prefix: string
    
    :param strip: Boolean to signal that the original filename of the astrodata
                  object prior to processing should be used. adinput MUST be 
                  defined for this to work.
    :type strip: Boolean
    
    ::
    
     filename_updater(adinput=myAstrodataObject, suffix='_prepared', strip=True)
     result: 'N20020214S022_prepared.fits'
        
     filename_updater(infilename='N20020214S022_prepared.fits',
         suffix='_biasCorrected')
     result: 'N20020214S022_prepared_biasCorrected.fits'
        
     filename_updater(adinput=myAstrodataObject, prefix='testversion_')
     result: 'testversion_N20020214S022.fits'
    
    """
    log = gemLog.getGeminiLog() 

    # Check there is a name to update
    if infilename=='':
        # if both infilename and adinput are not passed in, then log critical msg
        if adinput==None:
            log.critical('A filename or an astrodata object must be passed '+
                         'into filename_updater, so it has a name to update')
        # adinput was passed in, so set infilename to that ad's filename
        else:
            infilename = adinput.filename
            
    # Strip off any path that the input file name might have
    basefilename = os.path.basename(infilename)

    # Split up the filename and the file type ie. the extension
    (name,filetype) = os.path.splitext(basefilename)
    
    if strip:
        # Grabbing the value of PHU key 'ORIGNAME'
        phuOrigFilename = adinput.phu_get_key_value('ORIGNAME') 
        # If key was 'None', ie. store_original_name() wasn't ran yet, then run
        # it now
        if phuOrigFilename is None:
            # Storing the original name of this astrodata object in the PHU
            phuOrigFilename = adinput.store_original_name()
            
        # Split up the filename and the file type ie. the extension
        (name,filetype) = os.path.splitext(phuOrigFilename)
        
    # Create output filename
    outFileName = prefix+name+suffix+filetype
    return outFileName
    

def log_message(function, name, message_type):
    if function == 'ulf':
        full_function_name = 'user level function'
    else:
        full_function_name = function
    if message_type == 'calling':
        message = 'Calling the %s %s' \
                  % (full_function_name, name)
    if message_type == 'starting':
        message = 'Starting the %s %s' \
                  % (full_function_name, name)
    if message_type == 'finishing':
        message = 'Finishing the %s %s' \
                  % (full_function_name, name)
    if message_type == 'completed':
        message = 'The %s %s completed successfully' \
                  % (name, full_function_name)
    if message:
        return message
    else:
        return None


def make_dict(key_list=None, value_list=None):
    """
    The make_dict function creates a dictionary with the elements in 'key_list'
    as the key and the elements in 'value_list' as the value to create an
    association between the input science dataset (the 'key_list') and a, for
    example, dark that is needed to be subtracted from the input science
    dataset. This function also does some basic checks to ensure that the
    filters, exposure time etc are the same.

    :param key: List containing one or more AstroData objects
    :type key: AstroData

    :param value: List containing one or more AstroData objects
    :type value: AstroData
    """
    # Check the inputs have matching filters, binning and SCI shapes.
    ret_dict = {}
    if not isinstance(key_list, list):
        key_list = [key_list]
    if not isinstance(value_list, list):
        value_list = [value_list]
    if len(key_list) == 1 and len(value_list) == 1:
        # There is only one key and one value - create a single entry in the
        # dictionary
        ret_dict[key_list[0]] = value_list[0]
    elif len(key_list) > 1 and len(value_list) == 1:
        # There is only one value for the list of keys
        for i in range (0, len(key_list)):
            ret_dict[key_list[i]] = value_list[0]
    elif len(key_list) > 1 and len(value_list) > 1:
        # There is one value for each key. Check that the lists are the same
        # length
        if len(key_list) != len(value_list):
            msg = """Number of AstroData objects in key_list does not match
            with the number of AstroData objects in value_list. Please provide
            lists containing the same number of AstroData objects. Please
            supply either a single AstroData object in value_list to be applied
            to all AstroData objects in key_list OR the same number of
            AstroData objects in value_list as there are in key_list"""
            raise Errors.InputError(msg)
        for i in range (0, len(key_list)):
            ret_dict[key_list[i]] = value_list[i]
    
    return ret_dict

def mark_history(adinput=None, keyword=None):
    """
    The function to use near the end of a python user level function to 
    add a history_mark timestamp to all the outputs indicating when and what
    function was just performed on them, then logging the new historyMarkKey
    PHU key and updated 'GEM-TLM' key values due to history_mark.
    
    Note: The GEM-TLM key will be updated, or added if not in the PHU yet, 
    automatically everytime wrapUp is called.
    
    :param adinput: List of astrodata instance(s) to perform history_mark 
                      on.
    :type adinput: Either a single or multiple astrodata instances in a 
                     list.
    
    :param keyword: The PHU header key to write the current UT time 
    :type keyword: Under 8 character, all caps, string.
                          If None, then only 'GEM-TLM' is added/updated.
    """
    # Instantiate the log
    log = gemLog.getGeminiLog()
    # If adinput is a single AstroData object, put it in a list
    if not isinstance(adinput, list):
        adinput = [adinput]
    # Loop over each input AstroData object in the input list
    for ad in adinput:
        # Add the 'GEM-TLM' keyword (automatic) and the keyword specified by
        # the 'keyword' parameter to the PHU. If 'keyword' is None,
        # history_mark will still add the 'GEM-TLM' keyword
        ad.history_mark(key=keyword, stomp=True)
        if keyword is not None:
            log.fullinfo("PHU keyword %s = %s added to %s" \
                         % (keyword, ad.phu_get_key_value(keyword),
                            ad.filename), category='header')
        log.fullinfo("PHU keyword GEM-TLM = %s added to %s" \
                     % (ad.phu_get_key_value("GEM-TLM"), ad.filename),
                     category='header')

def parse_sextractor_param():

    # Get path to default sextractor parameter files
    default_dict = Lookups.get_lookup_table(
                             "Gemini/source_detection/sextractor_default_dict",
                             "sextractor_default_dict")
    param_file = lookup_path(default_dict["dq"]["param"])
    if param_file.endswith(".py"):
        param_file = param_file[:-3]
    
    columns = []
    fp = open(param_file)
    for line in fp:
        fields = line.split()
        if len(fields)==0:
            continue
        if fields[0].startswith("#"):
            continue
        
        name = fields[0]
        columns.append(name)

    return columns

def trim_to_data_section(adinput=None):
    """
    This function trims the data in each SCI extension to the
    the section returned by its data_section descriptor.  VAR and DQ
    planes, if present, are trimmed to the same section as the
    corresponding SCI extension.
    This is intended for use in removing overscan sections, or other
    unused parts of the data array.
    """
    # Instantiate the log. This needs to be done outside of the try block,
    # since the log object is used in the except block 
    log = gemLog.getGeminiLog()
    
    # The validate_input function ensures that the input is not None and
    # returns a list containing one or more AstroData objects
    adinput = validate_input(adinput=adinput)

    # Initialize the list of output AstroData objects
    adoutput_list = []
 
    try:

        for ad in adinput:

            for sciext in ad["SCI"]:
                
                # Get matching VAR, DQ, OBJMASK planes if present
                extver = sciext.extver()
                varext = ad["VAR",extver]
                dqext = ad["DQ",extver]
                objmask = ad["OBJMASK",extver]
                
                # Get the data section from the descriptor
                try:
                    # as a string for printing
                    datasecStr = str(sciext.data_section(pretty=True))

                    # as int list of form [x1,x2,y1,y2],
                    # 0-based and non-inclusive
                    dsl = sciext.data_section().as_pytype()

                    # Get the keyword associated with the data_section
                    # descriptor, for later updating.  This keyword 
                    # may be instrument specific.
                    ds_kw = sciext.data_section().keyword
                except:
                    raise Errors.ScienceError("No data section defined; " +
                                              "cannot trim to data section")

                # Check whether data needs to be trimmed
                sci_shape = sciext.data.shape
                if (sci_shape[1]==dsl[1] and 
                    sci_shape[0]==dsl[3] and
                    dsl[0]==0 and
                    dsl[2]==0):
                    sci_trimmed = True
                else:
                    sci_trimmed = False
               
                if sci_trimmed:
                    log.fullinfo("No changes will be made to %s[*,%i], since "\
                                 "the data section matches the data shape" %
                                 (ad.filename,sciext.extver()))
                    continue

                # Update logger with the section being kept
                log.fullinfo("For "+ad.filename+" extension "+
                             str(sciext.extver())+
                             ", keeping the data from the section "+
                             datasecStr,"science")
                
                # Trim the data section from input SCI array
                # and make it the new SCI data
                sciext.data=sciext.data[dsl[2]:dsl[3],dsl[0]:dsl[1]]
                
                # Update header keys to match new dimensions
                newDataSecStr = "[1:"+str(dsl[1]-dsl[0])+",1:"+\
                                str(dsl[3]-dsl[2])+"]" 
                sciext.set_key_value("NAXIS1",dsl[1]-dsl[0],
                                     comment=keyword_comments["NAXIS1"])
                sciext.set_key_value("NAXIS2",dsl[3]-dsl[2],
                                     comment=keyword_comments["NAXIS2"])
                sciext.set_key_value(ds_kw,newDataSecStr,
                                     comment=keyword_comments[ds_kw])
                sciext.set_key_value("TRIMSEC", datasecStr, 
                                     comment=keyword_comments["TRIMSEC"])
                
                # Update WCS reference pixel coordinate
                try:
                    crpix1 = sciext.get_key_value("CRPIX1") - dsl[0]
                    crpix2 = sciext.get_key_value("CRPIX2") - dsl[2]
                except:
                    log.warning("Could not access WCS keywords; using dummy " +
                                "CRPIX1 and CRPIX2")
                    crpix1 = 1
                    crpix2 = 1
                sciext.set_key_value("CRPIX1",crpix1,
                                     comment=keyword_comments["CRPIX1"])
                sciext.set_key_value("CRPIX2",crpix2,
                                     comment=keyword_comments["CRPIX2"])

                # If other planes are present, update them to match
                for ext in [dqext, varext, objmask]:
                    if ext is not None:
                        # Check that ext does not already match the science
                        # (eg. gireduce DQ planes)
                        if ext.data.shape!=sciext.data.shape:
                            # Trim the data
                            ext.data=ext.data[dsl[2]:dsl[3],dsl[0]:dsl[1]]
                            # Set NAXIS keywords
                            ext.set_key_value("NAXIS1",dsl[1]-dsl[0],
                                          comment=keyword_comments["NAXIS1"])
                            ext.set_key_value("NAXIS2",dsl[3]-dsl[2],
                                          comment=keyword_comments["NAXIS2"])
                            # Skip the rest for object masks
                            if ext.extname() in ["VAR","DQ"]:
                                # Set section keywords
                                ext.set_key_value(ds_kw,newDataSecStr,
                                            comment=keyword_comments[ds_kw])
                                ext.set_key_value("TRIMSEC", datasecStr, 
                                            comment=keyword_comments["TRIMSEC"])
                                # Set WCS keywords
                                ext.set_key_value("CRPIX1",crpix1,
                                            comment=keyword_comments["CRPIX1"])
                                ext.set_key_value("CRPIX2",crpix2,
                                            comment=keyword_comments["CRPIX2"])

            adoutput_list.append(ad)

        return adoutput_list    

    except:
        # Log the message from the exception
        log.critical(repr(sys.exc_info()[1]))
        raise


def update_key_from_descriptor(adinput=None, descriptor=None, 
                               keyword=None, extname=None):
    """
    This function updates keywords in the headers of the input dataset,
    performs logging of the changes and writes history keyword related to the
    changes to the PHU.
    
    :param adinput: astrodata instance to perform header key updates on
    :type adinput: an AstroData instance
    
    :param descriptor: string for an astrodata function or descriptor function
                       to perform on the input ad.
                       ie. for ad.gain(), descriptor='gain()'
    :type descriptor: string 
    
    :param extname: Set to 'PHU', 'SCI', 'VAR' or 'DQ' to update the given
                    keyword in the PHU, SCI, VAR or DQ extension, respectively.
                    
    :type extname: string
    """
    log = gemLog.getGeminiLog()
    historyComment = None

    # Make sure a valid extname is specified
    if extname is None:
        extname = "SCI"

    # Use exec to perform the requested function on full AD 
    # Allow it to raise the error if the descriptor fails
    exec('dv = adinput.%s' % descriptor)
    if dv is None:
        log.fullinfo("No value found for descriptor %s on %s" % 
                     (descriptor,adinput.filename))
    else:

        if keyword is not None:
            key = keyword
        else:
            key = dv.keyword
            if key is None:
                raise Errors.ToolboxError(
                    "No keyword found for descriptor %s" % descriptor)

        # Get comment from lookup table
        # Allow it to raise the KeyError if it can't find it
        comment = keyword_comments[key]
            
        if extname == "PHU":
            # Set the keyword value and comment
            adinput.phu_set_key_value(key, dv.as_pytype(), comment)
        else:
            # Use the dictionary form of the descriptor value
            dv_dict = dv.dict_val

            for ext in adinput[extname]:
                # Get value from dictionary
                dict_key = (ext.extname(),ext.extver())
                value = dv_dict[dict_key]
        
                # Set the keyword value and comment
                ext.set_key_value(key, value, comment)
            


def validate_input(adinput=None):
    """
    The validate_input helper function is used to validate the inputs given to
    the user level functions.
    """
    # If the adinput is None, raise an exception
    if adinput is None:
        raise Errors.InputError("The adinput cannot be None")
    # If the adinput is a single AstroData object, put it in a list
    if not isinstance(adinput, list):
        adinput = [adinput]
    # If the adinput is an empty list, raise an exception
    if len(adinput) == 0:
        raise Errors.InputError("The adinput cannot be an empty list")
    # Now, adinput is a list that contains one or more AstroData objects
    return adinput
