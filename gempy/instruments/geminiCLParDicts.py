# Author: Kyle Mede, February 2011
# This module contains a function to return a dictionary of all the default
# parameters for each iraf (CL) function used by the recipe system that may 
# have its important parameters updated and then passed into the iraf function.

from astrodata.adutils.gemutil import pyrafLoader
from gempy.instruments import geminiTools  as gemt
from astrodata.adutils import gemLog
from astrodata.Errors import ToolboxError

def CLDefaultParamsDict(CLscript):
    """
    A function to return a dictionary full of all the default parameters 
    for each CL script used so far in the Recipe System.
    
    :param CLscript: The name of the iraf script/function the default parameters
                     are needed for.
    :type CLscript: String, all lower case. ex. 'gireduce'.
    """
    log = gemLog.getGeminiLog()
    
    # loading and bringing the pyraf related modules into the name-space
    pyraf, gemini, yes, no = pyrafLoader()
    
    # Ensuring that if a invalide CLscript was requested, that a critical
    # log message be made and exception raised.
    if (CLscript != 'gemcombine') and (CLscript != 'gireduce') and \
        (CLscript != 'giflat') and (CLscript != 'gmosaic') and \
                        (CLscript != 'gdisplay') and (CLscript != 'gifringe'):
        log.critical('The CLscript '+CLscript+' does not have a default'+
                     ' dictionary')
        raise ToolboxError('The CLscript '+CLscript+
                              ' does not have a default'+' dictionary')
        
    if CLscript == 'gemcombine':
        defaultParams = {
            'input'      :'',            # Input MEF images
            'output'     :'',            # Output MEF image
            'title'      :'DEFAULT',     # Title for output SCI plane
            'combine'    :'average',     # Combination operation
            'reject'     :'avsigclip',   # Rejection algorithm
            'offsets'    :'none',        # Input image offsets
            'masktype'   :'none',        # Mask type
            'maskvalue'  :0.0,           # Mask value
            'scale'      :'none',        # Image scaling
            'zero'       :'none',        # Image zeropoint offset
            'weight'     :'none',        # Image weights
            'statsec'    :'[*,*]',       # Statistics section
            'expname'    :'EXPTIME',     # Exposure time header keyword
            'lthreshold' :'INDEF',       # Lower threshold
            'hthreshold' :'INDEF',       # Upper threshold
            'nlow'       :1,             # minmax: Number of low pixels to reject
            'nhigh'      :1,             # minmax: Number of high pixels to reject
            'nkeep'      :1,             # Minimum to keep or maximum to reject
            'mclip'      :yes,           # Use median in sigma clipping algorithms?
            'lsigma'     :3.0,           # Lower sigma clipping factor
            'hsigma'     :3.0,           # Upper sigma clipping factor
            'key_ron'    :'RDNOISE',     # Keyword for readout noise in e-
            'key_gain'   :'GAIN',        # Keyword for gain in electrons/ADU
            'ron'        :0.0,           # Readout noise rms in electrons
            'gain'       :1.0,           # Gain in e-/ADU
            'snoise'     :'0.0',         # ccdclip: Sensitivity noise (electrons
            'sigscale'   :0.1,           # Tolerance for sigma clipping scaling correction                                
            'pclip'      :-0.5,          # pclip: Percentile clipping parameter
            'grow'       :0.0,           # Radius (pixels) for neighbor rejection
            'bpmfile'    :'',            # Name of bad pixel mask file or image.
            'nrejfile'   :'',            # Name of rejected pixel count image.
            'sci_ext'    :'SCI',         # Name(s) or number(s) of science extension
            'var_ext'    :'VAR',         # Name(s) or number(s) of variance extension
            'dq_ext'     :'DQ',          # Name(s) or number(s) of data quality extension
            'fl_vardq'   :no,            # Make variance and data quality planes?
            'logfile'    :'',            # Log file
            'fl_dqprop'  :no,            # Propagate all DQ values?
            'verbose'    :yes,           # Verbose output?
            'status'     :0,             # Exit status (0=good)
            'Stdout'     :gemt.IrafStdout(),
            'Stderr'     :gemt.IrafStdout()
                       }
        
    if CLscript == 'gireduce':
        defaultParams = {
            'inimages'   :'',                # Input GMOS images 
            'outpref'    :'DEFAULT',         # Prefix for output images
            'outimages'  :'',                # Output images
            'fl_over'    :no,                # Subtract overscan level
            'fl_trim'    :no,                # Trim off the overscan section
            'fl_bias'    :no,                # Subtract bias image
            'fl_dark'    :no,                # Subtract (scaled) dark image
            'fl_flat'    :no,                # Do flat field correction?
            'fl_vardq'   :no,                # Create variance and data quality frames
            'fl_addmdf'  :no,                # Add Mask Definition File? (LONGSLIT/MOS/IFU modes)
            'bias'       :'',                # Bias image name
            'dark'       :'',                # Dark image name
            'flat1'      :'',                # Flatfield image 1
            'flat2'      :'',                # Flatfield image 2
            'flat3'      :'',                # Flatfield image 3
            'flat4'      :'',                # Flatfield image 4
            'key_exptime':'EXPTIME',         # Header keyword of exposure time
            'key_biassec':'BIASSEC',         # Header keyword for bias section
            'key_datasec':'DATASEC',         # Header keyword for data section
            'rawpath'    :'',                # GPREPARE: Path for input raw images
            'gp_outpref' :'g',               # GPREPARE: Prefix for output images
            'sci_ext'    :'SCI',             # Name of science extension
            'var_ext'    :'VAR',             # Name of variance extension
            'dq_ext'     :'DQ',              # Name of data quality extension
            'key_mdf'    :'MASKNAME',        # Header keyword for the Mask Definition File
            'mdffile'    :'',                # MDF file to use if keyword not found
            'mdfdir'     :'',                # MDF database directory
            'bpm'        :'',                # Bad pixel mask
            #'giandb'     :'default',        # Database with gain data
            'sat'        :65000,             # Saturation level in raw images [ADU]
            'key_nodcount':'NODCOUNT',       # Header keyword with number of nod cycles
            'key_nodpix' :'NODPIX',          # Header keyword with shuffle distance
            'key_filter' :'FILTER2',         # Header keyword of filter
            'key_ron'    :'RDNOISE',         # Header keyword for readout noise
            'key_gain'   :'GAIN',            # Header keyword for gain (e-/ADU)
            'ron'        :3.5,               # Readout noise in electrons
            'gain'       :2.2,               # Gain in e-/ADU
            'fl_mult'    :no, #$$$$$$$$$     # Multiply by gains to get output in electrons
            'fl_inter'   :no,                # Interactive overscan fitting?
            'median'     :no,                # Use median instead of average in column bias?
            'function'   :'chebyshev',       # Overscan fitting function
            'nbiascontam':4, #$$$$$$$        # Number of columns removed from overscan region
            'biasrows'   :'default',         # Rows to use for overscan region
            'order'      :1,                 # Order of overscan fitting function
            'low_reject' :3.0,               # Low sigma rejection factor in overscan fit
            'high_reject':3.0,               # High sigma rejection factor in overscan fit
            'niterate'   :2,                 # Number of rejection iterations in overscan fit
            'logfile'    :'',                # Logfile
            'verbose'    :yes,               # Verbose?
            'status'     :0,                 # Exit status (0=good)
            'Stdout'     :gemt.IrafStdout(),
            'Stderr'     :gemt.IrafStdout()
                           }
        
    if CLscript == 'giflat':
        defaultParams = { 
            'inflats'    :'',            # Input flat field images
            'outflat'    :'',            # Output flat field image
            'normsec'    :'default',     # Image section to get the normalization.
            'fl_scale'   :yes,           # Scale the flat images before combining?
            'sctype'     :'mean',        # Type of statistics to compute for scaling
            'statsec'    :'default',     # Image section for relative intensity scaling
            'key_gain'   :'GAIN',        # Header keyword for gain (e-/ADU)
            'fl_stamp'   :no,            # Input is stamp image
            'sci_ext'    :'SCI',         # Name of science extension
            'var_ext'    :'VAR',         # Name of variance extension
            'dq_ext'     :'DQ',          # Name of data quality extension
            'fl_vardq'   :no,            # Create variance and data quality frames?
            'sat'        :65000,         # Saturation level in raw images (ADU)
            'verbose'    :yes,           # Verbose output?
            'logfile'    :'',            # Name of logfile
            'status'     :0,             # Exit status (0=good)
            'combine'    :'average',     # Type of combine operation
            'reject'     :'avsigclip',   # Type of rejection in flat average
            'lthreshold' :'INDEF',       # Lower threshold when combining
            'hthreshold' :'INDEF',       # Upper threshold when combining
            'nlow'       :0,             # minmax: Number of low pixels to reject
            'nhigh'      :1,             # minmax: Number of high pixels to reject
            'nkeep'      :1,             # avsigclip: Minimum to keep (pos) or maximum to reject (neg)
            'mclip'      :yes,           # avsigclip: Use median in clipping algorithm?
            'lsigma'     :3.0,           # avsigclip: Lower sigma clipping factor
            'hsigma'     :3.0,           # avsigclip: Upper sigma clipping factor
            'sigscale'   :0.1,           # avsigclip: Tolerance for clipping scaling corrections
            'grow'       :0.0,           # minmax or avsigclip: Radius (pixels) for neighbor rejection
            'gp_outpref' :'g',           # Gprepare prefix for output images
            'rawpath'    :'',            # GPREPARE: Path for input raw images
            'key_ron'    :'RDNOISE',     # Header keyword for readout noise
            'key_datasec':'DATASEC',     # Header keyword for data section
            #'giandb'     :'default',    # Database with gain data
            'bpm'        :'',            # Bad pixel mask
            'gi_outpref' :'r',           # Gireduce prefix for output images
            'bias'       :'',            # Bias calibration image
            'fl_over'    :no,            # Subtract overscan level?
            'fl_trim'    :no,            # Trim images?
            'fl_bias'    :no,            # Bias-subtract images?
            'fl_inter'   :no,            # Interactive overscan fitting?
            'nbiascontam':4, #$$$$$$$    # Number of columns removed from overscan region
            'biasrows'   :'default',     # Rows to use for overscan region
            'key_biassec':'BIASSEC',     # Header keyword for overscan image section
            'median'     :no,            # Use median instead of average in column bias?
            'function'   :'chebyshev',   # Overscan fitting function.
            'order'      :1,             # Order of overscan fitting function.
            'low_reject' :3.0,           # Low sigma rejection factor.
            'high_reject':3.0,           # High sigma rejection factor.
            'niterate'   :2,             # Number of rejection iterations.
            'Stdout'      :gemt.IrafStdout(),
            'Stderr'      :gemt.IrafStdout()
                       }    
          
    if CLscript == 'gmosaic':
        defaultParams = { 
            'inimages'   :'',                     # Input GMOS images 
            'outimages'  :'',                     # Output images
            'outpref'    :'DEFAULT',              # Prefix for output images
            'fl_paste'   :no,                     # Paste images instead of mosaic
            'fl_vardq'   :no,                     # Propagate the variance and data quality planes
            'fl_fixpix'  :no,                     # Interpolate across chip gaps
            'fl_clean'   :yes ,                   # Clean imaging data outside imaging field
            'geointer'   :'linear',               # Interpolant to use with geotran
            'gap'        :'default',              # Gap between the CCDs in unbinned pixels
            'bpmfile'    :'gmos$data/chipgaps.dat',   # Info on location of chip gaps ## HUH??? Why is variable called 'bpmfile' if it for chip gaps??
            'statsec'    :'default',              # Statistics section for cleaning
            'obsmode'    :'IMAGE',                # Value of key_obsmode for imaging data
            'sci_ext'    :'SCI',                  # Science extension(s) to mosaic, use '' for raw data
            'var_ext'    :'VAR',                  # Variance extension(s) to mosaic
            'dq_ext'     :'DQ',                   # Data quality extension(s) to mosaic
            'mdf_ext'    :'MDF',                  # Mask definition file extension name
            'key_detsec' :'DETSEC',               # Header keyword for detector section
            'key_datsec' :'DATASEC',              # Header keyword for data section
            'key_ccdsum' :'CCDSUM',               # Header keyword for CCD binning
            'key_obsmode':'OBSMODE',              # Header keyword for observing mode
            'logfile'    :'',                     # Logfile
            'fl_real'    :no,                     # Convert file to real before transforming
            'verbose'    :yes,                    # Verbose
            'status'     :0,                      # Exit status (0=good)
            'Stdout'     :gemt.IrafStdout(),
            'Stderr'     :gemt.IrafStdout()
                       }
    
    if CLscript == 'gdisplay':
        defaultParams = { 
            'image'         :'',                # GMOS image to display, can use number if current UT
            'frame'         :1,                 # Frame to write to
            'output'        :'',                # Save pasted file to this name if not blank
            'fl_paste'      :'no',              # Paste images to one for imexamine
            'fl_bias'       :'no',              # Rough bias subtraction
            'rawpath'       :'',                # Path for input image if not included in name
            'gap'           :'default',         # Size of the gap between the CCDs (in pixels)
            'z1'            :0.0,               # Lower limit if not autoscaling
            'z2'            :0.0,               # Upper limit if not autoscaling
            'fl_sat'        :'no',              # Flag saturated pixels
            'fl_imexam'     :'yes',             # If possible, run imexam
            'signal'        :'INDEF',           # Flag pixels with signal above this limit
            'sci_ext'       :'SCI',             # Name of extension(s) to display
            'observatory'   :'',                # Observatory (gemini-north or gemini-south)
            'prefix'        :'auto',            # File prefix, (N/S)YYYYMMDDS if not auto
            'key_detsec'    :'DETSEC',          # Header keyword for detector section
            'key_datasec'   :'DATASEC',         # Header keyword for data section
            'key_ccdsum'    :'CCDSUM',          # Header keyword for CCD binning
            'gaindb'        :'default',         # Database with gain data
            'verbose'       :yes,               # Verbose
            'status'        :0,                 # Exit status (0=good)
            'Stdout'        :gemt.IrafStdout(), 
            'Stderr'        :gemt.IrafStdout()  
                       }
    
    if CLscript == 'gifringe':
        defaultParams = {
            'inimages'  :'',              # Input GMOS images
            'outimage'  : '',             # Output fringe frame
            'typezero'  : 'mean',         # Operation to determine the sky level or zero point
            'skysec'    : 'default',      # Zero point statistics section
            'skyfile'   : '',             # File with zero point values for each input image
            'key_zero'  : 'OFFINT',       # Keyword for zero level
            'msigma'    : 4.0,            # Sigma threshold above sky for mask
            'bpm'       : '',             # Name of bad pixel mask file or image
            'combine'   : 'median',       # Combination operation
            'reject'    : 'avsigclip',    # Rejection algorithm
            'scale'     : 'none',         # Image scaling
            'weight'    : 'none',         # Image Weights
            'statsec'   : '[*,*]',        # Statistics section for image scaling
            'expname'   : 'EXPTIME',      # Exposure time header keyword for image scaling
            'nlow'      : 1,              # minmax: Number of low pixels to reject
            'nhigh'     : 1,              # minmax: Number of high pixels to reject
            'nkeep'     : 0,              # Minimum to keep or maximum to reject
            'mclip'     : yes,            # Use median in sigma clipping algorithms?
            'lsigma'    : 3.0,            # Lower sigma clipping factor
            'hsigma'    : 3.0,            # Upper sigma clipping factor
            'sigscale'  : 0.1,            # Tolerance for sigma clipping scaling correction
            'sci_ext'   : 'SCI',          # Name of science extension
            'var_ext'   : 'VAR',          # Name of variance extension
            'dq_ext'    : 'DQ',           # Name of data quality extension
            'fl_vardq'  : no,             # Make variance and data quality planes?
            'logfile'   : '',             # Name of the logfile
            'glogpars'  : '',             # Logging preferences
            'verbose'   : yes,            # Verbose output
            'status'    : 0,              # Exit status (0=good)
            'Stdout'    :gemt.IrafStdout(),
            'Stderr'    :gemt.IrafStdout()
                       }
    return defaultParams