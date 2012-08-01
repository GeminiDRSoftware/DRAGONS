# This module contains a function to return a dictionary of all the default
# parameters for each iraf (CL) function used by the recipe system that may 
# have its important parameters updated and then passed into the iraf function.

from astrodata.adutils.gemutil import pyrafLoader
from gempy import managers as man
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
    if ((CLscript != 'display') and 
        (CLscript != 'gemcombine') and 
        (CLscript != 'gireduce') and
        (CLscript != 'giflat') and
        (CLscript != 'gmosaic') and
        (CLscript != 'gdisplay') and 
        (CLscript != 'gifringe') and 
        (CLscript != 'gsappwave') and 
        (CLscript != 'gscrrej') and 
        (CLscript != 'gsextract') and
        (CLscript != 'gsflat') and
        (CLscript != 'gsskysub') and 
        (CLscript != 'gstransform') and 
        (CLscript != 'gswavelength')):
        log.critical('The CLscript '+CLscript+' does not have a default'+
                     ' dictionary')
        raise ToolboxError('The CLscript '+CLscript+
                              ' does not have a default'+' dictionary')
        
    if CLscript == 'display':
        defaultParams = {
            'image'        :'',      # image to be displayed
            'frame'        :1,       # frame to be written into
            'bpmask'       :"BPM",   # bad pixel mask
            'bpdisplay'    :"none",  # bad pixel display (none|overlay|interpolate)
            'bpcolors'     :"red",   # bad pixel colors
            'overlay'      :"",      # overlay mask
            'ocolors'      :"green", # overlay colors
            'erase'        :yes,     # erase frame
            'border_erase' :no,      # erase unfilled area of window
            'select_frame' :yes,     # display frame being loaded
            'repeat'       :no,      # repeat previous display parameters
            'fill'         :no,      # scale image to fit display window
            'zscale'       :yes,     # display range of greylevels near median
            'contrast'     :0.25,    # contrast adjustment for zscale algorithm
            'zrange'       :yes,     # display full image intensity range
            'zmask'        :"",      # sample mask
            'nsample'      :1000,    # maximum number of sample pixels to use
            'xcenter'      :0.5,     # display window horizontal center
            'ycenter'      :0.5,     # display window vertical center
            'xsize'        :1.0,     # display window horizontal size
            'ysize'        :1.0,     # display window vertical size
            'xmag'         :1.0,     # display window horizontal magnification
            'ymag'         :1.0,     # display window vertical magnification
            'order'        :0,       # spatial interpolator order (0=replicate, 1=linear)
            'z1'           :0.0,     # minimum greylevel to be displayed
            'z2'           :0.0,     # maximum greylevel to be displayed
            'ztrans'       :"linear",# greylevel transformation (linear|log|none|user)
            'lutfile'      :"",      # file containing user defined look up table
            'Stdout'       :man.IrafStdout(),
            'Stderr'       :man.IrafStdout()
                       }

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
            'Stdout'     :man.IrafStdout(),
            'Stderr'     :man.IrafStdout()
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
            #'gaindb'     :'default',        # Database with gain data
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
            'Stdout'     :man.IrafStdout(),
            'Stderr'     :man.IrafStdout()
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
            #'gaindb'     :'default',    # Database with gain data
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
            'Stdout'      :man.IrafStdout(),
            'Stderr'      :man.IrafStdout()
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
            'bpmfile'    :'gmos$data/chipgaps.dat',   # Info on location of chip gaps
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
            'Stdout'     :man.IrafStdout(),
            'Stderr'     :man.IrafStdout()
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
            'Stdout'        :man.IrafStdout(), 
            'Stderr'        :man.IrafStdout()  
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
            'Stdout'    :man.IrafStdout(),
            'Stderr'    :man.IrafStdout()
                       }

    if CLscript == 'gsappwave':
        defaultParams = {
            'inimages': '',            # Input images
            'gratingdb': "gmos$data/GMOSgratings.dat", # Gratings database file
            'filterdb': "gmos$data/GMOSfilters.dat",   # Filters database file
            'key_dispaxis': "DISPAXIS",# Keyword for dispersion axis
            'dispaxis': 1 ,            # Dispersion axis
            'logfile' : '',            # Logfile
            'verbose' : yes,           # Verbose?
            'status'  : 0,             # Exit status (0=good)
            'Stdout'    :man.IrafStdout(),
            'Stderr'    :man.IrafStdout()
            }

    if CLscript == 'gscrrej':
        defaultParams = {
            'inimage' : '',            # Input image
            'outimage': '',            # Output image with cosmic rays removed
            'datares' : 4.0,           # Instrumental FWHM in x-direction
            'fnsigma' : 8.0,           # Sigma clipping threshold for fitting
            'niter'   : 5,             # Number of fitting iterations
            'tnsigma' : 10.0,          # Sigma clipping threshold for mask
            'fl_inter': no,            # Examine spline fit interactively?
            'logfile' : '',            # Logfile
            'verbose' : yes,           # Verbose?
            'status'  : 0,             # Exit status (0=good)
            'Stdout'    :man.IrafStdout(),
            'Stderr'    :man.IrafStdout()
            }

    if CLscript == 'gsextract':
        defaultParams = {
            'inimages' : '',            # Input image
            'outimages': '',            # Output image with cosmic rays removed
            'outprefix' : "e",          # Output prefix
            'refimages' : "",           # Reference images for tracing apertures
            'apwidth' : 1.0,            # Extraction aperture in arcsec (diameter)
            'fl_inter' : no,            # Run interactively?
            'database' : "database",    # Directory for calibration files
            'find' : yes,               # Define apertures automatically?
            'recenter' : yes,           # Recenter apertures?
            'trace' : yes,              # Trace apertures?
            'tfunction' : "chebyshev",  # Trace fitting function
            'torder' : 5,               # Trace fitting function order
            'tnsum' : 20,               # Number of dispersion lines to sum for trace
            'tstep' : 50,               # Tracing step
            'weights' : "none",         # Extraction weights (none|variance)\n
            'clean' : no,               # Detect and replace bad pixels?
            'lsigma' : 3.0,             # Lower rejection threshold for cleaning
            'usigma' : 3.0,             # Upper rejection threshold for cleaning\n
            'background' : "none",      # Background subtraction method
            'bfunction' : "chebyshev",  # Background function
            'border' : 1,               # Order of background fit
            'long_bsample' : "*",       # LONGSLIT: backgr sample regions, WRT aperture
            'mos_bsample' : 0.9,        # MOS: fraction of slit length to use (bkg+obj)
            'bnaverage' : 1,            # Number of samples to average over
            'bniterate' : 2,            # Number of rejection iterations
            'blow_reject' : 2.5,        # Background lower rejection sigma
            'bhigh_reject' : 2.5,       # Background upper rejection sigma
            'bgrow' : 0.0,              # Background rejection growing radius (pix)\n
            'fl_vardq' : no,            # Propagate VAR/DQ planes? (if yes, must use variance weighting)
            'sci_ext' : "SCI",          # Name of science extension
            'var_ext' : "VAR",          # Name of variance extension
            'dq_ext' : "DQ",            # Name of data quality extension
            'key_ron' : "RDNOISE",      # Keyword for readout noise in e-
            'key_gain' : "GAIN",        # Keyword for gain in electrons/ADU
            'ron' : 3.5,                # Default readout noise rms in electrons
            'gain' : 2.2,               # Default gain in e-/ADU
            'logfile' : '',             # Logfile
            'verbose' : yes,            # Verbose?
            'status'  : 0,              # Exit status (0=good)
            'Stdout'    :man.IrafStdout(),
            'Stderr'    :man.IrafStdout()
            }

    if CLscript == 'gsflat':
        defaultParams = {
            'inflats': '',            # Input flatfields
            'specflat':'',            # Output normalized flat (MEF)
            'fl_slitcorr': no,        # Correct output for Illumination/Slit-Function
            'slitfunc': '',           # Slit Function (MEF output of gsslitfunc)
            'fl_keep': no,            # Keep imcombined flat?
            'combflat': "",           # Filename for imcombined flat
            'fl_over': no,            # Subtract overscan level
            'fl_trim': no,            # Trim off overscan region
            'fl_bias': no,            # Subtract bias image
            'fl_dark': no,            # Subtract (scaled) dark image
            'fl_fixpix': yes,         # Interpolate across chip gaps
            'fl_vardq': no,           # Create variance and data quality frames
            'bias': "",               # Bias image
            'dark': "",               # Dark image
            'key_exptime': "EXPTIME", # Exposure time header keyword
            'key_biassec': "BIASSEC", # Header keyword for overscan strip image section
            'key_datasec': "DATASEC", # Header keyword for data section (excludes the overscan)
            'rawpath': "",            # GPREPARE: Path for input raw images
            'sci_ext': "SCI",         # Name of science extension
            'var_ext': "VAR",         # Name of variance extension
            'dq_ext': "DQ",           # Name of data quality extension
            'key_mdf': "MASKNAME",    # Header keyword for the MDF
            'mdffile': "",            # MDF to use if keyword not found
            'mdfdir': "gmos$data/",   # MDF database directory
            'bpm': "",                # Name of bad pixel mask file or image
            'gaindb': "default",      # Database with gain data
            'gratingdb': "gmos$data/GMOSgratings.dat", # Gratings database file
            'filterdb': "gmos$data/GMOSfilters.dat", # Filters database file
            'bpmfile': "gmos$data/chipgaps.dat", # Info on location of chip gaps
            'refimage': "",           # Reference image for slit positions
            'sat': "default",         # Saturation level in raw images
            'xoffset': 'INDEF',       # X offset in wavelength [nm]
            'yoffset': 'INDEF',       # Y offset in unbinned pixels
            'yadd': 0.0,              # Additional pixels to add to each end of MOS slitlet lengths
            'fl_usegrad': no,         # Use gradient method to find MOS slits
            'fl_emis': no,            # mask emission lines from lamp (affected pixels set to 1. in output)
            'nbiascontam': "default", # Number of columns removed from overscan region
            'biasrows': "default",    # Rows to use for overscan region
            'fl_inter': no,           # Fit response interactively?
            'fl_answer': yes,         # Continue interactive fitting?
            'fl_detec': yes,           # Fit response detector by detector rather than slit by slit?
            'function': "spline3",    # Fitting function for response
            'order': "15",            # Order of fitting function, minimum value=1
            'low_reject': 3.0,        # Low rejection in sigma of response fit
            'high_reject': 3.0,       # High rejection in sigma of response fit
            'niterate': 2,            # Number of rejection iterations in response fit
            'combine': "average",     # Combination operation
            'reject': "avsigclip",    # Rejection algorithm
            'masktype': "goodvalue",  # Mask type
            'maskvalue': 0.0,         # Mask value
            'scale': "mean",          # Image scaling
            'zero': "none",           # Image zeropoint offset
            'weight': "none",         # Image weights
            'statsec': "",            # Statistics section
            'lthreshold': 'INDEF',    # Lower threshold
            'hthreshold': 'INDEF',    # Upper threshold
            'nlow': 1,                # minmax: Number of low pixels to reject
            'nhigh': 1,               # minmax: Number of high pixels to reject
            'nkeep': 0,               # Minimum to keep or maximum to reject
            'mclip': yes,             # Use median in sigma clipping algorithms?
            'lsigma': 3.0,            # Lower sigma clipping factor
            'hsigma': 3.0,            # Upper sigma clipping factor
            'key_ron': "RDNOISE",     # Keyword for readout noise in e-
            'key_gain': "GAIN",       # Keyword for gain in electrons/ADU
            'ron': 3.5,               # Readout noise rms in electrons
            'gain': 2.2,              # Gain in e-/ADU
            'snoise': "0.0",          # ccdclip: Sensitivity noise (electrons)
            'sigscale': 0.1,          # Tolerance for sigma clipping scaling correction
            'pclip': -0.5,            # pclip: Percentile clipping parameter
            'grow': 0.0,              # Radius (pixels, for neighbor rejection)
            'ovs_flinter': no,        # Interactive overscan fitting?
            'ovs_med': no,            # Use median instead of average in column bias?
            'ovs_func': "chebyshev",  # Overscan fitting function
            'ovs_order': 1,           # Order of overscan fitting function
            'ovs_lowr': 3.0,          # Low sigma rejection factor
            'ovs_highr': 3.0,         # High sigma rejection factor
            'ovs_niter': 2,           # Number of rejection iterations
            'fl_double': no,          # Make double flats for nod-and-shuffle science
            'nshuffle': 0,            # Number of shuffle pixels (unbinned)
            'logfile': "",            # Logfile name
            'verbose': yes,           # Verbose
            'status': 0,              # Exit status (0=good)
            'Stdout'    :man.IrafStdout(),
            'Stderr'    :man.IrafStdout()
                       }

    if CLscript == 'gsskysub':
        defaultParams = {
            'input'       : '',            # Input GMOS spectra
            'fl_answer'   : '',            # Continue with interactive fitting
            'output'      : '',            # Output spectra
            'outpref'     : "s",           # Output prefix
            'sci_ext'     : "SCI",         # Name of science extension
            'var_ext'     : "VAR",         # Name of variance extension
            'dq_ext'      : "DQ",          # Name of data quality extension
            'fl_vardq'    : no,            # Propagate VAR/DQ planes
            'long_sample' : "*",           # Sky sample for LONGSLIT
            'mos_sample'  : 0.9,           # MOS: Maximum fraction of slit length to use as sky sample
            'mosobjsize'  : 1.0,           # MOS: Size of object aperture in arcsec
            'naverage'    : 1,             # Number of points in sample averaging
            'function'    : "chebyshev",   # Function to fit
            'order'       : 1,             # Order for fit
            'low_reject'  : 2.5,           # Low rejection in sigma of fit
            'high_reject' : 2.5,           # High rejection in sigma of fit
            'niterate'    : 2,             # Number of rejection iterations
            'grow'        : 0.0,           # Rejection growing radius in pixels
            'fl_inter'    : no,            # Fit interactively
            'logfile'     : "",            # Logfile name
            'verbose'     : yes,           # Verbose?
            'status'      : 0,             # Exit status (0=good)
            'Stdout'      :man.IrafStdout(),
            'Stderr'      :man.IrafStdout()
            }

    if CLscript == 'gstransform':
        defaultParams = {
            'inimages' :  "",              # Input GMOS spectra
            'outimages' : "",              # Output spectra
            'outprefix' : "t",             # Prefix for output spectra
            'fl_stran' : no,               # Apply S-distortion correction
            'sdistname' : "",              # Names of S-distortions calibrations
            'fl_wavtran' : yes,            # Apply wavelength calibration from arc spectrum
            'wavtraname' : "",             # Names of wavelength calibrations
            'database' : "database",       # Directory for calibration files
            'fl_vardq' : no,               # Transform variance and data quality planes
            'interptype' : "linear",       # Interpolation type for transform
            'lambda1' : 'INDEF',             # First output wavelength for transform (Ang)
            'lambda2' : 'INDEF',             # Last output wavelength for transform (Ang)
            'dx' : 'INDEF',                  # Output wavelength to pixel conversion ratio for transform (Ang/pix)
            'nx' : 'INDEF',                  # Number of output pixels for transform (pix)
            'lambdalog' : no,              # Logarithmic wavelength coordinate for transform
            'ylog' : no,                   # Logarithmic y coordinate for transform
            'fl_flux' : yes,               # Conserve flux per pixel in the transform
            'gratingdb' : "gmos$data/GMOSgratings.dat", # Gratings database file
            'filterdb' : "gmos$data/GMOSfilters.dat", # Filters database file
            'key_dispaxis' : "DISPAXIS",   # Keyword for dispersion axis
            'dispaxis' : 1,                # Dispersion axis
            'sci_ext' : "SCI",             # Name of science extension
            'var_ext' : "VAR",             # Name of variance extension
            'dq_ext' : "DQ",               # Name of data quality extension
            'logfile'     : "",            # Logfile name
            'verbose'     : yes,           # Verbose?
            'status'      : 0,             # Exit status (0=good)
            'Stdout'      :man.IrafStdout(),
            'Stderr'      :man.IrafStdout()
            }

    if CLscript == 'gswavelength':
        defaultParams = {
            'inimages' : "",             # Input images
            'crval' : "CRVAL1",          # Approximate wavelength at coordinate reference pixel
            'cdelt' : "CD1_1",           # Approximate dispersion
            'crpix' : "CRPIX1",          # Coordinate reference pixel
            'key_dispaxis' : "DISPAXIS", # Header keyword for dispersion axis
            'dispaxis' : 1,              # Dispersion axis
            'database' : "database",     # Directory for files containing feature data
            'coordlist' : "gmos$data/CuAr_GMOS.dat", # User coordinate list, line list
            'gratingdb' : "gmos$data/GMOSgratings.dat", # Gratings database file
            'filterdb' : "gmos$data/GMOSfilters.dat", # Filters database file
            'fl_inter' : no,             # Examine identifications interactively
            'section' : "default",       # Image section for running identify
            'nsum' : 10,                 # Number of lines or columns to sum
            'ftype' : "emission",        # Feature type
            'fwidth' : 10.0,             # Feature width in pixels
            'gsigma' : 1.5,              # Gaussian sigma for smoothing
            'cradius' : 12.0,            # Centering radius in pixels
            'threshold' : 0.0,           # Feature threshold for centering
            'minsep' : 5.0,              # Minimum pixel separation for features
            'match' : -6.0,              # Coordinate list matching limit, <0 pixels, >0 user
            'function' : "chebyshev",    # Coordinate fitting function
            'order' : 4,                 # Order of coordinate fitting function
            'sample' : "*",              # Coordinate sample regions
            'niterate' : 10,             # Rejection iterations
            'low_reject' : 3.0,          # Lower rejection sigma
            'high_reject' : 3.0,         # Upper rejection sigma
            'grow' : 0.0,                # Rejection growing radius
            'refit' : yes,               # Refit coordinate function when running reidentify
            'step' : 10,                 # Steps in lines or columns for reidentification
            'trace' : yes,               # Use fit from previous step rather than central aperture
            'nlost' : 15,                # Maximum number of lost features
            'maxfeatures' : 150,         # Maximum number of features
            'ntarget' : 30,              # Number of features used for autoidentify
            'npattern' : 5,              # Number of features used for pattern matching (autoidentify)
            'fl_addfeat' : yes,          # Allow features to be added by reidentify
            'aiddebug' : "",             # Debug parameter for aidpars
            'fl_dbwrite' : "YES",        # Write results to database?
            'fl_overwrite' : yes,        # Overwrite existing database entries?
            'fl_gsappwave' : no,         # Run GSAPPWAVE on all images?
            'fitcfunc' : "chebyshev",    # Function for fitting coordinates
            'fitcxord' : 4,              # Order of fitting function in X-axis
            'fitcyord' : 4,              # Order of fitting function in Y-axis
            'logfile'     : "",          # Logfile name
            'verbose'     : yes,         # Verbose?
            'status'      : 0,           # Exit status (0=good)
            'Stdout'      :man.IrafStdout(),
            'Stderr'      :man.IrafStdout()
            }

    return defaultParams
