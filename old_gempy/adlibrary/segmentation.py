import numpy as np
import scipy.ndimage as nd

from matplotlib import pyplot as pl

from ..library import gfit

try:
   from skimage.morphology import skeletonize
except ImportError:
   from ..library.skeletonize import skeletonize

# Lookup replacements; Gemini-specific functions only.
from astrodata_Gemini.ADCONFIG_Gemini.lookups.GMOS import StandardGMOSGratings
from astrodata_Gemini.ADCONFIG_Gemini.lookups.GMOS import GMOSfilters
from astrodata_Gemini.ADCONFIG_Gemini.lookups.GMOS import GMOSgratingTilt
from astrodata_Gemini.ADCONFIG_Gemini.lookups.GMOS import GMOSPixelScale

from astrodata_Gemini.ADCONFIG_Gemini.lookups.F2   import F2offsets

# ------------------------------------------------------------------------------
allowed_functions = ['polynomial','legendre','chebyshev']

# ------------------------------------------------------------------------------
def print_timing(func):
    # Decorator function to time a function
    def wrapper(*arg,**kargs):
        t1 = time.time()
        res = func(*arg,**kargs)
        t2 = time.time()
        print '%s took %0.3fs' % (func.func_name, (t2-t1))
        return res
    return wrapper

class Edge(object):
    """ Edge present functionality to store one edge 
        (x_array,y_array) coordinates, its orientation,
        fitting a function to it and resetting the fitting
        function and order.
    
        Attributes
        ----------
           trace:   Tuple with edge coordinates (x_array,y_array)
           orientation:
                    Integer. Values allow are 0 (horizontal) and 
                    90 (vertical).
           function: String. Function name for edge fitting. Values are
                     ['polynomial','legendre','chebyshev']
           order:    Int. Order of the fitting function.
           ylim:     Tuple. (min_y,max_y) values from the trace y_array.
           xlim:     Tuple. (min_x,max_x) values from the trace x_array.
           coefficients:
                     List with polynomial coefficients from the fitting.
                     For a polynomial, the list starts with the higher
                     first.
           evalfunction:
                     Function. Function evaluator. It takes one value
                     or list of values, returning an array.
        
        Methods
        -------
    """
    def __init__(self,x_array=[],y_array=[]):
        """
          Instantiation method to set the default function to
          polynomial with order 2. Also set the attribute
          trace with the input tuple (x_array,y_array).
        """
        self.function = 'polynomial'
        self.order = 2
        self.orientation = None
        self.coefficients = None
        self.evalfunction = None

        if x_array != []:
            self.trace=(x_array,y_array)
            self.ylim = (min(y_array), max(y_array))
            self.xlim = (min(x_array), max(x_array))
        

    def setfunction(self,function):
        """ Set a fitting function name.
            Use this before call to self.fitfunction
        """
        if function not in allowed_functions:
            raise RuntimeError,\
                 'setfunction: '+function+' is not:'+allowed_functions
        else:
            self.function = function
   
    def setorder(self,order): 
        """ Set a fitting function name.
            Use this before call to self.fitfunction
        """
        if order <= 0:
            raise RuntimeError,\
                 'setorder: '+order+' is not positive.'
        else:
            self.order = order

    def fitfunction(self):
        """ Call the actual fit function defined with 
            self.setfunction or use the default function if not.
            The order of the arrays is determine by the attribute
            'orientation'.

            Input: 
               Attribute Edge object having the trace attribute set.

            Output attributes:
               coefficients: Fitting coefficients.
               xlim:    (xmin,xmax)
               ylim:    (ymin,ymax)
               evalfunction: function to evaluate.
                                  
        """

        xx,yy = self.trace
      
        if self.orientation == 0:
            z = gfit.Gfit(xx,yy,fitname=self.function, order=self.order)
            # Evaluate the function with the xx array
            zy = z(xx)

            # Get the index of the minimum and maximum value in the zy array.
            imin = np.argmin(zy)
            imax = np.argmax(zy)
            self.ylim = (zy[imin], zy[imax])
            self.xlim = (xx[imin], xx[imax])
     
        elif self.orientation == 90:
            z = gfit.Gfit(yy,xx,fitname=self.function, order=self.order)
            zx = z(yy)
            # For a vertical edge, the minimum or maximum x is not
            # necessarily at the bottom or top 
            imin = np.argmin(zx)
            imax = np.argmax(zx)
            self.xlim = (zx[imin], zx[imax])
            self.ylim = (yy[imin], yy[imax])

        else:
            raise ValueError("Edge: orientation member is not 0 nor 90.")

        self.coefficients = z.coeff 
        self.evalfunction = z.__call__

def _plot_edges_dev(edges):
        """
          NOTE: This is for development. Not for 
                public release
          Plot the edges in the list array self.trace
          i: sequential number, can be edge number
        """
        for i,ed in enumerate(edges):
            x,y = ed.trace
            pl.plot(x,y,['b','r'][np.mod(i,2)]) 



# Module variable
allowed_filter_kernels = ['sobel']    # More filter can be added in the future

def edge_detector_data(ad,filter_kernel='Sobel'):
    
    """
      Edge_detector_data is a function that reads data from 
      the AstroData object and setups a dictionary with the MDF
      extension data returning the dictionary.

      ad: Input Astrodata object
      
      output: A dictionary with some of the Astrodata object MDF
              extension.

      mdf: Dictionary with:

       filter_kernel: String. The edge enhancement filter_kernel to use.
       image_data:    The image data ndarray.
       instrument:    String with the instrument name
       pixel_scale    Scalar. The pixel scale of the observation
       slitpos_mx:    An array with the slits x position
       slitpos_my:    An array with the slits y position 
       slitsize_mx:   An array with the slits width
       slitsize_my:   An array with the slits length
       speclen:       Contains an instrument dependent structure to
                      help derive the footprint_low and footprint_high
                      values.
       xccd:          Array with the MDF x_ccd values
       yccd:          Array with the MDF y_ccd values
       xybin:         (x_bin,y_bin). The image data binning
        
    """
    # check for supported astrodata type
    if ad.instrument() not in ['F2','GMOS','GMOS-N','GMOS-S','GNIRS']:
         raise SystemError, ('Astrodata type not supported: '+
                              ad.instrument())
    if ad['MDF'] == None:
         raise SystemError, ('Input astrodata object does not '
                             'have  an "MDF" extension')

    tb = ad['MDF'].data

    # Setup a dictionary of metadata to pass to the
    # subclass instantiation.
    mdf = {}
                                                # Slits parameter from the MDF.
    mdf['slitpos_mx']  = tb.field('slitpos_mx')  # The slits x position
    mdf['slitpos_my']  = tb.field('slitpos_my')  # The slits y position
    mdf['slitsize_mx'] = tb.field('slitsize_mx') # The slits width
    mdf['slitsize_my'] = tb.field('slitsize_my') # The slits length
    mdf['xccd']        = tb.field('x_ccd')       # 
    mdf['yccd']        = tb.field('y_ccd')

    # Get the string value for the instrument name.
    # Mainly use for GMOS-N, GMOS-S.
    mdf['instrument'] = '%s'%ad.instrument()

    # Get binning
    mdf['xybin'] = (ad.detector_x_bin(),ad.detector_y_bin())

    mdf['pixel_scale'] = ad['SCI'].pixel_scale()

    # Set default edge enhancement kernel name 
    mdf['filter_kernel'] = filter_kernel.lower()
    mdf['image_data'] = ad['SCI',1].data

    # Instantiate the appropiate class
    if 'F2' in ad.types:
        # Set the footprint_low, footprint_high values from the
        # instrument setting.
        mdf['speclen'] = f2_fplen(ad)
    elif 'GNIRS' in ad.types:
        # Set the footprint_low, footprint_high values from the
        # instrument setting.
        mdf['speclen'] = gnirs_fplen(ad)
    elif 'GMOS' in ad.types:
        # Set the footprint_low, footprint_high values from the
        # instrument setting.
        # Single hdu
        hdu = ad['SCI',1]
        mdf['speclen'] = gmos_fplen(hdu)
        #mdf['speclen'] = gmos_fplen(ad)
    else:
        raise ValueError("Astrodata type not supported: "+str(ad.types))

    return mdf
    


class EdgeDetector(object):
    """
      The EdgeDetector is a base class that provides the 
      following functionality:

      - Prefilter the input data if necessary
      - Enhance edges in the image
      - Binarize the enhanced image
      - Setup a reference set of edges
      - Scan the spectrum edges setting lists of
        (x,y) coordinates.

      Methods
      -------
       prefilter    -  Apply a smoothing filter to the input
                       image to decrese the noise level.

       enhance_edges-  Apply a filter_kernel, threshold
                       and turn to a binary image.

       binarize     -  Just a method to call prefilter and
                       enhance_edges methods.

       set_reference_edges - 
                       Calculate the list of footprint's left or bottom edge positions
                       at the middle row or columns of the input binary image.

       get_peaks    -  Get edge locations from a FLAT spectrum that is nearly 
                       horizontal or nearly vertical. (as the GMOS, F2 and
                       GMOS spectra are).

       get_edgesxy  -  Setup the peak searching algorithm for footprint edges

       _get_xy      -  Find the (x,y) pairs of all the peaks belonging
                       to an edge. Return a list of list.

       get_slices   -  Form a tuple with slice objects in order to select a
                       section of the binary_image.

       find_edges   -  High level method that puts together binarize, 
                       set_reference_edges and get_edgesxy methods returning
                       a list of lists of edges location.
       pixel_scale  -  pixel_scale arcs/pixel

      Attributes
      ----------
       image:         Input image ndarray
       filter_kernel: (string) 'sobel' 
       ref_edge_positions: 
                      List of footprint's left or bottom coordinate
                      in the spatial direction at the middle coordinate
                      in the dispersion direction.
                      #List of left footprint's x_coord at the
                      #middle row (for vertical footprints) or
                      #list of bottom footprint's y_coord at the
                      #middle column (for horizontal footprints).
                      
       bin_image:     Binary image ndarray. The result of the
                      binarize() method.

       footprint_spatial:
       footprint_dispersion:
                      List of of coordinates at the slit middle point
                      in the spatial and dispersion direction.
       footprint_low, footprint_high:
                      Lowest and highest pixel position in the dispersion
                      direcction. These marks are given by the instrument
                      setup at the exposure time.

       slitlengths:   List of slit widths.
       footprints_median_dispersion: 
                      Median of all the slit positions in the dispersion
                      direction.

       reference_ndispersion:
                      The number of rows about the middle of the image to
                      collapse in order to look for the reference edge positions.
       ref_threshold: The threshold applied in the collapse array in order 
                      to pick an edge position.
 
       xyskip:        The number of rows to skip in the image when traveling
                      up and dowm from the middle_dispersion.
       
       spatial_tolerance: 
                      The numbers of pixels of tolerance before adjusting
                      the local array of slit centers in the spatial direction.
                      (Default value is 3) (Note. Maybe it can be a function
                      of the slit width)

    """
    def __new__(cls,ad):
       """ 
         Overload the __new__ method, which
         allow us to create instances of F2, GNIRS or
         GMOS (_edge_detector) class according to the value
         of 'ad.is_type'.
         The EdgeDetector class has one argument and the __init__
         method of EdgeDetector and the subclass should have one
         arguments as well. (Need to see if it possible to change
         this requirements)
       """
       mdf = edge_detector_data(ad)
       if 'F2' in ad.types:
          inst = object.__new__(F2_edge_detector)
       elif 'GNIRS' in ad.types:
          inst = object.__new__(GNIRS_edge_detector)
       elif 'GMOS' in ad.types:
          inst = object.__new__(GMOS_edge_detector)

       inst.mdf = mdf
       return inst

    def __init__(self,foo=None):
        """
          Init method for EdgeDetector class. The
          argument 'foo' is a dummy argument and is
          needed when using __new__.
        """

        filter_kernel = self.mdf['filter_kernel']
        self.setfilter_kernel(filter_kernel)
        self.ref_edge_positions = []
        self.footprint_spatial = []
        self.slitlengths = []
        self.footprints_median_dispersion = None
        self.reference_ndispersion = None
        self.ref_threshold = None
        self.xyskip = None
        self.spatial_tolerance = None
        self.bin_image = None

    def setfilter_kernel(self,filter):
        """
          Check that the input parameter fiilter is one 
          of the supported filter kernel.

          Set the attribute filter_kernel to the
          input value.
        """
          
        if filter.lower() not in allowed_filter_kernels:
            raise ValueError("filter_kernel is not one of:"
                        (allowed_filter_kernels))
        else:
            self.filter_kernel = filter

    def binarize(self):
        """ Turn the grey input image into a
            binary image after prefiltering and
            enhancing edges with the Sobel filter.
        """
        self.prefilter()
        self.enhance_edges()
   
    def enhance_edges(self):
	"""
         Applies the selected filter_kernel to the image. We then take
         this output and calculate its standard deviation (we use numpy
         *std* function) to  threshold clipped it. Each value that is 
         higher than the threshold is set to one, otherwise to zero.
 
         NOTE: 
          The Sobel edge enhancement filter returns positive and negative
          values at the footprints edges. Here we take only the positive 
          edges (left/bottom); since considering both edges and looking
          where they are, can result in edge merging when a right-left
          (top-bottom) from two neighboring footprints are too close.

         Input
         -----
         self.image: Prefilter image if necessary.
         self.axis:  Integer: zero for horizontal edges
                              one for vertical edges.

         Output
         ------
         self.bin_image: Binary image

        """

        if self.filter_kernel == 'sobel':
            sdata=nd.sobel(self.image,axis=self.axis)
            # Get sigma from a narrow strip
            ny,nx=sdata.shape
            s1,s2=((slice(0,ny),slice(nx/2,nx/2+20)),
                   (slice(ny/2,ny/2+20),slice(0,nx)))[self.axis]
            
            sigma = np.std(sdata[s1,s2])
            # Binarize
            bdata=np.where(sdata>sigma,1,0)
            self.bin_image=bdata

        else:
            raise ValueError("Bad filter_kernel name")

    def prefilter(self):
        # The subclasses would do work if necessary
        pass

    def get_peaks(self,bin_image,r1,r2,threshold):
        """
           Inputs
           ------
             bin_image: Binary image
             r1,r2:     Locations in the dispersion direction
                        to collapse over.
             threshold: Lower limit. Any value in the collapse
                        section greater than this is a potential
                        edge.

           Output
           ------
             peaks_location: A list with peak's pixel locations.

           Get edge locations from a FLAT spectrum that
           is nearly horizontal or nearly vertical as the
           GMOS, F2 and GMOS are.
           Sum (collapse) all the pixels in the dispersion direction 
           between location r1 and r2. Get the indices of values in the
           sum ndarray that are higher that threshold.

        """

        # Collapsing in the dispersion direction r2-r1 rows/cols might
        # result in spreading when the edges are slanted as in the case
        # of GNIRS, resulting in short sections of a few pixels 
        # (the spreading) with value greater than one.

        # From r1,r2 form image slices to get the sections to be collapse.
        slice_y,slice_x = self.get_slices(r1,r2)
        line = np.sum(bin_image[slice_y,slice_x],axis=(not self.axis))
        
        # This line is one-pixel thick with values greater than one
        # for those sections containing potential peaks. Pick those
        # and change the values to one.
        binary_line = np.where(line > threshold,1,0) 
        
        # Make sure there are no holes in these short sections.
        binary_line = nd.binary_closing(binary_line)
        
        # Put labels on each of these continuous short sections.
        # 'label' puts a different integer values to different
        # sections.
        labels,nlabels = nd.label(binary_line)

        # Get the first element of each section as the position 
        # of the edge at this r1 location in the dispersion 
        # direction.

        if nlabels <=1: 
            return []
        peaks = [np.where(labels==k)[0][0] for k in range(1,nlabels+1)]
         
        return np.asarray(peaks)

    def set_reference_edges(self):
	"""Calculate the list of footprint's left/bottom edge positions
	   at the middle position (dispersion_center) of the input binary image.
           The bin_image in summary is the Sobel thresholded output
           of the input data, taking only the positive peaks
           (left/bottom edge).

           Algorithm to find the footprints' left/bottom edge positions.
           1. Take the median of 20 rows above and below
              the 'mrow' in the image, and collapse them for a
              good signal to eliminate spurious small traces.
           2. Given that we have at most a value of 40 in this
              list, look for indices where the value is greater than 20
              (to be conservative), since some of the footprints in this 40
              rows stretch might end at the top or bottom.
           3. These indices correspond to the footprints' left/bottom edges. 
           4. We compare these positions with the position of the
              footprints' middle position (slitpos_mx) from the MDF table.
           5. There should be one left/bottom edge per slitpos_mx value.
           5.5 If we have more or less edges that lenght of MDF array 
              slitpos_mx then the algorithm fails mostly due to poor signal
              to noise ratio in the input image or too close of a separation
              between footprints (slits).
           6. Return the list of left/bottom edges and the list of corrected
              footprints' middle position.
           
        """

        # The middle location of each slit in the spatial direction.
        fp_spatial = np.asarray(self.footprint_spatial,dtype=int)

        # Slit_widths
        slitlengths = self.slitlengths

        middle_dispersion = self.footprints_median_dispersion
        ny,nx = self.image.shape

        # If this value is larger than nx then just take the middle.
        # This can happen if the user is looking for footprint on one
        # GMOS extension rather than the whole mosaicked image.
        if middle_dispersion > nx:
            middle_dispersion = nx/2

        # r1 and r2 are the positions in the dispersion direction
        # marking the section to collapse.
        nref_dispersion = self.reference_ndispersion
        r1 = middle_dispersion - nref_dispersion/2
        r2 = middle_dispersion + nref_dispersion/2

        # Reduce the width of the edges to 1 pixel (skeletonize).
        # This function is slow, to take only a stripe about the
        # middle.
        if self.axis==1:  # F2, GNIRS
            skel_bar = skeletonize(self.bin_image[r1:r2,:])
        else:
            skel_bar = skeletonize(self.bin_image[:,r1:r2])

        edges_peak_pos = self.get_peaks(skel_bar,
                                      0,r2-r1, self.ref_threshold) 

        # Get the edges location in this section.
        #edges_peak_pos = self.get_peaks(self.bin_image,
        #                              r1,r2,self.ref_threshold) 

        # Get the indices of edges_peak_pos where each of the fp_spatial
        # elements are to be position to maintain order.
        ss = np.searchsorted(edges_peak_pos, fp_spatial)

      
        # If the fp_spatial position are to the right of the edges_peak_pos
        # elements the indices start at 1.
        if ss[0] >0: ss=ss-1

        # These are the real footprint indices in the edges_peak_pos list.
        # If we have some repeated indices then we can have one or more missing
        # edges in the edges_peak_pos list.
        badref_message = ('Could not determine a reference set of edges '
                         'from the input image.\nPlease make sure you have'
                         ' an input image with a signal to noise above 3.')
        
        if (len(ss) != len(np.unique(ss))) or \
           (len(np.unique(ss))!=len(fp_spatial)):
            raise ValueError(badref_message)


        edges_lb = list(edges_peak_pos[ss])

        # Make sure the left/bottom edge is no further away
        # more than slitlengths/2 away from the spatial.

        edar = np.asarray(edges_lb)
        for k,fs in enumerate(fp_spatial):
            g, = np.where(abs(edges_lb-fs)<slitlengths[k])
            if g.size == 0:
                raise ValueError(badref_message)

        # Set the reference edges positions.
        self.ref_edge_positions = edges_lb

        # ----Center the spatial values between left/bottom 
        #     and right/top edges
        # Make a right_edge array
        edges_rt = [(le+w+1.) for le,w in zip(edges_lb,slitlengths)]
        diff_lb = np.median(fp_spatial - edges_lb)
        diff_rt = np.median(edges_rt - fp_spatial)
        sptemp = fp_spatial - (diff_lb-diff_rt)/2.

        # Adjust values
        self.footprint_spatial = sptemp     

        return

    def get_edgesxy(self):
        """
          Setup the peak searching algorithm for footprint edges.

          - Divide the image in strips along the spatial direction. Each stripe
            start, end pixel number is determined by a given width (xyskip).

          - Make 2 sets of stripe markings. One starting from the footprint
            midpoint in the disppersion direction (middlerow)
            towards the last highest pixel number and another set from the 
            'middlerow' pixel number  toward the first pixel number.

          - Call _get_xy to find the (x,y) pairs of all the peaks belonging
            to an edge. Return a list of (x_array, y_array) list.

          - Merge correspondings upward and downward list into one.

          - Return the list.

        """
        bin_image = self.bin_image
        left_edges = self.ref_edge_positions 
        # The amount of rows/cols to skip to get the next dispersion pair (x,y)
        xyskip = self.xyskip

        # dispersion axis length
        ny,nx = bin_image.shape
        dispersion_size = [nx,ny][self.axis]

        # --- From middle dispersion to dispersion_size

        # Form tuples from middle row to last row every xyskip rows

        middle_dispersion = int(self.footprints_median_dispersion)
        span = range(middle_dispersion, dispersion_size, xyskip)
        dispersion_range = [(span[i],span[i+1]) for i in range(0,len(span)-1)]

        # Get (x,y) pairs at the footprint's edge 
        xylistu = self._get_xy(dispersion_range)

        # --- From middle dispersion to zero in the dispersion direction

        # Form tuples from middle row to first row every xyskip span of rows.
        span = range(middle_dispersion, 0, -xyskip)
        dispersion_range = [(span[i+1],span[i]) for i in range(0,len(span)-1)]

        xylistd = self._get_xy(dispersion_range)
        
        if (len(xylistu)!=len(left_edges)) or (len(xylistd)!=len(left_edges)):
            raise ValueError(
               'Could not determine the left or bottom set of edges from the '
               'input image.\nPlease make sure you have an image with a '
               'to noise above 3.')

        # Add each slit up and down to slit from 0 to dispersion_size
        xylist_1 = [xyu+xyd for xyu,xyd in zip(xylistu, xylistd)]

        # Sort the array using the independent coordinate x: for axis 0 
        # y for axis 1

        for k,xy in enumerate(xylist_1):
            xx = np.asarray([x for x,y in xy])
            yy = np.asarray([y for x,y in xy])
            order = np.argsort([xx,yy][self.axis])
            xx = xx[order]
            yy = yy[order]
            xylist_1[k] = [(x,y) for x,y in zip(xx,yy)]

        return xylist_1

    def _get_xy(self,rc_pairs):
        """
          Input
          -----
           rc_pairs: Tuple of row/column numbers (r1,r2) in the image.
	  
	  1. Sum all the rows/columns between a pair (r1,r2) along the
             columns. 
          2. In resulting list find peaks with value (r2-r1)/2 or larger.
          3. Each peak location found in the sum list is compare with the
             ref_edge_position list and if they are within a given width
	     then they are the starting points to travel along the edge
	     from the middle row/col up/right and down/left, appending
             the (x,y) location to a list. There are as many lists as 
             the length of ref_edge_position list.. 
	 
        """
         
        bin_image = self.bin_image

        # Make a local copy of the list
        ref_edges = self.ref_edge_positions[:]

        # Initialize the list of lists.
        edges_xy = [[] for _ in range(len(ref_edges))]

        # width: The maximum space allow between a reference edge
        # marker and a peak for it to be considered as
        # a slit edge location.

        width = self.slitlengths/self.slit_ratio
        # For rows between r1 and r2 find slit edges that are
        # close (within 'width') to the current reference edges.

        for r1,r2 in rc_pairs:

            # Find peaks by collapsing (sum) rows/columns from r1 to r2,
            # since bin_image is binary, the sum would be at most
            # (r2-r1+1). Any point in the sum that is larger than
            # threshold is consider a potential edge.
            threshold = abs(r2-r1)/2.
	    peaks = self.get_peaks(bin_image,r1,r2,threshold)
            if len(peaks) == 0: continue

            # For each reference location find a close peak in the 
            # current list.
            for k, redge in enumerate(ref_edges):

                diff = redge - peaks

                # The indices of possible edges that are within
                # 'width' pixels.
                cindx, = np.where(abs(diff) < width[k])

                if len(cindx) > 0: 
                    ic = 0 
                    if len(cindx)>1:
                       # Choose the closest peak to ref_edges[k]
                       ic = np.argmin(abs(peaks[cindx]-ref_edges[k]))
                    peak_location = peaks[cindx[ic]]

                    # Adjust the reference edge location (k)
                    # for slit-edge curvature by substracting the
                    # difference in location between the reference and
                    # peak_location
              
                    peak_diff = redge - peak_location
                    if abs(peak_diff) > self.spatial_tolerance:     
                        ref_edges[k] = ref_edges[k] - peak_diff

                    # Add the pair (x,y) to the list for row r1.
                    x,y = [(r1,peak_location),(peak_location,r1)][self.axis]
                    edges_xy[k].extend([(x,y)])

        return edges_xy

    def get_rightTop_edge(self,xylist_1):
        """
           Giving the left/bottom xylist form the
           rightTop xylist by adding to each edge the
           corresponding slitlength.

           Input
           -----
            xylist_1:   List of [x_arrray, y_array] for each
                      left/bottom edge.

        """
        # Initialize an xylists for edge 2.
        xylist_2 = [[] for _ in range(len(xylist_1))]
        
        for k,xy in enumerate(xylist_1):
            xx = np.asarray([x for x,y in xy])
            yy = np.asarray([y for x,y in xy])
            w = self.slitlengths[k]+1.
            xw,yw = [np.asarray([0,1])*w,np.asarray([1,0])*w][self.axis]
            xylist_2[k] = [(x+xw,y+yw) for x,y in zip(xx,yy)]

        return xylist_2


    def get_slices(self, r1,r2):
        """ Form slice objects for a 2 dimensional image.

            r1,r2: Input row numbers to slice. We select all columns

        """
        slice_y = slice(r1,r2)
        slice_x = slice(None,None)
        return (slice_y,slice_x)

    def find_edges(self):
        """ 
          Function that takes the input image data and finds where the 
          footprint edges are by forming a tuple (x_array, y_array)
          of coordinates for each footprint's edges.
          Returns a list of these tuples for the left/bottom edges
          and a list of these tuples for the right/top edges.
          
        """
        # Take in input image into an image of zeros and ones.
        self.binarize()

        # Finds a starting set of left/bottom (whether the footprints
        # are vertical/horizontal) edges.
        self.set_reference_edges()

        # Forms the lists of tuples (x_array,y_array) for each 
        # left/bottom edges.
        xylist_1 = self.get_edgesxy()

        # Add the footprint widths to the left/bottom edges and forms
        # a list tuples (x_array,y_array) for each right/top edges.
        xylist_2 = self.get_rightTop_edge(xylist_1)

        return (xylist_1,xylist_2)

class F2_edge_detector(EdgeDetector): 
    """
        Class for F2 data. Inherits all methods

    """
    def __init__(self,foo=None):
        """ 
           Initialize attributes for F2 footprint edges location
           and the processing of the maximum and minimum coordinate in the
           dispersion direction.
           Note: The 'foo' parameter is needed for subclass instantiation
                 via the __new__ overloading method. It is not part of the
                 Edge Detection set of attributes.
        """

        # mdf attribute is set in the __new__ method and after the
        # instrument specific class is instantiated.
        mdf = self.mdf

        # Get data from the parent function
        EdgeDetector.__init__(self)

        self.axis = 1
        self.image = mdf['image_data']
        slitpos_mx = mdf['slitpos_mx']
        slitpos_my = mdf['slitpos_my']
        slitsize_mx = mdf['slitsize_mx']
        y1_off = mdf['speclen']['y1_off']
        y2_off = mdf['speclen']['y2_off']

        # Sort to make sure they are in order
        xorder = np.argsort(slitpos_mx)
        slitpos_mx = slitpos_mx[xorder]
        slitpos_my = slitpos_my[xorder]
        slitsize_mx = slitsize_mx[xorder]

 
        pixscale = mdf['pixel_scale']
        asecmm = 1.611444     # Arcsec per mm

        x_bin,y_bin = mdf['xybin']
        yscale = pixscale*y_bin
        xscale = pixscale*x_bin

        imsize = self.image.shape
        ny,nx = self.image.shape

        # Correct for small distorsion in the spatial direction.
        slitpos_mx = (slitpos_mx * 1.00035894) - 0.86698091

        # Now convert to pixel units and offset to the footprint middle
        # position. These are the position at the middle of each footprint in
        # the spatial direction.
        self.footprint_spatial = np.sort((slitpos_mx*asecmm / pixscale) + nx/2)

        # Convert the arcsec slit position in the dispersion direction
        # to pixel.
        footprint_dispersion = (slitpos_my * asecmm / pixscale) + (ny / 2)

        # With y1_off and y2_off as the offsets from the slit position
        # calculate the minimum (x1) and maximum (x2) footprint
        # extend from the slit.
        y1 = footprint_dispersion + y1_off
        y2 = footprint_dispersion + y2_off
        dy = y2 - y1
        y_cen = y1 + (dy / 2)    # This should be the position where to start
                                 # collapsing, up and down 
        self.footprint_low = np.asarray(y1,dtype=np.int)
        self.footprint_high = np.asarray(y2,dtype=np.int)

        # The middle position in the dispersion direction.
        self.footprints_median_dispersion = np.median(
                        (slitpos_my*asecmm/yscale) + ny/2)
        self.slitlengths = slitsize_mx*asecmm/xscale    # To pixel

        # The number of rows about the middle of the image to
        # collapse in order to look for the reference edge positions.
        self.reference_ndispersion = 40

        # The threshold applied in the collapse array in order to pick
        # an edge position.
        self.ref_threshold = 15  

        # The number of rows to skip in the image when traveling up and
        # dowm from the middle middle_dispersion.
        self.xyskip = 20

        # The numbers of pixels of tolerance before adjusting the local 
        # array of slit centers in the spatial direction.
        # See _get_xy method.
        self.spatial_tolerance = 3
        self.slit_ratio = 2



class GNIRS_edge_detector(EdgeDetector):
    """
      GNIRS_edge_detector subclass
    """

    def __init__(self,foo=None):

        mdf = self.mdf
        EdgeDetector.__init__(self,foo)

        # COMMENTS: See attribute docstring in the
        #            base class.
        self.axis = 1  
        self.image = mdf['image_data']
        slitpos_mx = mdf['slitpos_mx']
        slitpos_my = mdf['slitpos_my']
        slitsize_mx = mdf['slitsize_mx']
        xccd = mdf['xccd']
        yccd = mdf['yccd']

        pixscale = mdf['pixel_scale']
        self.footprint_spatial = xccd
        self.footprints_median_dispersion = np.median(yccd)

        # NOTE!! The slitsize in the MDF is already in pixel and the
        #        value is half of the slitwidth.
        self.slitlengths  = slitsize_mx*2

        self.ref_threshold = 2    # for GNRIS
        self.reference_ndispersion = 10
        self.xyskip = 5
        self.spatial_tolerance = 3
        self.slit_ratio = 4        # factor to divide the slitlenght array
                                     # Any pixel closer to slitlenght/4.
                                     # is consider part of the edge.

        self.footprint_high = np.zeros(len(slitsize_mx),dtype=np.int)
        self.footprint_low = np.zeros(len(slitsize_mx),dtype=np.int)

    
    def prefilter(self):
        # Clip to normalize the orders; otherwise the
        # high orders would be masked by thresholding done
        # in enhance_edges(). 

        data = self.image
        for k in range(2):    # Do normalization only once.
            dmean = np.mean(data)
            data = np.where(data>dmean,dmean,data)
            # Eliminate any negative pixels
            data = np.where(data<0,0,data)
        self.image = data

    def find_edges(self):
        """GNIRS does not have reliable slit widths.
           Need to find the right edges by using the 
           binary image with the right edges only.
        """
        self.binarize()
        self.bin_image = self.left_bin_image
        self.set_reference_edges()
        xylist_1 = self.get_edgesxy()
        self.bin_image = self.right_bin_image
        self.set_reference_edges()
        xylist_2 = self.get_edgesxy()

        # find the min max
        k=0
        for xy1,xy2 in zip(xylist_1,xylist_2):
            min1 = min([y for x,y in xy1])
            min2 = min([y for x,y in xy2])
            max1 = max([y for x,y in xy1])
            max2 = max([y for x,y in xy2])
            self.footprint_low[k] = min(min1,min2)
            self.footprint_high[k] = max(max1,max2)
            k += 1
        return (xylist_1,xylist_2)
        
    def enhance_edges(self):
        """ GNIRS_edge_detector method to enhance the footprint edges using the Sobel
            kernel. Generate two binary images, one with the left edges
            and the other showing the right edges only. This is because the
            MDF information about the footprints location is not well 
            determined.
        """
        sdata=nd.sobel(self.image,axis=self.axis)
        std = np.std(sdata)
        bdata=np.where(sdata>std,1,0)
 
        # Make the edges one pixel wide
        self.left_bin_image = skeletonize(bdata)

        bdata=np.where(sdata < -std,1,0)
        self.right_bin_image = skeletonize(bdata)


class GMOS_edge_detector(EdgeDetector):
    """
     Edge Detector class for GMOS MOS data.

     Methods
     -------
       get_slices: Overload the parent method

    """ 
    def __init__(self,foo=None):

        mdf = self.mdf
        EdgeDetector.__init__(self,foo)

        # COMMENTS: See attribute docstring in the
        #            base class.
        self.axis = 0     # for GMOS 
        self.image = mdf['image_data']
        #self.image = image
        slitpos_mx =  mdf['slitpos_mx']
        slitpos_my =  mdf['slitpos_my']
        slitsize_mx = mdf['slitsize_mx'] 
        slitsize_my = mdf['slitsize_my'] 
        xccd = mdf['xccd']
        instrument =  mdf['instrument']
        x_bin,y_bin = mdf['xybin']
        # The mdf['speclen'] was set in init function edge_detector_data()
        wave1,wave2,wavoffset,nmppx,a,cwave,l_yoff = mdf['speclen']
        
        pixscale = mdf['pixel_scale']
        asecmm = 1.611444     # arc seconds per mm
        npix_y,npix_x = self.image.shape

        xscale = pixscale*x_bin
        yscale = pixscale*y_bin

        Y = slitpos_my
        if instrument == 'GMOS-S':
            yccd = 0.99911*Y - 1.7465E-5*Y**2 + 3.0494E-7*Y**3
        else:  # GMOS-N
            yccd = 0.99591859227*Y + 5.3042211333437E-8*Y**2 + \
                            1.7447902551997E-7*Y**3
        yccd = yccd * asecmm/yscale + npix_y/2 + l_yoff

        # sort the values
        order = np.argsort(yccd)
        yccd = yccd[order]
        slitsize_my = slitsize_my[order]
        slitpos_mx = slitpos_mx[order]

        # Set the attribute members 
        self.footprint_spatial = yccd
        self.slitlengths = (slitsize_my*asecmm/yscale)

        # Let's use the half point in the x direction instead
        # of median(slitpos_mx).
        self.footprints_median_dispersion = np.median(xccd)

        self.ref_threshold = 15 
        self.reference_ndispersion = 40
        self.xyskip = 20
        self.spatial_tolerance = 3
        self.slit_ratio = 2

        #  -- Calculates the low and high pixels values in the dispersion 
        #     direction based on the exposure configuration.

        # Set xccd to pixel unit
        xcen = npix_x/2.
        xccd = xcen + slitpos_mx * asecmm/xscale
        
        # Footprint length in pixels
        speclen = int((wave2-wave1)/nmppx)

        # Center of footprint in pixels 
        pixcwave = speclen - (cwave-wave1)/nmppx

        # 
        yccd = yccd - l_yoff
        y = yccd/npix_y -.5
        dx = npix_x * (0.0014*y - 0.0167*y**2)

        refpix = pixcwave
        low = []
        high = []
        for xcc,dd in zip(xccd,dx):
            x1 = int(xcen-(xcen-xcc)/a-pixcwave) + wavoffset/nmppx + dd
            x2 = x1 + speclen-1
            if x1 < 1:
               refpix = refpix+x1-1
               x1 = 0

            if x2 > npix_x:
               x2 = npix_x-1

            low.append(x1)
            high.append(x2)

        self.footprint_low = low 
        self.footprint_high = high

    def get_slices(self, c1,c2):
	""" Form slice object for a 2 dimensioncal image.
	
 	    c1,c2: Input column numbers to slice. 
                   We select all rows.

	"""
	slice_x = slice(c1,c2)
	slice_y = slice(None,None)
	return (slice_y,slice_x)


def gmos_fplen(ad):
    """
      GMOS minimum and maximum offsets from
      the slit position in the dispersion direction
      This is a take from gscut.cl
    """
    filters = GMOSfilters.GMOSfilters
    gratings = StandardGMOSGratings.StandardGMOSGratings
    grating_tilt = GMOSgratingTilt.grating_tilt
    pixscale = GMOSPixelScale.gmosPixelScales

    # Define the spectral cut-off limit (red limit) according to the iccd
    # (detector type). Value is in nm. If needed, this can be changed to also
    # accomodate different values for GMOS-N and GMOS-S, see pixscale
    detector_upper_spec_limit = {
       'SDSU II CCD':             1025,   # EEV CCDs
       'SDSU II e2v DD CCD42-90': 1050,   # e2vDD CCDs
       'S10892-01':               1080,   # Hamamatsu CCDs
    }

    npix_y, npix_x = ad.data.shape
    xbin = ad.detector_x_bin()
    ybin = ad.detector_y_bin()
    instrument = ad.instrument().as_str()

    # Get header values. The input AD should have been
    # verified to contain a MOS image.
    phu = ad.phu_get_key_value
    dettype = phu('DETTYPE')
    grating_name = phu('GRATING')
    filter1 = phu('FILTER1')
    filter2 = phu('FILTER2')
    cwave = phu('GRWLEN')
    tilt = phu('GRTILT')
    tilt = np.radians(tilt) 

    xscale = pixscale[instrument, dettype] * xbin
    yscale = pixscale[instrument, dettype] * ybin

    # Get grating info from lookup table
    grule, gblaze, gR, gcoverage, gwave1, gwave2,\
        wavoffset, l_yoff = gratings[grating_name]

    greq=(cwave * grule) / 1.e6

    # grating_tilt is a list of tuples
    greqs  = [g for g, t in grating_tilt]
    gtilts = [t for g, t in grating_tilt]
    # Interpolate at greq
    gtilt = np.interp(greq, greqs, gtilts)

    gtilt = np.radians(gtilt)
    a = np.sin(gtilt + 0.872665) / np.sin(gtilt)
    gR = 206265. * greq / (0.5 * 81.0 * np.sin(gtilt))
    nmppx = a*xscale*cwave*81.0*np.sin(gtilt)/(206265.*greq)
    wave1 = gwave1
    wave2 = gwave2

    # get filter information
    fwave1 = 0.0     ; wmn1 = 0.0     ; wmn2 = 0.0
    fwave2 = 99999.0 ; wmx1 = 99999.0 ; wmx2 = 99999.0
    if filter1 != '' and 'open' not in filter1:
        wmn1, wmx1, ffile = filters[filter1]
    if filter2 != '' and 'open' not in filter2:
        wmn1, wmx1, ffile = filters[filter2]

    fwave1 = max(wmn1,wmn2)
    fwave2 = min(wmx1,wmx2)

    # determine whether filter or grating limits wavelength coverage
    wave1 = max(wave1,fwave1)
    wave2 = min(wave2,fwave2)

    # This sets the hard red limit according to detector type if user doesn't
    # supply an upper limit
    if wave2 > detector_upper_spec_limit[dettype]:
        wave2 = detector_upper_spec_limit[dettype]

    speclist= (wave1,wave2,wavoffset,nmppx,a,cwave,l_yoff)
    return speclist


def f2_fplen(ad):
    """ Calculates the minimum (x1) and maximum (x2) footprint
        extend from the slit position for F2 MOS data
        (This code is a take from f2cut.cl)
    """
    yoffset_delta = F2offsets.yoffset_delta
    filter_table  = F2offsets.filter_table

    header = ad.phu.header
    if 'grism' in header:
        grism = header['grism']
    elif 'grismpos' in header:
        grism = header['grismpos']
    else:
        raise KeyError('Keyword "GRISM" nor "GRISMPOS" not found in PHU.')

    if 'filter' in header:
        filter = header['filter']
    else:
        raise KeyError('Keyword "FILTER" not found in PHU.')

    if 'mospos' in header:
        slit = header['mospos'][:-1]      # All chars but last.
    else:
        raise KeyError('Keyword "MOSPOS" not found in PHU.')

    # Get the tuple value (yoffset,delta)
    yoffset,delta = yoffset_delta[(grism, filter, slit)]

    # From filter_table calculates filter_width based on cuton50 and cutoff50.
    filt_names=['center','width','cuton80','cutoff80','cuton50',
                'cutoff50','transmission']
    filter_lower = filter_table[filter][filt_names.index('cuton50')]
    filter_upper = filter_table[filter][filt_names.index('cutoff50')]
    filter_width = filter_upper - filter_lower

    # Rename delta value to orig_dispersion (more meaning).
    orig_dispersion = delta 

    # The dispersion is required in microns, but the value
    # in the tuple (yoffset,delta) is in (negative) Angstroms
    dispersion = orig_dispersion / -10000
 
    # Form dictionary to output
    y1_off = yoffset - (filter_width / dispersion) / 2
    y2_off = yoffset + (filter_width / dispersion) / 2
    out = {'y1_off':y1_off,'y2_off':y2_off}
    return out

def gnirs_fplen(ad):
    """ This code will calculate the minimum and maximum footprint
        extend from the slit position for GNIRS XD data.
        At this time, looking at the nscut.cl the footprints
        extends all the y-range.
        Need more discussion with the instrument scientist to
        see if ymin, ymax can be determine to be other than the
        default's.
    """
    return None

