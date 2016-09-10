.. _ed_det:

class EdgeDetector
==================

The EdgeDetector is a base class that provides the 
following functionality:
  
- The :ref:`edge_detector_data <ed_data>` is a function that reads the input AstroData object and set the necessary parameters to each subclass. 
- Prefilter the input data if necessary
- Enhance edges in the image
- Binarize the enhanced image
- Setup a reference set of edges
- Scan the footprint edges setting lists of (x,y) coordinates.

EdgeDetector methods
--------------------

These are the EdgeDetector class methods than can be overrriden by the subclasses methods witht the same name:

- :ref:`binarize <binarize>`
- :ref:`enhance_edges <enh_edges>`
- :ref:`find_edges <find_edges>`
- :ref:`prefilter <prefilter>`
- :ref:`get_edgesxy <get_edgesxy>`
- :ref:`get_peaks <get_peaks>`
- :ref:`set_reference_edges <set_ref_edges>`


EdgeDetector instantiates the following classes according to the value of *ad.istype()* descriptor.

.. _ed_det_subs:

- :ref:`GMOS_edge_detector <gmos_class>`
- :ref:`F2_edge_detector <f2_class>`
- :ref:`GNIRS_edge_detector <gnirs_class>`

.. _binarize:

**binarize()** 

 Binarize is an EdgeDetector method that turns the grey input image into a binary image after prefiltering and enhancing edges with a filter_kernel (Sobel kernel as default). The steps to achieve this are as follows:

 1. **Prefilter()**. 
  An instrument dependent function that applies a smoothing filter to decrease the raw data noise. For example in the GNIRS images, some of the orders have higher fluxes than the low orders; a normalization allows to calculate a sigma value that is well below the low orders.

.. _enh_edges:

 2. **enhance_edges()**
  Apply the selected filter_kernel to a prefiltered image. Take this output and calculate its standard deviation (we use numpy *std* function) to threshold clipped it. Each value that is higher than the threshold is set to one, otherwise to zero. The result is a binary image containing the left edge of each footprint.
   
 **Note**
 The Sobel edge enhancement filter returns positive and negative values at the footprints edges. Here we take only the positive edges (left/bottom); since considering both edges and looking where they are, can result in edge merging when a right-left (top-bottom) from two neighboring footprints are too close.

.. _find_edges:

**find_edges()**

 Find_edges is an EdgeDetector method that calls the functions:

 - :ref:`binarize <binarize>`
 - :ref:`set_reference_edges <set_ref_edges>`
 - :ref:`get_edgesxy <get_edgesxy>`

 It returns a list of lists of edges coordinates.

.. _prefilter:

**prefilter()**

 Prefilter is an EdgeDetector that will be overloaded by the subclasses method. It usually is a smoothing function to decrease sharp pixels avoiding large intensities when the Sobel filter is applied.

.. _get_edgesxy:

**get_edgesxy()**

 Get_edgesxy is an EdgeDetector method that setups the peak searching algorithm to find footprint edges by using the list of reference coordinates from the set_reference_edges function. Here is a summry to obtain the list of (x_array,y_array) for each footprint edge that matches the reference coordinates:

 1. Starting from middle of the image where we already have the reference edges moves toward one of the image in the dispersion direction, collapsing a given number of row/columns.

 2. From this collapsed line get the set of peaks coordinates.

 3. Compare these coordinates with the reference list and select those that fall within a given tolerance.

 4. Append these coordinates (row/column, peak_coordinate) to the tuple (x_array,y_array). There is one tuple per footprint edge.

 5. Repeat steps 1. through 4. now moving from the middle of the image toward the other end of the image in the dispersion direction making a second list of (x_array,y_array).

 6. Merge the two lists into one.

 7. Return the (x_array,y_array) list.

.. _get_peaks:

**get_peaks()**

 Get_peaks is an EdgeDetector method that finds peaks in a collapsed line from a binary image. Any point that is larger than a given threshold is consider a peak. 

 Usage
 ::
 
  peaks = ed.get_peaks(bin_image,r1,r2,threshold)

  parameters
  ----------
    bin_image:
          Binary image 
          
    r1,r2: Locations in the dispersion direction to collapse over.

    threshold:
          Any value in the collapse section greater than this is
          a potential edge.

  Output
  ------
    peaks: A list with peak's pixel locations.

.. _set_ref_edges:

**set_reference_edges()**          

 Set_reference_edges is an EdgeDetector method to find the left/bottom footprints' edge positions along a line crossing the dispersion axis. The line is chosen to be in the center of the image. Notice that the image is here is the binary image.

 Algorithm to find the footprints left/bottom edge positions.

 1. Take the median of 20 rows above and below the center of the image, and collapse them into a line. 
 2. Given that we have at most a value of 40 in this line (the image is zeros and ones, with one were there is an edge) where edges are present, look for coordinates where the value is greater than 20 (to be conservative).
 3. These coordinates correspond to the footprints' left/bottom edges.
 4. We compare these positions with the position of the centers of the footprints in the spatial direction. This comparison is within the width of the footprint.
 5. There should be one left/bottom edge per slitpos_mx value.
 6. Return the list of left/bottom edges and the list of corrected footprints' middle position. We correct the middle positions to be at the middle between the footprint's edges coordinate.

.. _ed_data:

**edge_detector_data(ad, filter_kernel='sobel')**

 Edge_detector_data is a function to setup instrument dependent parameters and to read selected columns from the MDF table in the AstroData object. See below for the returning dictionary description.
 ::

  parameters
  ----------

  - ad: Input AstroData object

  - filter_kernel: 'sobel' is the only supported kernel at this time. 'sobel' refers to the scipy.ndimage.sobel filter. 

  output
  ------

.. _mdf:

  mdf: A dictionary with the following information
  ::

   mdf = {
       filter_kernel: String. The edge enhancement filter_kernel to use.
       image_data:    The image data ndarray.
       instrument:    String with the instrument name.
       pixel_scale    Scalar. The pixel scale of the observation
       slitpos_mx:    An array with the slits x position. Is the slit position
                      in the spatial direction.
       slitpos_my:    An array with the slits y position. Is the slit position
                      in the dispersion direction.
       slitsize_mx:   An array with the slits width.
       slitsize_my:   An array with the slits length.
       speclen:       Contains an instrument dependent structure to
                      help derive the footprint_low and footprint_high
                      values.
       xccd:          Array with the MDF x_ccd values.
       yccd:          Array with the MDF y_ccd values.
       xybin:         (x_bin,y_bin). The image data binning.
	  }


.. _gmos_class:

GMOS_edge_detector subclass
===========================

Subclass of EdgeDetector that offers facilities to detect footprint edges in a GMOS flat field.  All methods, with the exception of __init__ and get_slices are defined in the parent class. This subclass is instantiated by the parent class EdgeDetector when the Astrodata object contains GMOS data.

GMOS_edge_detector methods
---------------------------

**get_slices()**

 GMOS_edge_detector method to form a pair of 'slice' python objects to be used when collapsing columns in the input image.

.. _f2_class:

F2_edge_detector subclass
===========================

EdgeDetector subclass that offers facilities to detect footprint edges in F2 flat fields.  All methods, with the exception of __init__ and get_slices are defined in the parent class.

F2_edge_detector methods
-------------------------

**get_slices()**

 F2_edge_detector method to form a pair of 'slice' python objects to be used when collapsing rows in the input image.

.. _gnirs_class:

GNIRS_edge_detector subclass
=============================

EdgeDetector subclass that offers facilities to detect footprint edges in GNIRS flat fields. The methods defined in this class will override the parent methods.

GNIRS_edge_detector methods
----------------------------

**enhance_edges()**

 GNIRS_edge_detector method to enhance the footprint edges using the Sobel kernel. Generate two binary images, one with the left edges and the other showing the right edges only. This is because the MDF information about the footprints location is not well determined.


**find_edges()**

 GNIRS_edge_detector method to determine the left and right footprint edges. 

**prefilter()**

 The GNIRS flat fields have orders with very different intensities. To make sure we detect edges in weak orders we normalize by clipping at the mean value of the image; i.e. any value greater than the mean is replaced by it. We repeat this process again.

