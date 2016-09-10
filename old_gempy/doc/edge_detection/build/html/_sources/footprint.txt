
.. _foot_class:

Footprint class
===============

Small class with facilities to put together the two footprint edges.
::

 USAGE

 fp = Footprint(edge_1,edge_2)

 parameters
 ----------
 edge_1:  Edge object corresponding to the left or bottom footprint edge.
 edge_2:  Edge object corresponding to the right or top footprint edge.
  

class members
---------------

- **id**     Integer reference number for the footprint.
- **region** Section of the image where the footprint solution is valid:
             (x1, x2, y1, y2), the origin of these coordinates is the
             lower left of the input image.
- **edges**  Tuple of Edge objects (edge_1,edge_2) defining footprint edges.
- **width**  Average width of the footprint.

.. _footpt_class:

class FootprintTrace
=====================

FootprintTrace offers functionality to create a FITS binary table with information about each footprint in the image suitable to be read by the CutFootprints class methods.
::

 USAGE

 fpt = FootprintTrace(footprints)

 parameters
 -----------
 footprints: List of footprint objects

FootprintTrace attributes
--------------------------

- **Footprints** Footprint objects list

.. _fp_asbintable:

FootprintTrace methods
------------------------

**as_bintable()**

 Creates the *TRACEFP*  FITS BINTABLE from the FootprintTrace object.

 The columns description are:

 ::

    'id'       : integer reference number for the footprint.
    'region'   : (x1,x2,y1,y2), window of pixel co-ords enclosing this footprint.
                 The origin of these coordinates could be the lower left of the
                 original image.
    'range1'   : (x1,x2,y1,y2), range where edge_1 is valid.
                 The origin of these coordinates is the lower left of the
                 original image.
    'function1': Fit function name (default: polynomial) fitting edge_1.
    'coeff1'   : Arrray of coefficients, high to low order, such that
                 pol(x) = c1*x**2 + c2*x + c3   (for order 2).
    'order1'   : Order or polynomial (default: 2).
    'range2'   : ditto for edge_2.
    'function2': ditto for edge_2
    'coeff2'   : ditto for edge_2
    'order2'   : ditto for edge_2

    'cutrange1'   : (x1,x2,y1,y2), range where edge_1 is valid.
                    The origin of these coordinates is the lower left of the
                    cutout region.
    'cutfunction1': Fit function name (default: polynomial).
    'cutcoeff1'   : Arrray of coefficients, high to low order, such that
                    pol(x) = c1*x**2 + c2*x + c3   (for order 2)
    'cutorder1'   : Order or polynomial (default: 2).
    'cutrange2'   : ditto for edge_2
    'cutfunction2': ditto for edge_2
    'cutcoeff2'   : ditto for edge_2
    'cutorder2'   : ditto for edge_2

.. _cutfp_class:

Class CutFootprints
====================

CutFootprint provides functionality to to build a list of footprint sections from the input *TRACEFP* table in the Astrodata object.
::

 USAGE

 cut = CutFootprints(ad)

 parameter
 ---------
 ad: The AstroData object containing a *TRACEFP* table extension.

CutFootprints attributes
-------------------------

- **ad**. AstroData object containing the extension *TRACEFP*.
- **debug**. Same is input parameter
- **cut_list**. List of CutFootprint objects.
- **dq_section**. DQ image ndarray.
- **filename**. Original fits filename. 
- **has_dq**. Boolean flag stating whether a DQ extension is present in the input AstroData object.
- **has_var**. Boolean flag stating whether a VAR extension is present in the input AstroData object.
- **instrument**. Is the CutFootprintsInstance.instrument
- **nregions**. Number of records in the *TRACEFP* table.
- **orientation**. Value of 90 is the footprints are vertical, zero if they are horizontal.
- **region**. (x1,x2,y1,y2) Coordinates of the region enclosing the footprint.
- **sci_data**. SCI image ndarray.
- **var_section**. VAR image ndarray.



CutFootprints methods
-------------------------

.. _cut_out:

**cut_out(rec,science,dq,var)**

 Cut a region enclosing a footprint. Each cut is defined by *region* and the footprint in it is defined by the edges fitting functions.  The science section is zero out between the rectangle borders and the footprint edge. The DQ section is bitwise ORed with 1. The results are sci_data, dq_section and var_section ndarrays.
::

 USAGE

 CutFootprints.cut_out(rec,science,dq,varcut)

 parameters
 ----------
 rec:     *TRACEFP* record
 science: SCI entire frame.
 dq:      DQ entire frame. Value is None if not available.
 var:     VAR entire frame. Value is None if not available.

.. _cutl_regions:

**cut_regions()**

 Loop through the records of the *TRACEFP* table creating one CutFootprint object per iteration setting the science data, dq and var data sections.  Then it appends each object to a list of cuts.

.. _cutl_initas:

**init_as_astrodata()**

 Initializes parameters to be used by as_astrodata method.  Creates a WCS object (pywcs) from the SCI header and form the output AD object with the PHU and MDF from the input AD. We are adding the *TRACEFP* extension as well for later use on the spectral reduction process.  

.. _cutl_astr:

**as_astrodata()**

 With each cut object in the cut_list having the SCI, DQ, VAR image data, form an hdu and append it to the output AstroData object.  Update keywords EXTNAME= 'SCI', EXTVER=<slit#>, CCDSEC, DISPAXIS, CUTSECT, CUTORDER in the header and reset WCS information if there was a WCS in the input AD header.

