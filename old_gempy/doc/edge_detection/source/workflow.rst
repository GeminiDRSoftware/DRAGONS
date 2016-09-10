
.. _work_flow:

Basic functionality
===================

The upper level functions :ref:`trace_footprints <trace_footprints>` and :ref:`cut_footprints <cut_footprints>` use all the necessary functionality to produce footprints cutouts.

The steps below is a summary of the *trace_footprint* functionality:


1) Instantiate an EdgeDetector object using the AstroData object as input.

2) The :ref:`edge_detector_data() <ed_data>` function sets up the :ref:`mdf <mdf>` dictionary containing instrument specific parameters.

3) The class method find_edges() produces a tuple (edges1_xy,edges2_xy) where edges1_xy is the list of (x_array,y_array) coordinates for all the left/bottom edges and edges2_xy is the list of all the corresponding right/top edges for each footprint. For more details please see :ref:`find_edges <find_edges>`.

4) Loop through the list of (x_array,y_array) tuples and instantiate one :ref:`Edge <edge_class>` object. Call the fitfunction() Edge method as well.

5) With the list of Edge objects for the (left/bottom) edges and the list of Edge objects for the (right/top) edges, instantiate as many a :ref:`Footprint class <foot_class>` objects as there are footprints represented by the (left/bottom),(right/top) Edge objects.

6) Instantiate a :ref:`FootprintTrace class <footpt_class>` using the list of Footprint objects as input calling the method :ref:`FootprintTrace.as_bintable <fp_asbintable>` to build the *TRACEFP* table within an output AstroData object.

The steps below is a summary of the *cut_footprints* functionality:

1) Open the FITS file containing the *TRACEFP* table extension.

2) If you have a target spectrum with footprints that matches the ones created by *trace_footprint* then copy the *TRACEFP* extension to the AstroData object with target spectrum.

3) Instantiate a :ref:`CutFootprints <cutfp_class>` object using the input AstroData object.

4) Execute the method :ref:`CutFootprints.cut_regions <cutl_regions>` to read the input image and table creating a list of :ref:`CutFootprint class <cutfp_class>` objects.

5) Execute the method :ref:`CutList.as_astrodata <cutl_astr>` to create as many image extension with one footprint as there are records in the table. Append each extension to the output AstroData object.


