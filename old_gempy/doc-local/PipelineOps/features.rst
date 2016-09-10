.. features:

.. _features:

*********************************************
Features and Capabilities Currently Available
*********************************************

The list available features and capabilities will continuously grow with each
iterative deployment.  Here we describe the features and capabilities as of
September 2013.

Core QAP Infrastructure
=======================
**Recipe System**
   This is the core of the automation.

**Calibration Service**
   This is served by FitsStorage.  The Recipe System makes calibration requests to FitsStorage.

**Automatic Polling**
   The system can poll the DHS directory for new dataset and automatically launch a reduction.

**Nighttime QA Metrics Reporting**
   QA Metrics are being reported in a GUI.  The GUI also provides plots of metrics through the night.

**QA Metrics Database**
   QA Metrics are stored in FitsStorage for permanent records.  Staff are able to access the database at any time.

Supported Instrument and Mode
=============================
**GMOS Imaging**
   
   * Processing of **acquisition** images
   * Processing of **science** observations
   * Processing of **biases**, to create processed biases
   * Processing of imaging **flats**, to create processed flats
   * Processing of science observations, to create processed **fringe frames**
   * Variance and DQ planes propagation
   * Display of current image and stacked image, if available (ds9)
   * Calibration request supported:
      * Retrieval of processed bias matching overscan and trim status
      * Retrieval of processed imaging flats
      * Retrieval of processed fringe frame
   * Data reduction steps supported:
      * Overscan correction
      * Trim overscan section
      * Bias correction
      * Flat correction
      * Fringe correction
      * Detection of point sources
      * Registration and alignment of images from same observation
      * Stacking of images from same observation
   * CCDs support
      * Current (Aug 2012) EEV CCDs and E2V CCDs
   * Note that stacking has been removed from the standard recipe for the time
     being.  The CL task used to stack is too slow.  Stacking will be re-enable
     once a more performant Python or C/C++ version is available.  (No plans yet.)


Supported Metrics Measurements
==============================
**Automatic measurement of image quality metrics to determine IQ band**
   This includes **seeing** and **PSF ellipticity**.

   Available for:   

   * GMOS Imaging

**Automatic astrometry**
   This is available for:

   * GMOS Imaging

**Automatic measurement of cloud cover conditions to determine the CC band**
   This includes **zero-points** and **extinction**
   
   This is available for:

   * GMOS Imaging

**Automatic measurement of sky background to determine the BG band**
   This is available for:

   * GMOS Imaging

**Automatic storage of QA Metrics to database**
   This is available for:
   
   * GMOS Imaging
   
   
Notable features NOT INCLUDED yet
=================================
* GMOS Hamamatsu CCDs
