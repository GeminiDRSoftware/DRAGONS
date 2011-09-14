.. features:

*********************************************
Features and Capabilities Currently Available
*********************************************

The list available features and capabilities will continuously grow with each
iterative deployment.  Here we describe the features and capabilities as of
September 2011.

Core QAP Infrastructure
=======================
**Recipe System**
   This is the core of the automation.

**Calibration Service**
   This is served by FitsStorage.  The Recipe System makes calibration requests to FitsStorage.

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
      * Current (Sep 2011) E2V CCDs


Supported Metrics Measurements
==============================
**Automatic measurement of image quality metrics to determine IQ band**
   This includes **seeing** and **PSF ellipticity**.
   
   This is available for:

   * GMOS Imaging

Notable features NOT INCLUDED yet
=================================
* Automatic polling for new data
* Automatic launching of a reduction upon arrival of new data
* Graphical User Interfaces
* Automatic measurement of zero-points to determine CC
* Automatic measurement of sky background to determine BG
* New GMOS North CCDs, E2Vs or Hamamatsu
