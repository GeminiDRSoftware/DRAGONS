.. ex1_igrins2_pointsrc_dataset.rst

.. _igrins2_pointsrc_dataset:

********************************
Example 1 - Datasets description
********************************

Dithered Point Source Observation
=================================
In this example, we will reduce an IGRINS-2 observation of a CEMP-no star,
a carbon-enhanced metal-poor start that lack enhncements in neutron-capture
element.

The dither pattern is one standard ABBA sequence.

The calibrations we use for this example are:

* **Flats** taken a the end of the night as series of 20 lamp-on and 10
  lamp-off exposures.  The flats are used for the science, the zenith sky
  observation, and the telluric standard.
* **Zenith Sky for wavelength calibration** taken at the beginning of the
  night to serve as a wavelength calibration reference.  The sky lines will
  be used to calculate the wavelength solution.
* **Telluric standard** observation taken just after the science observation.
  This observation will also be used to refine the wavelength solution using
  the telluric absorption lines.

Here is the breakdown of the files.  All the files are included in the tutorial data
package.  They can also be downloaded from the Gemini Observatory Archive (GOA).

+--------------------------------+-----------------------------------+
| Science                        || N20260130S0179-182               |
+--------------------------------+-----------------------------------+
| Flats                          || N20260130S0369,373,377,382,385,  |
|                                || 388,391,396,401,407-427          |
+--------------------------------+-----------------------------------+
| Sky for wavelength calibration || N20260130S0021                   |
+--------------------------------+-----------------------------------+
| Telluric                       || N20260130S0183-186               |
+--------------------------------+-----------------------------------+


