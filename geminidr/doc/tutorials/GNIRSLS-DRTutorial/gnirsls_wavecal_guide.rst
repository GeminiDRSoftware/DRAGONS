.. gnirsls_wavecal_guide.rst

.. include:: symbols.txt

.. |br| raw:: html

   <br />

.. _gnirsls_wavecal_guide:

*****************************************
Wavelength Calibration Guide for GNIRS LS
*****************************************

This is a quick reference table to guide you towards the wavelength calibration
scenario that most likely applies based on the instrument configuration.

Below the table, we discuss the various ways to do the wavelength solution
calculation.

+-------------+------------+--------------+-------------------------------------------------------------------------------+
| Bandpass    | Grating    | Central |br| | Advise                                                                        |
|             |            | Wavelength   |                                                                               |
+-------------+------------+--------------+-------------------------------------------------------------------------------+
| X-band |br| | 10/mm |br| |   Any        | **most likely** The arcs will normally have enough lines and |br|             |
| J-band |br| | 32/mm |br| |              | reasonable coverage. |br| |br|                                                |
| H-band      |            |              | **possible improvement** If the OH and O\ :sub:`2`\  emission lines are |br|  |
|             |            |              | visible, they can be used instead of the arc. |br| |br|                       |
|             |            |              | **unlikely** If emission lines are not visible (short exposures), |br|        |
|             |            |              | and the arc is not sufficient, the telluric absorption lines  |br|            |
|             |            |              | can be used.                                                                  |
|             +------------+--------------+-------------------------------------------------------------------------------+
|             | 111/mm     |   Any        | **most likely** The arc lines will be very few, 1 to 5, and will |br|         |
|             |            |              | not offer a good coverage of the spectral range.  OH and O\ :sub:`2`\  |br|   |
|             |            |              | lines should be visible, use them. |br| |br|                                  |
|             |            |              | **less likely** If the exposure times are short, the OH and O\ :sub:`2`\ |br| |
|             |            |              | lines might not be visible, the only solution will be to use the |br|         |
|             |            |              | telluric absorption features.                                                 |
+-------------+------------+--------------+-------------------------------------------------------------------------------+
| K-band      | 10/mm |br| |   Any        | **most likely** The arcs will normally have enough lines and |br|             |
|             | 32/mm |br| |              | reasonable coverage. |br| |br|                                                |
|             |            |              | **possible improvement** If the OH and O\ :sub:`2`\  emission lines are |br|  |
|             |            |              | visible, they can be used instead of the arc. |br| |br|                       |
|             |            |              | **unlikely** If emission lines are not visible (short exposures), |br|        |
|             |            |              | and the arc is not sufficient, the telluric absorption lines  |br|            |
|             |            |              | can be used.                                                                  |
|             +------------+--------------+-------------------------------------------------------------------------------+
|             | 111/mm     |  < 2.3 |um|  | **most likely** The arc lines will be very few, 1 to 5, and will |br|         |
|             |            |              | not offer a good coverage of the spectral range.  OH and O\ :sub:`2`\  |br|   |
|             |            |              | lines should be visible, use them. |br| |br|                                  |
|             |            |              | **less likely** If the exposure times are short, the OH and O\ :sub:`2`\ |br| |
|             |            |              | lines might not be visible, the only solution will be to use the |br|         |
|             |            |              | telluric absorption features.                                                 |
|             |            +--------------+-------------------------------------------------------------------------------+
|             |            |  > 2.3 |um|  | **most likely** The arc lines will be very few, 1 to 5, and will |br|         |
|             |            |              | not offer a good coverage of the spectral range. There will be |br|           |
|             |            |              | no OH or O\ :sub:`2`\  lines available. The telluric absorption features |br| |
|             |            |              | **must** be used.                                                             |
+-------------+------------+--------------+-------------------------------------------------------------------------------+
| L-band |br| | Any        |  Any         | No arc lamp observations are taken for the thermal bands. |br| |br|           |
| M-band      |            |              | The emission features in the sky spectrum must be used |br|                   |
|             |            |              | for wavelength calibration.  Note that for L-band > 3.8 |um|, |br|            |
|             |            |              | the automatic line identification is often wrong and the use |br|             |
|             |            |              | of the interactive mode is required.                                          |
+-------------+------------+--------------+-------------------------------------------------------------------------------+




Usage
=====

Need for processed flat
-----------------------
While optional it is highly recommended to use processed flat when producing
a wavelength solution.  When using an arc, the flat needs to be passed
manually (there are no calibration association rules yet between the arcs
and the flats).  When using the sky lines, emission or absorption, the flat will be
retrieved automatically.

The processed flat stores a mask that defines the illuminated region.  That's
the information that helps calculating the wavelength solution.

From the Arc Lamp
-----------------
Producing a wavelength solution from the arc observations is fairly
straightforward.  Just call reduce on the raw arcs.

The use of the interactive mode is recommended to verify the solution and
ensure that the lines offer a good coverage the entire spectral range.

::

  reduce @arcs.lis -p interactive=True

From the Emission Lines
-----------------------
When OH and O\ :sub:`2`\  lines are present in the science data, or, in the thermal bands,
when emission features in the sky spectrum are present, it is possible to use those
to calculate the wavelength solution.

This is done by running reduce on the science frames and specifying the use
of the ``makeWavecalFromSkyEmission`` recipe.  The interactive mode is
recommended to verify and ensure correct line identification.

::

  reduce @sci.lis -r makeWavecalFromSkyEmission -p interactive=True

If the OH and O\ :sub:`2`\  sky lines are bright, it is possible to use only one science
observation instead of the stack.  This could lead to higher precision as the
stack might make the lines a touch thicker.


From the Absorption Lines
-------------------------
When the arc lamp offers very few lines, or poor coverage (eg. all the lines at
one end of the spectrum), and there are no emission lines, one has to resort
to using the telluric absorption features to measure the wavelength solution.

This is done by running reduce on the science frames and specifying the use
of the ``makeWavecalFromSkyAbsorption`` recipe.

This recipe requires a solution from the arc lamp to serve as initial
condition.  So in this case, there are two steps to the process.

::

  reduce @arcs.lis -p interactive=True
  caldb remove N20210407S0181_arc.fits

  reduce @sci.lis -r makeWavecalFromSkyAbsorption --user_cal processed_arc:N20210407S0181_arc.fits -p interactive=True

The solution from the arc is really poor and used only to help calculate the
solution from the telluric absorption.  Therefore it is removed from the
calibration manager to ensure that no science reduction ever uses it.

The interactive mode is recommended to verify and ensure correct line
identification.