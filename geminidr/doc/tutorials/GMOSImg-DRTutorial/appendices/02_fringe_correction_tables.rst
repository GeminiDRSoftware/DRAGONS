.. 02_fringe_correction_tables.rst

.. _fringe_correction_tables:

************************
Fringe Correction Tables
************************

Here you will find what are the detector-filter combinations that requires a
Processed Fringe Frame for the data reduction. Below are one table for
GMOS-N and one table for GMOS-S. Each row of these tables corresponds to one 
of the detectors used in the instrument during its life-time. The five columns
in the right contains
the broadband filters used in imaging mode. The intersection of the 
detector rows with the filter columns contains the following
information:

- `Yes`: Requires a Processed Fringe Frame for data reduction;

- `No`: Does not require a Processed Fringe Frame for data reduction;

- `---`: Does not have data with these filters.


.. table:: GMOS-N Table for Detector/Filter Configurations that require
    Processed Fringe Frame for data reduction.  Filters bluer than i' do
    not require fringe correction.

    +----------------+-------------------------+-----+-----+-----+-----+-----+
    | GMOS-N         |                         | i'  | CaT | Z   | z'  | Y   |
    +================+=========================+=====+=====+=====+=====+=====+
    | EEV CCDs       | Aug 2001 - Nov 2011     | Yes | Yes | --- | Yes | --- |
    +----------------+-------------------------+-----+-----+-----+-----+-----+
    | E2V DD CCDs    | Nov 2011 - Feb 2017     | Yes | Yes | Yes | Yes | Yes |
    +----------------+-------------------------+-----+-----+-----+-----+-----+
    | Hamamatsu CCDs | February 2017 - Present | No  | No  | No  | Yes | Yes |
    +----------------+-------------------------+-----+-----+-----+-----+-----+

More: `GMOS-N Fringe Information <https://www.gemini.edu/sciops/instruments/gmos/imaging/fringing/gmosnorth>`_


.. table:: GMOS-S Table for Detector/Filter Configurations that require
    Processed Fringe Frame for data reduction. Filters bluer than i' do
    not require fringe correction.

    +----------------+---------------------------+-----+-----+-----+-----+-----+
    | GMOS-S         |                           | i'  | CaT | Z   | z'  | Y   |
    +================+===========================+=====+=====+=====+=====+=====+
    | EEV CCDs       | Commissioning - June 2014 | Yes | Yes | --- | Yes | --- |
    +----------------+---------------------------+-----+-----+-----+-----+-----+
    | Hamamatsu CCDs | June 2014 - Present       | No  | No  | No  | Yes | Yes |
    +----------------+---------------------------+-----+-----+-----+-----+-----+

More: `GMOS-S Fringe Information <https://www.gemini.edu/sciops/instruments/gmos/imaging/fringing/gmossouth>`_
