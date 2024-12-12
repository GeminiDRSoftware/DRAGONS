.. showd.rst

.. _showd:

showd
=====
The ``showd`` command line tool helps the user gather information about files
on disk.  The "d" in ``showd`` stands for "descriptor".  ``showd`` is used to
show the value of specific AstroData descriptors for the files requested.

Its basic usage can be printed using the following command::

    $ showd --help
    usage: showd [-h] --descriptors DESCRIPTORS [--long] [--csv] [--adpkg ADPKG] [--debug]
                 [inputs ...]

    For each input file, show the value of the specified descriptors.

    positional arguments:
      inputs                Input FITS files

    optional arguments:
      -h, --help            show this help message and exit
      --descriptors DESCRIPTORS, -d DESCRIPTORS
                            comma-separated list of descriptor values to return
      --long                Long format for the descriptor value
      --csv                 Format as CSV list.
      --adpkg ADPKG         Name of the astrodata instrument package to useif not
                            gemini_instruments
      --debug               Toggle debug mode
          --debug               Toggle debug mode


One or more descriptors can be printed together. Here is an example:::

    $ showd -d object,exposure_time *.fits
    ----------------------------------------------
    filename                object   exposure_time
    ----------------------------------------------
    N20160102S0275.fits    SN2014J          20.002
    N20160102S0276.fits    SN2014J          20.002
    N20160102S0277.fits    SN2014J          20.002
    N20160102S0278.fits    SN2014J          20.002
    N20160102S0279.fits    SN2014J          20.002
    N20160102S0295.fits      FS 17          10.005
    N20160102S0296.fits      FS 17          10.005
    N20160102S0297.fits      FS 17          10.005
    N20160102S0298.fits      FS 17          10.005
    N20160102S0299.fits      FS 17          10.005

Above is a human-readable table.  It is possible to return a comma-separated
list, CSV list, with the ``--csv`` tag::

    $ showd -d object,exposure_time *.fits --csv
    filename,object,exposure_time
    N20160102S0275.fits,SN2014J,20.002
    N20160102S0276.fits,SN2014J,20.002
    N20160102S0277.fits,SN2014J,20.002
    N20160102S0278.fits,SN2014J,20.002
    N20160102S0279.fits,SN2014J,20.002
    N20160102S0295.fits,FS 17,10.005
    N20160102S0296.fits,FS 17,10.005
    N20160102S0297.fits,FS 17,10.005
    N20160102S0298.fits,FS 17,10.005
    N20160102S0299.fits,FS 17,10.005

The ``showd`` command also integrates well with ``dataselect``. You can use
:ref:`dataselect` together with ``showd`` if you want to print
the descriptors values of a data subset::

    $ dataselect raw/*.fits --tag FLAT | showd -d object,exposure_time
    ----------------------------------------------
    filename                object   exposure_time
    ----------------------------------------------
    N20160102S0363.fits   GCALflat          42.001
    N20160102S0364.fits   GCALflat          42.001
    N20160102S0365.fits   GCALflat          42.001
    N20160102S0366.fits   GCALflat          42.001
    N20160102S0367.fits   GCALflat          42.001

The "pipe" `` | `` gets the ``dataselect`` output and passes it to ``showd``.
