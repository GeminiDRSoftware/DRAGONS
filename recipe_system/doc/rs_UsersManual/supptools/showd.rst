
showd
-----

This command list the values stored in a given descriptor for the input data.
Its basic usage can be printed using the following command:::

    $ showd --help
    usage: showd [-h] --descriptors DESCRIPTORS [--debug] [inputs [inputs ...]]

    For each input file, show the value of the specified descriptors.

    positional arguments:
      inputs                Input FITS files

    optional arguments:
      -h, --help            show this help message and exit
      --descriptors DESCRIPTORS, -d DESCRIPTORS
                            comma-separated list of descriptor values to return
      --debug               Toggle debug mode

One or more descriptors can be printed together. Here is an example:::

    $ showd -d object,exposure_time *.fits
    filename:   object   exposure_time
    ------------------------------
    S20150609S0022.fits: Dark 150.0
    S20150609S0023.fits: Dark 150.0
    ...
    S20170504S0031.fits: Domeflat 7.0
    S20170504S0032.fits: Domeflat 7.0
    S20170504S0033.fits: Domeflat 7.0
    ...
    S20170503S0115.fits: HP 1 - control field 30.0
    S20170503S0116.fits: HP 1 - control field 30.0
    ...

You can use dataselect_ together with showd_ if you want to print
the descriptors values in a data subset:::

    $ dataselect raw/*.fits --tag FLAT | showd -d object,exposure_time
    filename:   object   exposure_time
    ------------------------------
    S20170504S0027.fits: Domeflat 7.0
    S20170504S0028.fits: Domeflat 7.0
    ...
    S20171208S0066.fits: Domeflat 7.0
    S20171208S0067.fits: Domeflat 7.0

The Pipe `` | `` gets the ``dataselect`` output and passes it to ``showd``.

.. todo::
    The ``showd`` now uses space to separate columns. This may be a problem
    to the user if they want to use a table and any value also contains spaces
    within itself. A possible solution would be replacing the space to separate
    columns by comma.
