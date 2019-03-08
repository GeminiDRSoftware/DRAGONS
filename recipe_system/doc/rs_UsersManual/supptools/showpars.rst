.. _showpars:

showpars
--------

Inheritance and class overrides within the primitive and parameter hierarchies
means that one cannot simply look at any given primitive function and its
parameters and extrapolate those to all such named primitives and parameters.
Primitives and their parameters are tied to the particular classes designed for
those datasets identified as a particular kind of data.

The ``showpars`` application is a simple command line utility allowing users
to see the available parameters and defaults for a particular primitive
function applicable to a given dataset. Since the applicable primitives
for a dataset are dependent upon the `tagset` of the identified dataset
(i.e. ``NIRI IMAGE`` , ``F2 SPECT`` , ``GMOS BIAS``, etc.), which is
to say, the `kind` of data we are looking at, the parameters available on a
named primitive function can vary across data types, as can the primitive function
itself. For example, F2 IMAGE ``stackFlats`` uses the generic implementation of
the function, while GMOS IMAGE ``stackFlats`` overrides that generic method.

We examine the help on the command line of showpars::

 $ showpars.py -h
 usage: showpars.py [-h] [-v] filen primn

 Primitive parameter display, v2.0.0 (beta)

 positional arguments:
   filen          filename
   primn          primitive name

 optional arguments:
   -h, --help     show this help message and exit
   -v, --version  show program's version number and exit

Two arguments are requiered: the dataset filename, and the primitive name of
interest. As readers will note, ``showpars`` provides a wealth of information
about the available parameters on the specified primitive, including allowable
values or ranges of values::

    $ showpars FR39441/S20180516S0237.fits stackFlats
    Dataset tagged as set(['RAW', 'GMOS', 'GEMINI', 'SIDEREAL', 'FLAT', 'UNPREPARED',
    'IMAGE', 'CAL', 'TWILIGHT', 'SOUTH'])

    Settable parameters on 'stackFlats':
    ========================================
    Name			Current setting

    suffix               '_stack'             Filename suffix
    apply_dq             True                 Use DQ to mask bad pixels?
    separate_ext         False                Handle extensions separately?
    statsec              None                 Section for statistics
    operation            'mean'               Averaging operation
    Allowed values:
      wtmean	variance-weighted mean
      mean	arithmetic mean
      median	median
      lmedian	low-median

    reject_method        'minmax'             Pixel rejection method
    Allowed values:
        minmax	reject highest and lowest pixels
    none	no rejection
    varclip	reject pixels based on variance array
    sigclip	reject pixels based on scatter

    hsigma               3.0                  High rejection threshold (sigma)
    Valid Range = [0,inf)
    lsigma               3.0                  Low rejection threshold (sigma)
        Valid Range = [0,inf)
    mclip                True                 Use median for sigma-clipping?
    max_iters            None                 Maximum number of clipping iterations
        Valid Range = [1,inf)
    nlow                 1                    Number of low pixels to reject
        Valid Range = [0,inf)
    nhigh                1                    Number of high pixels to reject
        Valid Range = [0,inf)
    memory               None                 Memory available for stacking (GB)
        Valid Range = [0.1,inf)
    scale                False                Scale images to the same intensity?

With this information, users can now confidently adjust parameters for
particular primitive functions. As we have seen already, this can be done
easily from the `reduce` command line. Building on material covered in this
manual, and continuing our example from above::

  $ reduce -p stackFlats:nhigh=3 <fitsfiles> [ <fitsfile>, ... ]

And the reduction proceeds. When the ``stackFlats`` primitive begins, the
new value for nhigh will be used.

.. note::
   ``showpars`` is not considered the final tool for users to examine and set
   parameters for dataset reduction. Plans are in the works to develop a more
   graphical tool to help users view and adjust parameters on primitive functions.
   But it does show users the important information: the parameters available
   on a primitive's interface, the current (default) settings of the named
   parameters, and allowed ranges of values where appropriate.

.. todo:: Update showpars to list the parameters related to a primitive that
belongs to a no-default recipe.