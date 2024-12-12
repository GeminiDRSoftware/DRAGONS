.. discuss:
.. include install

In the works
============
.. _fitsstore:

Fits Storage
------------
In the context of the DRAGONS Recipe System and ``reduce``, FitsStorage provides
a calibration management and association feature. When given a science frame (or
any frame that requires calibration) and a calibration type requested, FitsStorage
is able to automatically choose the best available calibration of the required type
to apply to the science frame. The Recipe System uses a machine-oriented calibration
manager interface in order to select calibration frames to apply as part of pipeline
processing.

The URLs that appear in ``test_one`` recipe example (Sec. :ref:`test`), reference
web services available within the Gemini Observatory's operational environment.
This specific webservice will `not` be available directly to users running
``reduce`` outside of the Gemini Observatory environment. Users external to the
Gemini firewall will be able to use a `local calibration manager` service, which is
a "stand alone" version of the FitsStorage calibration manager.

Local Calibration Service
+++++++++++++++++++++++++
A local 'fitsstore' service will be delivered as part of future Recipe System
releases and will provide the calibration management and association features of
:ref:`fitsstore`: for use with the public release of the DRAGONS data reduction
package. This feature will provide automatic calibration selection for both pipeline
(recipe) operations and in an interactive processing environment.

.. _future:

Future Enhancements
-------------------

Intelligence
++++++++++++
One enhancement long imagined is what has been generally termed 'intelligence'.
That is, an ability for either ``reduce`` or some utility to automatically do
AstroDataType classification of a set of data, group them appropriately, and
then pass these grouped data to the Recipe System.

As things stand now, it is up to the user to pass commonly typed data to
``reduce``. As shown in the previous section, :ref:`typewalk`, ``typewalk``
can help a user perform this task and create a 'ready-to-run' @file that can
be passed directly to ``reduce``. Properly implemented 'intelligence' will
`not` require the user to determine the AstroData tags of datasets.
