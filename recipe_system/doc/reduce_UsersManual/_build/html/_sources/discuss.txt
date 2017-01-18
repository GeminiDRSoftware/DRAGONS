.. discuss:
.. include supptools
.. include userenv

Discussion
==========
.. _fitsstore:

Fits Storage
------------

The URLs that appear in ``test_one`` recipe example (Sec. :ref:`test`), reference web services 
available within the Gemini Observatory's operational environment. They will 
`not` be available directly to users running ``reduce`` outside of the Gemini 
Observatory environment.

In the context of ``reduce`` and the Astrodata Recipe System, FitsStorage provides 
a calibration management and association feature. Essentially, given a science 
frame (or any frame that requires calibration) and a calibration 
type requested, FitsStorage is able to automatically choose the best available 
calibration of the required type to apply to the science frame. The Recipe System
uses a machine-oriented calibration manager interface in order to select 
calibration frames to apply as part of pipeline processing.

Though this service is not currently available to general gemini_python users,
plans to provide this as a local calibration service are in place and expected
for :ref:`future`. 

.. _adcc:

The adcc
--------

As a matter of operations, ``reduce`` and the Recipe System depend upon the 
services of what is called the ``adcc``, the Automated Data Communication Center.
The ``adcc`` provides services to pipeline operations through two proxy servers,
an XML-RPC server and an HTTP server. The XML_RPC server serves calibration 
requests made on it, and retrieves calibrations that satisfiy those requests 
from the Gemini FITS Store, a service that provides automated calibration
lookup and retrieval.

The ``adcc`` can be run externally and will run continuously until it is 
shutdown. Any instances of ``reduce`` (and the Recipe System) will employ 
this external instance of the ``adcc`` to service a pipeline's calibration 
requests. However, a user of ``reduce`` need not start an instance of the
``adcc`` nor, indeed, know anytihng about the ``adcc`` `per se`. If one is not
available, an instance of the ``adcc`` will be started by ``reduce`` itself,
and will serve that particular ``reduce`` process and then terminate.

This note is provided should users notice an ``adcc`` process and wonder what 
it is.

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
`not` require the user to determine the AstroDataTypes of datasets.


Local Calibration Service
+++++++++++++++++++++++++
The Fits Storage service will be delivered as part of a future release and will
provide the calibration management and association features of :ref:`fitsstore`: for
use with the public release of the `gemini_python` data reduction package. This 
feature will provide automatic calibration selection for both pipeline (recipe)
operations and in an interactive processing environment.
