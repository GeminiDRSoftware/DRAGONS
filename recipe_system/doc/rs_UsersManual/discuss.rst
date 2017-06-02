.. discuss:
.. include install

Discussion
==========
.. _fitsstore:

Fits Storage
------------

The URLs that appear in ``test_one`` recipe example (Sec. :ref:`test`), reference 
web services available within the Gemini Observatory's operational environment. 
They will `not` be available directly to users running ``reduce`` outside of the 
Gemini Observatory environment.

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
provide the calibration management and association features of :ref:`fitsstore`: 
for use with the public release of the `gemini_python` data reduction package. 
This feature will provide automatic calibration selection for both pipeline 
(recipe) operations and in an interactive processing environment.
