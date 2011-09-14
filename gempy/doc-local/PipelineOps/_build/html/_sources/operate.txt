.. operate:

****************************************************
How to operate the Gemini Quality Assurance Pipeline
****************************************************

Nighttime Operations
====================

For the impatient: ::

   any-terminal-on-mkocon: telops>  startqap

   new-xterm-from-mkopipe1: pipeops>  redux 35
   new-xterm-from-mkopipe1: pipeops>  redux 36
   ...

That's it. 

Now let's talk about it.

At night, the QA pipeline runs on the dedicated pipeline computer.  This 
helps avoid resource conflicts between the pipeline and the rest of the observing software.
The summit machine is named ``mkopipe1``.  As a backup, the base facility machine is
named ``hbfpipe1``.   The pipeline runs on the pipeline computer but displays on the ``telops``
monitors

There is also a dedicated pipeline operator's account, ``pipeops``.

At this time, the pipeline does not detect arrival of new datasets.  Therefore the
operator must manually launch the reduction of the new dataset.  The command is
``redux``, followed by the image number.

To run the pipeline, log in and get ready:

   1. Log into the ``mkocon`` computer with the telops account as usual.
   2. Type ``startqap``.

   This needs to be done only once.  This will launch ``ds9`` and an ``xterm`` on the ``mkopipe1`` 
   machine, and display both on the ``telops`` monitors.  The session in the ``xterm`` starts 
   in ``/pipestore`` directory automatically; this is your work ``xterm`` and ``directory``.

From now on, when a dataset comes in:

   3. In the ``xterm``, type ``redux <img number>`` , gacq-style

   The reduction will be launched by the ``redux`` command.  The reduced, new dataset will be 
   measured and displayed. Compatible datasets previously reduced with the pipeline, and 
   available in ``/pipestore``, will be combined with the current dataset and the deep frame 
   will be measured and displayed.

   No interactivity is available at this time.

Repeat for next image.


Daytime Operations
==================

The pipeline can be used for reducing data in general.  For example, it can be used
to create processed biased and processed pre-imaging data.  At the moment though,
there are memory issues that prevents the stacking of very large numbers of datasets 
at once like it is needed to make the GN processed biases.

If SOS-DA's and staff in general wish to use the pipeline for something other than
nighttime operations, please check with the Data Processing Software Group first to
confirm what can be done and what cannot.

