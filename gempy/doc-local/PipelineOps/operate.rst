.. operate:

.. _operate:

****************************************************
How to operate the Gemini Quality Assurance Pipeline
****************************************************

Nighttime Operations
====================

Automatic Operation
-------------------
For the impatient: ::

   any-terminal-on-mkocon: telops>  startqap

   new-xterm-from-mkopipe1: pipeops>  autoredux
   
   Point browser to: http://mkopipe1:8777/qap/nighttime_metrics.html

That's it. 

At CPO, the equivalent of the above is: ::

   any-terminal-on-cpocon:  telops>  startqap
   
   Point browser to:  http://cpopipe01:8777/qap/nighttime_metrics.html

(Oct 2013: autoredux now starts automatically in the pipeline xterm at GS)

Now let's talk about it.

At night, the QA pipeline runs on the dedicated pipeline computer.  This 
helps avoid resource conflicts between the pipeline and the rest of the observing software.
The summit machine is named ``mkopipe1``.  As a backup, the base facility machine is
named ``hbfpipe1``.   The pipeline runs on the pipeline computer but displays on the ``telops``
monitors

There is also a dedicated pipeline operator's account, ``pipeops``.

To launch the automatic polling and reduction, the command is ``autoredux``.

To run the pipeline, log in and get ready:

   1. Log into the ``mkocon`` computer with the telops account as usual.
   2. Type ``startqap``, or use the QAP item in the yellow G menu.

   This needs to be done only once.  This will launch ``ds9`` and an ``xterm`` on the ``mkopipe1`` 
   machine, and display both on the ``telops`` monitors.  The session in the ``xterm`` starts 
   in ``/pipestore`` directory automatically; this is your work ``xterm`` and ``directory``.
   
   Now you are ready to start the automatic polling and reduction:
   
   3. In the ``xterm``, type ``autoredux``.
   
   A new reduction will be launched by the ``autoredux`` process every time a new datasets shows
   up in the DHS directory.  The reduced, new dataset will be measured and displayed. 
   
   [*Stacking has been disabled until a more performant Python or C/C++ routine becomes available.*] 
   If stacking is enable compatible datasets previously reduced with the pipeline, and available in 
   ``/pipestore``, will be combined with the current dataset and the deep frame will be measured 
   and displayed.

   No interactivity with the image in ``ds9`` is possible at this time.
   
   To monitor the QA metrics and the reduction status, access the GUI.
   
   4. Point your browser to: http://mkopipe1:8777/qap/nighttime_metrics.html
   
   The Nighttime QA Metrics GUI is introduced in the :ref:`nighttimeGUI` section.


Manual Operation
----------------
Now that the automatic mode is available, the manual mode should be a lot less
appealing.  If, for some reason, you need to switch to manual mode, first kill
any ``autoredux`` process and **wait for the reduction in progress to complete**.
Only then will you be able to start a new reduction manually without getting 
into trouble.

So, for the impatient, and assuming that the QAP has already been started: ::

   - Kill autoredux, if running.
   - Wait for reduction in progress to complete (check the Reduction Status from the GUI).
   new-xterm-from-mkopipe1: pipeops>  redux 35
   new-xterm-from-mkopipe1: pipeops>  redux 36
   ...


To manually launch a reduction on a specific dataset, the command is
``redux``, followed by the image number.

**The pipeline should be running already**, and the pipeline's ``xterm`` and ``ds9`` should be up.
**If not**:

   1. Log into the ``mkocon`` computer with the telops account as usual.
   2. Type ``startqap``.

   Again do this **only if** the pipeline had **not** been launched already.  This needs to be done only once.  
   This will launch ``ds9`` and an ``xterm`` on the ``mkopipe1`` 
   machine, and display both on the ``telops`` monitors.  The session in the ``xterm`` starts 
   in ``/pipestore`` directory automatically; this is your work ``xterm`` and ``directory``.

Disable the automatic mode:

   3. If autoredux is running, kill it.
   4. If autoredux was running, open the GUI and monitor the reduction in progress.
   5. Only once the reduction in progress completes, can you move on to the next step.

From now on, when a dataset comes in:

   6. In the ``xterm``, type ``redux <img number>`` , gacq-style

   The reduction will be launched by the ``redux`` command.  The reduced, new dataset will be 
   measured and displayed. The QA metrics will scroll down the screen. 

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

