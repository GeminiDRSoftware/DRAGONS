.. faq:

**************************
Frequently Asked Questions
**************************

**1. What can the QAP do?**

   In a nutshell, it can process GMOS Imaging data and measure QA metrics.  For complete list of
   what it can do, see :ref:`features`.

**2. Where does the QAP run?**

   The software can be run from any Linux or Mac desktop properly configured.
   However, for night operations use mkopipe1 or hbfpipe1. 

**3. How do I start the QAP?**

   From a ``telops`` session, set up the QAP environment with ``startqap`` or
   use the QAP item in the yellow G menu.  Then for each new dataset, type ``autoredux``
   to launch the automatic polling and reduction, and point your brownser to the
   GUI (for the summit pipeline: http://mkopipe1:8777/qap/nighttime_metrics.html).  
   For more information, see :ref:`operate`.

**4. Do I have to start the QAP?**

   To run in automatic mode, a process called ``adcc`` must be running in the background
   and ``autoredux`` must have been launched in the shell.  To view the QA metrics a browser
   must be pointed to the Nighttime QA Metrics GUI.  Also, a ``ds9`` must be launched
   and the QAP must run in a specific directory.
   
   The short story.
   
   Use ``startqap`` or the QAP item in the yellow G menu.  Don't worry about the ``adcc``, 
   type ``autoredux`` in the pipeline's ``xterm``, and point your browser to the GUI.
   
   The long story.
   
   The ``adcc`` is the control center and handles communications between the various 
   components of the pipeline, and it serves the web-based GUI.  The ``adcc`` is launched
   transparently to the telops user when ``startqap`` or the QAP item in the yellow G menu
   is used.  You shouldn't have to worry about the ``adcc``.  But if things go belly up,
   that could be a sign that the ``adcc`` is either in a bad state or dead.  If that happens
   on your watch, call your local pipeline contact.  If you feel brave, check the 
   :ref:`troubleshoot` section.
   
   What you do need to start is ``autoredux``.  This command will start the polling of
   the DHS directory and, for each acquisition and science dataset, it will launch a
   reduction of that dataset.  The QA metrics will ultimately be sent to the GUI.
   
   In order to monitor the QA metrics, you will need to point a browser to 
   the Nighttime QA Metrics GUI served by the ``adcc``.  
   The URL is typically http://mkopipe1:8777/qap/nighttime_metrics.html
   
   The ``ds9`` and the pipeline's ``xterm`` will be launched when you to ``startqap`` or
   select the QAP item in the yellow G menu.  You will also be automatically taken the
   the QAP data reduction directory (``/pipestore``)
   
   For step by step instructions, see :ref:`operate`.

**5. How do I operate the QAP?**

   Once set up, just watch the GUI.

**6. How to process the next dataset?**

   Once set up, the pipeline will automatically detect the next dataset and process it.  
   Just sit back, relax, and watch the GUI.

**7. How to process the previous dataset?**

   Why would you want to do that?  Okay, you might have a valid reason, but I'm afraid
   that for the time being, in automatic mode you cannot go back and reprocess the data
   while ``autoredux`` is running.

**8. Is the QAP interactive?**

   Nope.  PyRAF should be used for interactivity.

