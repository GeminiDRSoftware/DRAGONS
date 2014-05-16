.. logging:

***********
Log Utility
***********

The Astrodata Log Utility
=========================
Astrodata uses a logging utility based on the Python logging facility. 

.. todo::
   write the astrodata log utility section.

.. note::
   ``astrodata/adutils/logutils.py``
   creates logfile, default name reduce.log

   config() method
      mode: standard, stream, null, debug
      consolve_lvl: controls the console logging level
      file_name: logfile name (default=reduce.log)
      stomp:  clobber

   get_logger()

   update_indent(): control indenting during recipe/primitive execution.

   logger mode standard: default console-> stdinfo, default to file-> fullinfo

Writing to Log
==============

.. todo::
   write the section about writing to log

.. note::
   Using the logging facility involves *getting* the logger, *configuring* the logger,

   ``log.<loglevel>(<message_to_log>)``

   default name: reduce.log

::

   from astrodata.adutils import logutils
  
   log = logutils.get_logger(__name__)

   ??
   log = self.log
   self.log.stdinfo()

In primitives, call logger "once at the top"


Log Levels
==========
Several log levels are supported, some are directly from the Python logging facility,
others are defined in Astrodata.  Here are definitions of the log levels and usage
examples.

critical
   A serious error, indicating that the program itself may be unable to 
   continue running. For example:

   .. code-block:: python
    
       try:
           ...
       except:
           # Log the message from the exception
           log.critical(repr(sys.exc_info()[1]))
  
       # or simply
       ...
       log.critical("Something really bad happened.  Exiting now.")
   
error
   Due to a serious problem, the software has not been able to perform some function.
   The error does not necessarily prevent the program from continuing.
   
   .. code-block:: python
   
       log.error('An error occurred while trying to calculate the \
                  nbiascontam, using default value = 4')
   
warning
   An indication that something unexpected happened, or indicative of some problem 
   in the near future. The software is still working as expected, but might be using
   some default or recovery settings.
   
   .. code-block:: python
   
       log.warning("A [DQ,%d] extension already exists in %s" %
                   (extver, ad.filename))
   
status
   Start and end processing information, number of files, name of the input or output 
   files.  In other words, "What's happening? What's being processed?"
   
   .. code-block:: python
   
       log.status("List for stack id=%s" % sid)
       if len(stacklist) > 0:
           for f in stacklist:
               log.status("    %s" % os.path.basename(f))
       else:
           log.status("No datasets in list")

stdinfo
   Scientific information like seeing measurements, statistics, etc. or what
   is scientifically being done to the data.  This is information that an
   astronomer might want to see displayed on the screen.
   
   .. code-block:: python
   
       log.stdinfo("Adding the read noise component of the variance")
       log.stdinfo("RA: %.2f +- %.2f    Dec: %.2f +- %.2f   arcsec" % 
                    (ra_mean, ra_sigma, dec_mean, dec_sigma))
   
info
   Confirmation that things are working as expected.  The information here is
   more programmatical than scientific.
   
fullinfo
   Detailed information on the processing, like input parameters, header 
   changes.  Useful information for a log file but not necessary for standard
   output (screen output).
   
   .. code-block:: python
   
       log.fullinfo("Tiling extensions together to get statistics from CCD2")
       log.fullinfo("Using data section [%i:%i,%i:%i] from CCD2 for statistics" %
                     (xborder,sci_data.shape[1]-xborder,
                      yborder,sci_data.shape[0]-yborder))
                          
debug
   Very detailed engineering information for used in debugging.
   For example:
   
   .. code-block:: python
       
       ...
       log.debug("SplotETI __init__")
       ...
       log.debug("SplotETI.execute()")
       ...
       log.debug("SplotETI.run()")
       ...
       log.debug("SplotETI.recover()")
       ...
   
