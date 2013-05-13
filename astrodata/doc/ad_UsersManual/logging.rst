.. logging:

***********
Log Utility
***********

The Astrodata Log Utility
=========================
astrodata/adutils/logutils.py
creates logfile, default name reduce.log

config() method
   mode: standard, stream, null, debug
   consolve_lvl: controls the console logging level
   file_name: logfile name (default=reduce.log)
   stomp:  clobber

get_logger()

update_indent(): control indenting during recipe/primitive execution.


Writing to Log
==============

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
critical
error
warning
status
stdinfo
info
fullinfo
debug

logger mode standard: default console-> stdinfo, default to file-> fullinfo