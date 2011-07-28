"""
This module provides various utility functions for add_to_ingest_queue.py, 
inotify_ingest_queue.py and local_add_to_ingest_quueue.py in the Fits Storage 
System.
"""
from FitsStorage import *
from FitsStorageLogger import logger
from sqlalchemy.orm.exc import ObjectDeletedError


def addto_ingestqueue(session, filename, path):
  """
  Adds a file to the ingest queue
  """
  iq = IngestQueue(filename, path)
  session.add(iq)
  session.commit()
  try:
    logger.debug("Added id %d for filename %s to ingestqueue" % (iq.id, iq.filename))
    return iq.id
  except ObjectDeletedError:
    logger.debug("Added filename %s to ingestqueue which was immediately deleted" % filename)
