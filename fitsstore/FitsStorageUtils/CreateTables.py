"""
This module provides various utility functions for create_tables.py 
in the Fits Storage System.
"""
from FitsStorage import *


def create_tables(session):
  """
  Creates the database tables and grants the apache user
  SELECT on the appropriate ones
  """
  # Create the tables
  File.metadata.create_all(bind=pg_db)

  if fsc_localmode == False:
      # Now grant the apache user select on them for the www queries
      session.execute("GRANT SELECT ON file, diskfile, diskfilereport, header, fulltextheader, gmos, niri, michelle, gnirs, nifs, tape, tape_id_seq, tapewrite, taperead, tapefile, notification TO apache");
      session.execute("GRANT INSERT,UPDATE ON tape, tape_id_seq, notification, notification_id_seq TO apache");
      session.execute("GRANT DELETE ON notification TO apache");
  session.commit()

def drop_tables(session):
  """
  Drops all the database tables. Very unsubtle. Use with caution
  """
  session.execute("DROP TABLE gmos, niri, nifs, gnirs, michelle, header, fulltextheader, diskfile, diskfilereport, file, tape, tapewrite, taperead, tapefile, notification, standards, ingestqueue CASCADE")
  session.commit()
