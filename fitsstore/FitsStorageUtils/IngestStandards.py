"""
This module provides various utility functions for ingest_standards.py
in the Fits Storage System.
"""
from FitsStorage import *


def ingest_standards(session, filename):
  """
  Load the standards text file into the Standards table
  """
  # open the standards text file
  f = open(filename, 'r')
  list = f.readlines()
  f.close()

  # Loop through entries, adding to table
  for line in list:
    if(line[0]!='#'):
      fields = line.strip().split(',')

      # Create and populate a standard instance
      std = PhotStandard()
      try:
        std.name = fields[0]
        std.field = fields[1]
        std.ra = 15.0*float(fields[2])
        std.dec = float(fields[3])
        if(fields[4]!='None'):
          std.u_mag = float(fields[4])
        if(fields[5]!='None'):
          std.v_mag = float(fields[5])
        if(fields[6]!='None'):
          std.g_mag = float(fields[6])
        if(fields[7]!='None'):
          std.r_mag = float(fields[7])
        if(fields[8]!='None'):
          std.i_mag = float(fields[8])
        if(fields[9]!='None'):
          std.z_mag = float(fields[9])
        if(fields[10]!='None'):
          std.y_mag = float(fields[10])
        if(fields[11]!='None'):
          std.j_mag = float(fields[11])
        if(fields[12]!='None'):
          std.h_mag = float(fields[12])
        if(fields[13]!='None'):
          std.k_mag = float(fields[13])
        if(fields[14]!='None'):
          std.lprime_mag = float(fields[14])
        if(fields[15]!='None'):
          std.m_mag = float(fields[15])
      except ValueError:
        print "Fields: %s" % str(fields)
        raise
    
      # Add to database session
      session.add(std)
      session.commit()
