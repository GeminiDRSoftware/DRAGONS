"""
This module contains the functions for curation_report.py that compare items in the tables 
Header and DiskFile.
"""
from FitsStorage import *
from sqlalchemy import *
from sqlalchemy.orm import aliased

checkonly = None
exclude = None

def duplicate_datalabels(session, checkonly, exclude):
  """
  Returns a list of df_ids as ids that were created to represent the join of DiskFile and 
  Header tables and identify which rows have duplicate datalabels.
  """
  # Form a connection with session
  conn = session.connection()
  # A select statement that joins the Header and DiskFile tables and reduces its size with filters
  s = """SELECT a.df_id 
                FROM (Diskfile JOIN Header ON DiskFile.id = Header.diskfile_id) AS a (df_id), 
                     (DiskFile JOIN Header ON DiskFile.id = Header.diskfile_id) AS b (df_id) 
                WHERE a.df_id != b.df_id AND
                      a.canonical = 'True' AND 
                      b.canonical = 'True' AND 
                      a.data_label = b.data_label"""
  if checkonly:
    s += """ AND a.data_label LIKE '%%%s%%'""" % (checkonly)
  if exclude:
    s += """ AND a.data_label NOT LIKE '%%%s%%'""" % (exclude)
  
  s += """ ORDER BY a.diskfile_id ASC"""

  # Execute the select statement
  result = conn.execute(text(s))   
  # Makes a list of all the joined table ids, df_id, in the list "duplicates"
  duplicates = []
  for row in result:
    duplicates.append(row['df_id'])
  return duplicates  
  

def duplicate_canonicals(session):
  """
  Returns a list of all the values in a row for any rows where there is more than one row 
  with the same file_id that has canonical = True.
  """
  # Make an alias of DiskFile
  diskfile_alias = aliased(DiskFile)
  # Self join DiskFile with its alias and compare their file_ids
  query = session.query(DiskFile).select_from(diskfile_alias).filter(DiskFile.id==diskfile_alias.id).filter(DiskFile.canonical==True).filter(diskfile_alias.canonical==True).filter(DiskFile.file_id==diskfile_alias.file_id).order_by(DiskFile.id)
  # Creates a list of the rows identified by the query
  diskfiles = query.all()
  return diskfiles


def duplicate_present(session):
  """
  Returns a list of all the values in a row for any rows where there is more than one row 
  with the same file_id that has present = True.
  """
  # Make and alias of DiskFile
  diskfile_alias = aliased(DiskFile)
  # Self join DiskFile with its alias and compare the file_ids
  query = session.query(DiskFile).select_from(diskfile_alias).filter(DiskFile.id==diskfile_alias.id).filter(DiskFile.present==True).filter(diskfile_alias.present==True).filter(DiskFile.file_id==diskfile_alias.file_id).order_by(DiskFile.id)
  # Creates a list of the rows identified by the query
  diskfiles = query.all()
  return diskfiles
  

def present_not_canonical(session):
  """
  Returns a list of all the values in a row for any rows that have present=True AND 
  canonical=True.
  """
  # Filters the DiskFile table for any rows where present=True AND canonical=True
  query = session.query(DiskFile).filter(DiskFile.present==True).filter(DiskFile.canonical==False).order_by(DiskFile.id)
  # Creates a list of the rows identified by the query
  diskfiles = query.all()
  return diskfiles
  

