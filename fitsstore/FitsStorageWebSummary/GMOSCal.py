"""
This module contains the gmoscal html generator function. 
"""
from sqlalchemy.sql.expression import cast
from FitsStorageWebSummary.Selection import *

import os


def gmoscal(req, selection):
   """
   This generates a GMOS imaging twilight flat report.
   And a BIAS report.
   If no date or daterange is given, tries to find last processing date
   """

   title = "GMOS Cal (Imaging Twilight Flats and Biases) Report %s" % sayselection(selection)
   req.content_type = "text/html"
   req.write('<html><head><title>%s</title><link rel="stylesheet" href="/htmldocs/table.css"></head><body><h1>%s</h1>' % (title, title))
   if(fits_system_status == 'development'):
     req.write("<H1>This is the Development Server, not the operational system. If you're not sure why you're seeing this message, please consult PH</H1>")
 
   # If no date or daterange, look on endor or josie to get the last processing date
   if(('date' not in selection) and ('daterange' not in selection)):
     base_dir=das_calproc_path
     checkfile = 'Basecalib/biasall.list'
     enddate = datetime.datetime.now().date()
     oneday = datetime.timedelta(days=1)
     date = enddate
     found = -1000
     startdate = None
     while(found < 0):
       datestr = date.strftime("%Y%b%d").lower()
       file = os.path.join(base_dir, datestr, checkfile)
       if(os.path.exists(file)):
         found = 1
         startdate = date
       date -= oneday
       found += 1

       if(startdate):
         # Start the day after the last reduction
         startdate += oneday
         selection['daterange']="%s-%s" % (startdate.strftime("%Y%m%d"), enddate.strftime("%Y%m%d"))
         req.write("<H2>Auto-detecting Last Processing Date: %s<H2>" % selection['daterange'])

   # Get a database session
   session = sessionfactory()
   try:
     # First the Twilight Flats part
     req.write('<H2>Twilight Flats</H2>')

     # We do this twice, first for the science data, then for the twilight flat data
     # These are differentiated by being science or dayCal

     # Put the results into dictionaries, which we can then combine into one html table
     sci = {}
     tlf = {}
     for observation_class in (['science', 'dayCal']):

       # The basic query for this
       query = session.query(func.count(1), Header.filter_name, Gmos.detector_x_bin, Gmos.detector_y_bin).select_from(join(Gmos, join(Header, join(DiskFile, File))))
       query = query.filter(DiskFile.canonical == True)

       # Fudge and add the selection criteria
       selection['observation_class']=observation_class
       selection['observation_type']='OBJECT'
       selection['spectroscopy']=False
       selection['inst']='GMOS'
       if(observation_class == 'science'):
         selection['qa_state']='Win'
       else:
         selection['qa_state']='Pass'
         # Only select full frame dayCals
         query = query.filter(or_(Gmos.amp_read_area == '''["'EEV 9273-16-03, right':[1:2048,1:4608]", "'EEV 9273-20-04, right':[2049:4096,1:4608]", "'EEV 9273-20-03, left':[4097:6144,1:4608]"]''', Gmos.amp_read_area == '''["'EEV 2037-06-03, left':[1:2048,1:4608]", "'EEV 8194-19-04, left':[2049:4096,1:4608]", "'EEV 8261-07-04, right':[4097:6144,1:4608]"]'''))

       query = queryselection(query, selection)
  
       # Knock out ENG programs
       query = query.filter(~Header.program_id.like('%ENG%'))

       # Group by clause
       query = query.group_by(Header.filter_name, Gmos.detector_x_bin, Gmos.detector_y_bin).order_by(Gmos.detector_x_bin, Header.filter_name)

       list = query.all()

       # Populate the dictionary
       # as {'i-2x2':[10, 'i', '2x2'], ...}  ie [number, filter_name, binning]
       if(observation_class == 'science'):
         dict = sci
       else:
         dict = tlf

       for row in list:
         binning = "%dx%d" % (row[2], row[3])
         key = "%s-%s" % (row[1], binning)
         dict[key]=[row[0], row[1], binning]

     # Make the master dictionary
     # as {'i-2x2':[10, 20, 'i', '2x2'], ...}   [n_sci, n_tlf, filter_name, binning]
     all = {}
     for key in sci.keys():
       nsci = sci[key][0]
       ntlf = 0
       filter_name = sci[key][1]
       binning = sci[key][2]
       all[key] = [nsci, ntlf, filter_name, binning]
     for key in tlf.keys():
       if (key in all.keys()):
         all[key][1] = tlf[key][0]
       else:
         nsci = 0
         ntlf = tlf[key][0]
         filter_name = tlf[key][1]
         binning = tlf[key][2]
         all[key] = [nsci, ntlf, filter_name, binning]
     
     
     # Output the HTML table and links to summaries etc
     req.write('<TABLE border=0>')
     req.write('<TR class=tr_head>')
     req.write('<TH>Number of Science Frames</TH>')
     req.write('<TH>Number of Twilight Frames</TH>')
     req.write('<TH>Filter</TH>')
     req.write('<TH>Binning</TH>')
     req.write('</TR>')
     
     even=False
     keys = all.keys()
     keys.sort(reverse=True)
     for key in keys:
       even = not even
       if(even):
         if((all[key][0] > 0) and (all[key][1] == 0)):
           cs = "tr_warneven"
         else:
           cs = "tr_even"
       else:
         if((all[key][0] > 0) and (all[key][1] == 0)):
           cs = "tr_warnodd"
         else:
           cs = "tr_odd"

       req.write("<TR class=%s>" % cs)

       for i in range(4):
         req.write("<TD>%d</TD>" % all[key][i])

       req.write("</TR>")
     req.write("</TABLE>")
     datething=''
     if('date' in selection):
       datething = selection['date']
     if('daterange' in selection):
       datething = selection['daterange']
     req.write('<P><a href="/summary/GMOS/imaging/OBJECT/science/Win/%s">Science Frames Summary Table</a></P>' % datething)
     req.write('<P><a href="/summary/GMOS/imaging/OBJECT/dayCal/Pass/%s">Twilight Flat Summary Table</a></P>' % datething)
     req.write('<P>NB. Summary tables will show ENG program data not reflected in the counts above.</P>')

     # Now the BIAS report
     req.write('<H2>Biases</H2>')

     #tzoffset = datetime.timedelta(seconds=time.timezone)
     #hack to make hbffits1 look chilean for tmp work Dec 2010
     tzoffset = datetime.timedelta(seconds=14400)
     
     oneday = datetime.timedelta(days=1)
     offset = sqlalchemy.sql.expression.literal(tzoffset - oneday, sqlalchemy.types.Interval)
     query = session.query(func.count(1), cast((Header.ut_datetime + offset), sqlalchemy.types.DATE).label('utdate'), Gmos.detector_x_bin, Gmos.detector_y_bin, Gmos.amp_read_area).select_from(join(Gmos, join(Header, join(DiskFile, File))))

     query = query.filter(DiskFile.canonical == True)

     # Fudge and add the selection criteria
     selection['observation_type']='BIAS'
     selection['inst']='GMOS'
     selection['qa_state']='Pass'
     query = queryselection(query, selection)

     query = query.group_by('utdate', Gmos.detector_x_bin, Gmos.detector_y_bin, Gmos.amp_read_area).order_by('utdate', Gmos.detector_x_bin, Gmos.detector_y_bin, Gmos.amp_read_area)

     list = query.all()

     # OK, re-organise results into tally table dict
     # dict is: {utdate: {binning: {roi: Number}}
     dict={}
     for row in list:
       # Parse the element numbers for simplicity
       num = row[0]
       utdate = row[1]
       binning = "%dx%d" % (row[2], row[3])
       roi = row[4]
       if(roi == '''["'EEV 9273-16-03, right':[1:2048,1:4608]", "'EEV 9273-20-04, right':[2049:4096,1:4608]", "'EEV 9273-20-03, left':[4097:6144,1:4608]"]'''):
         roi = "Full"
       if(roi == '''["'EEV 9273-16-03, right':[1:2048,1792:2815]", "'EEV 9273-20-04, right':[2049:4096,1792:2815]", "'EEV 9273-20-03, left':[4097:6144,1792:2815]"]'''):
         roi = "Cent"
       if(roi == '''["'EEV 2037-06-03, left':[1:2048,1:4608]", "'EEV 8194-19-04, left':[2049:4096,1:4608]", "'EEV 8261-07-04, right':[4097:6144,1:4608]"]'''):
         roi = "Full"
       if(roi == '''["'EEV 2037-06-03, left':[1:2048,1792:2815]", "'EEV 8194-19-04, left':[2049:4096,1792:2815]", "'EEV 8261-07-04, right':[4097:6144,1792:2815]"]'''):
         roi = "Cent"

       if(utdate not in dict.keys()):
         dict[utdate]={}
       if(binning not in dict[utdate].keys()):
         dict[utdate][binning] = {}
       if(roi not in dict[utdate][binning].keys()):
         dict[utdate][binning][roi] = num

     # Output the HTML table 
     # While we do it, add up the totals as a simply column tally
     binlist = ['1x1', '2x2', '2x1', '1x2', '2x4', '4x2', '4x1', '1x4', '4x4']
     roilist = ['Full', 'Cent']
     req.write('<TABLE border=0>')
     req.write('<TR class=tr_head>')
     req.write('<TH rowspan=2>UT Date</TH>')
     for b in binlist:
       req.write('<TH colspan=2>%s</TH>' %b)
     req.write('</TR>')
     req.write('<TR class=tr_head>')
     for b in binlist:
       for r in roilist:
         req.write('<TH>%s</TH>'% r)
     req.write('</TR>')

     even=False
     utdates = dict.keys()
     utdates.sort(reverse=True)
     total=[]
     for i in range(0, len(binlist)*len(roilist)):
       total.append(0)

     for utdate in utdates:
       even = not even
       if(even):
         cs = "tr_even"
       else:
         cs = "tr_odd"

       req.write("<TR class=%s>" % cs)
       req.write("<TD>%s</TD>" % utdate)
       i=0
       for b in binlist:
         for r in roilist:
           try:
             num = dict[utdate][b][r]
           except KeyError:
             num = 0
           total[i] += num
           i += 1
           req.write("<TD>%d</TD>" % num)
       req.write("</TR>")

     req.write("<TR class=tr_head>")
     req.write("<TH>%s</TH>" % 'Total')
     for t in total:
       req.write("<TH>%d</TH>" % t)
     req.write("</TR>")
     req.write("</TABLE>")

     # OK, find if there were dates for which there were no biases...
     # Can only do this if we got a daterange selection, otherwise it's broken if there's none on the first or last day
     # utdates is a reverse sorted list for which there were biases.
     if('daterange' in selection):
       # Parse the date to start and end datetime objects
       daterangecre=re.compile('(20\d\d[01]\d[0123]\d)-(20\d\d[01]\d[0123]\d)')
       m = daterangecre.match(selection['daterange'])
       startdate = m.group(1)
       enddate = m.group(2)
       tzoffset = datetime.timedelta(seconds=time.timezone)
       oneday = datetime.timedelta(days=1)
       startdt = dateutil.parser.parse("%s 14:00:00" % startdate)
       startdt = startdt + tzoffset - oneday
       enddt = dateutil.parser.parse("%s 14:00:00" % enddate)
       enddt = enddt + tzoffset - oneday
       enddt = enddt + oneday
       # Flip them round if reversed
       if(startdt > enddt):
         tmp = enddt
         enddt = startdt
         startdt = tmp
       startdate = startdt.date()
       enddate = enddt.date()

       nobiases = []
       date = startdate
       while(date <= enddate):
         if(date not in utdates): 
           nobiases.append(str(date))
         date += oneday

       req.write('<P>There were %d dates with no biases: ' % len(nobiases))
       if(len(nobiases)>0):
         req.write(', '.join(nobiases))
       req.write('</P>')

     req.write("</body></html>")
     return apache.OK


   except IOError:
     pass
   finally:
     session.close()

