import urllib2
import re
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import datetime
from optparse import OptionParser

parser = OptionParser()
parser.add_option("--emailto", action="store", dest="toaddr", default="gnda@gemini.edu", help="Email Address to send to")
parser.add_option("--emailfrom", action="store", dest="fromaddr", default="Missing GMOS Arc Check <fitsdata@gemini.edu>", help="Email Address to send from")
parser.add_option("--replyto", action="store", dest="replyto", default="gnda@gemini.edu", help="Set a Reply-To email header")
parser.add_option("--ndays", action="store", type="int", dest="ndays", default=14, help="Number of days to query")
parser.add_option("--skipdays", action="store", type="int", dest="skipdays", default=4, help="Number of days ago to start query from")
parser.add_option("--httpserver", action="store", dest="httpserver", default="fits", help="hostname of FitsStorage http server to query")
(options, args) = parser.parse_args()

# Work out the date range to query
utcnow = datetime.datetime.utcnow()
utcend = utcnow - datetime.timedelta(days=options.skipdays)
utcstart = utcend - datetime.timedelta(days=options.ndays)
daterange="%s-%s" % (utcstart.date().strftime("%Y%m%d"), utcend.date().strftime("%Y%m%d"))

url = "http://%s/calibrations/GMOS/Win/%s/arc/warnings" % (options.httpserver, daterange)

f = urllib2.urlopen(url)
html = f.read()
f.close()

cremissing = re.compile('Counted (\d*) potential missing Calibrations')
crewarning = re.compile('Query generated (\d*) warnings')

warnings = int(crewarning.search(html).group(1))
missing = int(cremissing.search(html).group(1))

mailhost = "smtp.gemini.edu"

if(missing==0):
  if options.skipdays == 0:
    subject = "No missing calibrations today. Yay!"
  else:
    subject = "No missing calibrations this week. Yay!"
else:
  subject = "MISSING CALIBRATIONS: %d missing arcs" % missing

msg = MIMEMultipart()

text = "Calibration Check: %d missing, %d warnings.\n\n%s" % (missing, warnings, url)

part1 = MIMEText(text, 'plain')
part2 = MIMEText(html, 'html')

msg['Subject'] = subject
msg['From'] = options.fromaddr
msg['To'] = options.toaddr
msg['Reply-To'] = options.replyto

msg.attach(part1)
msg.attach(part2)

smtp = smtplib.SMTP("smtp.gemini.edu")
smtp.sendmail(options.fromaddr, [options.toaddr], msg.as_string())
smtp.quit()
