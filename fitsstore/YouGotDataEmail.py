import sys
sys.path=['/opt/sqlalchemy/lib/python2.5/site-packages', '/astro/iraf/x86_64/gempylocal/lib/stsci_python/lib/python2.5/site-packages']+sys.path

import FitsStorage
import FitsStorageConfig
from FitsStorageUtils import *

import urllib2
import re
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import datetime
from optparse import OptionParser

parser = OptionParser()
parser.add_option("--emailfrom", action="store", dest="fromaddr", default="fitsdata@gemini.edu", help="Email Address to send from")
parser.add_option("--replyto", action="store", dest="replyto", default="gnda@gemini.edu", help="Set a Reply-To email header")
(options, args) = parser.parse_args()

mailhost = "smtp.gemini.edu"
cre = re.compile('\.fits')

# The project / email list. Get from the database
session = sessionfactory()
notifs = session.query(Notification).all()

for notif in notifs:

  if(notif.internal):
    url = "http://fits/summary/today/%s" % notif.selection
  else:
    url = "http://fits/summary/nolinks/today/%s" % notif.selection

  f = urllib2.urlopen(url)
  html = f.read()
  f.close()

  match = cre.search(html)

  if(match):

    subject = "New Data for %s" % notif.selection

    msg = MIMEMultipart()

    text = "New data has been taken for %s. The attached html file gives details.\n\n" % (notif.selection)
    if(notif.internal):
      text += "The fits storage summary table for this data be found at: %s\n\n" % url
    else:
      text += "Access to all Gemini data is via the Gemini Science Archive at http://www1.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/gsa/\n"
      text += "Data Quality assessment and data package release will proceed as normal over the next few days."

    part1 = MIMEText(text, 'plain')
    part2 = MIMEText(html, 'html')

    msg['Subject'] = subject
    msg['From'] = options.fromaddr
    msg['To'] = notif.to
    msg['Cc'] = notif.cc
    msg['Reply-To'] = options.replyto

    msg.attach(part1)
    msg.attach(part2)

    fulllist = []
    tolist = notif.to.split(',')
    fulllist += tolist
    if(notif.cc):
      cclist = notif.cc.split(',')
      fulllist += cclist


    smtp = smtplib.SMTP(mailhost)
    smtp.sendmail(options.fromaddr, fulllist, msg.as_string())
    smtp.quit()
