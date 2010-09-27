#Author: Kyle Mede, May 2010
#this module is being used as the workhorse for the prepare primitives.

from astrodata.adutils import gemLog
from astrodata.AstroData import AstroData

log=gemLog.getGeminiLog() 

def stdInstHdrs(ad):  
    # adding the missing/needed keywords into the headers 
    if not ad.isType('GMOS_IMAGE'):
    # do the stuff to the headers that is for the MOS, those for IMAGE are 
    # taken care of with stdObsHdrs all ready 
        for ext in ad["SCI"]:
            ext.extSetKeyValue(('SCI',int(ext.header['EXTVER'])),'DISPAXIS', ext.dispersion_axis() , 'Dispersion axis')
            ext.extSetKeyValue(('SCI',int(ext.header['EXTVER'])),'DISPAXIS',ext.dispersion_axis(), "Dispersion axis")

def valInstData(ad):
    # to ensure structure is the normal PHU followed by 3 SCI 
    # extensions for GMOS
    length=ad.countExts('SCI')
    if length==1 or length==3 or length==6 or length==12:
        pass
    else: 
        log.critical("There are NOT 1, 3, 6 or 12 extensions in file = "+ad.filename)
        raise 
            
def addMDF(ad,mdf): 
        # so far all of this is done in the primitive, will 
        # figure this out later
        pass
        
         
       