import time
from IQTool.iq import detectSources as ds
#import iqUtil
import pyfits as pf
import pyraf
from pyraf import iraf
from pyraf.iraf import gemini
gemini()
gemini.gmos()

from astrodata.adutils.future import pyDisplay
from astrodata.adutils.future import starFilter

'''
!!!!!!!!!!!!!!!!!!!!!!! READ ME !!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
@author: River Allen

What is this? - An experiment in creating a 'star filter' that could learn.

Why did you do this? - I had read an article on spam filtering using Bayesian Techniques and was inspired.
The article can be found here: http://www.paulgraham.com/spam.html

How does it work? - In it's primitive unfinished form, you simply run it on a selected image and click inside
the boxes of things that you think should be not be detected. Then you close ds9. 
If you run it again, these things will not be detected the second time around. The goal was to have some 
implement a balanced probability based Bayesian type algorithm, but I never got around to actually getting 
a good one. Right now it is 'uneven' or too strong,and is not really learning except in the sense that it 
will not detect the EXACT same object again, rather than similar objects. For the most part, look in/modify 
star filter in the gemini_python/trunk/utils/future for improving this.

Why is this not working? - When I developed this, I did so entirely on my machine without any assumptions
for anyone else as I was rapidly programming it. The environment I had:
- ds9 installed.
- gemini_python checked out and the PYTHONPATH pointing to gemini_python_trunk.
- pyds9 installed.

Why is there such a stupid system for determining which stars are good and bad (i.e. clicking and then closing
ds9 is annoying)? - For some reason, I could not get the xpa command: 'imexam any coordinate' or 
'imexam key coordinate' to work. Because of rapid development, it was the only solution that came to mind for
giving human input.

Why is it so 'dirty'? - I wrote it very quickly.

Why is this even here? - I thought that I might as well upload it in case it had some future use.

'''
def main():
    allfiles = ['mo_flatdiv_biassub_trim_gN20091027S0133.fits','N20020214S059.fits','N20020606S0141.fits',
                'N20080927S0183.fits','N20091002S0219.fits','N20091027S0133.fits','rgN20031029S0132.fits',
                'rgS20031031S0035.fits','S20080922S0100.fits']
    
    files = allfiles
    files = ['N20091002S0219.fits']#['N20030427S0092.fits']#['mo_flatdiv_biassub_trim_gN20091027S0133.fits']
    
    
    paramset = [
                {'threshold':2, 'window':None, 'verbose':True, 'timing':True, 
                 'grid':False, 'rejection':None},
                ]
    labelStrs = pyraf.iraf.no
    time_dict = {}
    frameForDisplay = 1
    dispFrame = 0
    numtimes = 1
    drawWindows = True
    display = True
    tmpFilename = 'tmpfile.tmp'
    iraf.set(stdimage='imtgmos')
    
    pydisplay = pyDisplay.getDisplay()
    
    sfilt = starFilter.starFilter()
    for fil in files:
        ext = 1
        while ext < 2:
            indi = 0
            for param in paramset:
                
                indi += 1
                for i in range(numtimes):
                    ds9 = pydisplay.getDisplayTool( 'ds9', 'myds9' )
                    try:
                        print '-'*50
                        if display:
                            try:
                                iraf.display( fil+'['+str(ext)+']', frameForDisplay)
                            except:
                                break
                        else:
                            try:
                                pf.getdata( fil, ext )
                            except:
                                break
                        
                        if frameForDisplay > 0:
                            dispFrame = frameForDisplay - 1
                        
                        xyArray, filTime = ds.detSources( fil, dispFrame=dispFrame, drawWindows=drawWindows, exts=ext, **param  )
                        end_time = time.time()
                        
                        scidata = pf.getdata( fil, ext )
                        xyArray = sfilt.filterStars( scidata, xyArray)
                        key = fil +'['+str(ext)+'] P'+str(indi)
                        
                        if not time_dict.has_key(key):
                            time_dict[key] = []
                        
                        time_dict[key].append((filTime,len(xyArray)))
                        #print xyArray
                        if display:
                            tmpFile = open( tmpFilename, 'w' )
                            index = 0
                            toWrite = ''
                            for coords in xyArray:
                                toWrite += '%s %s %s\n' %(str(coords[0]), str(coords[1]), str(index))
                                index += 1
                            tmpFile.write( toWrite )
                            tmpFile.close()
                            
                            iraf.tvmark( frame=dispFrame,coords=tmpFilename, mark='point',
                                pointsize=1, color=204, label=labelStrs )
                        
                            #tutorExam( ds9, scidata, xyArray, ext)
                        
                    except:
                        print "'" + fil + "': Failed"
                        raise
                
                    
                
                #frameForDisplay += 1
            ext += 1
        
    timelisting = sorted(time_dict.keys())
    print timelisting
    print 'Number of Runs:', numtimes
    for entry in timelisting:
        sum = 0
        objsum = 0
        for val in time_dict[entry]:
            sum += val[0]
            objsum += val[1]
        
        print "'" + entry + "': Time was (" + str( sum/numtimes ) + ")\tsecs / Objects ("+str(objsum)+")"
    
    
def tutorExam( ds9, scidata, xyArray, ext=1 ):
    '''
    
    
    '''
    sfilt = starFilter.starFilter()
    
    print scidata, '\n', scidata.shape
    box_size = 14
    half_box_size = box_size / 2

    for xc, yc in xyArray:
        ds.draw_windows( [(xc - half_box_size, yc - half_box_size, box_size, box_size)], label=False )
    
    bad_star_list = []
    while True:
        try:
            val = ds9.get( 'imexam coordinate' )
        except:
            break
        
        print 'TDS130:', val
        if type( val ) == str:
            try:
                xcoord , ycoord = val.split( ' ' )
            except:
                break
            xcoord = float( xcoord )
            ycoord = float( ycoord )
            
            counter = 0
            
            for coord in xyArray:
                if counter in bad_star_list:
                    counter += 1
                    continue
                if abs( coord[0] - xcoord ) < half_box_size:
                    if abs( coord[1] - ycoord ) < half_box_size:
                        sfilt.nonStar(scidata[int(xyArray[counter][1]-half_box_size):int(xyArray[counter][1]+half_box_size),
                                              int(xyArray[counter][0]-half_box_size):int(xyArray[counter][0]+half_box_size)],
                                              xyArray[counter][0], xyArray[counter][1] )
                        bad_star_list.append( counter )
                counter += 1
        else:
            break
    
    goodstar_list = list( set(range(len(xyArray))) - set(bad_star_list) )
    for index in goodstar_list:
        sfilt.goodStar( scidata[xyArray[index][0]-half_box_size:xyArray[index][0]+half_box_size,
                               xyArray[index][1]-half_box_size:xyArray[index][1]+half_box_size], 
                               xyArray[index][0], xyArray[index][1] )
    
    sfilt.flushFilter()
    print 'KAPLAH'
    
    
if __name__ == '__main__':
    main()