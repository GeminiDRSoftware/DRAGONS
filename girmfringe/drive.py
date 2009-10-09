import girmfringe
reload(girmfringe)
import time
from pyraf import iraf
import os

def main():
    print 'benchmarking test for girmfringe begin:\n'
    print 'Gemini North Data test (two images)'
    os.system("rm testout*")   
    starttime = time.time()
    girmfringe.girmfringe("@inlist", "../../test_data/girmfringe_data/rgN20031023S0132_fringe.fits", \
             outimages="@outlist", logfile="testgirmfringePY.log")
    endtime = time.time()
    print "Time for PYTHON version of girmfringe:",(endtime-starttime)  
     
    starttime = time.time()
    iraf.girmfringe(inimages="@inlist", fringe="../../test_data/girmfringe_data/rgN20031023S0132_fringe", \
        outimages="@outlist_cl", outpref="", statsec="[SCI,2][*,*]", 
        scale=1.0, logfile="testgirmfringe.log", verbose = False)
    endtime = time.time()
    print "Time for CL version of girmfringe:",(endtime-starttime)
    print ''
    print ''
        
    print 'Gemini South Data test (one image)'
    starttime = time.time()
    girmfringe.girmfringe("../../test_data/girmfringe_data/rgS20031031S0035.fits", "../../test_data/girmfringe_data/S20031031S0034_fringe.fits", \
             outimages="testout_S_python")
    endtime = time.time()
    print "Time for PYTHON version of girmfringe :",(endtime-starttime)
    starttime = time.time()
    iraf.girmfringe(inimages="../../test_data/girmfringe_data/rgS20031031S0035.fits", fringe="../../test_data/girmfringe_data/S20031031S0034_fringe.fits", \
        outimages="testout_S_cl", outpref="", statsec="[SCI,2][*,*]", 
        scale=1.0, logfile="testgirmfringe.log", verbose = False)
    endtime = time.time()
    print "Time for CL version of girmfringe:",(endtime-starttime) 
    print 'benchmarking test for girmfringe end'
      
if __name__ == "__main__":
    main()