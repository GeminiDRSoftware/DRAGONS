import time
import iqtool
from iqtool.iq import getiq
import os

def test_main():

    test_data_path = os.path.abspath("../../test_data/iqtooldata")
    print test_data_path
    #'''
    filelist = ['mo_flatdiv_biassub_trim_gN20091027S0133.fits']#["N20080927S0183.fits","S20080922S0100.fits"]
    lazyFileList = filelist
    #'''
    '''
    lazyFileList = os.listdir( test_data_path )
    #'''
    lazyFileList2 = []
    
    for lazy in lazyFileList:
        if lazy.find(".fits") != -1:
            lazyFileList2.append( lazy )
            
    time_list = []
    
    # Use 'filelist' and manually put in files if lazyFileList isn't working for whatever reason
    for file in lazyFileList2:
        start_time = time.time()
        try:
            #print getiq.__doc__
            getiq.gemiq( file, function='moffat', rawpath=test_data_path, display=True, pymark=True, mosaic=True )
            end_time = time.time()
            time_list.append( (file, (end_time-start_time)) )
        except:
            print "'" + file + "': Failed"
            raise
    for entry in time_list:
        print "'" + entry[0] + "': Time was (" + str( entry[1] ) + ") secs"
if __name__ == '__main__':
    test_main()

