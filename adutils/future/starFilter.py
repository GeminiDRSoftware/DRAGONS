'''
@author: River Allen
@contact: riverallen@gmail.com
@organization: Gemini Observatory

A better name for this would be 'BayesionNonStarFilter', but star filter sounded better.
'''
import pickle
from numpy import *
import numpy as np
import os
import hashlib

class starFilter(object):
    '''
    
    
    '''    
    def __init__( self, filterFile='', boxSize=14 ):
        self.filter = {}
        if filterFile == '':
            self.filterFile = os.path.join( os.path.dirname( __file__ ), 'starfilter.bay' )
        else:
            self.filterFile = filterFile
        self.boxSize = boxSize
        self.halfBoxSize = boxSize / 2
        self.loadFilter()
        self.hashFunc = hashlib.sha1()
        
    
    def flushFilter( self ):
        print 'DUMPING FILTER TO DISK'
        try:
            pickle.dump( self.filter, open(self.filterFile, 'w') )
        except:
            raise
    
    def loadFilter( self ):
        try:
            self.filter = pickle.load( open(self.filterFile) )
        except:
            self.flushFilter()
    
    def filterStars( self, data, xyArray ):
        '''
        
        
        '''
        good_star_list = []
        index = 0
        for xcoord, ycoord in xyArray:
            tempData = data[int(xyArray[index][1]-self.halfBoxSize):int(xyArray[index][1]+self.halfBoxSize),
                            int(xyArray[index][0]-self.halfBoxSize):int(xyArray[index][0]+self.halfBoxSize)]
            
            hashsr = self.hashStar( tempData )
            val = self.retrieveValueFromFilter( hashsr )
            if val >= 0:
                good_star_list.append( (xcoord, ycoord) )
            '''
            sum = 0
            counter = 0 
            for point in hashsr:
                key = (counter,point)
                val = self.retrieveValueFromFilter( key )
                sum += val
                counter += 1
            
            if sum >= 0:
                good_star_list.append( index )
            '''
            index += 1
        ''' 
        zz = []
        
        for ind in good_star_list:
            zz.append( xyArray[ind] )
        '''
        return good_star_list
        
    def nonStar( self, data, xcoord, ycoord ):
        print 'NON STAR', xcoord, ycoord
        #print 'DATA', data
        flatHash = self.hashStar( data )
        #print flatHash
        '''
        counter = 0 
        for point in flatHash:
            key = (counter,point)
            val = self.retrieveValueFromFilter( key )
            self.filter.update( {key:val-2} )
            counter += 1
        '''
        val = self.retrieveValueFromFilter( flatHash )
        self.filter.update( {flatHash:val-5} )
        
    def goodStar( self, data, xcoord, ycoord ):
        print 'STAR', xcoord, ycoord
        #print 'DATA', data
        flatHash = self.hashStar( data )
        #print flatHash
        '''
        counter = 0 
        for point in flatHash:
            key = (counter,point)
            val = self.retrieveValueFromFilter( key )
            self.filter.update( {key:val+1} )
            counter += 1
        '''
        val = self.retrieveValueFromFilter( flatHash )
        self.filter.update( {flatHash:val+1} )
    
    def retrieveValueFromFilter(self, key):
        if self.filter.has_key( key ):
            return self.filter[key]
        else:
            self.filter.update( {key:0} )
            return 0
    
    def hashStar( self, starData ):
        '''
        
        
        '''
        sumstardata = sum( starData )
        
        hashsr = starData / sumstardata * 1000
        hashsr = hashsr.flatten()
        
        hashsr = hashsr.astype( int )

        zz = np.where( hashsr >= 7 )
        tmp = np.zeros( self.boxSize * self.boxSize )
        tmp[zz] = 3
        xx = np.where( (hashsr >=5) & (hashsr < 7) )
        tmp[xx] = 2
        yy = np.where( (hashsr >=4) & (hashsr < 5) )
        tmp[yy] = 1
        
        
        #print 'hashsr', hashsr
        #print tmp
        return hash( str(tmp.tolist()) )
    
    