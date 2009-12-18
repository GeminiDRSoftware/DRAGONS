'''
@author: River Allen
@contact: riverallen@gmail.com
@organization: Gemini Observatory

A better name for this would be 'BayesionNonStarFilter', but star filter sounded better.
'''
import pickle
from numpy import *

class starFilter(object):
    '''
    
    
    '''
    def __del__( self ):
        self.flushFilter()
    
    def __init__( self, filterFile='starfilter.bay', boxSize=14 ):
        self.filter = {}
        self.filterFile = filterFile
        self.boxSize = boxSize
    
    def flushFilter( self ):
        try:
            pickle.dump( self.filterFile, open(self.filterFile, 'w') )
        except:
            raise
    
    def loadFilter( self ):
        try:
            self.filter = pickle.load( open(self.filterFile) )
        except:
            self.flushFilter()
    
    def filterStars( self, data, xyCoords, interactive=False ):
        '''
        
        
        '''
        starData = []
        
    def nonStar( self, data, xcoord, ycoord ):
        pass
        print 'NON STAR', xcoord, ycoord
        
        
    def goodStar( self, data, xcoord, ycoord ):
        pass
        print 'STAR', xcoord, ycoord
        
    def hashStar( self, starData ):
        '''
        
        
        '''
        sumstardata = sum( starData )
        
        hashStar = starData / sumstardata * 100
        hashStar = int( hashStar )
        
        print hashStar
        
    
    
    