
_displayObj = None

class DisplayException:
    pass

def getDisplay():
    global _displayObj
    
    if _displayObj is None:
        _displayObj = display()
    
    return _displayObj

def supported():
    '''
    Get list of tool/package system can support.
    
    @return: List of all packages that can be imported.
    @rtype: list of str
    '''
    support_list = []
    
    try:
        import ds9
        support_list.append( 'pyds9' )
    except:
        pass
    
    try:
        from pyraf import iraf
        support_list.append( 'iraf' )
    except:
        pass
    
    return support_list

class display(object):
    '''
    
    
    '''
#------------------------------------------------------------------------------ 
    def __init__(self):
        self.displayDict = {}
        self.support_list = supported()
#------------------------------------------------------------------------------ 
    def __addtool__(self, tool, id=None):
        tooltype = str(type(tool))
        
        if not self.displayDict.has_key( tooltype ):
            self.displayDict.update( {tooltype:[]} )
            
        self.displayDict[tooltype].append(DS9)
        
        if id is not None:
            self.displayDict.update( {id:tool})
#------------------------------------------------------------------------------ 
    def __getitem__(self, id):
        return self.displayDict[id]
#------------------------------------------------------------------------------ 
    def getDisplayTool(self, tool, id=None, **toolargs ):
        '''
        
        
        '''
        if type( tool ) == str:
            toolMethod = 'create'+tool.upper()
            try:
                toolCreator = self.__getattribute__( toolMethod )
            except:
                raise DisplayException( 'The display program: "%s" is not supported.' %(tool) ) 
            
            newtool = toolCreator( **toolargs )
            self.__addtool__( newtool, id )
            
        else:
            raise DisplayException( 'Invalid Argument type: "%s", must be str' %( str(type(tool)) ) )
#------------------------------------------------------------------------------ 
    def createDS9(self, **args):
        if 'iraf' in self.support_list or 'pyds9' in self.support_list:
            return DS9( **args )
        else:
            raise DS9Exception( 'No packages to support ds9 are installed or configured. This tool requires ' + \
                                'iraf and/or pyds9 (both for maximum functionality). If installed, check ' + \
                                'your PYTHONPATH.' )
        

'''
This ds9 stuff should probably be moved to another file to keep things more organized as more
tools are added/supported. In here out of convenience and laziness.

'''
class DS9Exception:
    pass

class DS9(object):
    '''
    
    '''
    
    
#------------------------------------------------------------------------------ 
    def __init__(self, *args, **kargs):
        self.idlist = range(1,17)
        self.iddict = {}
        for elem in self.idlist:
            self.iddict.update( {str(elem):elem} )
        
        
        self.iraf_sup = False
        self.pyds9_sup = False
        self.ds9 = None
        
        tmpSupport = supported()
        
        if 'iraf' in tmpSupport:
            self.iraf_sup = True
            from pyraf import iraf
        if 'pyds9' in tmpSupport:
            import ds9
            self.pyds9_sup = True
            
            self.ds9 = ds9.ds9( *args, **kargs )
            
            ourkeys = set( self.__dict__.keys() )
            ds9keys = set( self.ds9.__dict__.keys() )
            diffkeys = ds9keys - ourkeys
            for key in diffkeys:
                self.__setattr__( key, self.ds9.__dict__[key] )
                
        self.current_frame = 1
        self.current_frame = self.frame()
        self.max_frame = self.current_frame
#------------------------------------------------------------------------------ 
    def __getitem__(self, item):
        '''
        
        '''
        itemtype = type( item )
        if itemtype == str:
            return self.iddict[item]
        elif itemtype == int:
            return self.idlist[item-1]
        else:
            raise DS9Exception( 'Bad argument. Expecting str or int, received "%s"' %(str(itemtype)))
#------------------------------------------------------------------------------ 
    def __setitem__(self, item, value):
        '''
        
        '''
        itemtype = type( item )
        if itemtype == str:
            self.iddict.update( {item:value} )
            self.idlist[value] = item
        elif itemtype == int:
            self.idlist[item-1] = value
            self.iddict.update( {str(item):value} )
        else:
            raise DS9Exception( 'Bad argument. Expecting str or int, received "%s"' %(str(itemtype)))
#------------------------------------------------------------------------------ 
    def reset(self):
        '''
        The idea behind this is to reset the ds9 in the case of long running (i.e. get rid of all ids, delete
        all frames, etc.) so the ds9 is not cluttered with frames from hours ago.
        '''
        pass
#------------------------------------------------------------------------------ 
    def displayFile(self, filename, frame=None, **kargs ):
        assert type(frame) == int
        
        self.frame( frame )
        
        if self.pyds9_sup:
            self.set( 'file ' + str(filename) )
        else:
            iraf.display( filename, frame )
#------------------------------------------------------------------------------ 
    def findFile(self, filename):
        '''
        Find a file currently being displayed, and return frame number.
        
        '''
        pass
#------------------------------------------------------------------------------ 
    def frame(self, frame=None):
        '''
        The code for this function seems overly confusing and does not really work. Most of the 
        confusion comes from the fact that it not entirely obvious how any of this will be used
        in the long run. (i.e. user IDs, no IDs, intermixed, ds9 can only support 16 frames so 
        is a new one created, reset?, if someone writes to frame 1 then to frame 7 without 2-6, 
        etc., etc.)
        
        The goal behind this code is to simply switch the frame.
        
        '''
        if type(frame) == int:
            if frame == 0:
                frame = 1
            elif frame <= -1:
                frame = self.max_frame
            elif frame > self.max_frame:
                if frame <= 16:
                    self.max_frame = frame
            
            if frame not in self.iddict:
                self[frame] = frame
            
            self.current_frame = frame
            self.set( 'frame %d' %(self.current_frame) )
            

        elif type(frame) == str:
            print self[1], type(self[1])
            try:
                self.current_frame = self[frame]
                
            except:
                if self.current_frame == 1 and self[1] == 1:
                    pass
                else:
                    self.max_frame += 1
                    self.current_frame = self.max_frame-1
                    
                self[frame] = self.current_frame-1
            
            self.set( 'frame %d' %(self.current_frame) )
                
            
        elif frame is None:
            try:
                frame = int(self.get( 'frame' ))
            except:
                frame = 1
            # This is in case whoever went outside using these methods, and thus
            # changed the frame through ds9.set, etc.
            if self.current_frame != frame:
                self.current_frame = frame
            
            return frame
#------------------------------------------------------------------------------ 
    def frames(self):
        return self.idlist
#------------------------------------------------------------------------------ 
    def set(self, *args, **kargs ):
        if self.pyds9_sup:
            print args
            self.ds9.set( *args, **kargs )
        else:
            print 'Cannot use XPA set. Only IRAF is working on this machine.'
#------------------------------------------------------------------------------ 
    def get(self, *args, **kargs ):
        if self.pyds9_sup:
            '''
            I am thinking this try/except block should be able to handle multiple exceptions. For example,
            if the ds9 is closed an exception relating to it will be generated. Perhaps this is not the best
            place to put it, but this should create a new one...etc.
            '''
            try:
                print args
                return self.ds9.get( *args, **kargs )
            except:
                raise
        else:
            print 'Cannot use XPA set. Only IRAF is working on this machine.'
#------------------------------------------------------------------------------ 
    def zoomto(self, level):
        if type( level ) == str:
            level = float(level)
        
        if level <= 0:
            return 
        else:
            self.set( 'zoom to %s' %(str(level)) )
#------------------------------------------------------------------------------ 
    def draw(self, shape, xcoord, ycoord, properties=None, color='red', text='', crossout=False, fixed=False, 
             **kargs):
        '''
        
        '''
        xcoord = str( xcoord )
        ycoord = str( ycoord )
        
        # Eventually, it would be nice to check that lengths matches shapes.
        if type(properties) == list:
            prop_size = len( properties )
            for i in range(prop_size):
                properties[i] = str( properties[i] )
                
            properties = ' '.join( properties )
        else:
            properties = str( properties )
        
        if crossout:
            crossout = '-'
        else:
            crossout = ''
            
        if fixed:
            fixed = '0'
        else:
            fixed = '1'
        
        additional_args = ''
        for arg in kargs.keys():
            tempval = str( kargs[arg] )
            additional_args += '%s=%s ' %(str(arg), tempval)
        
        command = '%(crossout)s%(shape)s %(xcoord)s %(ycoord)s %(props)s # color="%(colour)s" text="%(text)s" fixed=%(fix)s %(params)s' %{'crossout':crossout, 
                                                 'shape':shape, 'xcoord':xcoord,
                                                 'ycoord':ycoord, 'props':properties, 'colour':color,
                                                 'text':text, 'fix':fixed, 'params':additional_args
                                                 }
        
        self.setRegion( command )
#------------------------------------------------------------------------------ 
    def markPixel(self, xcoord, ycoord, badPixel=True, color='red'):
        self.draw( 'box', xcoord, ycoord, [1,1], crossout=badPixel, color=color, fixed=True, edit='0',
                    select='0', move='0' )
#------------------------------------------------------------------------------ 
    def setRegion(self, command):
        self.set( 'regions', command )
#------------------------------------------------------------------------------ 
    def pan(self, xoord, ycoord, relative=False):
        if relative:
            command = 'pan %s %s' 
        else:
            command = 'pan to %s %s'
        
        command = command %( str(xcoord), str(ycoord) )
        self.set( command )


if __name__ == "__main__":
    zz = getDisplay()
    idz = 'metal gear'
    frameid = 'snake'
    fid = 'plissken'
    zz.getDisplayTool( 'ds9', idz )#, existing=True )
    print zz[idz].frame()
    tomogatchi = zz[idz]
    tomogatchi.set( 'file mo_flatdiv_biassub_trim_gN20091027S0133.fits' )
    tomogatchi.set( 'tile yes' )
    tomogatchi.frame( frameid )
    tomogatchi.frame( fid )
    print tomogatchi.frames()
    tomogatchi.frame( frameid )
    tomogatchi.set( 'file mo_flatdiv_biassub_trim_gN20091027S0133.fits' )
    '''
    print tomogatchi.frames()
    
    import random
    xcoord = random.randint(0,2000)
    ycoord = random.randint(0,2000)
    tomogatchi.zoomto( 1 )
    tomogatchi.pan( xcoord, ycoord )
    tomogatchi.markPixel( xcoord, ycoord, color='red' )
    tomogatchi.markPixel( xcoord-1, ycoord, color='red' )
    tomogatchi.markPixel( xcoord+1, ycoord, color='red' )
    tomogatchi.markPixel( xcoord, ycoord-1, color='red' )
    tomogatchi.markPixel( xcoord, ycoord+1, color='red' )
    
    tomogatchi.zoomto( 8 )
    '''
    #zz[idz].set( 'sleep 2' )
    #zz[idz].pan( 1500, 1000 )
    #zz[idz].zoomto( 1 )
    