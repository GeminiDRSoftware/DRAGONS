#!/usr/bin/env python

# EDITED for RTF tests. nz   Nov 28 2013

# Acceptance Tests
#
# This module puts together all the Aceptance Tests for the Mosaic project and it
# can be run in conjunction with the explanations in the MS Word document:
# ATPMosaic-Oct4.docx

# To run individual tests:
# UNIX shell:

#% atp_mosaic.py  at<n>   # where 'n' is a number from 1-8   (Mosaic tests)
#% atp_mosaic.py  atd<n>   # where 'n' is a number from 1-9   (MosaicAD tests)

# Test FITS files. 
# You need GMOS and GSAOI files. This module uses:
#
#    gmos_file = '../data/gS20120420S0033.fits'
#    gsaoi_file = '../data/guS20110324S0146.fits'
#    nongem_file='../data/kp620765.fits'             # To Non_gemini files.

def at1():
    """
    This test creates a list of 4 ndarrays of the same size as input to MosaicData.
    The test checks that the data is not corrupted during this process 
    by calculating the median of each ndarray before and after MosaicData instantiation.
    """

    import numpy
    from gempy.library.mosaic import MosaicData

    print '\n at1 REQUIREMENT.......'
    print ('*****MosaicData shall instantiate an object given a list of' 
         ' equal size ndarrays.')

    #     Make an ndarray of shape(2048,1024) of  pixel values 
    #     between 0 and 1000.
    data = numpy.linspace(0.,1000.,1024*2048).reshape(2048,1024)
    indata_list = [data*(-1)**k for k in numpy.arange(4)]

    print 'Median values before:',[numpy.median(data) for data in indata_list]

    #     Now create the object
    md = MosaicData(indata_list)
    print 'Median values after:',[numpy.median(data) for data in md.data_list]

    # The expected answer is: [-500.0, 500.0, -500.0, 500.0]
    print 'data_list attribute: number of ndarrays: ',len(md.data_list)
    print ' The expected answer is: 4'

    print 'data_list attribute: shape of 1st ndarray: ', md.data_list[0].shape
    print ' The expected answer is: (2048, 1024)'

def at2():
    """
    The test instantiates a MosaicGeometry object from an input Geometry
    dictionary     containing GSAOI  block constants.  
    """ 
    from gempy.library.mosaic import MosaicGeometry

    print '\n at2 REQUIREMENT.......'
    print ('*****MosaicGeometry shall instantiate an object given a Geometry Dictionary')

    geometry = {
             'transformation': {
                 'shift':   [(0.,0.), (43.60,-1.24), (0.02, 41.10), 
                               (43.42, 41.72)],  
                   'rotation': (0.0, -1.033606, 0.582767, 0.769542),         
                   'magnification': (1., 1.0013, 1.0052, 1.0159),        
                           },
                
    'gap_dict': { 
         'tile_gaps': {(0,0):(139,139),(1,0):(139,139),
                      (0,1):(139,139),(1,1):(139,139)}, 
           'transform_gaps': {(0,0):(139,139),(1,0):(139,139),
                        (0,1):(139,139),(1,1):(139,139)},
                }, 
                'blocksize':    (2048,2048),
            'interpolator': 'linear',
                'ref_block':    (1,0),      # 0-based         
                'mosaic_grid':  (2,2)  }

          #    Now create an object. The __init__ function will verify
    #    the input dictionary.
    geo = MosaicGeometry(geometry)

    #    Verify attributes for GSAO geometry values. geo.info()
    #    is a function that forms  a dictionary with the object
    #    attributes.
    print geo.info()

def at3():
    """
     Verify that a mosaic is created from a list of ndarrays
    """
    from gempy.library.mosaic import Mosaic, MosaicData
    import numpy

    print '\n at3 REQUIREMENT.......'
    print ('*****When positioning information is not available, Mosaic shall by '
           'default tile horizontally the list of input ndarrays of the same shape')

    #     Make one ndarray with pixel values between 0 and 1000
    data =numpy.linspace(0.,1000.,1024*2048).reshape(2048,1024)

    #     Replicate the ndarray 4 times
    data_list = [data*(-1)**k for k in numpy.arange(4)]

    #     Create a valid object with only a data list
    md = MosaicData(data_list)

    #     Create a Mosaic object using this MosaicData object
    mo = Mosaic(md)

    #     Now produce a mosaic by simply tiling the input 
    #     ndarrays since we do not have geometry information.
    mosaic_data = mo.mosaic_image_data()

    #     If using the above input data list, the output
    #     shloud be (2048,2048).
    print mosaic_data.shape

    list_average=[numpy.average(data_list[k]) for k in range(4)]
    print 'Input data average is:',numpy.average(list_average)
    print ' The expected result is: 0.0'

    print 'Output mosaic average:',numpy.average(mosaic_data)
    print ' The expected result is: 0.0'

def at4():
    """
    This test creates a list of 4 ndarrays where one of them as 
    a different shape than the other 3. This is input to MosaicData
    which raises an exception.
    """
    import numpy
    from gempy.library.mosaic import MosaicData

    print '\n at4 REQUIREMENT.......'
    print ('*****MosaicData shall raise an exception when given a list of ndarrays of  different shapes')

    #  Make an ndarray of shape (2048,1024) of pixel values between 0 #  and 1000.
    data = numpy.linspace(0.,1000.,1024*2048).reshape(2048,1024)
    data_list = [data*(-1)**k for k in numpy.arange(4)]
    data_list[1] = data_list[1][:,:1000]     # Change shape
    print "\n >>>>>The expected result is an exception with the message:"
    print ">>>>>MosaicData:: 'data_list' elements are not of the same size."
    #  Now create the object
    md = MosaicData(data_list)


def at5():
    """
    The test instantiates a MosaicGeometry object from an input dictionary,
    a MosaicData object from a set of 4 data ndarrays and with these 2 object
    as input, instantiates a Mosaic object.
    """

    
    import numpy as np
    from gempy.library.mosaic import Mosaic,MosaicGeometry, MosaicData
    try:
        from stsci.numdisplay import display
    except ImportError:
        from numdisplay import display

    print '\n at5 REQUIREMENT.......'
    print ('*****MosaicData shall correctly use the geometry dictionary to mosaic the input ndarrays.')

    geometry = {
    'transformation': {
    'shift':         [(0.,0.), (-10,20), (-10,20), (0,0)],
    'rotation':      (0.0,     0.0,        45.0,       45.),
    'magnification': (1.,      1.,      1.,           .5),
    },
    'interpolator': 'linear',
    'gap_dict': { 'tile_gaps': {(0,0):(20,30),(1,0):(20,30),
                        (0,1):(20,30),(1,1):(20,30)}, 
            'transform_gaps': {(0,0):(20,30),(1,0):(20,30),
                        (0,1):(20,30),(1,1):(20,30)}, 
            },
    'blocksize':    (200,300),
    'ref_block':    (0,0),      # 0-based
    'mosaic_grid':  (2,2) }

    mosaic_geometry = MosaicGeometry(geometry)

    #   Make a rectangle (200,300) (wide,high).
    data = np.ones((300,200),dtype=np.float32)
    data = data*20   # make the background 20

    #   Make a 2x2 array with this rectangle
    data_list = [data,data,data,data] 

    # Inside each block, make a small box 50x50
    # starting at (50,50) with value 100
    for k in range(4): 
          data_list[k][50:101,50:101] = 100. 
          # Mark the block borders with value 400
          data_list[k][:,0]  =400 
          data_list[k][:,199]=400 
          data_list[k][0,:]  =400 
          data_list[k][299,:]=400
     
    #     Now create the MosaicData object
    mosaic_data = MosaicData(data_list)

    #      With these two objects we instantiate a Mosaic object
    mo = Mosaic(mosaic_data, mosaic_geometry)

    #      Finally make the mosaic
    mosaic_data = mo.mosaic_image_data()

    #  Success Criteria 1.
    ref_block = mo.geometry.ref_block
    blksz_x,blksz_y = mo.geometry.blocksize
    print (int(mosaic_data.shape[0]/blksz_y),  int(mosaic_data.shape[1]/blksz_x))

    #   Success Criteria 2.
    x_gap,y_gap= mo.geometry.gap_dict['tile_gaps'][ref_block]
    nblkx,nblky = mo.geometry.mosaic_grid
    print 'nx:', nblkx*blksz_x + (x_gap)*( nblkx-1)
    print 'ny:', nblky*blksz_y + (y_gap)*( nblky-1)

    #   Success Criteria 3.
    # Now display the mosaic and the mask
    display(mosaic_data,frame=1,z1=0,z2=400.5) 

    # Results:
    # 1) Lower left block, top right corner of 
    #    test box: (101,101)
    # 2) Lower right block, top right corner
    #    of test box: (200+20-10+101, 101+20) or (311,121); 
    #    gap: (20,0), shift: (-10,20)
    # 3) Top left block, top right corner of 
    #    test box gets rotated 45 degrees w/r to the
    #    block center: (50*cos(45),(50-50*sin(45)) or (35.4,14.6)
    #    in distance change of this corner. Now the new 
    #    positioning with respect to the mosaic origin is
    #    (135.4-10,300+20+30+114.6) or (125.4, 464.6);
    #    gap: (0,30), shift: (-10,20)
    # 4) top right block, top right corner of 
    #    test box gets rotated 45 degrees but we have a
    #    magnification of 0.5 so the distance between the
    #    block center and the top right corner is 25 now.
    #    The distance change for rotation is then: (17.7,7.4)
    #    This corner position with no rotation is (100,125)
    #    now adding the rotation we have: (117.7,132.4).
    #    The position w/r to the mosaic origin is:
    #    (200+20+117.7, 300+30+132.4) or (337.7, 362.4)

def at6():
    """
    Please run test AT-mosaic-5. We display the mask and verify the values in the
    image display. The position of the test box corners should coincide with those 
    in the   mosaic image display in test AT-mosaic-5.
    """
    
    try:
        from stsci.numdisplay import display
    except ImportError:
        from numdisplay import display

    print '\n at6 REQUIREMENT.......'
    print ('*****The system shall create a mask ndarray with value 1 for no-data areas')
    print (' .........NEED TO RUN at5......')
    

    # Using the Mosaic object, display the mask
    #  
    # UNCOMMENT display after running at5
    #display(mo.mask,frame=2,z1=0,z2=1.5) 

    # MASK
    # Please position the ds9 pointer on the left 
    # corner of each test box
    # and verify that their's (x,y) values are:
    # 1) Lower left block, top right corner of 
    #    test box: (101,101)
    # 2) Lower right block, top right corner
    #    of test box:  (311,121); 
    # 3) Top left block, top right corner of 
    #    test box:  (125.4, 464.6)
    # 4) top right block, top right corner of 
    #    test box:  (337.7, 362.4)
    #    pixel values in the blocks is zero and everywhere else
    #    one, except for blocks and test boxes boundaries where
    #    the values are zero.

def at7():
    """
    AT-mosaic-7 Verify that Mosaic can return a given block.

    The test creates a MosaicGeometry object from just 2 input values a blocksize and 
    mosaic_grid tuples. Then it creates a MosaicData object with 4 data ndarrays
    and a  dictionary names 'coords' with coordinates information about the input data. 
    For Success Criteria 1), we create coords['amp_mosaic_coord'] such that
    we have 1 amplifier per block. 
    For Success Criteria 2), we create coords['amp_mosaic_coord'] such that
    we have 2 amplifiers per block
    """

    import numpy
    from gempy.library.mosaic import Mosaic,MosaicGeometry, MosaicData

    print '\n at7 REQUIREMENT.......'
    print ('*****The system shall return an ndarray from a given block number')

    geometry = {'blocksize':(1024,2048),'mosaic_grid':(4,1)}

    geometry = MosaicGeometry(geometry)

    # Now make four data ndarrays: (2048,1024) with pixel values
    # between 0 and 1000.

    data = numpy.linspace(0.,1000.,1024*2048).reshape(2048,1024)
    data_list = [data*(-1)**k for k in numpy.arange(4)]

    # Here is the coordinates description for each of the data
    # ndarray. The order is (x1,x2,y1,y2).
    #
    # Success Criteria 1.
    # Set 'amp_mosaic_coord' for Success Criteria 1)
    coords = {'amp_mosaic_coord': 
                  [(0, 1024, 0, 2048),   (1024, 2048, 0, 2048),
                   (2048, 3072, 0, 2048), (3072, 4096, 0, 2048)],
              'amp_block_coord': 
                   [(0, 1024, 0, 2048), (0, 1024, 0, 2048),
                    (0, 1024, 0, 2048), (0, 1024, 0, 2048)]}

    #     Now create the data object
    mosaic_data = MosaicData(data_list,coords)
   
    #    Instantiates Mosaic object with these 2 input objects
    mo = Mosaic(mosaic_data, geometry)


    #      Finally get the given block.
    block_data = mo.mosaic_image_data(block=(1,0)) 
    print 'Success Criteria 1..................'
    print 'Median value of input block(1,0):',numpy.median(data_list[1])
    print 'Median value of output block:', numpy.median(block_data)
    #     Print the shape: (height, width) in pixels. This should be
    #     (2048,1024)
    print 'Output block shape should be(2048,1024):'
    print block_data.shape
    #
    print '\nStarting Criteria 2 ...................'
    # Success Criteria 2.
    # Setting up 'amp_mosaic_coord' 
    # Make 2 amplifiers per block. We need 8 input ndarrays
    data = numpy.linspace(0.,1000.,512*2048).reshape(2048,512)
    data_list = [data*(-1)**k for k in numpy.arange(8)]
    amp_mosaic_coord = []
    amp_block_coord = []
    coords={}
    for k in range(8):
        x1 = 512*k
        bx1 = 512*(k%2)
        amp_mosaic_coord.append((x1,x1+512,0,2048))
        amp_block_coord.append((bx1,bx1+512,0,2048))
    coords['amp_mosaic_coord'] = amp_mosaic_coord
    coords['amp_block_coord'] = amp_block_coord

    #     Now create the data object
    mosaic_data = MosaicData(data_list,coords)

    #    Instantiates Mosaic object with these 2 input objects
    mo = Mosaic(mosaic_data, geometry)
    #      Finally get the given block.
    block_data = mo.mosaic_image_data(block=(1,0)) 

    print 'Success Criteria 2..................'
    print 'Median value of input block(1,0):',numpy.median(data_list[1])
    print 'Median value of output block:',numpy.median(block_data)
    #     Print the shape: (height, width) in pixels. This should be
    #     (2048,1024)
    print 'Output block shape should be(2048,1024):'
    print block_data.shape

def at8():
    """
    Create a 2 ndarrays list and mark a strip of the 2nd ndarray with a higher value.
     Set the Geometry dictionary with 'rotate' this ndarray by 5 degrees. Now we create
     the mosaic with  default interpolation function and again with the 'spline' function
     of order 5. We plot a column from each mosaic.
    """
    
    import numpy as np
    from gempy.library.mosaic import Mosaic,MosaicGeometry, MosaicData
    from matplotlib import pyplot as pl

    print '\n at8 REQUIREMENT.......'
    print ('*****It must be possible to select a different interpolator function')

    geometry = {
    'transformation': {  # shift and magnification will 
                   # have default values
         'rotation':      (0.0,     5.),
                      },
    'blocksize':   (100,100),
    'mosaic_grid':  (2,1) 
               }

    mosaic_geometry = MosaicGeometry(geometry)

    #   Make an ndarray
    data = np.zeros((100,100),dtype=np.float32)

    #   put a stripe of 5 rows with value 5
    data[45:50,:] = 5

    #   Make a 2x1 array with this rectangle.
    data_list = [data,data] 
    mosaic_data = MosaicData(data_list)

    #   With these two objects we instantiate a Mosaic object
    mo = Mosaic(mosaic_data, mosaic_geometry)

    #   Finally make the mosaic
    mosaic_linear = mo.mosaic_image_data()

    #   Now reset the interpolator function
    mo.set_interpolator('spline',5)
 
    #   Create the mosaic. 
    mosaic_spline = mo.mosaic_image_data()

    #   Now plot one column across the stripes
    pl.plot(mosaic_linear[:,140])

    #   The transformation of block two to the reference
    #   block (one) was done now using a spline interpolator
    #   of order 5 which will create edge effects as seen in
    #   this plot.
    pl.plot(mosaic_spline[:,140])
    #
    #   The tester should see the following values:
    #   mosaic_linear[43] is  0.0 
    #   mosaic_spline[43] is -0.3
    #
    #   mosaic_linear[48] is 5.0
    #   mosaic_spline[48] is 5.3
    jj = raw_input("Press Enter to exit the test")
    pl.close()

def atd1():
    """
    With a GMOS AstroData object, the test instantiates a MosaicAD object
    containing 'coords' as one of the attributes. The test verify that 
    coords['amp_mosaic_coord'] and ad['SCI'].detector_array.as_dict() values
    match.
    """
   
    print '\n atd1 REQUIREMENT.......'
    print ('*****MosaicAD shall instantiate an object from a supported AstroData objec')
 
    gmos_file = '../data/gS20120420S0033.fits'
    gsaoi_file = '../data/guS20110324S0146.fits'
    nongem_file='../data/kp620765.fits'
        
    from astrodata import AstroData
    from gempy.adlibrary.mosaicAD import MosaicAD
    #     This is the default Mosaic function
    from gempy.mosaic.gemMosaicFunction import gemini_mosaic_function

    #    Success Criteria 1. (GMOS data)
    #    The tester should provide her/his own GMOS file.
    for file in [gmos_file,gsaoi_file,nongem_file]:
        ad = AstroData(file)
        print '\n ...........CASE for:',file,ad.instrument()

        #    Create the MosaicAD object
        mo = MosaicAD(ad,gemini_mosaic_function)

        #    print DETECTOR values for all the 'SCI' extensions as
        #    a dictionary.
        print ad['SCI'].detector_section().as_dict()
          
        #    print the 'amp_mosaic_coord' key value from the 'coords'
        #    attribute. This list is in increasing order of extver.
        print mo.coords['amp_mosaic_coord']


def atd2():
    """
    AT-mosaicAD-1  Verify that mosaicAD can create a mosaic from extensions of
    a given name.
    The test uses the MosaicAD method mosaic_image_data  to create a mosaic 
    using the  extension name 'SCI'.
    """
    
    import numpy as np
    from astrodata import AstroData
    from gempy.adlibrary.mosaicAD import MosaicAD
    #    This is a user's Mosaic_function
    from gempy.mosaic.gemMosaicFunction import gemini_mosaic_function

    print '\n atd2 REQUIREMENT.......'
    print ('***** From a given extension name in the input AstroData object, the system'
            ' shall create a    mosaic')

    gmos_file = '../data/gS20120420S0033.fits'

    ad = AstroData(gmos_file)
    mo = MosaicAD(ad, gemini_mosaic_function)
    #    Now use the mosaic_image_data method to create
    #    the mosaic ndarray.
    mosaic_data = mo.mosaic_image_data(extname='SCI') 

    #    Get blocksize, gap_list and number of blocks in x and y 
    #    values from the MosaicGeometry object.
    blksz_x,blksz_y = mo.geometry.blocksize
    gap_dict = mo.geometry.gap_dict
    nblkx,nblky = mo.geometry.mosaic_grid

    #    Success Criteria 1.
    mszx,mszy=0,0
    for k in range(nblkx):
        gx,gy = gap_dict['transform_gaps'][(k,0)]
        mszx += blksz_x + gx
    for k in range(nblky):
        gx,gy = gap_dict['transform_gaps'][(0,k)]
        mszy += blksz_y + gy

    #    print the shape of the resulting mosaic and  (ny,nx)
    print mosaic_data.shape,' should be equal to:',(mszy,mszx)

    #    Success Criteria 2.
    #    Check the mean of the input AD for each 'SCI' extension
    in_mean = []
    for ext in ad['SCI']:
        in_mean.append(np.mean(ext.data))
    print 'Mean of input data:',np.mean(in_mean)

    #    Get the coordinates of data areas. Does not count 
    #    gaps nor no-data areas.
    g = np.where(mo.mask == 0)
    print 'Mean of mosaic:',np.mean(mosaic_data[g])

def atd3():
    """
     Verify that MosaicAD can merge associated binary tables

     Create a mosaic from the input AD object. It is up to the tester to 
     see if there is one    IMAGE extension name 'SCI'  and one BINTABLE
     extension with the same number    and values of EXTVER -these are 
     associated. The as_astrodata method creates the     mosaic. Please
     see the Procedure for the steps performed to verify the correctness
     of the merging.  

     Resources:
     1) gmos_file = '../data/N20120121S0175_ccMeasured.fits'
     2) Uses pysao
      
    """
        
    from astrodata import AstroData
    from gempy.adlibrary.mosaicAD import MosaicAD
    #     This is the default Mosaic function
    from gempy.mosaic.gemMosaicFunction import gemini_mosaic_function

    print '\n atd3 REQUIREMENT.......'
    print ('***** Given an AstroData object with associated binary table, the system '
           'shall merge the tables')

    gmos_file = '../data/N20120121S0175_ccMeasured.fits'
    ad = AstroData(gmos_file)
    # 1) Make sure that we have an IMAGE and BINTABLE extension
    #    with the same number and values of extver in one IMAGE and
    #    BINTABLE extension.
    # 2) Creates a mosaicAD object using the input AD and the 
    #    default mosaic function named gemini_mosaic_function.
    # 3) The 'column_names' parameter in MosaicAD has as 
    #    default values the X_IMAGE, Y_IMAGE and X_WORLD, 
    #    Y_WORLD column names. The reference catalog have 
    #    RAJ2000 and DEJ2000. For more information please the 
    #    documentation about MosaicAD.

    mo = MosaicAD(ad, gemini_mosaic_function)

    #    Now  create the output AstroData object.
    outad = mo.as_astrodata()

    #    Check that the output table has been merged correctly by 
    #    verifying that the duplicates rows are deleted and that the
    #    new X,Y values for objects have been calculated correctly
    #    based on the new WCS.
    #
    #    Get input rows and remove duplicates. Output the content
    #    of the resulting AD object which should show the 
    #    associated IMAGE and BINTABLE extensions.

    print outad.info()

    #    Save to a filename
    from os import path
    nn = path.basename(ad.filename)
    outad.write(path.splitext(nn)[0]+'_AD.fits',clobber=True)

    #    Verify that (X,Y) values in the merged tables correspond
    #    the object positions. 
    #    The output catalog table name should be 'OBJCAT' and
    #    pixel coordinates X_IMAGE and Y_IMAGE with corresponding
    #    world coordinates Y_WORLD and Y_WORLD.

    tab = outad['OBJCAT'].data

    #    Form a list of tuples (x_pixel,y_pixel)
    xy=[(x,y) for x,y in zip(tab.field('X_IMAGE'), tab.field('Y_IMAGE'))]

    #    Now you can use any program to draw points and check for
    #    location of the xy points on the objects.

    #    Using DS9 PYSAO module
    #    Assuming you have the module in your PYTHONPATH
    import pysao

    # Bring up a ds9 display by instantiating the pysao object
    ds9 = pysao.ds9()

    # Display the image
    ds9.view(outad['SCI'].data)

    # ***NOTE***
    #    Use file tab to display the 'SCI' extension of file
    #    you save previously. Check for some of these pixel 
    #    coordinates with DS9.
    #    Now draw the point

    tmp=[ds9.set_region('arrow point '+str(x)+' '+str(y)) for x,y in xy]

    #    Verify for correctly removing duplicates when merging. 
    #    Use python sets()
    #    Get the first all extensions from the input 
    #    AstroData object. Form a list of tuples (ra,dec) 
    #    from the reference tables in all extensions.

    # test for 'REFCAT' existance
    if ad['REFCAT'] == None:
        raise ValueError('"REFCAT" extension is not in the AstroData object')

    rd = []
    for tab in ad['REFCAT']:
       rd += [(x,y) for x,y in zip(tab.data.field('RAJ2000'),\
                      tab.data.field('DEJ2000'))]

    #    Turning rd list to a set will eliminate duplicates.
    print 'Number of unique elements from the list:',len(set(rd))

    #    Now from the outad object
    tab = outad['REFCAT'].data
    radec=[(x,y) for x,y in zip(tab.field('RAJ2000'), tab.field('DEJ2000'))]

    #    The number of elements in the output table
    print 'The number of elements in the output table:',len(radec)
    jj = raw_input("Press Enter to exit the test")

def atd4():
    """
    Verify that a  mosaicAD class  method can create a tiled array from 
    extensions of a given name.

    The test creates a mosaic ndarray using the method mosaic_image_data 
    with the parameter 'tile=True'  which avoids the transformation step.

    """
    from astrodata import AstroData
    from gempy.adlibrary.mosaicAD import MosaicAD
    #     This is the default Mosaic function
    from gempy.mosaic.gemMosaicFunction import gemini_mosaic_function

    print '\n atd4 REQUIREMENT.......'
    print ('***** Given an AstroData object, the system shall tile all IMAGE extensions '
          'matching a given extension name')

    gmos_file='../data/gS20120420S0033.fits'
    gsaoi_file='../data/guS20120413S0048.fits'

    ad = AstroData(gmos_file)
    mo = MosaicAD(ad, gemini_mosaic_function)
    #     Now use the mosaic_image_data method to create
    #     the mosaic tile array from the 'SCI' extname.
    tile_data = mo.mosaic_image_data(tile=True, extname='SCI') 

    # ----- Comparing input and output. GMOS image 

    #     The tester should feel free to verify any input and output
    #     pixel location.

    #     For example: A GMOS image:
    #     The lower left corner (2x2) pixels the first GMOS 
    #     data extension. For a GSAOI is the second extension

    corner_gmos = ad['SCI',1].data   # For a GMOS file    
    print 'ad["SCI",1].data[:2,:2]\n',corner_gmos[:2,:2]

    #     From the output mosaic. We should get the same values.
    print 'tile_data[:2,:2]\n',tile_data[:2,:2]

    # The top right corner of the mosaic
    nexts = ad.count_exts('SCI')
    block = ad['SCI',nexts].data    # There is one amp per block
    print '\nad["SCI",last].data[-2:-2]\n',block[-2:,-2:]

    # The mosaic top corner
    print '\ntile_data[-2:,-2:]\n',tile_data[-2:,-2:]

    # ----- GSAOI data
    ad = AstroData(gsaoi_file)
    mo = MosaicAD(ad, gemini_mosaic_function)

    #     Now use the mosaic_image_data method to create
    #     the mosaic tile array from the 'SCI' extname.
    tile_data = mo.mosaic_image_data(tile=True, extname='SCI') 

    print '\nGSAOI data'
    corner_gsaoi = ad['SCI',2].data   # For a GSAOI file    
    print 'ad["SCI",2].data\n',corner_gsaoi[:2,:2]
    print 'tile_data[:2,:2]\n', tile_data[:2,:2]

    #     The top right corner of the mosaic
    block4 = ad['SCI',4].data    # There is one amp per block
    print '\nblock4[-2:,-2:]\n',block4[-2:,-2:]
    print 'tile_data[-2:,-2:]\n',tile_data[-2:,-2:]

def atd5():
    """
    Verify that mosaicAD gives the correct WCS information for the mosaiced data.

    Given a GMOS input file, the MosaicAD object method as_astrodata
    creates an output AstroData object. This object 'SCI' header have the
    CRPIX1 and CPRIX2 for the reference extension header. The value
    CRPIX1  should match the value explained in the Success Criteria
    section. The value CRPIX2 is unchanged.
    
    Resources:
    gmos_file='../data/gS20120420S0033.fits'
    gsaoi_file='../data/guS20120413S0048.fits'

    ds9 running        

    """
    import pywcs
    try:
        from stsci.numdisplay import display
    except ImportError:
        from numdisplay import display

    print '\n atd5 REQUIREMENT.......'
    print ('***** Given an AstroData object, the system shall update the header keywords '
         ' CRPIX1 and CRPIX2  in the output mosaiced AD object to match the requested '
            'transformations')

    gmos_file='../data/gS20120420S0033.fits'
    gsaoi_file='../data/guS20120413S0048.fits'

    from astrodata import AstroData
    from gempy.adlibrary.mosaicAD import MosaicAD
    #    This is the default Mosaic function
    from gempy.mosaic.gemMosaicFunction import gemini_mosaic_function

    ad = AstroData(gmos_file)
    #    Creates a mosaicAD object using the input ad and the
    #    default mosaic function name gemini_mosaic_function.
    #    'SCI' is the default extname.
    mo = MosaicAD(ad, gemini_mosaic_function)

    #         
    outad = mo.as_astrodata()

    # NOTE: The ref_ext is the left most amplifier in
    #       reference block. For GMOS the reference block
    #       (2,1). E.G. for a 6-amp GMOS exposure the left
    #       most exposure is 3.
    refblk = mo.geometry.ref_block
    col,row = refblk[0], refblk[1]
    amp_per_block = mo._amps_per_block
    ref_ext = col*amp_per_block+1
    ocrpix1 = ad['SCI',ref_ext].header['CRPIX1']
    xgap = mo.geometry.gap_dict['transform_gaps'][col,row][0]
    blksz_x,blksz_y = mo.geometry.blocksize

    #    Success Criteria 1.

    #    Get the x2 value from coords['amp_mosaic_coord'][refblk]
    xoff = mo.coords['amp_mosaic_coord'][max(0,col-1)][1]
    print ocrpix1 + xoff + xgap, 'should match: '
    print outad['SCI',1].header['CRPIX1']

    #    Success Criteria 2.
    #    For a GSAOI file, 
    ad = AstroData(gsaoi_file)

    if ad.instrument() != 'GSAOI':
        print '******** file is not GSAOI ************'
    mo = MosaicAD(ad, gemini_mosaic_function)
    outad = mo.as_astrodata()
    outhdr = outad['SCI'].header 
    inhdr = ad['SCI',2].header 

    #    The values should be the same.
    print 'Crpix1 values (in,out):',inhdr["CRPIX1"],outhdr['CRPIX1']
    print 'Crpix2 values (in,out):',inhdr["CRPIX2"],outhdr['CRPIX2']

    #    Success Criteria 3.
    #    For a GMOS file in extension #2
    hdr = ad['SCI',2].header

    wcs = pywcs.WCS(hdr)

    import pysao

    # Bring up a ds9 display by instantiating the pysao object
    ds9 = pysao.ds9()
    # Display the image
    ds9.view(ad['SCI',2].data, frame=1)

    # display(ad['SCI',2].data,frame=1)
    print 'Click on any object:'
    X,Y,f,k = ds9.readcursor()

    #    Get X,Y values from an object on the ds9 display
    #    Get ra,dec
    ra,dec = wcs.wcs_pix2sky(X,Y,1)

    #    Generate the mosaic_data for this ad using 
    #    as_astrodata method.
    #    Display the mosaic mosaic_data for the 'SCI' extension

    mosaic_data = mo.mosaic_image_data()
    # Display the mosaic
    ds9.view(mosaic_data,frame=2)

    # display(ad['SCI',2].data,frame=1)
    print 'Click on the same object:'
    MX,MY,f,k = ds9.readcursor()

    #display(mosaic_data,frame=2)

    #    Measure X,Y of the same object, named this MX,MY
    #    Get the wcs from the mosaic header

    mhdr = outad['SCI'].header
    mwcs = pywcs.WCS(mhdr)
    mra,mdec = mwcs.wcs_pix2sky(MX,MY,1)
    print 'These RA,DEC should be pretty close:',(ra[0],mra[0]),(dec[0],mdec[0])

def atd6():
    """
    Verify that a MosaicAD method can create a block from a given extension name.

    The test creates a mosaic ndarray from the MosaicAD method mosaic_image_data 
    with the block parameter value (0,0), indicating to output the lower left block.

    NOTE: Having one amp per block, the actual extension data is not the same 
          as the block since it would be trim by the DATASEC image section.

    gmos_file='../data/gS20120420S0033.fits'
    gsaoi_file='../data/guS20120413S0048.fits'
    """
        
    from astrodata import AstroData
    from gempy.adlibrary.mosaicAD import MosaicAD
    #    This is the default Mosaic function
    from gempy.mosaic.gemMosaicFunction import gemini_mosaic_function
    print '\n atd6 REQUIREMENT.......'
    print ('***** From a given AstroData object, the system shall create a block from '
          'a given extension name')

    gmos_file='../data/gS20120420S0033.fits'
    gsaoi_file='../data/guS20120413S0048.fits'

    for file in [gmos_file,gsaoi_file]:
        ad = AstroData(file)
        print 'Instrument: ',ad.instrument()
        mo = MosaicAD(ad, gemini_mosaic_function)
        #    Now use the mosaic_image_data method to generate
        #    an output block by using the parameter block and
        #    value as a tuple (col,row) (0-based) of the block
        #    you want returned. For GMOS the block values are
        #    (0,0), (1,0), (2,0).

        block=(1,0)
        block_data = mo.mosaic_image_data(block=block,tile=True)
         
        #    Get the shape: (height, width) in pixels.
        print 'block_data.shape:',block_data.shape

        extn = block[0] + block[1]*mo.geometry.mosaic_grid[0] + 1
        print 'Input shape for 1-amp per detector:',ad['SCI',extn].data.shape

        #    Check values of the lower 2x2 pixels
        print 'Output block [0:2,0:2] pixels:\n',block_data[:2,:2]
        if ad.instrument() == 'GSAOI':
            # GSAOI FITS extension 1 correspond to block (1,0)
            # and extension 2 to block (0,0). DETSEC image section
            # indicates this.
            extn = [2,1,3,4][extn-1]

        # To get the correct segment we need to look at
        # DATASEC, in case the data is trimm -as it appears in data_list.
        x1,x2,y1,y2 = ad.data_section().as_dict()['SCI',extn]
        print 'Input amp DATASEC[x1:x1+2 pixels:\n',\
              ad['SCI',extn].data[x1:x1+2,y1:y1+2]
        print '\n'

            
def atd7():
    """
    Verify that blocks are laid out correctly


    NOTE: This test depends on gmosaic.cl and gamosai.cl. We can use 
          atd5 and meet the requirement.
          
    """
    print '\n atd7 REQUIREMENT.......'
    print ('***** Verify that blocks are laid out correctly')
    print ('***** THIS IS ALREADY COVERED BY atd5')

def atd8():
    """
    From a given AstroData object, the system shall offer an option to
    prevent the creation of merged table associations.

    The test creates an output AstroData object using the method 
    as_astrodata with the     parameter 'return_associated_bintables'
    set to False which prevents the creation of     associated binary 
    tables to the reference image extension name.

    Resources:

    file: N20120121S0175_ccMeasured.fits. Contains SCI,VAR,DQ
         OBJCAT (Bintable) and REFFACT (Bintable)
    """
    
    from astrodata import AstroData
    from gempy.adlibrary.mosaicAD import MosaicAD
    #    This is the default Mosaic function
    from gempy.mosaic.gemMosaicFunction import gemini_mosaic_function

    print '\n atd8 REQUIREMENT.......'
    print ('***** From a given AstroData object, the system shall offer an option to prevent the creation of    merged table associations')

    file = '../data/N20120121S0175_ccMeasured.fits'
    ad = AstroData(file)
    #    Creates a mosaicAD object using the input ad
    #    and the default mosaic function named 
    #    gemini_mosaic_function. 'SCI' is the default extname
    mo = MosaicAD(ad, gemini_mosaic_function)
    outad = mo.as_astrodata()
    print '.......OUTPUT AD with all extensions'
    print outad.info()

    outad = mo.as_astrodata(return_associated_bintables=False)

    #    The tester should see that there is no BINTABLE 
        #    extension name associated with the reference image 
          #    extension name in this output.
    print '.......OUTPUT AD with no associated BINTABLE extensions'
    print outad.info()

def atd9():
    """
    Verify that transformation preserve flux.

    The test creates a mosaic from an input AstroData object and displays it 
    in the DS9 display. Using IRAF imexam we measure objects that are on the
    the middle block for GMOS or the lower left block for GSAOI. We also measure
    the same objects in the input amplifers. We calculate the magnitude difference.

    """
    from astrodata import AstroData
    from gempy.adlibrary.mosaicAD import MosaicAD
    #    This is the default Mosaic function
    from gempy.mosaic.gemMosaicFunction import gemini_mosaic_function
    try:
        from stsci.numdisplay import display
    except ImportError:
        from numdisplay import display

    print '\n atd9 REQUIREMENT.......'
    print ('***** From a given AstroData object, the system shall conserve flux'
           ' after "transforming"')

    ad = AstroData(file)
    #    Now make a mosaicAD object using the input ad
    #    and the default mosaic function named 
    #    gemini_mosaic_function.
    mo = MosaicAD(ad, gemini_mosaic_function)

    mosaic_data = mo.mosaic_image_data()
    display(mosaic_data,frame=1)

    #    display input amplifier 3. Assumning that there is one
    #    amplifier per block
    display(ad['SCI',3].data,frame=2)

    #    Now use IRAF imexam to measure the magnitude for some
    #    object and calculate their difference.

if __name__ == '__main__':

   import sys
   if len(sys.argv) == 2:
       print '\n'
       eval(sys.argv[1]+'()')
       sys.exit(0)

   [eval('at'+str(k)+'()') for k in range(1,4)]
   try:
     at4()
   except:
     pass
   # Continue 
   [eval('at'+str(k)+'()') for k in range(5,9)]
   try: atd1()
   except: pass
   [eval('atd'+str(k)+'()') for k in range(2,10)]
