#
import numpy as np
import transformation as tran

class Mosaic(object):
    """ 
      Mosaic is a base class that provides functionality to layout a list 
      of data ndarrays of the same size into an output mosaic. The main 
      characteristics are:

      - Input data and their location in the output mosaic is done via 
        MosaicData objects.

      - Information about gaps between the blocks and transformation is 
        given by the MosaicGeometry  object.

      - Mosaic object can generate masks associated with the output mosaic

      - Can reset the interpolated function use in tranformation via Mosaic
        class function.

      - Preserving flux when transforming a block

    """

    def __init__(self,mosaic_data, mosaic_geometry=None):
        """
          USAGE
                mosaic=Mosaic(mosaic_data, mosaic_geometry)

          Create a list of ordered ndarrays that form the mosaic. The 
          ordering system is given by list of coordinates describing
          the layout of the ndarrays into block and the layout of 
          the block into mosaics.

          Parameters
          ----------
         
            **mosaic_data** 
                 MosaicData class object containing the data_list and list
                 of coordinates. The members of this class are:

                 data_list:
                   List of ndarrays representing the pixel image data,
                   all the elements are of the same shape.
            
                 coords: See Attributes.

            **mosaic_geometry** 
                 MosaicGeometry class object that will contain in 
                 a standard way the relevant information to perform
                 the requested mosaic. mosaic_geometry is optional.
		 See class definition below for more details.
                   

          Attributes
          ----------
          
            **coords** 
                   A dictionary with keys 'amp_mosaic_coord' and 'amp_block_coord'. 
                   The 'amp_mosaic_coord' values contain a list of tuples 
                   describing the corners of the ndarrays, i.e.,  
                   (x1,x2,y1,y2) with respect to the mosaic lower left 
                   corner (0,0). The 'amp_block_coord' values contain a list
                   of tuples. describing the corners of the ndarrays, 
                   i.e.,  (x1,x2,y1,y2) with respect to the block
                   lower left corner (0,0). Notice that we can have more 
                   than one ndarray per block.

            Other attributes are explained below.

            Methods
            -------
          
             **mosaic_image_data:**
                   Layout the data blocks in a mosaic grid
                   correction for rotation, translation and
                   magnification is performed. Returns the mosaic.

             **set_transformations:**
                  Instantiates the Transformation class objects for
                  each block that needs correction for rotation, shift
                  and/or magnification. Set a dictionary with (col,row)
                  as a key and value the Transformation object.

             **set_blocks:**
                   Initializes the block order, amplifier's indexes
                   in blocks and block coordinates.

             **get_blocks:**
                   Return a dictionary of block data arrays using their
                   mosaic grid (column,row) position as keys. Data blocks
                   are necessary when applying transformation.
             
          EXAMPLE
          -------
          >>> from gemp.library.mosaic import Mosaic
              # Now import you own 'my_mosaic_function' to compose the input
              # data_list coordinate descriptors and the geometry object. 
          >>> from my_module import my_mosaic_function  
          >>>
          >>> modata, geometry = my_mosaic_function('filename.fits')
          >>> mo = Mosaic(modata, geometry)
          >>> outdata = mo.mosaic_image_data()


        """
        # Set attributes
        self.mosaic_data = mosaic_data           # For use in calling methods.
        self.data_list = mosaic_data.data_list  # Lists of ndarrays with 
                                               # pixel data.

        # Check whether we can initialize coords
        mosaic_data.init_coord(mosaic_geometry)
        self.coords = mosaic_data.coords    # Dictionary with keys
                                           # 'amp_mosaic_coord','amp_block_coord'.
        # Generate default geometry object given 
        # the coords dictionary.
        if mosaic_geometry == None:            # No geometry object.
            mosaic_geometry = self.set_default_geometry() 

        # We migth have a None value for mosaic_geometry if self.coords is None.
        self.geometry = mosaic_geometry

        # If both objects are not None set up block attributes.
        if not ((mosaic_geometry==None) and (self.coords==None)):
            self.set_blocks()             

            # Check the mosaic_data and mosaic_geometry attribute
            # values for consistency.
            #
            self.verify_inputs()

            # Now that we have coords, get the order of the amps
            # according to x and y position. 
            self.coords['order'] = mosaic_data.get_coords_order()   

        self.return_ROI = True      # Boolean to set a minimum area enclosing
                                    # all the data_list elements in the mosaic.

        self.mask = None            # Mask array for the resulting mosaic.
                                    # 0: good data, 1: no-data

        self.__do_mask = True       # Hidden flag. To be set to False when we
                                    # have the mask for the reference extension
                                    # already made.
        self.transform_objects = None   # When set by set_transformations it is
                                        # a dictionary with keys (col,row) of the
                                        # mosaic block location.
        self.as_iraf = True         # Set the Transformation class method
                                    # behaves the same way as the IRAF geotran
                                    # task.

    def set_default_geometry(self):
        """
          We are here because the object MosaicGeometry instantiated
          in Mosaic __init__ function is None.
 
        """
        if not hasattr(self,'coords'):
            raise ValueError("From set_default_geometry: 'coords' not defined.")

        if self.coords == None:
            return

        # The coords dictionary is defined.
        # This is the list of positions w/r to the mosaic lower left (0,0) origin
        #
        amp_mosaic_coord = self.coords['amp_mosaic_coord']

        # This is the list of positions w/r to the block lower left (0,0) origin
        #
        bcoord = self.coords['amp_block_coord']

        # Set blocksize from amp_block_coord. Notice that we are taken the 
        # maximum value for each axis.
        dnx,dny = max([(cc[1],cc[3]) for cc in bcoord])
        blocksize = (dnx,dny)

        # Get minimum origin coordinates from all the data_list elements.
        x0, y0 = min([(dd[0],dd[2]) for dd in amp_mosaic_coord])

        # Calculate the mosaic array grid tuple (ncols,nrows).
        nblocksx, nblocksy = \
                max([((dd[1]-x0)/dnx,(dd[3]-y0)/dny) for dd in amp_mosaic_coord])

        # Set the attribute
        mosaic_grid =  (max(1,int(nblocksx)), max(1,int(nblocksy)))

        # This is the minimum dictionary necessary to create a
        # MosaicGeometry object
        geodict = {'mosaic_grid':mosaic_grid,'blocksize':blocksize}

        # Now instantiate the object
        return  MosaicGeometry(geodict) 

    def verify_inputs(self):
        """
          Verify that mosaic_data and geometry object atributes
          are consistent.
        """
        if (self.coords==None) and (self.geometry==None):
            return
 
        # coord1 ... amp_mosaic_coord
        # coord2 ... amp_block_coord

        # blocksize should be equal or larger that any value in coord2.
        #
        if self.coords == None:
            # We do not have coords but we do have geometry values.
            # Set coords1 and 2 then.
            dszy,dszx = self.data_list[0].shape
            datalen = len(self.data_list)
            bcoord = [(0,dszx,0,dszy) for k in range(datalen)]
            # amp_mosaic_coord: Make the sections run along the x-axis.
            mcoord = [(k*dszx,k*dszx+dszx,0,dszy) for k in range(datalen)]
            self.coords = {'amp_mosaic_coord':mcoord,'amp_block_coord':bcoord}
        else:
            bcoord = np.asarray(self.coords['amp_block_coord'])


        # Get the minimum and maximum value of corners
        # for each block data_list element.  
        #
        minx = min([cc[0] for cc in bcoord])
        maxx = max([cc[1] for cc in bcoord])
        miny = min([cc[2] for cc in bcoord])
        maxy = min([cc[3] for cc in bcoord])
        blcx, blcy = self.geometry.blocksize

        # Now check that given blocksize shape is consistent 
        # with these limits. 
        #
        if not((minx >= 0) and (maxx <= blcx) and 
            (miny >= 0) and (maxy <= blcy)):
            line1="Geometry tuple 'blocksize' does not agree with "
            line1 += "coordinates range."
            line2='\n\t\tblocksize_x:'+str(blcx)+', blocksize_y:'+str(blcy)
            line3='\n\t\tminX,maxX,minY,maxY:(%d,%d,%d,%d)'%(minx,maxx,miny,maxy)
            raise ValueError(line1+line2+line3)

    def set_transformations(self):
        """
          Instantiates the Transformation class objects for
          each block that needs correction for rotation, shift
          and/or magnification. Set a dictionary with (col,row)
          as a key and value the Transformation object.
        """
        # The correction parameters are coming from the 
        # MosaicGeometry object dictionary.
        # 
        geo = self.geometry
        nblocksx,nblocksy = geo.mosaic_grid
        mag = geo.transformation['magnification']
        rot = geo.transformation['rotation']   # (x_rot,y_rot) 
        shift = geo.transformation['shift']    # (x_shift,y_shift) 
        order = 1
        if geo.interpolator == 'spline':
            order = geo.spline_order

        transform_objects = {}
        # Use the keys valid (col,row) tuples as there could be some
        # tuples with no data hence no tuple.
        for col,row in self.data_index_per_block:
            # Turn block tuple location (col,row) into list index
            indx = col + row*nblocksx

            # Instantiates a Transformation object. The interpolator
            # order can be reset if needed.
            trf =  tran.Transformation(rot[indx][0], shift[indx],
                         mag[indx], order=order, as_iraf=self.as_iraf)
            # Add a key to the dictionary with value the
            # object.
            transform_objects[col,row] = trf
        # Reset the attribute
        self.transform_objects = transform_objects

    def mosaic_image_data(self, tile=False, block=None, return_ROI=True,
                          dq_data=False, jfactor=None):

        """
          Main method to layout the block of data in a mosaic grid. 
          Correction for rotation, shifting and magnification is performed with
          respect to the reference block. 
          A Mask is also created containing value zero for positions were
          there are pixel data and one for everywhere else -like gaps and
          areas of no-data due to shiftingr; for example, after transforming.

          Parameters
          ----------

          :param tile: If True, layout the block in the mosaic grid
                       with no correction for rotation nor shift. 
                       Gaps are included.
          :ptype tile: boolean, default: False

          :param block: Allows a specific block to be returned as the 
                        output mosaic. The tuple notation is (col,row)
                        (zero-based) where (0,0) is the lower left block.
                        The blocks layout is given by the attribute mosaic_grid.
          :ptype block: Tuple, default: None

          :param return_ROI: 
                        Flag to use the minimum frame enclosing all the block_data
                        elements. 
          :ptype return_ROI: Boolean, default: True
          :ptype dq_data: (bool). If True, then the input data is transformed
                        bit-plane by bit-plane. DQ is 8-bit planes so far.
          :param jfactor: Factor to multiply transformed block to conserve flux. 

          Output
          ------

          :return: An ndarray with the mosaiic. The Mask created is available
                   as an attribute with name 'mask'.


        """
        self.return_ROI = return_ROI
        geo = self.geometry

        # If we don't have coordinates then do Horizontal tiling
        if self.coords == None:
            return(np.hstack(self.data_list))

        # See if we want to return a block only.
        if block:
            return self.get_blocks(block).values()[0]

        # Number of blocks in (x,y)_direction in mosaic
        nblocksx,nblocksy = geo.mosaic_grid

        # Blocksize in x and y 
        blocksize_x, blocksize_y = geo.blocksize
        
        # Initialize multiplicative factor to conserve flux.
        if jfactor == None:
            jfactor = [1.]*blocksize_x*blocksize_y

        # Get gap dictionary for either tile or transform mode.
        if tile: 
            gap_mode = 'tile_gaps'
        else:   
            gap_mode = 'transform_gaps'
            
            # -- Set up the transform_object dictionary
            self.set_transformations()
           
        gaps = geo.gap_dict[gap_mode]

        # Get the maximum value for x_gap and y_gap to
        # determine the maximum mosaic size
        #
        gap_values = gaps.values()
        max_xgap = max([g[0] for g in gap_values])
        max_ygap = max([g[1] for g in gap_values])

        # This is the entire mosaic area.
        # Number of pixels in x and in y.
        #
        mosaic_nx = blocksize_x*nblocksx + max_xgap*(nblocksx-1)
        mosaic_ny = blocksize_y*nblocksy + max_ygap*(nblocksy-1)

        # Form a dictionary of blocks from the data_list.
        # The keys are tuples (column,row)
        #
        block_data = self.get_blocks()

        # If we have ROI, not all blocks in the block_data list
        # are defined.
        # 
        # Get the 1st defined block_data element.
        def_key = block_data.keys()[0]

        # Setup mosaic output array. Same datatype as block_data's
        outtype = block_data[def_key].dtype  
        outdata = np.zeros((mosaic_ny,mosaic_nx), dtype=outtype)

        # Create the output mask. We do not create another one for
        # this self (instance).
        if self.__do_mask:
            self.mask = np.ones((mosaic_ny,mosaic_nx),dtype=np.int8)

        bny,bnx = block_data[def_key].shape    # First input data shape

        # ------- Paste each block (after transforming if tile=False)
        #         into the output mosaic array considering the gaps.

        md = self.mosaic_data                       # MosaicData object
        self.block_mosaic_coord = md.block_mosaic_coord

        # Initialize coordinates of the box to contain
        # all the blocks.
        rx1 = mosaic_nx
        rx2 = 0
        ry1 = mosaic_ny
        ry2 = 0

        bszx,bszy = blocksize_x, blocksize_y
    
        for col,row in block_data:

            data = block_data[col,row]

            if not tile:
		# Correct data for rotation, shift and magnification
                tr = self.transform_objects[col,row]
                if dq_data:
                    tr.set_dq_data()
                data = tr.transform(data)
                # Divide by the jacobian to conserve flux
                indx = col + row*nblocksx
                data = data/jfactor[indx]

	    # Get the block corner coordinates plus the gaps w/r to mosaic origin
            x_gap,y_gap = gaps[(col,row)]
            my1,my2,mx1,mx2 = self._get_block_corners(bszx,bszy,col,row,x_gap,y_gap)

            if dq_data:
                # When transforming DQ data, we set the no-data values 
                # -resulting from the shifting or rotation areas, to Nans
                # Set the nodata value (nans) to zero
                gnan = np.where(np.isnan(data))
                data[gnan] = 0

            # Position block_data in the output mosaic ndarray.
            outdata[my1:my2,mx1:mx2] = data

            if self.__do_mask: 
		# -- mask ndarray values are zero for pixel data, one for no-data.
                self.mask[my1:my2,mx1:mx2] = np.where(data==0,1,0)
                
            # ------ ROI
            
            # Coordinates of the current block including gaps w/r to the mosaic
            # lower left corner. 
            x1,x2,y1,y2 = self.block_mosaic_coord[col,row]
            x1 = int(x1 + x_gap*col)
            x2 = int(x2 + x_gap*col)
            y1 = int(y1 + y_gap*row)
            y2 = int(y2 + y_gap*row)

            # Boundaries enclosing all the current blocks in the mosaic.
            rx1 = min(rx1,x1)
            rx2 = max(rx2,x2)
            ry1 = min(ry1,y1)
            ry2 = max(ry2,y2)
        if return_ROI:
            outdata   = outdata[ry1:ry2,rx1:rx2]      # Crop data
            self.mask = self.mask[ry1:ry2,rx1:rx2]    # Crop masks

        del block_data      # We no longer need this list in memory
        # mask is already done. Reset the flag to not do another for any
        # other extension.
        self.__do_mask = False

        return outdata 

    def _get_block_corners(self,nx,ny,col,row,x_gap,y_gap):
        """ 
          Determines the output section in the mosaic
          array where the block array is going to be
          copied.

          Parameters
          ----------
            ny,nx:    block shape
            col,row:  Column, row position of the block in the
                      mosaic grid.
            x_gap,y_gap: (x,y) block gap.
        
          Output
          ------
            Section:  Pixel coordinates (y1,y2,x1,x2)
        """ 
        
        mx1 = int(col*nx + x_gap*col)
        mx2 = int(mx1 + nx)
        my1 = int(row*ny + y_gap*row)
        my2 = int(my1 + ny)
        return  (my1,my2,mx1,mx2)

    def set_blocks(self):
        """
          1) Set position of blocks within a mosaic.
          2) Set indexes of input amplifier list belonging to each block.
          
          Look where an amplifier is positioned within a block by
          looking at its image-section amp_mosaic_coord corners. If the amplifier data
          is within the block limits, add it to data_index_per_block.
 
        """
        # Set things up only if we have coords

        if self.coords == None:
           return

        #NOTE: Here we use the word 'amplifier' to refer to one element
        #      of the input data_list.    

        geo = self.geometry
        mosaic_grid = geo.mosaic_grid
        self.blocksize = geo.blocksize

        md = self.mosaic_data                       # MosaicData object
     
        # See where each extension data (aka amps) goes in the
        # block (ccdsec).
        md.position_amps_in_block(mosaic_grid,self.blocksize)
        

        # Indices from the amp_mosaic_coord list in each block.
        # E.g. In a 2-amp per CCD, we would have 2 indices per block
        # entry in 'data_index_per_block'
        #
        self.data_index_per_block = md.data_index_per_block

        mcoord = np.asarray(self.coords['amp_mosaic_coord'])

        # Number of amplifiers per block
        #
        # mark the 1st block that have data. In ROI cases, not
        # all blocks have data.
        data_index = self.data_index_per_block
        for key in data_index:
           namps = len(data_index[key])
           if namps > 0:
               self._amps_per_block = namps
               break

        return

    def get_blocks(self,block=None):
        """From the input data_list and the position of the amplifier
           in the block array, form a dictionary of blocks. Forming 
           blocks is necessary for transformation. 
           
           :param  block: default is (None).
		     Allows a specific block to be returned as the 
		     output mosaic. The tuple notation is (col,row) 
		     (zero-based) where (0,0) is the lower left block.
                     This is position of the reference block w/r 
		     to mosaic_grid.
 
           :return block_data: 
                     Block data dictionary keyed in by (col,row) of the
                     mosaic_grid layout.
                
        """
        # set an alias for dictionary of data_list elements
        data_index = self.data_index_per_block 

        # If a block has been requested, modified the data_index_per_block
        # dictionary to have only the data_list indices corresponding to
        # this block.
        if block:
            try:
                # Get a dictionary subset.
                tdic={}
                tdic[block]=data_index[block]
                self.data_index_per_block = tdic    # Reset dictionary values.
            except:
                line = 'Block number requested:%s out of range:%s'%\
                       (block,max(data_index.keys()))
                raise ValueError(line)

        blocksize_x,blocksize_y = self.blocksize

        mcoord = np.asarray(self.coords['amp_mosaic_coord'])
        bcoord = np.asarray(self.coords['amp_block_coord'])

        data_list = self.data_list

        block_data = {}     # Put the ndarrays blocks in this dictionary
                            # keyed in by (i,j)          
        dtype = data_list[0].dtype

        # Loop over the dictionary keys with are the block tuples (col,row) in 
        # the mosaic layout.
        for key in data_index:
            detarray = np.zeros((blocksize_y,blocksize_x),dtype=dtype)
            for id in data_index[key]:
                x1,x2,y1,y2 = bcoord[id]
                # Convert to trimmed coordinates
                detarray[y1:y2,x1:x2] = data_list[id]
            block_data[key] = detarray    
            

        # Free up memory here.
        #del self.data_list
        
        return block_data

    def set_interpolator(self,tfunction='linear',spline_order=2):
        """
           Changing the interpolation method to use when
           correcting the blocks for rotation, shifting and
           magnification.

           Parameters
           ----------

           :param tfunction:
                   Interpolator name. The supported values are:
                   'linear', 'nearest', 'spline'.
                   The order for 'nearest' is set to 0.  The order
                   for 'linear' is set to 1.  Anything greater
                   than 1 will give the specified order of spline
                   interpolation.
           :param  spline_order: Used when tfunction is 'spline'. The order
                     of the spline interpolator.  (default is 2).

        """
        if tfunction == 'linear':
            order = 1
        elif tfunction == 'nearest':
            order = 0
        else:
            order = min(5, max(spline_order,2))  # Spline. Reset order to if <=1.

        self.geometry.interpolator = tfunction
        if order > 1: self.geometry.interpolator='spline'
        self.geometry.spline_order = order
    
class MosaicData(object):
    """
       Provides functionality to create a data object suitable for
       input to the class Mosaic.
                        
       Usage:  mosaic_data = MosaicData(<optional list of parameters>)

       Attributes
       ----------

       data_list: Same as input parameter
       coords:    Same as input parameter
       order:     Index list for amp_mosaic_coord based on the
                  the amplifier coordinate value for ascending x-coord
                  first then y-coord.
                  We mainly need the order for a methods that needs
                  information about amp_mosaic_coord list indices and order.
       data_index_per_block = {}
       block_mosaic_coord = {}    # Block Coordinates w/r to the mosaic
                                  # lower left corner

       Methods
       -------

       get_coords_order       # Find the order of coords

       init_coord             # Initialize coords if not supplied
       position_amps_in_block # Set a dictionary to contain the list indexes
                              # of each data_list element that falls in one block.
    """
    def __init__(self, data_list=None, coords=None):
        """

          Parameters
          ----------

          :param data_list:
                   List of ndarrays representing the pixel image data,
                   all the elements are of the same shape.

          :param coords:
                   A dictionary with keys 'amp_mosaic_coord' and 'amp_block_coord'.
                   The 'amp_mosaic_coord' 
                   values contain a list of tuples describing the corners
                   of the ndarrays, i.e.,  (x1,x2,y1,y2) with respect to
                   the mosaic lower left corner (0,0). The 'amp_block_coord' values
                   contain a list of tuples. describing the corners of the
                   ndarrays, i.e.,  (x1,x2,y1,y2) with respect to the block
                   lower left corner (0,0). Notice that we can have more 
                   than one ndarray per block.


          EXAMPLES:        

          >>> from gemp.adlibrary.mosaicAD import MosaicData
          >>> # Make 4 data arrays of size nx:1024, ny:2048
          >>> data = numpy.linspace(0.,1000.,1024*2048).reshape(2048,1024)
          >>> data_list = [data*(-1)**k for k in numpy.arange(4)]
          >>> # Image section are: (x1,x2,y1,y2)
          >>> coords = {'amp_mosaic_coord': [(0, 1024, 0, 2048), (1024, 2048, 0, 2048),
                                    (2048, 3072, 0, 2048), (3072, 4096, 0, 2048)],
                        'amp_block_coord': [(0, 1024, 0, 2048), (0, 1024, 0, 2048),
                                    (0, 1024, 0, 2048),(0, 1024, 0, 2048)]}

          >>> md = MosaicData(data_list,coords)


        """

        self.data_index_per_block = {}

        if data_list:
            self.data_list = data_list
            # Check that data elements are of the same size
            shapes = [data.shape for data in data_list]
            if len(set(shapes)) > 1:
                raise ValueError(
                   ("MosaicData:: 'data_list' elements are "
			 "not of the same size."))

        # Define the attributes
        self.coords = None

        # If we have positions defined for the data_list items.
        if coords:
            if data_list==None:
                raise ValueError("MosaicData:: 'data_list' cannot be None.")

            if type(coords) is not dict:
                raise ValueError(" 'coords' needs to be a dictionary.")
            self.coords = coords

            ndata = len(data_list)
            for key in coords:
                if len(coords[key]) != ndata:
                    raise ValueError(
                          ( "MosaicData:: Number of coordinates in '"+str(key)+
                           "' is not the same as the number of arrays"
			   "in 'data_list'."))

    def init_coord(self,geometry_object):
        """ Initialize coords if not supplied
        """
        if (geometry_object != None) and (self.coords == None):
            # We have a MosaicGeometry object but no 'coords'
            # with 'blocksize' and 'mosaic_grid' we can generate it.
            ncols,nrows = geometry_object.mosaic_grid
            npix_x,npix_y = geometry_object.blocksize
            bcoord = [(0,npix_x,0,npix_y) for k in range(ncols*nrows)]
            mcoord=[]
            for row in range(nrows):
                for col in range(ncols):
                    x1 = col*npix_x
                    x2 = x1 + npix_x
                    y1 = row*npix_y
                    y2 = y1 + npix_y
                    mcoord.append((x1,x2,y1,y2))
            self.coords = {'amp_mosaic_coord':mcoord, 'amp_block_coord':bcoord}
        

    def position_amps_in_block(self,mosaic_grid,blocksize):
        """
           Set a dictionary to contain the list indexes of each data_list
           element that falls in one block.
        """
        # Number of blocks in (x,y)_direction in mosaic
        nblocksx = mosaic_grid[0]
        nblocksy = mosaic_grid[1]
       
        ampmcoord = np.asarray(self.coords['amp_mosaic_coord'])
        ampbcoord = np.asarray(self.coords['amp_block_coord'])

        blocksize_x, blocksize_y = blocksize

        # Form a list of block corner positions in the x-direction within
        # a mosaic with no gaps.
        # For example for a 1024 width of a block in a mosaic of 3 blocks
        # the list is: (0,1024,2048,3072)
        #
        blocksxlim = np.asarray([k*blocksize_x for k in range(nblocksx+1)])

        # Same for the y-direction.
        blocksylim = np.asarray([k*blocksize_y for k in range(nblocksy+1)])

        # Form an array of each amplifier x2 positions.
        # Each amp_mosaic_coord has (x1,x2,y1,y2)
        dataxmax = np.asarray([k[1] for k in ampmcoord])

        # Same for y2 positions.
        dataymax = np.asarray([k[3] for k in ampmcoord])

        ######  AMPLIFIERS IN BLOCK

        # Now look where each amplifier dataxmax and dataymax position 
        # falls on. Keep these indices (ix list) 
        wx = blocksize_x
        wy = blocksize_y
        bxmin = wx
        bxmax = 0
        bymin = wy
        bymax = 0

        self.block_mosaic_coord = {}
        for row in range(nblocksy):
            for col in range(nblocksx):
                bxmin,bxmax,bymin,bymax = (wx,0,wy,0)
                ix, = np.where((blocksxlim[col] < dataxmax) &\
                           (dataxmax <= blocksxlim[col+1]) &\
                           (blocksylim[row] < dataymax) &\
                           (dataymax <= blocksylim[row+1]))
                if len(ix) > 0:     # Stamp images might not have data 
                                    # on some blocks
                    self.data_index_per_block[(col,row)] = ix
                    bxmin =  min(bxmin,min([k[0] for k in ampbcoord[ix]]))
                    bxmax =  max(bxmax,max([k[1] for k in ampbcoord[ix]]))
                    bymin =  min(bymin,min([k[2] for k in ampbcoord[ix]]))
                    bymax =  max(bymax,max([k[3] for k in ampbcoord[ix]]))
                x1 = bxmin + col*wx
                x2 = bxmax + col*wx
                y1 = bymin + row*wy
                y2 = bymax + row*wy
                self.block_mosaic_coord[(col,row)] = (x1,x2,y1,y2)


    def get_coords_order(self):
        """ Find the amplifier order in ascending x-coord first then y-coord
            from the list of 'mosaic_coord' tuples.
            We need this order whenever we are looking for the reference block
            amplifiers. The left most amplifier header contains the reference
            CRPIX1, CRPIX2 values to determine the mosaic header crpixs'.
            We need to make sure that the order of the amplifiers in the blocks
            follows the amp_mosaic_coord (x,y) values in ascending order.
 
        """

        # list of major coordinates
        ampmcoord = np.asarray(self.coords['amp_mosaic_coord'])   
        # list of minor coordinates
        bcoord = np.asarray(self.coords['amp_block_coord'])

        coordx1 = [sec[0] for sec in ampmcoord] # x1 location from each tuple
        coordy1 = [sec[2] for sec in ampmcoord] # y1 location from each tuple

        anumbers = range(len(coordx1))
        # Find the order of x and y coords.
        ampsorder = np.array(zip(anumbers, coordx1,coordy1),dtype=
                    [('ext',np.int),('coordx1',np.int),('coordy1',np.int)])
        ampsorder.sort(order=('coordy1','coordx1'))
        order = np.asarray([amp[0] for amp in ampsorder]) + 1
        return order

class MosaicGeometry(object):
    """
      The MosaicGeometry class provides functionality to verify the 
      input dictionary elements and set all the require attributes.

      Usage:
      mosaic_geometry = MosaicGeometry(dictionary)
    """

    def __init__(self,geometry_dict):

        """
        .. _help_mgeo:

        Parameters
        ----------

          geometry_dict:  Dictionary with the following keys:
          -------------

          NOTE: The only require keys are blocksize and mosaic_grid.

          blocksize:        # Requireq
              Tuple of (npixels_x,npixels_y). Size of one block.
          mosaic_grid:      # Required
              Tuple (ncols,nrows). Number of blocks per row and
              number of rows in the output mosaic array.

          transformation: dictionary with: {
              'shift': 
                  List of tuples (x_shift,y_shift). Amount in pixels (floating
                  numbers) to shift to align with the ref_block. There 
                  are as many tuples as number of blocks.
              'rotation':
                  (Degrees). List of real numbers. Amount to rotate each
                  block to align with the ref_block. There are as 
                  many numbers as number of blocks. Angle is counter wise
                  from the x-axis.
              'magnification': List of real numbers. Amount to magnify each
                  block to align with the ref_block. There are as
                  many numbers as number of blocks. Magnification is about
                  the center of the block.
                                            }

          ref_block:
              Reference block tuple. The block location (x,y) coordinate 
              in the mosaic_grid. This is a 0-based tuple. 'x' increases to
              the right, 'y' increases in the upwards direction. The origin 
              (0,0) is the lower left block.
          interpolator:
              (String). Default is 'linear'. Name of the transformation
              function used for translation,rotation, magnification of the 
              blocks to be aligned with the reference block. Allow values
              are: 'linear', 'nearest', 'spline'.
          spline_order
              (int). Default 2. Is the 'spline' interpolator order. Values 
              should be in the range [0-5].
          gaps:
              A dictionary with key the block tuple and values
              a tuple (x_gap, y_gap) with the gap for each block. Where
              the gap is added to the block is left to the handling method.
          gap_dict:
              If you have different set of gaps for 'tiling' and 
              'transform', then use this entry. It takes precedence over
              the 'gaps' entry.
              'gap_dict' is a dictionary with two keys: 'tile_gaps' and
              'transform_gaps', with value a dictionary with key the block
              tuple and values a tuple (x_gap, y_gap) with the gap for 
              each block. Where the gap is added to the block is left to
              the handling method.

          Output
          ------
           An object with about the same attributes as the above keys and 
           values verified for consistency.
        
           attributes
           -----------

           blocksize:    Same as input
           mosaic_grid:  Same as input
           interpolator: Same as input
           ref_block:    Same as input
           transformation: dictionary {'shift':shift ,'rotation':rotation,
                         'magnification':magnification }


         EXAMPLE
         -------

         >>> from gempy.library.mosaic import MosaicGeometry
         >>> geometry_dict = {
              'transformation': {
                'shift':[(43.60,-1.24), (0.,0.),
                       (0.02, 41.10), (43.42, 41.72)], # List of (x_shift,y_shift)
                'rotation': (-1.033606, 0.0, 
                            0.582767, 0.769542),   # List of degrees, counterwise
                                                   # w/r to the x_axis
                'magnification': (1.0013, 1.,
                                1.0052, 1.0159),   # List of magnification
              },
              'interpolator': 'linear',       # transformation function
              'blocksize':    (1024,2048),    # shape of block (naxis1,naxis2)
              'ref_block':    (1,0),    # Position of reference block w/r to mosaic
                                        # grid. The lower left block is (0,0).
              'mosaic_grid':  (4,1)          # output mosaic layout: number
                                             # of blocks in (x,y)
                             } 
         >>> geo = MosaicGeometry(geometry_dict)
         >>> geo.info()    # prints out the dictionary

        """            
        # The required keys are mosaic_grid and blocksize
        mosaic_grid   = geometry_dict['mosaic_grid']
        blocksize     = geometry_dict['blocksize']

        if (type(mosaic_grid) is not tuple) or (len(mosaic_grid) < 2):           
            raise ValueError('Mosaic grid: Should be a 2 value tuple')
        else:
            nblocks = mosaic_grid[0]*mosaic_grid[1]     # The number of blocks 
                                                        # in the mosaic
        self.mosaic_grid = mosaic_grid

        if (type(blocksize) is not tuple) or (len(blocksize) < 2):           
            raise ValueError('Blocksize: Should be a 2 value tuple')
        else:
            self.blocksize = blocksize
        

        # Linear interpolator is the default
        interpolator = 'linear'
        if 'interpolator' in geometry_dict:
            interpolator     = geometry_dict['interpolator']
        if interpolator == 'spline':
            if 'spline_order' in geometry_dict:
                spline_order = geometry_dict['spline_order']
                if not(0 <= spline_order <= 5):
                    raise ValueError('spline_order: Should be >0 or <6')
            else:
                spline_order = 2
            self.spline_order = spline_order
        self.interpolator = interpolator

        gap_dict = None
        if 'gap_dict' in geometry_dict:
            gap_dict = geometry_dict['gap_dict']
            if not ('tile_gaps' in gap_dict or
                    ('transform_gaps' in gap_dict)):
                raise ValueError(\
                   "gap_dict: key is not 'tile_gaps' or 'transform_gaps'")
            for gt in ['tile_gaps','transform_gaps']:
                for k,v in zip(gap_dict[gt].keys(),gap_dict[gt].values()):
                    if (type(k) is tuple) and (type(v) is tuple):
                        if (len(k)!=2) or (len(v)!=2):
                            raise ValueError("Gaps values are not of length 2")
                    else:
                        raise ValueError("Gaps keys are not tuples")

        elif 'gaps' in geometry_dict:
            # Simples 'gaps' format.
            # This is dictionary {(col,row): (x_gap,y_gap)....}
            # that applies to tile and transform
            gaps = geometry_dict['gaps']
            if len(gaps) != nblocks:
                raise ValueError("Gaps dictionary length is not: "+str(nblocks))
            for k,v in zip(gaps.keys(),gaps.values()):
                if (type(k) is tuple) and (type(v) is tuple):
                    if (len(k)!=2) or (len(v)!=2):
                        raise ValueError("Gaps values are not of length 2")
                else:
                    raise ValueError("Gaps keys are not tuples")

            gap_dict = {'tile_gaps':gaps,'transform_gaps':gaps}
        else:
           # No gaps giving in input dictionary.
            # Make a gap dictionary. x_gap to the left and y_gap at the bottom of
            # each block. First row is zero at the bottom, first column is zero to
            # left of each block
            dval = {} 
            dval[0,0]=(0,0)
            for col in range(1,mosaic_grid[0]): dval[col,0] = (0,0)
            for row in range(1,mosaic_grid[1]): dval[0,row] = (0,0)
            for row in range(1,mosaic_grid[1]):
               for col in range(1,mosaic_grid[0]):
                   dval[col,row] = (0,0)
            gap_dict = {'tile_gaps':dval,'transform_gaps':dval} 
        self.gap_dict = gap_dict

        # 'transformation' key is a dictionary with keys shift, rotation
        # and magnification. All these are optional.
        #
        trans = {}
        if 'transformation' in geometry_dict:
            trans = geometry_dict['transformation']

        if 'shift' in trans:
            shift = trans['shift']
            if len(shift) != nblocks:     # Must have nblocks tuples
				          # of size 2.  
                raise ValueError(
                'shift: There must be '+str(nblocks)+' values.')
            for val in shift:             # check that each element is tuple
                if len(val) != 2: raise ValueError('One of the tuples in\
                                     "shift" is not of length 2.')
        else:
            shift = [(0.,0.)]*nblocks

        if 'rotation' in trans:
            rotation = trans['rotation']
            if len(rotation) != nblocks:  # Must have nblocks tuples
				          # of size 2.
                raise ValueError(
                'rotation: There must be '+str(nblocks)+' values.')
            if type(rotation[0]) is not tuple:
               rotation = [(r,r) for r in rotation]   # Turn to tuple
        else:
            rotation = [(0.,0.)]*nblocks

        if 'magnification' in trans:
            magnification = trans['magnification']
            if len(magnification) != nblocks:     # must have nblocks tuples of size 2.
                raise ValueError(
                'magnification: There must be '+str(nblocks)+' values.')
            if type(magnification[0]) is not tuple:
                magnification = [(m,m) for m in magnification]  # turn to tuple
        else:
            magnification = [(1.,1.)]*nblocks

        self.transformation = {'shift':shift ,'rotation':rotation,
                               'magnification':magnification }

        # If the key is not defined, set it to the lower left
        ref_block = (0,0)
        if 'ref_block' in geometry_dict:
            ref_block     = geometry_dict['ref_block']
        self.ref_block = ref_block    

    def info(self):
        """Return the geometry dictionary


          blocksize:
              Tuple of (npixels_x,npixels_y). Size of a block.
          mosaic_grid: 
              Tuple (ncols,nrows). Number of blocks per row and
              number of rows in the output mosaic array.

          transformation: dictionary with: {
              'shift': 
                  List of tuples (x_shift,y_shift). Amount in pixels (floating
                  numbers) to shift to align with the ref_block. There 
                  are as many tuples as number of blocks.
              'rotation':
                  (Degrees). List of real numbers. Amount to rotate each
                  block to align with the ref_block. There are as 
                  many numbers as number of blocks.The angle is counter
                  clockwise from the x-axis.
              'magnification': List of real numbers. Amount to magnify each
                  block to align with the ref_block. There are as
                  many numbers as number of blocks.
                                            }

          ref_block:
              Reference block tuple. The block location (ref_block_col,
              ref_block_row) is the position of the reference block w/r
              to mosaic_grid. This is a 0-based tuple and (0,0) refers to
              lower left block in the mosaic.

          interpolator:
              (String). Default is 'linear'. Name of the transformation
              function used for translation,rotation, magnification of the 
              blocks to be aligned with the reference block.
          spline_order
              (int). Default is 2. Is the 'spline' interpolator order. Values 
              should be in the range [0-5].
          gap_dict:
              A dictionary with two keys: 'tile_gaps' and 'transform_gaps', 
              with value a dictionary with key the block tuple and values
              a tuple (x_gap, y_gap) with the gap for each block. Where
              the gap is added to the block is left to the handling method.

        """
        geo_tab = {
	          'blocksize':      self.blocksize, 
                  'transformation': self.transformation,
	          'mosaic_grid':    self.mosaic_grid,
		  'ref_block':      self.ref_block,
                  'interpolator':   self.interpolator,
                  'gap_dict':       self.gap_dict,
                   }
        if self.interpolator == 'spline':
            geo_tab['spline_order'] = self.spline_order

        return geo_tab

