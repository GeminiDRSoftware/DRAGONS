import numpy as np

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
          >>> coords = {'amp_mosaic_coord': [(0, 1024, 0, 2048),
                                             (1024, 2048, 0, 2048),
                                            (2048, 3072, 0, 2048),
                                            (3072, 4096, 0, 2048)],
                        'amp_block_coord': [(0, 1024, 0, 2048),
                                            (0, 1024, 0, 2048),
                                            (0, 1024, 0, 2048),
                                            (0, 1024, 0, 2048)]}

          >>> md = MosaicData(data_list, coords)


        """

        self.data_index_per_block = {}

        if data_list:
            self.data_list = data_list
            # Check that data elements are of the same size
            shapes = [data.shape for data in data_list]
            if len(set(shapes)) > 1:
                raise ValueError("MosaicData:: 'data_list' elements are not "
                                 "of the same size.")

        # Define the attributes
        self.coords = None

        # If we have positions defined for the data_list items.
        if coords:
            if data_list is None:
                raise ValueError("MosaicData:: 'data_list' cannot be None.")

            if type(coords) is not dict:
                raise ValueError(" 'coords' needs to be a dictionary.")
            self.coords = coords

            ndata = len(data_list)
            for key in coords:
                if len(coords[key]) != ndata:
                    raise ValueError("MosaicData:: Number of coordinates "
                                     "in '" + str(key) + "' is not the same as "
                                     "the number of arrays in 'data_list'.")

    def init_coord(self, geometry_object):
        """
        Initialize coords if not supplied
        """
        if (geometry_object is not None) and (self.coords is None):
            # We have a MosaicGeometry object but no 'coords'
            # with 'blocksize' and 'mosaic_grid' we can generate it.
            ncols, nrows = geometry_object.mosaic_grid
            npix_x, npix_y = geometry_object.blocksize
            bcoord = [(0, npix_x, 0, npix_y)] * ncols*nrows
            mcoord = []
            for row in range(nrows):
                for col in range(ncols):
                    x1 = col * npix_x
                    x2 = x1 + npix_x
                    y1 = row * npix_y
                    y2 = y1 + npix_y
                    mcoord.append((x1, x2, y1, y2))
            self.coords = {'amp_mosaic_coord': mcoord,
                           'amp_block_coord': bcoord}


    def position_amps_in_block(self, mosaic_grid, blocksize):
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
        blocksxlim = np.asarray([k * blocksize_x for k in range(nblocksx+1)])

        # Same for the y-direction.
        blocksylim = np.asarray([k * blocksize_y for k in range(nblocksy+1)])

        # Form an array of each amplifier x2 positions.
        # Each amp_mosaic_coord has (x1, x2, y1, y2)
        dataxmax = np.asarray([k[1] for k in ampmcoord])

        # Same for y2 positions.
        dataymax = np.asarray([k[3] for k in ampmcoord])

        ######  AMPLIFIERS IN BLOCK

        # Now look where each amplifier dataxmax and dataymax position
        # falls on. Keep these indices (idx list)

        self.block_mosaic_coord = {}
        for row in range(nblocksy):
            for col in range(nblocksx):
                bxmin, bxmax, bymin, bymax = (blocksize_x, 0, blocksize_y, 0)
                idx, = np.where((blocksxlim[col] < dataxmax) &
                               (dataxmax <= blocksxlim[col+1]) &
                               (blocksylim[row] < dataymax) &
                               (dataymax <= blocksylim[row+1]))
                if len(idx) > 0:     # Stamp images might not have data
                                    # on some blocks
                    self.data_index_per_block[(col, row)] = idx
                    bxmin = min(bxmin, min([k[0] for k in ampbcoord[idx]]))
                    bxmax = max(bxmax, max([k[1] for k in ampbcoord[idx]]))
                    bymin = min(bymin, min([k[2] for k in ampbcoord[idx]]))
                    bymax = max(bymax, max([k[3] for k in ampbcoord[idx]]))
                x1 = bxmin + col * blocksize_x
                x2 = bxmax + col * blocksize_x
                y1 = bymin + row * blocksize_y
                y2 = bymax + row * blocksize_y
                self.block_mosaic_coord[(col, row)] = (x1, x2, y1, y2)


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

        coordx1 = [sec[0] for sec in ampmcoord]  # x1 location from each tuple
        coordy1 = [sec[2] for sec in ampmcoord]  # y1 location from each tuple

        anumbers = range(len(coordx1))
        # Find the order of x and y coords.
        ampsorder = np.array(zip(anumbers, coordx1, coordy1),
                             dtype=[('ext', np.int), ('coordx1', np.int),
                                    ('coordy1', np.int)])
        ampsorder.sort(order=('coordy1', 'coordx1'))
        order = np.asarray([amp[0] for amp in ampsorder]) + 1
        return order
