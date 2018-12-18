from __future__ import division
#
#                                                                        DRAGONS
#
#                                                                      mosaic.py
# ------------------------------------------------------------------------------
from builtins import str
from builtins import zip
from builtins import range
from builtins import object
from past.utils import old_div
import numpy as np

from .mosaicGeometry import MosaicGeometry
from .transformation import Transformation
from .transformation import DQMap

# temp import
from gempy.utils import logutils
log = logutils.get_logger(__name__)
# ------------------------------------------------------------------------------
class Mosaic(object):
    """
    Mosaic is a base class that provides functionality to layout a list
    of data ndarrays of the same size into an output mosaic. The main
    characteristics are:

    - Input data and their location in the output mosaic is done via
      MosaicData objects.

    - Information about gaps between the blocks and transformation is
      given by the MosaicGeometry  object.

    - Can reset the interpolator in tranformation via Mosaic class function.

    - Preserving flux when transforming a block

    """
    def __init__(self, mosaic_data, mosaic_geometry=None):
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
                   A dictionary with keys 'amp_mosaic_coord' and
                   'amp_block_coord'.  The 'amp_mosaic_coord' values contain a
                   list of tuples describing the corners of the ndarrays, i.e.,
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
          >>> from gempy.mosaic import Mosaic
              # Now import you own 'my_mosaic_function' to compose the input
              # data_list coordinate descriptors and the geometry object.
          >>> from my_module import my_mosaic_function
          >>>
          >>> modata, geometry = my_mosaic_function('filename.fits')
          >>> mo = Mosaic(modata, geometry)
          >>> outdata = mo.mosaic_image_data()

        """
        # Set attributes
        self.blocksize = None
        self.block_mosaic_coord = None
        self._amps_per_block = None
        self.mosaic_data = mosaic_data           # For use in calling methods.
        self.data_list = mosaic_data.data_list   # Lists of ndarray pixel data.
        self.data_index_per_block = None

        mosaic_data.init_coord(mosaic_geometry)
        self.coords = mosaic_data.coords
        if mosaic_geometry is None:
            mosaic_geometry = self.set_default_geometry()

        self.geometry = mosaic_geometry

        # If both objects are not None set up block attributes.
        if not (self.geometry is None and self.coords is None):
            self.set_blocks()
            self.verify_inputs()
            # Set order of amps according to x and y position.
            self.coords['order'] = mosaic_data.get_coords_order()

        self.return_ROI = True
        self.transform_objects = None
        self.as_iraf = True


    def mosaic_image_data(self, block=None, dq_data=False, jfactor=None,
                          tile=False, return_ROI=True):
        """
        Main method to layout the block of data in a mosaic grid.
        Correction for rotation, shifting and magnification is performed with
        respect to the reference block.

        A mask is created containing value zero for positions were
        there are pixel data and one for everywhere else -- like gaps and
        areas of no-data due to shifting; for example, after transforming.

        Parameters
        ----------

        block: <tuple>
            Allows a specific block to be returned as the output mosaic.
            The tuple notation is (col,row) (0-based), where (0,0) is the
            lower left block. The blocks layout is given by the attribute
            mosaic_grid. Default is None.

        dq_data: <bool>
            The input data are transformed bit-plane by bit-plane.
            Can handle both 8-bit and 16-bit planes. DQ is 16-bit.
            Default is False.

        tile: <bool>
            Layout the block in the mosaic grid with no correction for
            rotation nor shift. Gaps are included. Default is False.

        return_ROI: <bool>
            Use the minimum frame enclosing all the block_data elements.
            Default is True.

        jfactor: <list>
            Factor to multiply transformed block to conserve flux.

        Returns
        -------
        outdata: <ndarray>
            An ndarray containing the mosaic.

        """
        self.return_ROI = return_ROI
        # No coordinates, then do Horizontal tiling
        if self.coords is None:
            return np.hstack(self.data_list)

        if block:
            return list(self.get_blocks(block).values())[0]

        # N blocks in (x,y)_direction
        nblocksx, nblocksy = self.geometry.mosaic_grid
        blocksize_x, blocksize_y = self.geometry.blocksize

        if jfactor is None:
            jfactor = [1.] * blocksize_x * blocksize_y

        # Get gap dictionary for either tile or transform mode.
        if tile:
            gap_mode = 'tile_gaps'
        else:
            gap_mode = 'transform_gaps'
            self.set_transformations()

        gaps = self.geometry.gap_dict[gap_mode]
        gap_values = list(gaps.values())
        max_xgap = max([g[0] for g in gap_values])
        max_ygap = max([g[1] for g in gap_values])

        # This is the entire mosaic area, pixels in x and in y.
        mos_data = self.mosaic_data       # MosaicData object
        self.block_mosaic_coord = mos_data.block_mosaic_coord

        max_x = 0
        for coords in list(self.block_mosaic_coord.values()):
            max_x = max(max_x, coords[0], coords[1])

        mosaic_nx = max_x + max_xgap*(nblocksx-1)
        mosaic_ny = max(v[3] for k, v in list(self.block_mosaic_coord.items())) + \
                    max_ygap*(nblocksy-1)

        # Form a dictionary of blocks from the data_list, keys are tuples
        # (column,row)
        block_data = self.get_blocks()

        # If we have ROI, not all blocks in the block_data list are defined.
        # Get the 1st defined block_data element.
        def_key = list(block_data.keys())[0]

        # Setup mosaic output array. Same datatype as block_data's
        outtype = block_data[def_key].dtype
        outdata = np.zeros((int(mosaic_ny), int(mosaic_nx)), dtype=outtype)
        if dq_data:
            outdata += DQMap['no_data']

        # ------- Paste each block (after transforming if tile=False)
        #         into the output mosaic array considering the gaps.

        # Initialize coordinates of the box to contain all blocks.
        rx1 = mosaic_nx
        rx2 = 0
        ry1 = mosaic_ny
        ry2 = 0
        bszx, bszy = blocksize_x, blocksize_y
        for col, row in block_data:
            data = block_data[col, row]
            if not tile:
                # Correct data for rotation, shift and magnification
                trans_obj = self.transform_objects[col, row]
                if dq_data:
                    trans_obj.set_dq_data()

                data = trans_obj.transform(data)
                # Divide by the jacobian to conserve flux
                indx = col + row*nblocksx
                data = old_div(data, jfactor[indx])

            # Get the block corner coordinates plus gaps wrt to mosaic origin
            x_gap, y_gap = gaps[(col, row)]
            my1,my2,mx1,mx2 = self._get_block_corners(bszx,bszy,col,row,x_gap,y_gap)

            # Position block_data in the output mosaic ndarray.
            outdata[my1:my2, mx1:mx2] = data

            # ------ ROI
            # Coordinates of the current block including gaps w/r to the mosaic
            # lower left corner.
            x1, x2, y1, y2 = self.block_mosaic_coord[col, row]
            x1 = int(x1 + x_gap*col)
            x2 = int(x2 + x_gap*col)
            y1 = int(y1 + y_gap*row)
            y2 = int(y2 + y_gap*row)

            # Boundaries enclosing all the current blocks in the mosaic.
            rx1 = min(rx1, x1)
            rx2 = max(rx2, x2)
            ry1 = min(ry1, y1)
            ry2 = max(ry2, y2)

        if return_ROI:
            outdata = outdata[ry1:ry2, rx1:rx2]        # Crop data

        del block_data      # We no longer need this list in memory
        return outdata

    def _get_block_corners(self, xsize, ysize, col, row, x_gap, y_gap):
        """
          Determines the output section in the mosaic
          array where the block array is going to be
          copied.

          Parameters
          ----------
            ysize,xsize:    block shape
            col,row:  Column, row position of the block in the
                      mosaic grid.
            x_gap,y_gap: (x,y) block gap.

          Output
          ------
            Section:  Pixel coordinates (y1,y2,x1,x2)
        """

        mx1 = int(col * xsize + x_gap * col)
        mx2 = int(mx1 + xsize)
        my1 = int(row * ysize + y_gap * row)
        my2 = int(my1 + ysize)
        return my1, my2, mx1, mx2

    def get_blocks(self, block=None):
        """
        From the input data_list and the position of the amplifier in the block
        array, form a dictionary of blocks. Forming blocks is necessary for
        tiling and transformation. self.data_list should be first populated by
        a call on self.get_data-list(<extension_type>), where <extension_type>
        is one of 'data', 'variance', 'mask', or "OBJMASK'.

        Parameters
        ----------
        block: <2-tuple>
            Allows a specific block to be returned as the output mosaic. The tuple
            notation is ((col,row), zero-based) where (0,0) is the lower left block.
            This is position of the reference block wrt mosaic_grid.
            Default is None.

        Returns
        -------
        block_data: <dict> or None
            Block data dictionary keyed in by (col,row) of the mosaic_grid layout.

        """
        if not self.data_list:
            return None
        # set an alias for dictionary of data_list elements
        data_index = self.data_index_per_block

        # If a block has been requested, modified the data_index_per_block
        # dictionary to have only the data_list indices corresponding to
        # this block.
        if block:
            try:
                # Get a dictionary subset.
                tdic = {}
                tdic[block] = data_index[block]
                self.data_index_per_block = tdic    # Reset dictionary values.
            except:
                warn = 'Block number requested:{} out of range:{}'
                line = warn.format(block, max(data_index.keys()))
                raise ValueError(line)

        blocksize_x, blocksize_y = self.blocksize
        bcoord = np.asarray(self.coords['amp_block_coord'])
        data_list = self.data_list
        block_data = {}     # Put the ndarrays blocks in this dictionary
        dtype = data_list[0].dtype

        # Loop over the dictionary keys with are the block tuples (col,row) in
        # the mosaic layout.
        for key in data_index:
            detarray = np.zeros((int(blocksize_y), int(blocksize_x)), dtype=dtype)
            for i in data_index[key]:
                x1, x2, y1, y2 = bcoord[i]
                # Convert to trimmed coordinates
                detarray[int(y1):int(y2), int(x1):int(x2)] = data_list[i]
            block_data[key] = detarray

        return block_data

    def set_blocks(self):
        """
        1) Set position of blocks within a mosaic.
        2) Set indexes of input amplifier list belonging to each block.

        Look where an amplifier is positioned within a block by
        looking at its image-section amp_mosaic_coord corners. If the
        amplifier data is within the block limits, add it to
        data_index_per_block.

        """
        # Set things up only if we have coords
        if self.coords is None:
            return

        # NOTE: 'amplifier' is one element of input data_list.
        self.blocksize = self.geometry.blocksize
        mosaic_grid = self.geometry.mosaic_grid

        # Where extension data (amps) goes in the block (ccdsec).
        self.mosaic_data.position_amps_in_block(mosaic_grid, self.blocksize)

        # Indices from the amp_mosaic_coord list in each block.
        # E.g. In a 2-amp per CCD, we would have 2 indices per block
        # entry in 'data_index_per_block'
        self.data_index_per_block = self.mosaic_data.data_index_per_block

        # Number of amplifiers per block. Mark 1st block that has data.
        # For ROI, not all blocks have data.
        for key in self.data_index_per_block:
            namps = len(self.data_index_per_block[key])
            if namps > 0:
                self._amps_per_block = namps
                break

        return

    def set_default_geometry(self):
        """
        We are here because the object MosaicGeometry instantiated
        in Mosaic __init__ function is None.

        """
        if not hasattr(self, 'coords'):
            raise ValueError("From set_default_geometry: 'coords' not defined.")

        if self.coords is None:
            return

        # The coords dictionary is defined.
        # This is the list of positions w/r to the mosaic
        # lower left (0,0) origin
        amp_mosaic_coord = self.coords['amp_mosaic_coord']

        # This is the list of positions w/r to the block lower left (0,0) origin
        bcoord = self.coords['amp_block_coord']

        # Set blocksize from amp_block_coord. Notice that we are taken the
        # maximum value for each axis.
        dnx, dny = max([(cc[1], cc[3]) for cc in bcoord])
        blocksize = (dnx, dny)

        # Get minimum origin coordinates from all the data_list elements.
        xcoo_0, ycoo_0 = min([(dd[0], dd[2]) for dd in amp_mosaic_coord])

        # Calculate the mosaic array grid tuple (ncols,nrows).
        nblocksx, nblocksy = max([(old_div((dd[1]-xcoo_0), dnx), old_div((dd[3]-ycoo_0), dny))
                                  for dd in amp_mosaic_coord])

        # Set the attribute
        mosaic_grid = (max(1, int(nblocksx)), max(1, int(nblocksy)))

        # This is the minimum dictionary necessary to create a
        # MosaicGeometry object
        geodict = {'mosaic_grid':mosaic_grid, 'blocksize':blocksize}

        # Now instantiate the object
        return  MosaicGeometry(geodict)

    def set_interpolator(self, tfunction='linear', spline_order=2):
        """
        Changing the interpolation method to use when correcting the blocks for
        rotation, shifting and magnification.

        Parameters
        ----------

        :param tfunction:
            Interpolator name. The supported values are:
            'linear', 'nearest', 'spline'.
            The order for 'nearest' is set to 0.  The order
            for 'linear' is set to 1.  Anything greater
            than 1 will give the specified order of spline
            interpolation.
        :param  spline_order:
            Used when tfunction is 'spline'. The order
            of the spline interpolator.  (default is 2).

        """
        if tfunction == 'linear':
            order = 1
        elif tfunction == 'nearest':
            order = 0
        else:
            # Spline. Reset order to if <=1.
            order = min(5, max(spline_order, 2))

        self.geometry.interpolator = tfunction
        if order > 1:
            self.geometry.interpolator = 'spline'
        self.geometry.spline_order = order
        return

    def set_transformations(self):
        """
        Instantiates the Transformation class objects for each block that needs
        correction for rotation, shift and/or magnification. Set a dictionary 
        with (col,row) as a key and value the Transformation object.

        """
        # Correction parameters from the MosaicGeometry object dict.
        geo = self.geometry
        nblocksx, nblocksy = geo.mosaic_grid
        mag = geo.transformation['magnification']
        rot = geo.transformation['rotation']       # (x_rot, y_rot)
        shift = geo.transformation['shift']        # (x_shift, y_shift)
        order = 1
        if geo.interpolator == 'spline':
            order = geo.spline_order

        transform_objects = {}
        # Use the keys valid (col,row) tuples as there could be some
        # tuples with no data hence no tuple.
        for col, row in self.data_index_per_block:
            # Turn block tuple location (col,row) into list index
            indx = col + row*nblocksx

            # Instantiates a Transformation object. The interpolator
            # order can be reset if needed.
            trf = Transformation(rot[indx][0], shift[indx], mag[indx], 
                                 order=order, as_iraf=self.as_iraf)

            # Add a key to the dictionary with value the object.
            transform_objects[col, row] = trf

        # Reset the attribute
        self.transform_objects = transform_objects
        return

    def verify_inputs(self):
        """
        Verify that mosaic_data and geometry object atributes are consistent.

        """
        if self.coords is None and self.geometry is None:
            return

        # coord1 ... amp_mosaic_coord
        # coord2 ... amp_block_coord
        # blocksize should be equal or larger that any value in coord2.
        if self.coords is None:
            # We do not have coords but we do have geometry values.
            # Set coords1 and 2
            dszy, dszx = self.data_list[0].shape
            datalen = len(self.data_list)
            bcoord = [(0, dszx, 0, dszy) for k in range(datalen)]
            # amp_mosaic_coord: Make the sections run along the x-axis.
            mcoord = [(k*dszx, k*dszx + dszx, 0, dszy) for k in range(datalen)]
            self.coords = {'amp_mosaic_coord': mcoord,
                           'amp_block_coord': bcoord}
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
            line1 = "Geometry tuple 'blocksize' does not agree with "
            line1 += "coordinates range."
            line2 = '\n\t\tblocksize_x:'+str(blcx)+', blocksize_y:' + str(blcy)
            line3 = '\n\t\tminX,maxX,minY,maxY:(%d,%d,%d,%d)' % \
                    (minx, maxx, miny, maxy)
            raise ValueError(line1 + line2 + line3)

        return

# -------------
# The following 2 functions were added by James, to help with the co-ordinate
# bookeeping for GSAOI. I believe they are more general than the mosaicking
# functionality itself but there's no point building in unnecessary limitations
# and perhaps they'll be useful if we generalize the code to N-D later.

def reset_origins(ranges, per_block=False):
    """
    Given a sequence of (n-dimensional) co-ordinate range sequences
    (eg. ((x1a, x2a, y1a, x2a), (x1b, x2b, y1b, y2b)) ), shift their origins
    to 0. If per_block is False (default), a common origin will be preserved,
    such that the lowest x1 & y1 (etc.) values over all the sequences are 0;
    otherwise, each sequence will be adjusted to have its individual x1 & y1
    (etc.) equal to 0.

    """
    # This is a bit more general than needed for GSAOI, with a view to
    # moving it somewhere more generic later for other mosaicking stuff.

    # Require that the dimensionality of the input sequences is consistent:
    lens = set(len(coord_set) for coord_set in ranges)
    if ranges and len(lens) != 1:
        raise ValueError('input range sequences differ in length/'\
                         'dimensionality')

    # Rearrange the co-ordinate range sequences into (start, end) pairs
    # in order to iterate over their dimensions more easily:
    in_pairs = tuple(list(zip(*[iter(coord_set)] * 2)) for coord_set in ranges)

    # Derive the offset to apply to each co-ordinate of each sequence:
    if per_block:
        adj = (tuple(dim[0] for dim in coord_set for lim in (0, 1))
               for coord_set in in_pairs)
    else:
        # First invert the nested ordering of pairs (by dimension and then
        # original sequence, rather than vice versa):
        by_dim = list(zip(*[iter(coord_set) for coord_set in in_pairs]))
        adj = ([min(pair[0] for pair in dim) for dim in by_dim] * 2
               for coord_set in ranges)

    # Return the input sequence with the adjustments subtracted:
    return tuple(tuple(c-a for c, a in zip(coord_set, adj_set)) \
                 for coord_set, adj_set in zip(ranges, adj))


def combine_limits(ranges, to_FITS=False):
    """
    Given a sequence of (n-dimensional) co-ordinate range sequences
    (eg. ((x1a, x2a, y1a, x2a), (x1b, x2b, y1b, y2b)) ), reduce them to
    a single tuple spanning the combined range of the data.
    If to_FITS (default False) is True, also convert from an exclusive,
    zero-based Python range to an inclusive, 1-based FITS/IRAF range by
    adding 1 to each lower limit.

    """
    # Offset each lower limit to convert ranges to FITS conv. if requested:
    offset = 1 if to_FITS else 0

    # Split each set of limits into a set of lower & higher limits for each
    # axis, invert the nesting (by dimension first then co-ordinate set) &
    # find the min/max of the low/high limits, respectively, for each axis:
    llims = tuple(min(axis) + offset for axis in
                  zip(*[coord_set[::2] for coord_set in ranges]))
    hlims = tuple(max(axis) for axis in
                  zip(*[coord_set[1::2] for coord_set in ranges]))

    # Recombine the limits into a single tuple:
    return tuple(val for axis in zip(llims, hlims) for val in axis)

