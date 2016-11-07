#
#                                                                  gemini_python
#
#                                                              mosaicGeometry.py
# ------------------------------------------------------------------------------
"""
mosaicGeometry provides the MosaicGeometry class.

"""
class MosaicGeometry(object):
    """
    The MosaicGeometry class provides functionality to verify the
    input dictionary elements and set all the require attributes.

    Usage:
    mosaic_geometry = MosaicGeometry(dictionary)

    """

    def __init__(self, geometry_dict):
        """
        .. _help_mgeo:

        Parameters
        ----------

        geometry_dict:  Dictionary with the following keys:
        --------------------------------------------------
        NOTE: required keys are blocksize and mosaic_grid.

        blocksize:        # Required
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
            'gap_dict' is a dictionary with two keys: 'tile_gaps' and
            'transform_gaps', with value a dictionary with key the block
            tuple and values a tuple (x_gap, y_gap) with the gap for
            each block. Where the gap is added to the block is left to
            the handling method.

            If you have different set of gaps for 'tiling' and
            'transform', then use this entry. It takes precedence over
            the 'gaps' entry.

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
                'shift':[(43.60,-1.24),
                         (0.,0.),
                         (0.02, 41.10),
                         (43.42, 41.72)], # List of (x_shift,y_shift)
                'rotation': (-1.033606, 0.0,
                            0.582767, 0.769542), # List of degrees, counterwise
                                                 # w/r to the x_axis
                'magnification': (1.0013, 1.,
                                1.0052, 1.0159),   # List of magnification
              },
              'interpolator': 'linear',       # transformation function
              'blocksize':    (1024,2048),    # shape of block (naxis1,naxis2)
              'ref_block':    (1,0), # Position of reference block w/r to mosaic
                                     # grid. The lower left block is (0,0).
              'mosaic_grid':  (4,1)          # output mosaic layout: number
                                             # of blocks in (x,y)
                             }
         >>> geo = MosaicGeometry(geometry_dict)
         >>> geo.info()    # prints out the dictionary

        """
        # The required keys are mosaic_grid and blocksize
        mosaic_grid = geometry_dict['mosaic_grid']
        blocksize = geometry_dict['blocksize']

        if (type(mosaic_grid) is not tuple) or (len(mosaic_grid) < 2):
            raise ValueError('Mosaic grid: Should be a 2 value tuple')
        else:
            nblocks = mosaic_grid[0] * mosaic_grid[1]     # The number of blocks
                                                          # in the mosaic
        self.mosaic_grid = mosaic_grid

        if (type(blocksize) is not tuple) or (len(blocksize) < 2):
            raise ValueError('Blocksize: Should be a 2 value tuple')
        else:
            self.blocksize = blocksize


        # Linear interpolator is the default
        interpolator = 'linear'
        if 'interpolator' in geometry_dict:
            interpolator = geometry_dict['interpolator']

        if interpolator == 'spline':
            if 'spline_order' in geometry_dict:
                spline_order = geometry_dict['spline_order']
                if not (0 <= spline_order <= 5):
                    raise ValueError('spline_order: Should be >0 or <6')
            else:
                spline_order = 2

            self.spline_order = spline_order

        self.interpolator = interpolator

        gap_dict = None
        if 'gap_dict' in geometry_dict:
            gap_dict = geometry_dict['gap_dict']
            if not ('tile_gaps' in gap_dict or ('transform_gaps' in gap_dict)):
                raise ValueError("gap_dict: key is not 'tile_gaps' or "
                                 "'transform_gaps'")

            for gap_type in ['tile_gaps', 'transform_gaps']:
                for key, val in zip(gap_dict[gap_type].keys(),
                                    gap_dict[gap_type].values()):
                    if (type(key) is tuple) and (type(val) is tuple):
                        if (len(key) != 2) or (len(val) != 2):
                            raise ValueError("Gaps values are not of length 2")
                    else:
                        raise ValueError("Gaps keys are not tuples")

        elif 'gaps' in geometry_dict:
            # This is dictionary {(col,row): (x_gap,y_gap)....}
            # that applies to tile and transform
            gaps = geometry_dict['gaps']
            if len(gaps) != nblocks:
                raise ValueError("Gaps dictionary length is not: " +
                                 str(nblocks))
            for key, val in zip(gaps.keys(), gaps.values()):
                if (type(key) is tuple) and (type(val) is tuple):
                    if (len(key) != 2) or (len(val) != 2):
                        raise ValueError("Gaps values are not of length 2")
                else:
                    raise ValueError("Gaps keys are not tuples")

            gap_dict = {'tile_gaps':gaps, 'transform_gaps':gaps}
        else:
            # No gaps giving in input dictionary.
            # Make a gap dictionary. x_gap to the left and y_gap at the bottom
            # of each block. First row is zero at the bottom, first column is
            # zero to left of each block
            dval = {}
            dval[0, 0] = (0, 0)
            for col in range(1, mosaic_grid[0]):
                dval[col, 0] = (0, 0)
            for row in range(1, mosaic_grid[1]):
                dval[0, row] = (0, 0)
            for row in range(1, mosaic_grid[1]):
                for col in range(1, mosaic_grid[0]):
                    dval[col, row] = (0, 0)
            gap_dict = {'tile_gaps': dval,
                        'transform_gaps': dval}
        self.gap_dict = gap_dict

        # 'transformation' key is a dictionary with keys shift, rotation
        # and magnification. All these are optional.
        #
        trans = {}
        if 'transformation' in geometry_dict:
            trans = geometry_dict['transformation']

        if 'shift' in trans:
            shift = trans['shift']
            if len(shift) != nblocks:     # Must have nblocks tuples of size 2.
                raise ValueError('shift: There must be ' +
                                 str(nblocks) + ' values.')
            for val in shift:             # check that each element is tuple
                if len(val) != 2:
                    raise ValueError('One of the tuples in "shift" is not '
                                     'of length 2.')
        else:
            shift = [(0., 0.)] * nblocks

        if 'rotation' in trans:
            rotation = trans['rotation']
            if len(rotation) != nblocks:  # Must have nblocks tuples of size 2.
                raise ValueError('rotation: There must be ' +
                                 str(nblocks) + ' values.')
            if type(rotation[0]) is not tuple:
                rotation = [(r, r) for r in rotation]   # Turn to tuple
        else:
            rotation = [(0., 0.)] * nblocks

        if 'magnification' in trans:
            magnification = trans['magnification']
            # must have nblocks tuples of size 2.
            if len(magnification) != nblocks:
                raise ValueError('magnification: There must be ' +
                                 str(nblocks) + ' values.')
            if type(magnification[0]) is not tuple:
                magnification = [(m, m) for m in magnification] # turn to tuple
        else:
            magnification = [(1., 1.)] * nblocks

        self.transformation = {'shift': shift,
                               'rotation': rotation,
                               'magnification': magnification}

        # If the key is not defined, set it to the lower left
        ref_block = (0, 0)
        if 'ref_block' in geometry_dict:
            ref_block = geometry_dict['ref_block']
        self.ref_block = ref_block

    def info(self):
        """
        Return the geometry dictionary


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
