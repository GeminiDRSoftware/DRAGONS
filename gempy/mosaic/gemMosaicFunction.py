#
#                                                                  gemini_python
#
#                                                           gemMosaicFunction.py
# ------------------------------------------------------------------------------
import os
import numpy as np
from importlib import import_module

from .mosaicData import MosaicData
from .mosaicGeometry import MosaicGeometry

# ------------------------------------------------------------------------------
def gemini_mosaic_function(ad):
    """ 
    Default function returning data arrays and instrument geometry values. The
    geometry values are constant which are set for each instrument and they are 
    accesed using the Lookup dictionary. These dictionaries are named 
    'geometry_conf.py' for each instrument, currently GMOS and GSAOI.
        
    *Input*
    
    :param ad: Astrodata Object
    :type ad: <AstroData>

    :param ref_extname: Extension name to read data arrays from ad.
    :type ref_extname: <str>, Default is 'SCI'.

    *Output*
    
    **mosdata:**  Data object containing the following elements:
    
    :param data_list: List of extension data ndarrays.

    :param coords: Dictionary with coordinates describing each element
                   of data_list. The coordinates are tuples of the form 
                   (x1,x2,y1,y2), width and height zero-based.

    Example:
        {'amp_mosaic_coord':detsecs,'amp_block_coord':ccdsecs} 

    detsecs is a list of tuples.
    
    **geometry:** Geometry object containing the following attributes:
    
    :param gaps: (x_width,y_width) Tuple indicating the number of pixels
                 of separation between the detector chips.

    :param blocksize: (x_pixel, y_pixel). Size of a block (detector chip)

    :param mosaic_grid: Mosaic grid tuple (nblocks_x,n_rows). Number of blocks in
                        x-direction and the number of rows.

    :param ref_block: reference block tuple (colum,row) (0-based). Tuple (0,0)
                      refers to the lower left block in the mosaic.

    :param tranformation: Dictionary with key shift, rotation and magnification.
        shift: List of tuples (x_shift, y_shift) indicating the number of pixels 
               to shift each block with respect to the reference block.

        rotation: List of rotation values (degrees ) indicating the rotation of 
                  each block with respect to the reference block.
         magnification: List of magnification factor for each block.

    :param interpolator: ('linear'). Function name used to correct each block for
                                     shift, rotation and magnification.
        
    """
    if not 'GMOS' in ad.tags and not 'GSAOI' in ad.tags:
        raise TypeError('Input file is not supported by MosaicAD: ',
                            str(ad.filename),' Type:',str(ad.tags))

    md = MosaicData()          # Creates an empty object
    md.data_list = []

    detsecs = ad.detector_section()
    ccdsecs = ad.array_section()
    datasecs = ad.data_section()
    for n in range(len(datasecs)):
        (x1, x2, y1, y2) = datasecs[n]
        md.data_list.append(ad[n].data[y1:y2, x1:x2])

    # Get binning
    x_bin = ad.detector_x_bin()
    y_bin = ad.detector_y_bin()
    if x_bin is None:
        binning = (1, 1)
    else:
        binning = (x_bin, y_bin)

    # ccdsecs and detsecs are unbinned.
    ccdsecs = [(k[0]/x_bin, k[1]/x_bin, k[2]/y_bin, k[3]/y_bin) for k in ccdsecs]
    detsecs = [(k[0]/x_bin, k[1]/x_bin, k[2]/y_bin, k[3]/y_bin) for k in detsecs]

    md.coords = {'amp_mosaic_coord': detsecs,
                 'amp_block_coord' : ccdsecs,
                 'order':  range(len(ad))
    }

    # Reads and sets the geometry values from the Lookup table
    # returning a dictionary
    geo_dict = _set_geo_values(ad, ccdsecs, detsecs, binning)

    # Create a MosaicGeometry object with values from the geo_dict.
    geometry = MosaicGeometry(geo_dict)

    return md, geometry


def _set_geo_values(ad, ccdsecs, detsecs, binning):
    """
    Read geometric values from the geometry configuration lookup tables.
    These tables are dynamically imported based on instrument, currently either
    GMOS or GSAOI.
       
       *Input:*
         ad: AD object

       *Output:*
         dictionary:  Keys are: gaps, blocksize, mosaic_grid,
                      shift,rotation,magnification,interpolator,ref_block

    """
    geometry_module = 'geminidr.{}.lookups.geometry_conf'
    lookup_tables = geometry_module.format(ad.instrument_name.lower())
    geotable = import_module(lookup_tables)
    
    # key elements for chip geometries
    dettype      =  ad.phu['DETTYPE']
    detector     =  ad.phu['DETECTOR']
    instrument   = ad.instrument()
    x_bin, y_bin = binning
    if x_bin > 1 or y_bin > 1:
        bin_string = "binned"
    else:
        bin_string = "unbinned"

    binning_key  = (instrument, dettype, detector, bin_string)
    unbinned_key = (instrument, dettype, detector, 'unbinned')

    xshift, yshift = np.asfarray(geotable.shift[binning_key]).transpose()
    tab_rotation   = geotable.rotation[binning_key]
    rot      = np.asfarray(tab_rotation)
    xrot     = rot * x_bin / y_bin
    yrot     = rot * y_bin / x_bin
    rotation = [(x, y) for x, y in zip(xrot, yrot)]
    xshift   = xshift / x_bin
    yshift   = yshift / y_bin
    shift    = [(x, y) for x, y in zip(xshift, yshift)]

    # For x,y gap
    gaps_tile = geotable.gaps_tile[binning_key]
    gaps_transform = geotable.gaps_transform[binning_key]
    for k in gaps_tile.keys():                           # Bin the values
        gaps_tile[k] = (gaps_tile[k][0] / x_bin, 
                        gaps_tile[k][1] / y_bin)

    for k in gaps_transform.keys():                      # Bin the values
        gaps_transform[k] = (gaps_transform[k][0] / x_bin, 
                             gaps_transform[k][1] / y_bin)

    blocksize     = np.asfarray(geotable.blocksize[unbinned_key])
    nrows         = blocksize[0] / x_bin
    ncols         = blocksize[1] / y_bin
    blocksize     = (nrows, ncols)
    mosaic_grid   = geotable.mosaic_grid[unbinned_key]
    magnification = geotable.magnification[unbinned_key]
    ref_block     = tuple(geotable.ref_block['ref_block'])
    interpolator  = geotable.interpolator['SCI']
    gap_dict      = {'tile_gaps': gaps_tile, 'transform_gaps': gaps_transform}

    # Dictionary with all the values. Useful for printing.
    geodict = {'blocksize': blocksize,
               'mosaic_grid': mosaic_grid,
               'transformation': {'shift':shift,'rotation':rotation,
                                  'magnification':magnification},
               'interpolator': interpolator,
               'ref_block': ref_block,
               'gap_dict': gap_dict,
    }

    return geodict   

