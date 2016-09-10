#
#                                                                  gemini_python
#
#                                                                   gempy.gemini
#                                                           gemMosaicFunction.py
# ------------------------------------------------------------------------------
# $Id$
# ------------------------------------------------------------------------------
__version__      = '$Revision$'[11:-2]  # Changed by swapper, 22 May 2014
__version_date__ = '$Date$'[7:-3]
# ------------------------------------------------------------------------------
import numpy as np

from astrodata.utils import Lookups

from ..library.mosaic import MosaicData, MosaicGeometry

# ------------------------------------------------------------------------------
def gemini_mosaic_function(ad, ref_extname='SCI'):
    """ 
        Default function returning data arrays and instrument
        geometry values. The geometry values are constant which
        are set for each instrument and they are accesed using the
        AStrodata Lookup dictionary service. These dictionaries
        are in Gemini instrument directory with names 'geometry_conf.py'
        
        *Input*

        :param ad:       Astrodata Object
        :param ref_extname: Extension name to read data arrays from ad.
        :type ref_extname: Default is 'SCI'.

        *Output*

        **mosdata:**   Data object containing the following elements:

        :param data_list: List of 'SCI' extension ndarrays.
        :param coords:    Dictionary with coordinates describing each element
                          of data_list. The coordinates are tuples of the
                          form (x1,x2,y1,y2), width and height zero-based.
                          Example:
                               {'amp_mosaic_coord':detsecs,'amp_block_coord':ccdsecs} 
                               detsecs is a list of tuples.

        **geometry:** Geometry object containing the following attributes:

        :param gaps:      (x_width,y_width) Tuple indicating the number of pixels
                        of separation between the detector chips.
        :param blocksize:   (x_pixel, y_pixel). Size of a block (detector chip)
        :param mosaic_grid: Mosaic grid tuple (nblocks_x,n_rows). Number of blocks in
                            x-direction and the number of rows. 
        :param ref_block:  reference block tuple (colum,row) (0-based). Tuple (0,0)
                           refers to the lower left block in the mosaic.
        :param tranformation: Dictionary with key shift, rotation and magnification.
                              shift: List of tuples (x_shift, y_shift) indicating the
                                     number of pixels to shift each block with respect
                                     to the reference block.
                              rotation: List of rotation values (degrees ) indicating
                                     the rotation of each block with respect to the
                                     reference block.
                              magnification: List of magnification factor for each block.
        :param interpolator: ('linear'). Function name used to correct each block for
                                         shift, rotation and magnification.
        
    """
    
    # We only take GMOS and GSAOI for now

    if not 'GMOS' in ad.types and not 'GSAOI' in ad.types:
        raise RuntimeError('Input file is not supported by MosaicAD: ',
                            str(ad.filename),' Type:',str(ad.types))

    md = MosaicData()    # Creates an empty object
    md.data_list = []

    md._current_extname = ref_extname
    # Use as_dict in the meantime because with as_list it deletes
    # repeated elements.

    # We need to order the image_sections in increasing extv order. It can be done
    # as follows (assuming that we have the ccdsec,detsec and datasec keywords
    # in each extension.
    detsecsdict = ad[ref_extname].detector_section().as_dict()
    ccdsecsdict = ad[ref_extname].array_section().as_dict()
    datasecsdict = ad[ref_extname].data_section().as_dict()

    extlist = [s[1] for s in detsecsdict.keys()]    # Get extvers in a list

    # get dictionary values in lists
    extname = '*'
    dets_list = [detsecsdict[extname,k] for k in extlist]
    ccds_list = [ccdsecsdict[extname,k] for k in extlist]
    dats_list = [datasecsdict[extname,k] for k in extlist]
    
    # get the indices to sort the list in ascending order
    sort_indices = np.asarray(extlist).argsort()
    dets_list = [dets_list[k] for k in sort_indices]
    ccds_list = [ccds_list[k] for k in sort_indices]
    dats_list = [dats_list[k] for k in sort_indices]

    # Do datalist.......


    # Form the image_section lists first in extver order (we assumed, the extvers
    # are in a sequence starting from 1).
    nextensions = len(detsecsdict)
    detsecs = [detsecsdict[extname,extv+1] for extv in range(nextensions)]
    ccdsecs = [ccdsecsdict[extname,extv+1] for extv in range(nextensions)]
    datasecs = [datasecsdict[extname,extv+1] for extv in range(nextensions)]
    for extv in range(nextensions):
        (x1,x2,y1,y2) = tuple(datasecs[extv])
        md.data_list.append(ad[ref_extname,extv+1].data[y1:y2,x1:x2])
        
    if detsecs != dets_list:
        print "++++++++++++++++++++!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!****"
        print " **************** lists are not equal ***************" 
        print "\n >>>>>>>>>>>>>>>>>>>>>DETSEC from dictio:",detsecs
        print "\n >>>>>>>>>>>>>>>>>>>>>DETSEC from sorted_indx:",dets_list
        raise ValueError("detsecs list and dictionaries are differents!!!!!")

    # Get binning
    x_bin,y_bin = (ad[ref_extname].detector_x_bin().as_pytype(),
                   ad[ref_extname].detector_y_bin().as_pytype())
    binning = (x_bin,y_bin)
    if x_bin == None:
        binning = (1,1)

    # ccdsecs and detsecs are unbinned.
    ccdsecs = [(k[0]/x_bin,k[1]/x_bin,k[2]/y_bin,k[3]/y_bin) for k in ccdsecs]
    detsecs = [(k[0]/x_bin,k[1]/x_bin,k[2]/y_bin,k[3]/y_bin) for k in detsecs]

    md.coords = {'amp_mosaic_coord':detsecs,'amp_block_coord':ccdsecs}
    md.coords['order'] = range(nextensions)

    # Reads and sets the geometry values from the Lookup table
    # returning a dictionary
    geo_dict = _set_geo_values(ad,ccdsecs,detsecs,binning)

    # Create a MosaicGeometry object with values from the geo_dict.
    geometry = MosaicGeometry(geo_dict)

    return md, geometry


def _set_geo_values(ad,ccdsecs,detsecs,binning):
    """Read geometric values from the Lookup tables.
       
       *Input:*
         ad: AD object

       *Output:*
         dictionary:  Keys are: gaps,blocksize,mosaic_grid,
                      shift,rotation,magnification,interpolator,ref_block


    """
    _geoVal = {'gaps':None, 'blocksize':None, 'mosaic_grid':None,
                    'shift':None, 'rotation':None, 'magnification':None,
                    'interpolator':None,'ref_block':None}

    # Get the dictionary containing the GMOS geometric information 
    # needed by mosaicAD
    instrument = str(ad.instrument())
    
    dettype =  ad.phu_get_key_value("DETTYPE")
    detector =  ad.phu_get_key_value("DETECTOR")

    x_bin,y_bin = binning
    if (x_bin > 1) or (y_bin >1):
        bin_string = "binned"
    else:
        bin_string = "unbinned"


    if 'GMOS' in ad.types:  
        lookup = 'Gemini/GMOS/geometry_conf'
    else:
        lookup = 'Gemini/'+instrument+'/geometry_conf'

    # Now we used the Lookup table service to read a given
    # list of dictionaries in the list 'dnames' located
    # in modules 'geometry_conf.py'.
    # Then we form a key based on the instrument, dettype,
    # detector, and bin_string to load the correct value
    # in the hidden dictionary _geoVal.


    # These are names using 'binned', 'unbinned'
    dnames = ('gaps_tile', 'gaps_transform','shift')
    geo_table = Lookups.get_lookup_table(lookup, *dnames)

    key = (instrument, dettype, detector, bin_string)
    for dn,gdic in zip(dnames,geo_table):
        _geoVal[dn] = gdic[key]

    # These are names using 'unbinned' only
    dnames = ('blocksize', 'mosaic_grid',\
               'rotation','magnification')
    # Reset dictionary search key for these dnames.
    key = (instrument, dettype, detector, 'unbinned')
    geo_table = Lookups.get_lookup_table(lookup, *dnames)

    for dn,gdic in zip(dnames,geo_table):
        # Get the dictionary item value given the key
        _geoVal[dn] = gdic[key]
          

    geo_table = Lookups.get_lookup_table(lookup, 'interpolator')
    _geoVal['interpolator'] = geo_table['SCI']

    geo_table = Lookups.get_lookup_table(lookup, 'ref_block')
    _geoVal['ref_block'] = np.asarray(geo_table['ref_block'])


    # Now binned the appropiate parameters.


    xshift,yshift =  np.asfarray(_geoVal['shift']).transpose()
    rot  = np.asfarray(_geoVal['rotation'])
    xrot = rot*x_bin/y_bin
    yrot = rot*y_bin/x_bin
    xshift = xshift/x_bin
    yshift = yshift/y_bin
    rotation = [(x,y) for x,y in zip(xrot,yrot)]
    shift = [(x,y) for x,y in zip(xshift,yshift)]
    # For x,y gap
    gaps_tile      = _geoVal['gaps_tile']
    gaps_transform = _geoVal['gaps_transform']
    for k in gaps_tile.keys():          # Binn the values
           gaps_tile[k] = (gaps_tile[k][0]/x_bin,gaps_tile[k][1]/y_bin)
    for k in gaps_transform.keys():          # Binn the values
           gaps_transform[k] = (gaps_transform[k][0]/x_bin,
                                gaps_transform[k][1]/y_bin)

    blocksize = np.asfarray(_geoVal['blocksize'])
    nrows = blocksize[0]/x_bin
    ncols = blocksize[1]/y_bin
    blocksize = (nrows,ncols)
    mosaic_grid = _geoVal['mosaic_grid']
    magnification = _geoVal['magnification']
    ref_block = tuple(_geoVal['ref_block'])
    interpolator = _geoVal['interpolator']
    
    nblocksx,nblocksy = mosaic_grid
    # Determines the actual block size depending if we have
    # or not a ROI.
    # For return_ROI True
    # Get minimum lower left corner from all amplifier coords
    # Use as_dict in the meantime because with as_list it deletes
    # repeated elements.

    gap_dict = {'tile_gaps':gaps_tile, 'transform_gaps':gaps_transform}


    # Dictionary with all the values. Useful for printing.
    geodict = {'blocksize':blocksize,
               'mosaic_grid':mosaic_grid, 
               'transformation':{'shift':shift,'rotation':rotation,
                                 'magnification':magnification},
               'interpolator':interpolator,
               'ref_block':ref_block,'gap_dict':gap_dict,
               }

    return geodict   

