#
#                                                                        DRAGONS
#
#                                                                    mosaicAD.py
# ------------------------------------------------------------------------------
from os.path import join
from os.path import dirname

import numpy as np

import astropy.wcs as wcs

import astrodata
import gemini_instruments

from gempy.utils import logutils
from gempy.gemini.gemini_tools import tile_objcat
from geminidr.gemini.lookups.source_detection import sextractor_dict

from .mosaic import Mosaic
# ------------------------------------------------------------------------------
__version__ = "2.0"
# ------------------------------------------------------------------------------
class MosaicAD(Mosaic):
    """
    MosaicAD as a subclass of Mosaic extends its functionality by providing
    support for:

    - Astrodata objects with more than one extension name,
      i.e., 'SCI', 'VAR', 'DQ'.
    - Creating output mosaics in Astrodata objects.
    - Updating the WCS information in the output Astrodata object mosaic header.
    - A user_function as a parameter to input data and geometric values
      of the individual data elements.
    - A user_function (already written) to support GMOS and GSAOI data.

    Methods
    -------
    as_astrodata       - Output mosaics as AstroData objects.
    tile_as_astrodata  - Output tiled data as AstroData objects. Tile into
                         one extension per detector chip or all extensions
                         into a single extension.
    mosaic_image_data  - Create a single extension mosaic from extensions.
    mosaic_header      - Make a mosaic FITS header from reference extension
                         header.
    get_data_list      - Return a list of image data for a given extname
                         extensions in the input AstroData object.
    update_wcs         - Update the WCS information in the output header.
    info               - Creates a dictionary with coordinates, amplifier
                         and block information.

    Attributes   (In addition to parent class attributes)
    ----------
    ad            - Astrodata object
    data_list     - a list of array sections from all extensions.
    log           - Logutils object
    mosaic_data_array
                  - Dictionary of numpy arrays keyed by extension name.

    Notes
    -----
    The steps to creating a MosaicAD object are as follows:

    * Instantiate an Astrodata object with a GMOS or GSAOI fits file.

    * Import the function gemini_mosaic_function from the gemMosaicFunction
      module. This function reads the FITS extensions with amplifier data and
      create a list of ndarrays; it also reads a dictionary of geometry values
      from a module located in instrument lookups tables.

    """
    def __init__(self, ad, mosaic_ad_function):
        """
        Parameters
        ----------
        ad: <AstroData>, Input Astrodata object

        mosaic_ad_function: <func>, A required user function returning a
            MosaicData and a MosaicGeometry objects. This function that will act
            as an interface to the particular 'ad', e.g., knows which keywords
            represent the coordinate systems to use and whether they are binned
            or not, or which values in the geometry look-up table require to be
            binned. For instruments GMOS and GSAOI, DRAGONS/mosaic provides the
            'gemini_mosaic_function'.

        For help on this function please see its description in the mosaic.py
        module.

        """
        verr = "Nothing to mosaic. < 2 extensions found on file {}"
        self.ad = ad
        if len(ad) < 2:
            raise ValueError(verr.format(ad.filename))

        self.log = logutils.get_logger(__name__)
        mosaic_data, geometry = mosaic_ad_function(ad)  # Call geometry function.
        Mosaic.__init__(self, mosaic_data, geometry)
        self.jfactor = []               # Jacobians applied to interpolated pixels.
        self.calculate_jfactor()        # Fill the jfactor vector with the
                                        # jacobian of transformation matrix.
        self.mosaic_shape = None        # Shape of the mosaicked output frame.
        self.sx_dict = sextractor_dict.sx_dict.copy()
        # Prepend paths to SExtractor input files now
        self.sx_dict.update({k: join(dirname(sextractor_dict.__file__), v)
                             for k, v in self.sx_dict.items()})

    # --------------------------------------------------------------------------
    def as_astrodata(self, block=None, doimg=False, tile=False, return_ROI=True):
        """
        Returns an AstroData object  containing by default the mosaiced IMAGE
        extensions. WCS information in headers of the IMAGE extensions and
        updated appropriately. When tiling, OBJCATS are retained.

        Parameters
        ----------
        block: <2-tuple>
              Return a specific block as the output mosaic as (col, row).
              (0, 0) is lower left.

        tile: <bool>
              Tile rather than transform data blocks. Default is False.

        doimg: <bool>
              Process only science ("SCI') extensions. Default is False.

        return_ROI: <bool>
              Returns the minimum frame size calculated from the location of the
              amplifiers in a given block. If False, uses the blocksize value.
              Default is True.

        Returns
        -------
        adout: <AstroData> instance, mosaic or tiled as requested.

        """
        adout = astrodata.create(self.ad.phu)
        adout.phu['TILED'] = (repr(tile).upper(), "True: tiled; False: Mosaic")

        # image arrays mosaicked: 'data', 'variance', 'mask', 'OBJMASK'.
        # SCI
        self.data_list = self.get_data_list('data')
        if not self.data_list:
            emsg = "MosaicAD received a dataset with no data: {}"
            self.log.error(emsg.format(self.ad.filename))
            raise IOError("No science data found on file {}".format(self.ad.filename))
        else:
            self.log.stdinfo("MosaicAD working on data arrays ...")
            darray = self.mosaic_image_data(block=block,return_ROI=return_ROI,tile=tile)
            self.mosaic_shape = darray.shape
            header = self.mosaic_header(darray.shape, block, False)
            adout.append(darray, header=header)

        # VAR
        varray = None
        if not doimg:
            self.data_list = self.get_data_list('variance')
            if not self.data_list:
                self.log.stdinfo("No VAR array on {} ".format(self.ad.filename))
            else:
                self.log.stdinfo("Working on VAR arrays ...")
                varray = self.mosaic_image_data(block=block,return_ROI=return_ROI,
                                                tile=tile)
        # DQ
        marray = None
        if not doimg:
            self.data_list = self.get_data_list('mask')
            if not self.data_list:
                self.log.stdinfo("No DQ array on {} ".format(self.ad.filename))
            else:
                self.log.stdinfo("Working on DQ arrays ...")
                marray= self.mosaic_image_data(block=block,return_ROI=return_ROI,
                                               tile=tile, dq_data=True)

        adout[0].reset(data=darray, variance=varray, mask=marray)

        # Handle extras ...
        if not doimg:
            self.data_list = self.get_data_list('OBJMASK')
            if not self.data_list:
                self.log.stdinfo("No OBJMASK on {} ".format(self.ad.filename))
            else:
                self.log.stdinfo("Working on OBJMASK arrays ...")
                adout[0].OBJMASK = self.mosaic_image_data(block=block,
                                                          return_ROI=return_ROI,
                                                          tile=tile, dq_data=True)

        # When tiling, tile OBJCATS
        if not doimg and tile:
            self.log.stdinfo("Tiling OBJCATS ...")
            adout = self._tile_objcats(adout)

        # Propagate any REFCAT
        if not doimg:
            if hasattr(self.ad, 'REFCAT'):
                self.log.stdinfo("Keeping REFCAT ...")
                adout.REFCAT = self.ad.REFCAT

        return adout

    # --------------------------------------------------------------------------
    def tile_as_astrodata(self, tile_all=False, doimg=False, return_ROI=True):
        """
        Returns an AstroData object  containing by default the tiled IMAGE
        extensions. WCS information in headers of the IMAGE extensions and
        updated appropriately. When tiling, OBJCATS are retained.

        Parameters
        ----------
        tile_all: <bool>
            Tile data blocks into a single extension. If True, uses the standard
            as_astrodata() method with tile=True. Default is False.

        doimg: <bool>
            Process only science ("SCI') extensions. Default is False.

        return_ROI: <bool>
            Returns the minimum frame size calculated from the location of the
            amplifiers in a given block. If False, uses the blocksize value.
            Default is True.

        Returns
        -------
        adout: <AstroData> instance, tiled as requested.

        """
        block = None
        if tile_all:
            adout = self.as_astrodata(block=block, tile=True, doimg=doimg,
                                      return_ROI=return_ROI)
        else:
            adout = self._tile_blocks(block=block, doimg=doimg,
                                      return_ROI=return_ROI)

        return adout

    # --------------------------------------------------------------------------
    def calculate_jfactor(self):
        """
        Calculate the ratio of reference input pixel size to output pixel size.
        In practice this ratio is formulated as the determinant of the
        WCS tranformation matrix. This is the ratio that we will applied to each
        pixel to conserve flux in a feature.

        *Justification:*

        In general CD matrix element is the ration between partial derivative of
        the world coordinate (ra,dec) with respect to the pixel coordinate (x,y).
        We have 4 elements: cd11, cd12, cd21, and cd22.

        For an adjacent image in the sky (GMOS detectors 1,2,3 for example), the
        cd matrix elements will have slightly differents values.

        Given CD1 and CD2 as CD matrices from adjacent fields then the determinant
        of the dot product of the inverse CD1 times the CD2 will give the
        correcting factor.

        """
        # If there is no WCS return 1 list of 1.s
        try:
           ref_wcs = wcs.WCS(self.ad[0].hdr)
        except:
           self.jfactor = [1.0] * len(self.ad)
           return

        # Get CD matrix for each extension, calculate the transformation
        # matrix from composite of the reference cd matrix and the current one.
        self.jfactor.append(1.0)
        for ext in self.ad[1:]:
            header = ext.hdr
            if 'CD1_1' not in header:
                self.jfactor.append(1.0)
                continue

            try:
                img_wcs = wcs.WCS(header)
                # Cross product of both CD matrices
                matrix =  np.dot(np.linalg.inv(img_wcs.wcs.cd), ref_wcs.wcs.cd)
                matrix_det = np.linalg.det(matrix)
            except:
                jferr = "calculate_jfactor: Error calculating matrix_det."
                jferr += "Setting jfactor = 1"
                self.log.warning(jferr)
                matrix_det = 1.0

            # Fill out the values for each extension.
            self.jfactor.append(matrix_det)

    # --------------------------------------------------------------------------
    def get_data_list(self, attr):
        """
        Parameters
        ----------
        attr: Attribute of the member self.ad. This attribute is one of a number
              of image ndarrays that may be present on the instance of 'self.ad',
              and will be one of,
                  'data', 'mask', 'variance', 'OBJMASK'.
        type: <str>

        Return
        ------
        data_list: a list of actual array sections from 'attr'.
        type: <list>

        """
        data_list = []
        for ex in self.ad:
            xsec = ex.data_section()
            if hasattr(ex, attr):
                darray = getattr(ex, attr)
            else:
                darray = None

            if darray is not None:
                data_list.append(darray[xsec.y1: xsec.y2, xsec.x1: xsec.x2])

        return data_list

    # --------------------------------------------------------------------------
    def mosaic_image_data(self, block=None, dq_data=False, tile=False,
                          return_ROI=True):
        """
        Creates the output mosaic ndarray of the requested IMAGE extension.

        Parameters:
        ---------
        block: <2-tuple>, e.g., (<int>, <int>)
              Allows a specific block to be returned as the output mosaic.
              The tuple notation is (col, row) (0-based), where (0, 0) is the
              lower left block.  This is position of the reference block w/r to
              mosaic_grid. Default is None.

        tile: <bool>
              If True, the mosaics returned are not corrected for shifting and
              rotation.

        return_ROI: <bool>
              Returns the minimum frame size calculated from the location of
              the amplifiers in a given block. If False uses the blocksize
              value. Default is True.

        dq_data: <bool>
              Handle data in self.data_list as bit-planes.

        Return:
        ------
        out: <ndarray>, ndarray instance of the mosaiced data.

        """
        out = Mosaic.mosaic_image_data(self,block=block,dq_data=dq_data,tile=tile,
                                       jfactor=self.jfactor,return_ROI=return_ROI)
        return out
 
    # --------------------------------------------------------------------------
    def mosaic_header(self, mosaic_shape, block, tile):
        """
        Make the mosaic FITS header based on the reference extension header.
        Update CCDSEC, DETSEC, DATASEC, CRPIX1, CRPIX2 keywords to reflect the
        mosaic geometry.

        Parameters
        ----------
        mosaic_shape: <2-tuple>, (npixels_y, npixels_x)
            The output mosaic dimensionality

        tile: <bool>
            Transform blocks or not.True: blocks are not transformed.

        block: <tuple> (ncol, nrow)
            Block to return.

        Returns
        -------
        header: <Header>
            Mosaic Fits header object

        """
        mcomm = "Set by MosaicAD, v{}".format(__version__)
        fmat1 = "[{}:{},{}:{}]"
        fmat2 = "[1:{},1:{}]"

        mosaic_hd = self.ad[0].hdr.copy()         # ref ext header.
        ref_block = self.geometry.ref_block  
        amps_per_block = self._amps_per_block

        # ---- update CCDSEC,DETSEC and DATASEC keyword
        # Get keyword names for array_section and data_section
        arr_section_keyw = self.ad._keyword_for('array_section')
        dat_section_keyw = self.ad._keyword_for('data_section')
        det_section_keyw = self.ad._keyword_for('detector_section')
        # Get lowest x1 and y1
        min_x1 = np.min([k[0] for k in self.ad.detector_section()])
        min_y1 = np.min([k[2] for k in self.ad.detector_section()])

        # Unbin the mosaic shape
        x_bin, y_bin = (self.ad.detector_x_bin(), self.ad.detector_y_bin())
        if x_bin is None:
           x_bin, y_bin = (1, 1)

        unbin_width  = mosaic_shape[1] * x_bin
        unbin_height = mosaic_shape[0] * y_bin
        detsec = fmat1.format(min_x1 + 1, min_x1 + unbin_width,
                              min_y1 + 1, min_y1 + unbin_height)

        mosaic_hd[det_section_keyw] = detsec
        mosaic_hd[arr_section_keyw] = fmat2.format(unbin_width, unbin_height)
        mosaic_hd[dat_section_keyw] = fmat2.format(mosaic_shape[1],mosaic_shape[0])

        ccdname = self.ad.detector_name()
        mosaic_hd.set("CCDNAME", ccdname)

        # Remove these keywords from the mosaic header.
        remove = ['FRMNAME', 'FRAMEID', 'CCDSIZE', 'BIASSEC', 'DATATYP']
        for kw in remove:
            if kw in mosaic_hd:
                del mosaic_hd[kw]

        mosaic_hd.set('EXTVER', 1, comment=mcomm, after='EXTNAME')
        pwcs = wcs.WCS(mosaic_hd)

        # Update CRPIX1 and CRPIX2.
        crpix1, crpix2 = self.update_crpix(pwcs, tile)
        mosaic_hd.set("CRPIX1", crpix1, comment=mcomm)
        mosaic_hd.set("CRPIX2", crpix2, comment=mcomm)
        return mosaic_hd

    # --------------------------------------------------------------------------
    def info(self):
        """ 
        Creates a dictionary with coordinates, amplifier, and block information:
        The keys are:

        filename  (type: string) - The original FITS filename

        amps_per_block (type: int) - Number of amplifier in each block

        amp_mosaic_coord (type: list of tuples (x1,x2,y1,y2)) -
            The list of amplifier location within the mosaic. 
            These values do not include the gaps between the blocks

        amp_block_coord (type: list of tuples (x1,x2,y1,y2)) -
            The list of amplifier location within a block.

        interpolator (type: string) -
            The interpolator function name in use when transforming the blocks.

        reference_block - The block number containing the reference amplifier

        """
        info = {}
        info['filename'] = self.ad.filename
        # amp_mosaic_coord ==  DETSECS
        info['amp_mosaic_coord'] = self.coords['amp_mosaic_coord']

        # amp_block_coord == CCDSECS
        info['amp_block_coord'] = self.coords['amp_block_coord']
        info['amps_per_block']  = self._amps_per_block
        info['data_index_per_block'] = self.data_index_per_block
        # out data.data in (x,y) order
        info['amps_shape_no_trimming'] = [k.data.shape[::-1] for k in self.ad]

        geo = self.geometry
        info['interpolator']    = geo.interpolator
        info['reference_block'] = geo.ref_block
        if geo.interpolator not in ['linear', 'nearest']:
            info['spline_order'] = geo.spline_order

        return info

    # --------------------------------------------------------------------------
    def update_crpix(self, wcs, tile):
        """
        Update WCS elements CRPIX1 and CRPIX2 based on the input WCS header of
        the first amplifier in the reference block number.

        Parameters
        ----------
        wcs: <WCS object.>, Reference extension header's WCS object

        tile: <bool>, Tile or transform data. False transforms.

        Return
        ------
        (crpix1, crpix2): <2-tuple>, New pixel reference in output mosaic.

        """
        # Gaps have different values depending whether we have tile or not.
        gap_mode = 'tile_gaps' if tile else 'transform_gaps'
        o_crpix1, o_crpix2 = wcs.wcs.crpix
        ref_col, ref_row = self.geometry.ref_block          # 0-based

        # Get the gaps that are attached to the left and below a block.
        x_gap, y_gap = self.geometry.gap_dict[gap_mode][ref_col,ref_row]

        # The number of blocks in x and number of rows in the mosaic grid.
        nblocks_x, nrows  = self.geometry.mosaic_grid
        amp_mosaic_coords = self.coords['amp_mosaic_coord']
        amp_number = max(0, self.data_index_per_block[ref_col, ref_row][0] - 1)
        amp_index  = max(0, list(self.coords['order'])[amp_number] - 1)

        xoff = amp_mosaic_coords[amp_index][1] if ref_col > 0 else 0
        xgap_sum = 0
        ygap_sum = 0
        for cn in range(ref_col,0,-1):
            xgap_sum += self.geometry.gap_dict[gap_mode][cn,ref_row][0]
            ygap_sum += self.geometry.gap_dict[gap_mode][cn,ref_row][1]

        crpix1 = o_crpix1 + xoff + xgap_sum 

        # Don't change crpix2 unless the output have more than one row.
        crpix2 = o_crpix2
        if nrows > 1:
            yoff = 0
            if ref_col > 0:
               yoff = amp_mosaic_coords[amp_index][3]   # y2
            crpix2 = o_crpix2 + yoff + ygap_sum

        return (crpix1, crpix2)

    def _tile_objcats(self, adout):
        ampsorder = np.argsort([detsec.x1 for detsec in self.ad.detector_section()])
        ccdx1 = np.array([ccdsec.x1 for ccdsec in self.ad.array_section()])[ampsorder]

        # Make a list of the output extensions where each array goes
        num_ccd = 1
        ccd_map = [num_ccd]
        for i in range(1, len(ccdx1)):
            if ccdx1[i] <= ccdx1[i-1]:
                num_ccd += 1
            ccd_map.append(num_ccd)

        ccd_map = np.array(ccd_map)
        adoutput = tile_objcat(adinput=self.ad, adoutput=adout, ext_mapping=ccd_map,
                               sx_dict=self.sx_dict)

        return adoutput

    def _tile_blocks(self, block=None, doimg=False, return_ROI=True):
        """
        Tiles data into separate extensions for each CCD chip.
        In mosaicAD terms, these are called "blocks." Each block comprises
        all amplifier extensions that are part of the chip. Nominal GMOS
        Hamamatsu images result in a 3-extension FITS file, and GSAOI, a
        4-extension FITS file.

        This method is called when the primitive MosaicDetectors passes the
        tile_all parameter as False, which implies chip separated tiling.

        Parameters
        ----------
        block: <2-tuple>
            Users can select a particular block to tile. None indicates all
            blocks will be tiled. Default is None.

        doimg: <bool>
             Tile only the image (SCI) extension data. Fast for quicklook.

        return_ROI: <bool>
            Returns the minimum frame size calculated from the location of the
            amplifiers in a given block. If False, uses the blocksize value.
            Default is True.

        Returns
        -------
        adout: <AstroData>
            instance of astrodata containg the tiled data. Ready for writing.

        """
        warn = "No {} array for block {} on {}"
        adout = astrodata.create(self.ad.phu)
        adout.phu['TILED'] = (True, "True: tiled; False: Mosaic")

        # SCI
        self.data_list = self.get_data_list('data')
        if not self.data_list:
            emsg = "MosaicAD received a dataset with no data: {}"
            self.log.error(emsg.format(self.ad.filename))
            raise IOError("No science data found on file {}".format(self.ad.filename))

        self.log.stdinfo("MosaicAD v{} working on data arrays ...".format(__version__))
        dblocks = self.get_blocks()
        if not doimg:
            # VAR
            self.data_list = self.get_data_list('variance')
            varblocks = self.get_blocks()

            # DQ
            self.data_list = self.get_data_list('mask')
            maskblocks = self.get_blocks()

            # OBJMASK
            self.data_list = self.get_data_list('OBJMASK')
            objmaskblocks = self.get_blocks()

            blocks_indx = list(dblocks.keys())
            i = 0
            for iblock in blocks_indx:
                darray = dblocks[iblock]
                header = self._tile_header(darray.shape, iblock)
                adout.append(darray, header=header)

                varray = None
                if varblocks:
                    self.log.stdinfo("Working on VAR arrays ...")
                    varray = varblocks[iblock]
                else:
                    self.log.stdinfo(warn.format('VAR', iblock, self.ad.filename))

                marray = None
                if maskblocks:
                    self.log.stdinfo("Working on DQ arrays ...")
                    marray = maskblocks[iblock]
                else:
                    self.log.stdinfo(warn.format('DQ', iblock, self.ad.filename))

                adout[i].reset(data=darray, variance=varray, mask=marray)

                if objmaskblocks:
                    self.log.stdinfo("Working on OBJMASK arrays ...")
                    adout[i].OBJMASK = objmaskblocks[iblock]
                else:
                    self.log.stdinfo(warn.format('OBJMASK', iblock, self.ad.filename))
                i += 1

            # tile OBJCATS
            self.log.stdinfo("Tiling OBJCATS ...")
            adout = self._tile_objcats(adout)

            # Propagate any REFCAT
            if hasattr(self.ad, 'REFCAT'):
                self.log.stdinfo("Keeping REFCAT ...")
                adout.REFCAT = self.ad.REFCAT

        return adout

    def _tile_header(self, ashape, block_index):
        """
        Make the mosaic FITS header based on the reference extension header.
        Update CCDSEC, DETSEC, DATASEC, CRPIX1, CRPIX2 keywords to reflect the
        mosaic geometry.

        Parameters
        ----------
        ashape: <2-tuple>, (npixels_y, npixels_x)
            The tiled output shape.

        block_index: <2-tuple>, (<int>, <int>)
            Index of the current working block from attribute,
            .data_index_per_block, where a GMOS image gives,

            block_index
            -----------
                |
            { (2, 0): array([ 8,  9, 10, 11]),
              (1, 0): array([4, 5, 6, 7]),
              (0, 0): array([0, 1, 2, 3])
            }

            This is used to extract and adjust the header of the zeroth
            extension of a given block.

        Returns
        -------
        header: <Header>
            Mosaic Fits header object

        """
        mcomm = "Set by MosaicAD, v{}".format(__version__)
        fmat1 = "[{}:{},{}:{}]"
        fmat2 = "[1:{},1:{}]"

        hindex = self.data_index_per_block[block_index][0]
        theader = self.ad[hindex].hdr.copy()

        # ---- update CCDSEC, DETSEC, DATASEC keywords
        # Get keyword names for array_section and data_section
        arr_section_keyw = self.ad._keyword_for('array_section')
        dat_section_keyw = self.ad._keyword_for('data_section')
        det_section_keyw = self.ad._keyword_for('detector_section')

        # Get lowest x1 and y1
        min_x1 = np.min([k[0] for k in self.ad.detector_section()])
        min_y1 = np.min([k[2] for k in self.ad.detector_section()])

        # Unbin the mosaic shape
        x_bin, y_bin = (self.ad.detector_x_bin(), self.ad.detector_y_bin())
        if x_bin is None:
           x_bin, y_bin = (1, 1)

        unbin_width  = ashape[1] * x_bin
        unbin_height = ashape[0] * y_bin
        detsec = fmat1.format(1, unbin_width, 1, unbin_height)

        theader[det_section_keyw] = detsec
        theader[arr_section_keyw] = fmat2.format(unbin_width, unbin_height)
        theader[dat_section_keyw] = fmat2.format(ashape[1], ashape[0])

        # Can all remain for the tile extensions?
        # Remove these keywords from the mosaic header.
        remove = ['AMPNAME', 'FRMNAME', 'FRAMEID', 'CCDSIZE', 'BIASSEC', 'DATATYP',
                  'OVERSEC', 'TRIMSEC', 'OVERSCAN', 'OVERRMS']
        for kw in remove:
            if kw in theader:
                del theader[kw]

        theader.set('EXTVER', 1, comment=mcomm, after='EXTNAME')
        #pwcs = wcs.WCS(theader)

        # Update CRPIX1 and CRPIX2.
        #crpix1, crpix2 = self.update_crpix(pwcs, True)
        #theader.set("CRPIX1", crpix1, comment=mcomm)
        #theader.set("CRPIX2", crpix2, comment=mcomm)
        return theader
