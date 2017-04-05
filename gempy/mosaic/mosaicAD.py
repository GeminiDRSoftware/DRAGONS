#
#                                                                  gemini_python
#
#                                                                    mosaicAD.py
# ------------------------------------------------------------------------------
import numpy as np
import astropy.wcs as wcs

from copy import copy

import astrodata
import gemini_instruments

from astropy.io import fits
from gempy.utils import logutils
from .mosaic import Mosaic

# ------------------------------------------------------------------------------
__version__ = "2.0.0"
# ------------------------------------------------------------------------------
class MosaicAD(Mosaic):
    """
    MosaicAD as a subclass of Mosaic extends its functionality by providing
    support for:

    - Astrodata objects with more than one extension name; i.e. 'SCI',
      'VAR', 'DQ'.
    - Associating object catalogs in BINARY FITS extensions with
      the image extensions.
    - Creating output mosaics and merge tables in Astrodata objects.
    - Updating the WCS information in the output Astrodata object
      mosaic header.
    - A user_function as a parameter to input data and geometric values
      of the individual data elements.
    - A user_function (already written) to support GMOS and GSAOI data.

    Methods
    -------
    as_astrodata       - Output mosaics as AstroData objects.
    merge_table_data   - Merges catalogs extension that are associated
                         with image extensions.
    mosaic_image_data  - Create a mosaic from extensions.
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

    * Import the function gemini_mosaic_function from the
    gemMosaicFunction module. This function reads the FITS
    extensions with amplifier data and create a list of ndarrays;
    it also reads a dictionary of geometry values from a module
    located in the Astrodata Lookup.

    * If you want to merge object catalogs being associated to each
    input image extension, then provide a dictionary name to the
    parameter 'column_names'. (see __init__ for more details)

    """
    def_columns = { 
        'OBJCAT': ('X_IMAGE', 'Y_IMAGE', 'X_WORLD', 'Y_WORLD'),
        'REFCAT': (None, None, 'RAJ2000', 'DEJ2000')
    }

    def __init__(self, ad, mosaic_ad_function, column_names=def_columns,
                 dq_planes=False):
        """
        Parameters
        ----------

        :param ad: Input Astrodata object

        :param mosaic_ad_function:
              Is a user supplied function that will act as an interface
              to the particular ad, e.g., knows which keywords represent
              the coordinate systems to use and whether they are binned
              or not, or which values in the geometry look up table
              require to be binned.
              For help of this function please see its description
              in the mosaic.py module.

        :type mosaic_ad_function: <func>
              A required user function returning a MosaicData
              and a MosaicGeometry objects.

        :param column_names:
              Dictionary with bintable extension names that are associates
              with input images. The extension name is the key with value
              a tuple: (X_pixel_columnName, Y_pixel_columnName,
              RA_degrees_columnName, DEC_degrees_columnName)
              Example:
               column_names = {'OBJCAT': ('Xpix', 'Ypix', 'RA', 'DEC'),
                               'REFCAT': (None, None, 'RaRef', 'DecRef')}
        :type column_names: <dict>

        :param dq_planes:
              Boolean flag to transform bit_plane by bit_plane for rotation,
              shifting and scaling. At this the algorithmm has bad performance.
              Default is False.
        :type dq_planes: <bool>

        """
        self.ad = ad
        if len(ad) < 2:
            raise ValueError("Nothing to mosaic. < 2 extensions found.")

        self.log = logutils.get_logger(__name__)
        # Execute the input geometry function.
        mosaic_data, geometry = mosaic_ad_function(ad)
        Mosaic.__init__(self, mosaic_data, geometry)
        self.column_names = column_names
        self.dq_planes = dq_planes      # Transform bit_plane by bit_plane
        self.jfactor = []               # Jacobians applied to interpolated pixels.
        self.calculate_jfactor()        # Fill the jfactor vector with the
                                        # jacobian of transformation matrix.

    # --------------------------------------------------------------------------
    def as_astrodata(self,block=None,tile=False,return_associated_bintables=True, 
                     return_ROI=True, update_catalog_method='wcs'):
        """
        Returns an AstroData object  containing by default the mosaiced
        IMAGE extensions, the merged associated BINTABLEs and all other
        non-associated extensions of any other type. WCS information in
        the headers of the IMAGE extensions and any pixel coordinates in
        BINTABLEs will be updated appropriately.

        :param tile: (boolean). If True, the mosaics returned are not
                     corrected for shifting and rotation.

        :param block: See description below in method 'mosaic_image_data'.

        :param return_ROI: (True). Returns the minimum frame size calculated
        from the location of the amplifiers in a given block. If False uses
        the blocksize value.

        :param return_associated_bintables: (True). If a bintable is associated
        to the ref_extname then is returned as a merged table in the
        output AD.  If False, they are not returned in the output AD.

        :param return_non_associations: (True). Specifies whether to return
        extensions that are not deemed to be associated with the ref_extname.

        :param update_catalog_method: ('wcs').  Specifies if the X
        and Y pixel coordinates of any source positions in the BINTABLEs
        are to be recalculated using the output WCS and the sources R.A.
        and Dec. values within the table. If set to 'transform' the updated X
        and Y pixel coordinates will be determined using the transformations
        used to mosaic the pixel data. In the case of tiling, a shift is
        technically being applied and therefore update_catalog_method='wcs'
        should be set internally (Not yet implemented).

        :type update_catalog_method: <str>, values can be 'wcs', 'transform'

        """
        # Create mosaics of all image data in ad, merge bintables if they are
        # associated with the image extensions and 
        adout = astrodata.create(self.ad.header[0])
        adout.phu.set('TILED', ['FALSE', 'TRUE'][tile])
        adout.phu.set_comment('TILED', 'True: tiled; False: Image Mosaicked')

        # ------ Create mosaic ndarrays, update the output WCS, create an 
        #        AstroData object and append to the output list.
        # Mosaic the IMAGE extensions now
        mosarray = self.mosaic_image_data(block=block, tile=tile,
                                          return_ROI=return_ROI)

        # Create the mosaic FITS header using the reference extn header.
        header = self.mosaic_header(mosarray.shape, block, tile)

        # Generate WCS object to be used in the merging the object
        # catalog table for updating the objects pixel coordinates
        # w/r to the new crpix1,2.
        ref_wcs = wcs.WCS(header)

        # Setup output AD 
        new_ext = adout.append(data=mosarray, header=header)

        # Reset extver to 1.
        new_ext.rename_ext(name=extn, ver=1)
        adout.append(new_ext)

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
           ref_wcs = wcs.WCS(ad[0].header)
        except:
           self.jfactor = [1.0] * len(self.ad)
           return

        # Get CD matrix for each extension, calculate the transformation
        # matrix from composite of the reference cd matrix and the current one.
        self.jfactor.append(1.0)
        for ext in self.ad[1:]:
            header = ext.header[1]
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
    def get_data_list(self):
        """
        This could have been written as, 

            data_list = [ext.data[xsec.y1:xsec.y2, xsec.x1:xsec.x2] for 
                         ext, xsec in zip(ad, ad.data_section())]

        Pursuaded by arguments against code opacity, it is not.

        Return
        ------
        data_list: a list of actual data array sections for each extension.
        type: <list>

        """
        data_list = []
        for ex in self.ad:
            xsec = ex.data_section()
            data_list.append(ex.data[xsec.y1: xsec.y2, xsec.x1: xsec.x2])
        return data_list 

    # --------------------------------------------------------------------------
    def mosaic_image_data(self, block=None, tile=False, return_ROI=True):
        """
        Creates the output mosaic ndarray of the requested IMAGE extension.

        Parameters:
        ---------
        block: Allows a specific block to be returned as the output mosaic.
               The tuple notation is (col,row) (0-based), where (0,0) is the
               lower left block.  This is position of the reference block w/r to
               mosaic_grid. Default is None.
        type: <tuple>, e.g., (<int>, <int>)

        tile: If True, the mosaics returned are not corrected for shifting and
              rotation.
        type: <bool>

        return_ROI: Returns the minimum frame size calculated from the location of
                    the amplifiers in a given block. If False uses the blocksize
                    value. Default is True.
        type: <bool>

        Return:
        ------
        out: An ndarray of the mosaiced data.
        type: <ndarray>

        """
        dq_data = False
        self.data_list = self.get_data_list()
        if self.ad.mask.count('DQ') > 0 and self.dq_planes:
            dq_data = True

        out = Mosaic.mosaic_image_data(self, block=block, dq_data=dq_data, tile=tile,
                                       return_ROI=return_ROI, jfactor=self.jfactor)
        return out
 
    # --------------------------------------------------------------------------
    def mosaic_header(self, mosaic_shape, block, tile):
        """
        Make the mosaic FITS header based on the reference extension header.

        Update CCDSEC, DETSEC, DATASEC, CRPIX1, CRPIX2 keywords to reflect the
        mosaic geometry.

        :param mosaic_shape: The output mosaic dimensionality
                             (npixels_y, npixels_x)
        :type mosaic_shape: <tuple>

        :param tile: transform blocks or not.True: blocks are not transformed.
        :type tile: <bool>

        :param block: (ncol, nrow) indicating which block to return.
        :type block: <tuple>

        Output
        ------
        header:    Mosaic Fits header object

        """
        mcomm = "Set by MosaicAD, v{}".format(__version__)
        fmat1 = "[{}:{},{}:{}]"
        fmat2 = "[1:{},1:{}]"

        mosaic_hd = self.ad[0].header[1].copy()              # SCI ext header.
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

        ccdname = ",".join([ext.hdr.get('CCDNAME') for ext in self.ad])
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

        reference_extname (type: string) -
            The value of the EXTNAME header keyword use as reference.
            This is the output EXTNAME in the output mosaic FITS header.

        reference_extver (type: int) -
            The value of the EXTVER header keyword use as reference. Use this WCS
            information to generate the updated output mosaic FITS header.

        reference_block - The block number containing the reference amplifier

        """
        geo = self.geometry
        # Set up a dictionary 
        info = {}

        info['filename'] = self.ad.filename
        # amp_mosaic_coord:  same as DETSECS
        info['amp_mosaic_coord'] = self.coords['amp_mosaic_coord']
        # amp_block_coord: same as CCDSECS
        info['amp_block_coord'] = self.coords['amp_block_coord']
        info['reference_extname'] = self.ref_extname
        info['reference_block'] = geo.ref_block
        amps_per_block = self._amps_per_block
        info['amps_per_block'] = amps_per_block
        ref_block_number = (geo.ref_block[0] +
                              geo.ref_block[1]*geo.mosaic_grid[0])
        if amps_per_block == 1:
           ref_extver = ref_block_number
        else:
           ref_extver = amps_per_block*(ref_block_number - 1) + 1

        info['reference_extver']=ref_extver
        nx,ny = self.blocksize
        info['interpolator'] = geo.interpolator
        if geo.interpolator not in ['linear','nearest']:
            info['spline_order'] = geo.spline_order

        # out data.data in (x,y) order
        amp_shape = [k.data.shape[::-1] for k in self.ad[self.ref_extname]]
        info['amps_shape_no_trimming'] = amp_shape
        return info

    # --------------------------------------------------------------------------
    def merge_catalogs(self, ref_wcs, tile, merge_extvers, tab_extname, 
                        recalculate_xy='wcs',transform_pars=None):
        """
        This function merges together separate bintable extensions (tab_extname),
        converts the pixel coordinates to the reference extension WCS
        and remove duplicate entries based on RA and DEC.

        NOTE: Names used here so far: *OBJCAT:* Object catalog extension name

        Parameters
        ----------
        :param ref_wcs: wcs object containing the WCS from the output header.
        :param merge_extvers: List of extvers to merge from the tab_extname
        :param tab_extname: Binary table extension name to be merge over all
                            its ext_ver's.
        :param transform_pars: Dictionary  with rotation angle, translation
                               and magnification.
        :param recalculate_xy: Use reference extension WCS to recalculate the
                               pixel coordinates. If value is 'transform' use
                               the tranformation linear equations.
        :type recalculate_xy: (string, default: 'wcs').
                              Allow values: ('wcs', 'transform')

        Note
        ----
        For 'transform' mode these are the linear equations to use.

        X_out = X * mx * cosA - Y * mx * sinA + mx * tx
        Y_out = X * my * sinA + Y * my * cosA + my * ty

        mx, my: magnification factors.
        tx, ty: translation amount in pixels.
        A: Angle in radians.

        """
        column_names = self.column_names
        adoutput_list = []
        col_names = None
        col_fmts = None
        col_data = {}      # Dictionary to hold column data from all extensions
        newdata = {}
        
        # Get column names from column_names dictionary
        # EXAMPLE:
        #   column_names = {
        #      'OBJCAT': ('X_IMAGE', 'Y_IMAGE', 'X_WORLD', 'Y_WORLD'),
        #      'REFCAT': (None, None, 'RAJ2000', 'DEJ2000')
        #   }
        for key in column_names:
            if key == tab_extname:
               Xcolname, Ycolname = column_names[key][:2]
               ra_colname, dec_colname = column_names[key][2:4]

        # Get catalog data for the extension numbers in merge_extvers list.
        do_transform = (recalculate_xy == 'transform') and (Xcolname != None)
        if do_transform:
            dict = self.data_index_per_block
            nbx,nby=self.geometry.mosaic_grid

        for extv in merge_extvers:
            inp_catalog = self.ad[tab_extname,extv]
            # Make sure there is data. 
            if inp_catalog is None:
                continue

            if inp_catalog.data is None:
                continue

            if len(inp_catalog.data)==0:
                continue

            catalog_data = True

            # Get column names and formats for the first extv
            # and copy the data into the dictionary.
            if col_names is None:
                col_names = inp_catalog.data.names
                col_fmts = inp_catalog.data.formats
                # fill out the dictionary
                for name in col_names:
                    col_data[name] = []

                xx=[]; yy=[]

            for name in col_names:
                newdata[name] = inp_catalog.data.field(name)

            # append data from each column to the dictionary. 
            for name in col_names:
                col_data[name] = np.append(col_data[name],newdata[name])

            if do_transform:
                # Get the block tuple where an amplifier (extv) is located.
                block=[k for k, v in dict.iteritems() if extv-1 in v][0]
                if (extv-1) in self.data_index_per_block[block]:
                    # We might have more than one amplifier per block,
                    # so offset all these xx,yy to block's lower left.
                    x1,y1=[self.coords['amp_block_coord'][extv-1][k] for k in [0,2]]
                    # add it to the xx,yy
                    xx = np.append(xx,newdata[Xcolname]+x1)
                    yy = np.append(yy,newdata[Ycolname]+y1)
                    if extv%self._amps_per_block != 0:
                       continue


                # Turn tuples values (col,row) to index
                bindx = block[0]+nbx*block[1]
                nxx,nyy = self._transform_xy(bindx,xx,yy) 

                # Now change the origin of the block's (nxx,nyy) set to the 
                # mosaic lower left. We find the offset of the LF corner
                # by adding the width and the gaps of all the block to 
                # the left of the current block.
                #  

                if tile:
                    gap_mode = 'tile_gaps'
                else:
                    gap_mode = 'transform_gaps'

                gaps = self.geometry.gap_dict[gap_mode]
                # The block size in pixels.
                blksz_x,blksz_y = self.blocksize
                col,row = block
                # the sum of the gaps to the left of the current block
                sgapx = sum([gaps[k,row][0] for k in range(col+1)])
                # the sum of the gaps below of the current block
                sgapy = sum([gaps[col,k][1] for k in range(row+1)])
                ref_x1 = int(col*blksz_x + sgapx)
                ref_x2 = ref_x1 + blksz_x
                ref_y1 = int(row*blksz_y + sgapy)
                ref_y2 = int(ref_y1 + blksz_y)

                newdata[Xcolname] = nxx+ref_x1
                newdata[Ycolname] = nyy+ref_y1
                xx = []
                yy = []

        # Eliminate possible duplicates values in ra, dec columns
        ra, raindx  = np.unique(col_data[ra_colname].round(decimals=7),
                        return_index=True)
        dec, decindx = np.unique(col_data[dec_colname].round(decimals=7),
                        return_index=True)

        # Duplicates are those with the same index in raindx and decindx lists.
        # Look for elements with differents indices; to do this we need to sort
        # the lists.
        raindx.sort()
        decindx.sort()

        # See if the 2 arrays have the same length
        ilen = min(len(raindx), len(decindx))

        # Get the indices from the 2 lists of the same size
        v, = np.where(raindx[:ilen] != decindx[:ilen])
        if len(v) > 0:
            # Filter the duplicates
           try:
               for name in col_names:
                   col_data[name] = col_data[name][v]
           except:
               print 'ERRR:',len(v),name

        # Now that we have the catalog data from all extensions in the dictionary,
        # we calculate the new pixel position w/r to the reference WCS.
        # Only an Object table contains X,Y column information. Reference catalog
        # do not.
        #
        if (recalculate_xy == 'wcs') and (Xcolname != None):

            xx = col_data[Xcolname]
            yy = col_data[Ycolname]
            ra = col_data[ra_colname]
            dec = col_data[dec_colname]

            # Get new pixel coordinates for all ra,dec in the dictionary.
            # Use the input wcs object.
            newx,newy = ref_wcs.wcs_sky2pix(ra,dec,1)

            # Update pixel position in the dictionary to the new values.
            col_data[Xcolname] = newx
            col_data[Ycolname] = newy

        # Create columns information
        columns = {}
        table_columns = []
        for name,format in zip(col_names,col_fmts):
            # Let add_catalog auto-number sources
            if name=="NUMBER":
                continue

            # Define pyfits columns
            data = columns.get(name, fits.Column(name=name,format=format,
                            array=col_data[name]))
            table_columns.append(data)

        # Make the output table using pyfits functions
        col_def = fits.ColDefs(table_columns)
        tb_hdu = fits.new_table(col_def)

        # Now make an AD object from this table
        adout = AstroData(tb_hdu)
        adout.rename_ext(tab_extname,1)

        # Append to any other new table we might have
        adoutput_list.append(adout)

        return adoutput_list

    # --------------------------------------------------------------------------
    def merge_table_data(self,ref_wcs,tile,block=None,update_catalog_method='wcs'):
        """
        Merges input BINTABLE extensions of the requested tab_extname. Merging
        is based on RA and DEC columns. The repeated RA, DEC values in the output
        table are removed. The column names for pixel and equatorial coordinates
        are given in a dictionary with attribute name: column_names.

        Parameters
        ----------
        ref_wcs: reference WCS object.
        type:    <WCS object>

        block:
            Allows a specific block to be returned as the output mosaic.
            The tuple notation is (col,row) (0-based) where (0,0) is the lower
            left block. This is position of the reference block w/r to
            mosaic_grid.
        type: <2-tuple>, Default is None

        update_catalog_method:
            If 'wcs' use the reference extension header WCS to recalculate the x,y
            values. If 'transform', apply the linear equations using to correct
            the x,y values in each block.
        type: <str>

        Return
        ------
        adout: merged output BINTABLE of the requested tab_extname BINTABLE
            extension.

        """
        if block:
            if not isinstance(block, tuple):
                raise ValueError('Block number is not a tuple.')

            merge_extvers = self.data_index_per_block[block]

        #  Merge the bintables containing source catalogs.
        adout = self.merge_catalogs(ref_wcs, tile, merge_extvers,
                                    update_catalog_method, 
                                    transform_pars=self.geometry.transformation)
        return adout

    # --------------------------------------------------------------------------
    def update_crpix(self, wcs, tile):
        """
        Update WCS elements CRPIX1 and CRPIX2 based on the input WCS header of
        the first amplifier in the reference block number.

        Parameters
        ----------
        wcs: Reference extension header's WCS object
        type: WCS object.

        tile: Tile data or transform. tile = False transforms.
        type: <bool>

        Return
        ------
        (crpix1, crpix2): New pixel reference number in the output mosaic.
        type: <2-tuple>

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
