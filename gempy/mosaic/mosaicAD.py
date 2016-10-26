import pywcs
import numpy as np
import pyfits as pf

from ..library.mosaic import Mosaic

from astrodata import new_pyfits_version
from astrodata import AstroData
from astrodata.utils import logutils

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
        as_astrodata        - Output mosaics as AstroData objects.
        merge_table_data   - Merges catalogs extension that are associated
                             with image extensions.
        mosaic_image_data  - Create a mosaic from extensions.
        update_data        - Load image extensions from a different extension
        make_associations  - Look for associations between image extension
                             and bintables.
        get_extnames       - Make image and tables lists of the input AstroData
                             content.
        get_data_list      - Return a list of image data for a given extname
                             extensions in the input AstroData object.
        update_wcs         - Update the WCS information in the output header.
        info               - Creates a dictionary with coordinates, amplifier 
                             and block information.
        set_mosaic_data    - Loads a dictionary with a numpy array for a given 
                             extension name. This is to replace or skip the 
                             creation of a mosaic.

      Attributes   (In addition to parent class attributes)
      ---------- 
        log           - Logutils object
        ad            - Astrodata object
        ref_extname   - Is the IMAGE EXTNAME that should be used as the
                        primary reference when reading the ad data arrays.
                        Default values is 'SCI'
        extnames      - Contains all extension names in ad
        im_extnames   - All IMAGE extensions names in ad
        tab_extnames  - All BINTABLE extension names in ad
        associated_tab_extns
                      - List of binary extension names that have the same
                        number and values of extvers as the reference
                        extension name.
        associated_im_extns
                      - List of image extension names that have the same
                        number and values of extvers as the reference
                        extension name.
        non_associated_extns
                      - List of remaining extension names that are not
                        in the above 2 lists.
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

    def __init__(self, ad, mosaic_ad_function, ref_extname='SCI',
                 column_names='default',dq_planes=False):
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

          :type mosaic_ad_function: 
              A required user function returning a MosaicData
              and a MosaicGeometry objects.

          :param ref_extname:
              Is the IMAGE EXTNAME that should be used as the primary
              reference when reading the ad data arrays.
          :type ref_extname: string. Default is 'SCI'.

          :param column_names: 
              Dictionary with bintable extension names that are associates
              with input images. The extension name is the key with value
              a tuple: (X_pixel_columnName, Y_pixel_columnName, 
              RA_degrees_columnName, DEC_degrees_columnName)
              Example:
               column_names = {'OBJCAT': ('Xpix', 'Ypix', 'RA', 'DEC'),
                               'REFCAT': (None, None, 'RaRef', 'DecRef')}

          :param dq_planes: 
              (False). Boolean flag to transform bit_plane by bit_plane
              for rotation, shifting and scaling. At this the algorithmm 
              has a bad performance. Default value is False.

        """

        # Instantiate the log
        # logutils.config(mode='debug', console_lvl='stdinfo',file_name='mosaic.log') 
        self.log = logutils.get_logger(__name__)

        self.log.debug("******* Debug.STARTING MosaicAD **********")
        self.log.info( "******* INFO.STARTING MosaicAD ***********")

        # Make sure we have the default extension name in the input AD.
        if ad[ref_extname] == None:
            raise ValueError("Extension name: '"+ref_extname+"' not found.")

        self.ad = ad

        # The input file ougth to have more than one extension to mosaic.
        if ad.count_exts(ref_extname) <= 1:
            raise ValueError("Nothing to mosaic. Input file has 1 extension.")
            
        # Execute the input user function.
        mosaic_data, geometry = mosaic_ad_function(ad,ref_extname)

        self.ref_extname = ref_extname
        self.extnames = None                 # All extensions names in AD 
        self.im_extnames = None              # All IMAGE extensions names in AD 
        self.tab_extnames = None             # All BINTABLE extensions names in AD 
        self.get_extnames()                  # Form extnames, im_extnames, 
                                             # tab_extnames.
        self.dq_planes = dq_planes           # (False) Input parameter
                                             # Transform bit_plane byt bit_plane
        self.mosaic_data_array = {}          # attribute to reference 
                                             # numpy array by extension.
                                             # Set by method set_mosaic_data() 

        # Instantiate the Base class
        Mosaic.__init__(self,mosaic_data,geometry)  

        # Internal attribute to be used when loading data from another
        # extension name.
        #
        self.__current_extname = ref_extname

        # These are the default column names for merging catalogs.
        if column_names == 'default':
            self.column_names = \
                 {'OBJCAT': ('X_IMAGE', 'Y_IMAGE', 'X_WORLD', 'Y_WORLD'),
                  'REFCAT': (None, None, 'RAJ2000', 'DEJ2000') }
        else:
            self.column_names = column_names

        self.jfactor = []               # Jacobian factors applied to the 
                                        # interpolated pixels to conserve flux.

        self.calculate_jfactor()        # Fill out the jfactor list with the
                                        # jacobian of the transformation matrix.
                                        #  See transformation()

        self.associated_tab_extns = []  # List of binary extension names that
                                        # have the same number and values of 
                                        # extvers as the reference extension name.

        self.associated_im_extns = []   # List of image extension names that 
                                        # have the same number and values of 
                                        # extvers as the reference extension name.

        self.non_associated_extns = []  # List of remaining extension names that
                                        # are not in the above 2 lists.

        self.make_associations()    # Look for associations between image extension
                                    # and bintables filling out lists:
                                    # self.associated_tab_extns, 
                                    # self.associated_im_extns
                                    # and self.non_associated_extns
      

    def as_astrodata(self, extname=None, tile=False, block=None, return_ROI=True,
                    return_associated_bintables=True, return_non_associations=True,
                    update_catalog_method='wcs'):
        """

          Returns an AstroData object  containing by default the mosaiced 
          IMAGE extensions, the merged associated BINTABLEs and all other 
          non-associated extensions of any other type. WCS information in 
          the headers of the IMAGE extensions and any pixel coordinates in 
          BINTABLEs will be updated appropriately.

          :param extname: If None mosaic all IMAGE extensions. Otherwise 
              only the given extname. This becomes the ref_extname.

          :type extname: (string). Default is None

          :param tile: (boolean). If True, the mosaics returned are not 
              corrected for shifting and rotation.

          :param block: See description below in method 'mosaic_image_data'.

          :param return_ROI: (True). Returns the minimum frame size calculated
              from the location of the amplifiers in a given block. If False uses
              the blocksize value.

          :param return_associated_bintables: (True). If a bintable is associated
              to the ref_extname then is returned as a merged table in the 
              output AD.  If False, they are not returned in the output AD.

          :param return_non_associations (True). Specifies whether to return
              extensions that are not deemed to be associated with the ref_extname.

          :param update_catalog_method: ('wcs').  Specifies if the X 
              and Y pixel coordinates of any source positions in the BINTABLEs
              are to be recalculated using the output WCS and the sources R.A.
              and Dec. values within the table. If set to 'transform' the updated X 
              and Y pixel coordinates will be determined using the transformations
              used to mosaic the pixel data. In the case of tiling, a shift is 
              technically being applied and therefore update_catalog_method='wcs'
              should be set internally (Not yet implemented).

          :type update_catalog_method: (string). Possible values are 
                                                 'wcs' or 'transform'.
                     
        """
        # If extname is None create mosaics of all image data in ad, merge 
        # the bintables if they are associated with the image extensions 
        # and append to adout all non_associatiated extensions. Appending
        # these extensions to the output AD is controlled by 
        # return_associated_bintables and return_non_associations.

        # Make blank ('') same as None; i.e. handle all extensions.
        if extname == '': extname = None
        if (extname != None) and (extname not in self.extnames):
            raise ValueError("as_astrodata: Extname '"+extname+\
                        "' not found in AD object.")

        adin = self.ad      # alias
        
        # Load input data if data_list attribute is not defined. 
        #if not hasattr(self, "data_list"):
        #    self.data_list = self.get_data_list(extname)

        adout = AstroData()               # Prepare output AD
        adout.phu = adin.phu.copy()       # Use input AD phu as output phu

        adout.phu.header.update('TILED', ['FALSE', 'TRUE'][tile],
                 'False: Image Mosaicked, True: tiled')

        # Set up extname lists with all the extension names that are going to 
        # be mosaiced and table extension names to associate.
        #
        if extname is None:                     # Let's work through all extensions
            if self.associated_im_extns:
                extname_list = self.associated_im_extns
            else:
                extname_list = self.im_extnames
        else:
            self.ref_extname = extname          # Redefine reference extname
            if extname in self.associated_im_extns:
                self.associated_im_extns = [extname]    # We need this extname only
                extname_list = [extname]
            elif extname in self.non_associated_extns: 
                # Extname is not in associated lists; so clear these lists.
                extname_list = []                       
                self.associated_im_extns = []
                self.associated_tab_extns = []
            elif extname in self.associated_tab_extns:
                # Extname is an associated bintable.
                extname_list = []                       
                self.associated_im_extns = []
                self.associated_tab_extns = [extname]
            else:
                extname_list = [extname]

        # ------ Create mosaic ndarrays, update the output WCS, create an 
        #        AstroData object and append to the output list. 
        
        # Make the list to have the order 'sci','var','dq'
        svdq = [k for k in ['SCI','VAR','DQ'] if k in extname_list]
        # add the rest of the extension names.
        extname_list = svdq + list(set(extname_list)-set(svdq))

        for extn in  extname_list:
            # Mosaic the IMAGE extensions now
            mosarray = self.mosaic_image_data(extn,tile=tile,block=block,
                                          return_ROI=return_ROI)
            # Create the mosaic FITS header using the reference 
            # extension header.
            header = self.mosaic_header(mosarray.shape,block,tile)

            # Generate WCS object to be used in the merging the object
            # catalog table for updating the objects pixel coordinates
            # w/r to the new crpix1,2.
            ref_wcs = pywcs.WCS(header)

            # Setup output AD 
            new_ext = AstroData(data=mosarray,header=header)

            # Reset extver to 1.
            new_ext.rename_ext(name=extn,ver=1)
            adout.append(new_ext)

        if return_associated_bintables:
            # If we have associated bintables with image extensions, then
            # merge the tables.
            for tab_extn in self.associated_tab_extns:
                # adout will get the merge table
                new_tab = self.merge_table_data(ref_wcs, tile, tab_extn, block, 
                            update_catalog_method) 
                adout.append(new_tab[0])
        
        # If we have a list of extension names that have not tables extension
        # names associated, then mosaic them.
        #
        if return_non_associations:
            for extn in self.non_associated_extns:
                # Now get the list of extver to append
                if extn in self.im_extnames:   #  Image extensions
                    # We need to mosaic image extensions having more
                    # than one extver.
                    #
                    if adin.count_exts(extn) > 1:
                        mosarray = self.mosaic_image_data(extn,
                                    tile=tile,block=block,
                                    return_ROI=return_ROI)

                        # Get reference extension header
                        header = self.mosaic_header(mosarray.shape,block,tile)
                        new_ext = AstroData(data=mosarray,header=header)

                        # Reset extver to 1.
                        new_ext.rename_ext(name=extn,ver=1)
                        adout.append(new_ext)
                    else:
                        self.log.warning("as_astrodata: extension '"+extn+\
                                         "' has 1 extension.")
                        adout.append(adin[extn])

                if extn in self.tab_extnames:   # We have a list of extvers
                    for extv in self.tab_extnames[extn]:
                        adout.append(adin[extn,extv]) 
        # rediscover classifications.
        adout.refresh_types()
        return adout

    def merge_table_data(self, ref_wcs, tile, tab_extname, block=None,
                     update_catalog_method='wcs'):
        """
            Merges input BINTABLE extensions of the requested tab_extname. 
            Merging is based on RA and DEC columns. The repeated RA, DEC
            values in the output table are removed. The column names for 
            pixel and equatorial coordinates are given in a dictionary 
            with attribute name: column_names

          Input
          -----
          :param tab_extname: Binary table extname

          :param  block: default is (None).
                         Allows a specific block to be returned as the 
                         output mosaic. The tuple notation is (col,row) 
                         (zero-based) where (0,0) is the lower left block.
                         This is position of the reference block w/r
                         to mosaic_grid.

          :param  update_catalog_method:
                          If 'wcs' use the reference extension header
                          WCS to recalculate the x,y values. If 'transform', 
                          apply the linear equations using 
                          to correct the x,y values in each block.

          Output
          ------
          :param adout: merged output BINTABLE of the requested
                         tab_extname BINTABLE extension

        """
        adin = self.ad
        ref_extname = self.ref_extname
        if tab_extname not in self.associated_tab_extns:
            raise ValueError("merge_table_data: "+tab_extname+\
                        " not found in AD object.")
 
        if block:
            if type(block) is int: 
                raise ValueError('Block number is not a tuple.')
            merge_extvers = self.data_index_per_block[block]
        else: 
            # Merge all extension number in the bintable.
            # Get all version numbers from image and bintable in case the
            # user requested one extension.
            if ref_extname in self.im_extnames:
                merge_extvers = self.im_extnames[ref_extname]
            else:
                merge_extvers = self.tab_extnames[ref_extname]

        #  Merge the bintables containing source catalogs.
        adout = self.merge_catalogs(ref_wcs, tile, merge_extvers, 
                   tab_extname, update_catalog_method, 
                   transform_pars=self.geometry.transformation)

        return adout

    def mosaic_image_data(self, extname='SCI', tile=False, block=None,
                         return_ROI=True):
        """
          Creates the output mosaic ndarray of the requested IMAGE extension.

          :param extname: (default 'SCI'). Extname from AD to mosaic.

          :param tile: (boolean). If True, the mosaics returned are not 
              corrected for shifting and rotation.

          :param  block: default is (None). 
		  Allows a specific block to be returned as the output 
		  mosaic. The tuple notation is (col,row) (zero-based)
                  where (0,0) is the lower left block.  This is position 
                  of the reference block w/r to mosaic_grid.

          :param return_ROI: (True). Returns the minimum frame size calculated
              from the location of the amplifiers in a given block. If False uses
              the blocksize value.
                     
        """

        # Check if we have already a mosaic_data_array in memory. The user
        # has set this reference. Use this instead of re(creating)
        # it.
        #
        if len(self.mosaic_data_array) != 0:
            # It could be an arbitrary array shape; is up to the user
            # to correct for any implied problem.
            # Return the user supplied array only if it matches the current 
            # extname.
            if extname in self.mosaic_data_array:
                return self.mosaic_data_array[extname]

        # Here as oppose to the as_astrodata method we can handle only one
        # extension name.
        #
        if extname == '' or (extname == None): 
            extname = self.ref_extname
        if (extname != None) and (extname not in self.extnames):
            raise ValueError("mosaic: EXTNAME '"+extname+ \
			"' not found in AD object.")

        # If data_list in memory is different from requested then
        # read data in from the requested extname.
        self.update_data(extname)

        # Setup attribute for transforming a DQ extension if any.
        # We need to set dq_data here, since the base class method
        # does not know about extension names.
        #
        dq_data = False
        if extname == 'DQ' and self.dq_planes:
             dq_data = True

        # Use the base method. 
        out = Mosaic.mosaic_image_data(self,tile=tile,block=block,\
                 return_ROI=return_ROI, dq_data=dq_data, jfactor=self.jfactor)
        
        return out
 
    def set_mosaic_data(self, mosaic_data_array, ext_name):
        """
           Assign the mosaic image data for the given extension name
           to 'mosaic_data_array' dictionary with key 'ext_name'. This
           user method will avoid the mosaic creation method 'mosaic_image_data'.

           Parameter:
           ----------

           mosaic_data_array:
                     Numpy array with the mosaic data_array. It will
                     be used mainly by as_astrodata method as the
                     mosaic_array (adout.data)
           ext_name: Extension name of the mosaic_data_array.
        """

        self.mosaic_data_array[ext_name] = mosaic_data_array
      
    def calculate_jfactor(self):
        """
          Calculate the ratio of reference input pixel size to output
          pixel size.  
          In practice this ratio is formulated as the determinant of the
          WCS tranformation matrix.
          This is the ratio that we will applied to each pixel
          to conserve flux in a feature.

          *Justification:*

            In general CD matrix element is the ration between partial
            derivative of the world coordinate (ra,dec) with respect to the
            pixel coordinate (x,y). We have 4 elements cd11, cd12, cd21 and cd22.

            For an adjecent image in the sky (GMOS detectors 1,2,3 for example),
            the cd matrix elements will have slightly differents values.
            
            Given CD1 and CD2 as CD matrices from adjacent fields then
            the determinant of the dot product of the inverse CD1 times the CD2 
            will give the correcting factor. 
         
        """
        ad = self.ad
        ref_extn = self.ref_extname
        amps_per_block = self._amps_per_block

        # Reference block number location (1-based)
        ref_block = self.geometry.ref_block       # 0-based 
        nblocks_x = self.geometry.mosaic_grid[0]     
        ref_block_number = ref_block[0] + ref_block[1]*nblocks_x

        # We want to get the reference header.
        # Get the reference amplifier number; i.e. the reference extver
        if amps_per_block == 1:
            ref_extver = ref_block_number + 1   # EXTVERs starts from 1)
        else:
            # Get the first amplifier in the block
            ref_extver = amps_per_block*ref_block_number + 1
       
        # If there is no WCS return 1 list of 1.s
        try:
           ref_wcs = pywcs.WCS(ad[ref_extn,ref_extver].header)
        except:
           self.jfactor = [1.]*ad.count_exts(ref_extn)
           return
        
        # Loop through all the reference extension versions and get the
        # CD matrix. Calculate the transformation matrix from composite of 
        # the reference cd matrix and the current one.
        for ext in ad[ref_extn]:
            if ref_extver == ext.extver():
                # We do not tranform the reference block, hence 1.
                self.jfactor.append(1.) 
                continue 
            # see if we have CD matrix in header
            header = ad[ref_extn,ext.extver()].header
            if not ('CD1_1' in header):
                self.jfactor.append(1.0)
                continue
            try:
                img_wcs = pywcs.WCS(header)
                # Cross product of both CD matrices
                matrix =  np.dot(np.linalg.inv(img_wcs.wcs.cd),ref_wcs.wcs.cd)
                matrix_det = np.linalg.det(matrix)
            except:
                self.log.warning(\
                "calculate_jfactor: Error calculating matrix_det. "+ \
		 "Setting jfactor=1")
                matrix_det = 1.0

            # Fill out the values for each extension.
            self.jfactor.append(matrix_det)
        

    def verify_inputs(self):
        pass

    def update_data(self,extname):
        """
          Replaces the data_list attribute in the mosaic_data object with a new
          list containing ndarrays from the AD extensions 'extname'.The attribute
          data_list is updated with the new list.

          Input:

            extname: Reads all the image extensions from AD 
                     that matches the extname.
        """

        if (extname != self.__current_extname) or \
            (not hasattr(self, "data_list")):
            # We are requesting another data_list
            try:
                self.data_list = self.get_data_list(extname)
            except:
                raise ValueError("update_data:: "+extname+ 
			    ' not found in AD object. File:'+
                            str(self.ad.filename))

    def make_associations(self):
        """This determines three lists: one list of IMAGE extension EXTNAMEs and 
           one of BINTABLE extension EXTNAMEs that are deemed to be associated 
           with the reference extension. The third list contains the EXTNAMEs of
           extensions that are not deemed to be associated with the reference 
           extension. The definition of association is as follows: given the 
           ref_extname has n extension versions (EXTVER), then if any other 
           EXTNAME has the same number and exact values of EXTVER as the 
           ref_extname these EXTNAMEs are deemed to be associated to the 
           ref_extname.
        """
        imlist = []
        binlist = []
        ref_extn = self.ref_extname

        # self.im_extnames is a dictionary keyed by extension name and value
        # its list of extver's.
        #
        ref_extvers = self.im_extnames[ref_extn]    # List of extver's
       
        # Fill the association lists only if we have a bintable in the
        # input ad.
        #
        if self.tab_extnames:     
            # Associate images extension names if they have the same number 
            # and values of extension versions as the reference image.
            #
            for key_extn in self.im_extnames:      # is a dictionary
                if (len(ref_extvers)==len(self.im_extnames[key_extn])) and \
                   (ref_extvers==self.im_extnames[key_extn]):  
                    imlist.append(key_extn)
            
            # Associate binary tables extension names if they have the same number
            # and values of extension versions as the reference image.
            #
            for key_extn in self.tab_extnames:     # is a dictionary
                # check that number of extvers and extnames are the same.
                if (len(ref_extvers) == len(self.tab_extnames[key_extn])) and \
                   (ref_extvers ==  self.tab_extnames[key_extn]):  
                    binlist.append(key_extn)
            
            self.associated_tab_extns = binlist    # Bintable association
            self.associated_im_extns = imlist      # Images association

            # Use 'set' type to remove image and bintables extnames from
            # the list of all extnames.
            self.non_associated_extns = list(set(self.extnames)- \
				        set(binlist)-set(imlist))

    def get_extnames(self):
        """We should know what we have in AD.
           Form two dictionaries (images and bintables) with key the
           EXTNAME value and values the EXTVER values in a list.
           E.g.: {'VAR': [1, 2, 3, 4, 5, 6], 'OBJMASK': [1, 2, 3, 4, 5, 6]}
        """

        im_extnames = {}
        tab_extnames = {}

        for ext in self.ad:
            exttype = ext.header['xtension']
            if exttype == 'IMAGE':
                # Append the extver value to the im_extnames[ext.extname] list.
                im_extnames.setdefault(ext.extname(),[]).append(ext.extver())
            elif exttype == 'BINTABLE':
                tab_extnames.setdefault(ext.extname(),[]).append(ext.extver())
            else:
                self.log.warning(
                    '>>>> get_extnames: extension type not recorded: '+exttype)

        if len(im_extnames[self.ref_extname]) <= 1:
            raise ValueError("Nothing to mosaic with extension: "+self.ref_extname)

        # Sort the extension version values in each list. We need them in 
        # order for later comparison.
        #
        tmp = [im_extnames[k].sort() for k in im_extnames]
        tmp = [tab_extnames[k].sort() for k in tab_extnames]
        self.im_extnames = im_extnames
        self.tab_extnames = tab_extnames

        # Form a lists of extnames
        self.extnames = im_extnames.keys() + tab_extnames.keys()


    def get_data_list(self,extname='SCI'):
        """ Return a list of image data for
            all the extname extensions in ad.
            It assumes that the header keyword 'DATASEC'
            is present.
        """

        ad = self.ad
        data_list = []
        self.__current_extname = extname

        # Get DATASEC keyword value from the header of all extension
        # names matching the value of input parameter 'extname' and
        # form a dictionary.
        #
        datasecdict = ad.data_section(extname=extname).as_dict()

        # Loop thru each hdu matching the extname value, get its
        # extension version number and form a tuple of amplifier
        # corner locations from the above dictionary.
        #

        for ext in ad[extname]:
            extv = ext.extver()
            (x1,x2,y1,y2) = tuple(datasecdict['*',extv])
            data_list.append(ext.data[y1:y2,x1:x2])

        return data_list

    def update_crpix(self,wcs,tile):
        """ Update WCS elements CRPIX1 and CRPIX2
            based on the input WCS header of the first
            amplifier in the reference block number.

            *Input:*

            :param  wcs: Reference extension header's WCS object

            * Output:*

            :return crpix1, crpix2: New pixel reference number
                                    in the output mosaic
        """
        # Gaps have different values depending whether we have
        # tile or not.
        if tile: gap_mode = 'tile_gaps'
        else:    gap_mode = 'transform_gaps'

        # Get the crpix from the reference's WCS 
        o_crpix1,o_crpix2 = wcs.wcs.crpix

        ref_col,ref_row = self.geometry.ref_block   # 0-based
        
        # Get the gaps that are attached to the left and
        # below a block.
        x_gap,y_gap = self.geometry.gap_dict[gap_mode][ref_col,ref_row]

        # The number of blocks in x and number of rows in the mosaic grid.
        nblocks_x,nrows = self.geometry.mosaic_grid

        amp_mosaic_coords = self.coords['amp_mosaic_coord']

        amp_number = max(0, self.data_index_per_block[ref_col,ref_row][0]-1)
        amp_index = max(0, list(self.coords['order'])[amp_number] - 1)
        xoff = 0
        if ref_col > 0:
            xoff = amp_mosaic_coords[amp_index][1]     # x2

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

        return (crpix1,crpix2)
 
    def mosaic_header(self,mosaic_shape,block,tile):
        """
           Make the mosaic FITS header based on the reference
           extension header.

           Update CCDSEC,DETSEC,DATASEC, CRPIX1, CRPIX2 keywords to 
           reflect the mosaic geometry.

           *Input:*

           :param mosaic_shape: (tuple) The output mosaic dimensionality 
                             (npixels_y, npixels_x)
           :param tile: Boolean. If True, the blocks are not transformed.

           :param block:     Tuple (ncol,nrow) indicating which
                             block to return.
           
           Output
           ------
           header:    Mosaic Fits header

        """
        adin = self.ad

        ref_block = self.geometry.ref_block  

        # Pick the reference block as all block have the same number of
        # amplifiers.
        #
        amps_per_block = self._amps_per_block

        # EXTVER's starts from 1
        ref_extver = int(amps_per_block*ref_block[0]) + 1
       
        # Now get its header. 
        extn = self.__current_extname
        
        # The order of the amplifier according to their location 
        # in the output block origin.
        #
        order = list(self.coords['order'])
        try:
            extver = order[ref_extver-1]    # extver in order of coord1
        except:
            # For an ROI we might not have all the data extensions data
            # with some blocks being empty. We want to create the mosaic
            # anyways. For now pick the first extension header.
            extver = order[0]
        
        mosaic_hd  = adin[extn,extver].header.copy()

        if new_pyfits_version:
           mosaic_hd.update = mosaic_hd.set

        if block: 
            # Returning one block
            mosaic_hd.update('EXTVER',1,after='EXTNAME')
            return mosaic_hd

        # ---- update CCDSEC,DETSEC and DATASEC keyword

        # Get keyword names for array_section and data_section
        arr_section_keyw = adin[extn].array_section().keyword
        dat_section_keyw = adin[extn].data_section().keyword

        # Get the lower left corner coordinates from detector_section.
        det_section = adin[extn].detector_section()
        detdict = det_section.as_dict()

        # Sort by key value first before getting the dictionary values.
        o_detsec = [detdict[k] for k in sorted(detdict.iterkeys())]
        
        # Get lowest x1 and y1
        min_x1 = np.min([k[0] for k in o_detsec])
        min_y1 = np.min([k[2] for k in o_detsec])

        # Unbin the mosaic shape
        x_bin,y_bin = (adin[extn].detector_x_bin(),adin[extn].detector_y_bin())
        if x_bin is None: 
           x_bin,y_bin = (1,1)
        unbin_width =  mosaic_shape[1] * x_bin
        unbin_height = mosaic_shape[0] * y_bin

        detsec = "[%i:%i,%i:%i]" % (min_x1+1,min_x1+unbin_width,
                                    min_y1+1,min_y1+unbin_height)


        mosaic_hd[arr_section_keyw] = "[1:%i,1:%i]"%(unbin_width,\
                                            unbin_height)
        mosaic_hd[det_section.keyword] = detsec
        mosaic_hd[dat_section_keyw] = \
                      "[1:%i,1:%i]"%(mosaic_shape[1],mosaic_shape[0])
        
        ccdname = ''
        if 'CCDNAME' in adin[extn,1].header:
            for ext in adin[extn]:
                ccdname += ext.header['CCDNAME']+','
            mosaic_hd.update("CCDNAME", ccdname)
            
        # Remove these keywords from the mosaic header.
        for kw in ['FRMNAME','FRAMEID','CCDSIZE','BIASSEC','DATATYP']:
            if kw in mosaic_hd:
                del mosaic_hd[kw]

        del mosaic_hd['extver'] 
        mosaic_hd.update('EXTVER',1,after='EXTNAME')

        wcs = pywcs.WCS(mosaic_hd)

        # Update CRPIX1 and CRPIX2.
        crpix1,crpix2 = self.update_crpix(wcs,tile)
        mosaic_hd.update("CRPIX1", crpix1)
        mosaic_hd.update("CRPIX2", crpix2)

        return mosaic_hd

    def info(self):
        """ 
         Creates a dictionary with coordinates, 
         amplifier and block information:

         The keys are:

        filename  (type: string)
           The original FITS filename
	amps_per_block (type: int)
	   The number of amplifier in each block
	amp_mosaic_coord (type: list of tuples (x1,x2,y1,y2))
           The list of amplifier location within the mosaic. 
           These values do not include the gaps between the blocks
	amp_block_coord (type: list of tuples (x1,x2,y1,y2))
	   The list of amplifier location within a block.
	interpolator (type: string)
	   The interpolator function name in use when transforming
           the blocks.
	reference_extname (type: string)
	   The value of the EXTNAME header keyword use as reference.
	   This is the output EXTNAME in the output mosaic FITS header.
	reference_extver (type: int)
	   The value of the EXTVER header keyword use as reference. 
	   Use this WCS information to generate the updated output mosaic
           FITS header.
	reference_block	The block number containing the reference amplifier

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
        info['amps_shape_no_trimming']=\
            [k.data.shape[::-1] for k in self.ad[self.ref_extname]]
        return info

    def merge_catalogs(self, ref_wcs, tile, merge_extvers, tab_extname, 
                        recalculate_xy='wcs',transform_pars=None):
        """
          This function merges together separate bintable extensions (tab_extname), 
          converts the pixel coordinates to the reference extension WCS
          and remove duplicate entries based on RA and DEC. 

          NOTE: Names used here so far: *OBJCAT:* Object catalog extension name

          *Input:*

          :param ref_wcs: Pywcs object containing the WCS from the output header.
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
             For 'transform' mode this are the
             linear equations to use.

             X_out = X*mx*cosA - Y*mx*sinA + mx*tx
             Y_out = X*my*sinA + Y*my*cosA + my*ty

             mx,my: magnification factors.
             tx,ty: translation amount in pixels.
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
        #   column_names = 
        #      {'OBJCAT': ('X_IMAGE', 'Y_IMAGE', 'X_WORLD', 'Y_WORLD'),
        #      'REFCAT': (None, None, 'RAJ2000', 'DEJ2000') }


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

                if tile: gap_mode = 'tile_gaps'
                else:    gap_mode = 'transform_gaps'
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
        ra,  raindx  = np.unique(col_data[ra_colname].round(decimals=7),
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
            data = columns.get(name, pf.Column(name=name,format=format,
                            array=col_data[name]))
            table_columns.append(data)

        # Make the output table using pyfits functions
        col_def = pf.ColDefs(table_columns)
        tb_hdu = pf.new_table(col_def)

        # Now make an AD object from this table
        adout = AstroData(tb_hdu)
        adout.rename_ext(tab_extname,1)

        # Append to any other new table we might have
        adoutput_list.append(adout)

        return adoutput_list
