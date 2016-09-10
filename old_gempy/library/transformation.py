
import numpy as np
import scipy.ndimage as nd

class Transformation(object): 
    """
      Transformation provides facilities to transform a frame of 
      reference by rotation, translation (shifting) and/or 
      magnification. If the frame is an image, the input is 
      interpolatated using a Spline function.

      Members
      -------

      params: Dictionary with:
              {'rotation':         # Radians of rotation about the
                                   # frame center. It has reverse sign
                                   # as the input argument to be compatible
                                   # with IRAF.
               'shift':(xshift,yshift)  # Shift in pixels
               'magnification':(xmag,ymag) # (scalars). Magnification 
                                           # about the frame center.
               }
      matrix: Matrix rotation. Set when affine_transform is used.
      notransform:  Boolean flag indicating whether to apply the
                    transformation function or not.
      xy_coords:    A tuple is (x_array, y_array) contanining the
                    linear transformation equations. It is used when
                    the map_coordinates method is reuqested.
      affine:       Bool. Default is True. Use the affine_tranformation
                    function.
      map_coords:   Bool. Default is False. To use the map_coordinates
                    function.
      dq_data:      Bool. Default is False. To use the 'trasform_dq'
                    function.
      order:        Spline interpolator order. If 1 it use a linear
                    interpolation method, zero is the 'nearest' method.
                    Anything greater than 1 will give the specified order
                    of spline interpolation.

      mode:         string. Default is 'constant'. Points outside the 
                    boundaries of the input are filled according to the 
                    given mode ('constant', 'nearest', 'reflect' or 'wrap').
      cval:         scalar. Value used for points outside the boundaries of
                    the input if ``mode='constant'``. Default is 0.0

      Methods
      -------
        affine_init
           Set the affine_transformation function parameters:
           matrix and offset.

        map_coords_init
           Set the linear equations to transform the data
           For each (x,y) there is one (x_out,y_out) which
           are function of rotation, shift and/or magnification.

        affine_transform
           Front end method to the scipy.ndimage.affine_transform
           function.

        map_coordinates
           Front end method to the scipy.ndimage.affine_transform
           function.

        transform
          High level method to drive an already set transformation 
          function. The default one is 'affine'

        set_affine
           Set attribute affine to True or False

        set_map_coords
           Set attribute map_coords to True or False

        set_dq_data
           Set attribute dq_data to True or False

        set_interpolator
           Changing the interpolation method to use when
           correcting the blocks for rotation, shifting and
           magnification. 

        transform_dq
            Transform the data quality plane. Must be handled a little
            differently. DQ flags are set bit-wise, such that each pixel is the
            sum of any of the following values: 0=good pixel,
            1=bad pixel (from bad pixel mask), 2=nonlinear, 4=saturated, etc.
            To transform the DQ plane without losing flag information, it is
            unpacked into separate masks, each of which is transformed in the same
            way as the science data. A pixel is flagged if it had greater than
            1% influence from a bad pixel. The transformed masks are then added
            back together to generate the transformed DQ plane.

    """
    
    def __init__(self, rotation, shift, magnification, interpolator='affine',
                 order=1, as_iraf=True):
        """
          parameters
          ----------

          rotation:  Rotation in degrees from the x-axis in the
                     counterwise direction about the center.
          shift:     Tuple (x_shit,y_shift), amount to shift in pixel.
          magnification:
                     Tuple of floats, (x,y) magnification. Amount 
                     to magnify the output frame about the center.
          offset:    For affine transform, indicates the offset 
                     into the array where the transform is applied.
          interpolator: The interpolator function use, values are:
                     'affine': (Default), uses ndimage.affine_tranform
                               which uses a spline interpolator of order (order).
                     'map_coords': Uses ndimage.maps_coordinates using a
				 spline interpolator of order (order)
                     'dq_data': Uses the method transform_dq.
          order:     The order of the spline interpolator use in
                     ndimage.affine_transform and ndimage.maps_coordinates.
          as_iraf:   (bool). Default: True. If True, set the transformation
                     to be compatible with IRAF/geotran task, where the 
                     rotation is in the counterwise direction. 
             

        """

        # Set rotation angle (degrees) to radians
        rot_rads = np.radians(rotation)

        # Turn input tuples to single values
        xshift,yshift = shift
        xmag,ymag = magnification

        # Convert rotation and magnification values to IRAF/geotran 
        # types.
        if as_iraf:
            rot_rads = -rot_rads     # The rotation direction in Python
                                     # scipy.ndimage package is clockwise,
                                     # so we change the sign.
    
        # If the rot,shift and magnification have default values
        # then set member notransform to True.
        self.notransform = False
        if  (rot_rads==0.) and (shift==(0.,0.)) and (magnification==(1,1)):
            self.notransform = True

        # For clarity put the pars in a dictionary.
        self.params ={'rotation':rot_rads, 'shift':(xshift,yshift),
                      'magnification':(xmag,ymag)}

        if interpolator not in ['affine','map_coords']:
            raise ValueError('Bad input parameter "interpolator" value')

        # Set default values
        self.affine = True
        self.map_coords = False
        self.dq_data = False

        # Reset if parameter does not have the default value.
        if interpolator == 'map_coords':
            self.set_map_coords()

        if order>5:
            raise ValueError('Spline order cannot be greater than 5')

        self.order = min(5, max(order,1))  # Spline order

        # Set default values for these members. The methods affine_transform
        # and map_coordinates allow changing.
        self.mode = 'constant'
        self.cval = 0


    def affine_init(self,imagesize):
        """
          Set the affine_transformation function parameters:
          matrix and offset.

          Input
          -----
            imagesize:  
                Tuple with image.shape values or (npixel_y, npixels_x).
                These are used to put the center of rotation to the
                center of the frame.
        """
                        

        # Set rotation origin as the center of the image
        ycen,xcen = np.asarray(imagesize)/2.

        xmag,ymag = self.params['magnification']


        xshift,yshift = self.params['shift']
        rot_rads = self.params['rotation']

        cx = xmag*np.cos(rot_rads)
        sx = xmag*np.sin(rot_rads)
        cy = ymag*np.cos(rot_rads)
        sy = ymag*np.sin(rot_rads)
        self.matrix = np.array([[cy, sy], [-sx, cx]])

        # Add translation to this point
        xcen += xshift
        ycen += yshift

        xoff = (1.-cx)*xcen + sx*ycen
        yoff = (1.-cy)*ycen - sy*xcen

        # We add back the shift
        xoff -= xshift
        yoff -= yshift

        self.offset = (yoff, xoff)


    def affine_transform(self, image, order=None, mode=None, cval=None):
        """
          Front end method to the scipy.ndimage.affine_transform
          function.

          Inputs
          ------
              Image: ndarray with image data. 
              order:    Spline interpolator order. If 1 it use a linear
                        interpolation method, zero is the 'nearest' method.
                        Anything greater than 1 will give the specified order
                        of spline interpolation.
              mode:     string. Default is 'constant'. Points outside the 
                        boundaries of the input are filled according to the 
                        given mode ('constant', 'nearest', 'reflect' or 'wrap').
              cval:     scalar. Value used for points outside the boundaries of
                        the input if ``mode='constant'``. Default is 0.0

              If order, mode and/or cval are different from None, they 
              replace the default values.
        """
        
    
        if order == None:
            order = self.order
        if mode == None:
            mode = self.mode
        if cval == None:
            cval = self.cval
            
        if self.notransform:
            return image

        if not hasattr(self,'offset'):
            self.affine_init(image.shape)

        prefilter = order > 1 
        matrix = self.matrix
        offset = self.offset

        image = nd.affine_transform(image, matrix, offset=offset, 
                                    prefilter=prefilter,
                                    mode=mode, order=order, cval=cval)

        return image 

    def map_coords_init(self, imagesize):
        """
          Set the linear equations to transform the data
          For each (x,y) there is one (x_out,y_out) which
          are function of rotation, shift and/or magnification.

          Input
          -----
            imagesize: The shape of the frame where the (x,y)
                       coordinates are taken from. It sets
                       the center of rotation.
        """

        # Set rotation origin as the center of the image
        ycen, xcen = np.asarray(imagesize)/2.


        xsc,ysc       = self.params['magnification']
        xshift,yshift = self.params['shift']
        rot_rads      = self.params['rotation']

        cx = xsc*np.cos(rot_rads)
        sx = xsc*np.sin(rot_rads)
        cy = ysc*np.cos(rot_rads)
        sy = ysc*np.sin(rot_rads)


        # Create am open mesh_grid. Only one-dimension 
        # of each argument is returned.
        Y,X = np.ogrid[:imagesize[0],:imagesize[1]]

        # Add translation to this point
        xcen += xshift
        ycen += yshift

        xcc = xcen - X
        ycc = ycen - Y

        xx = -xcc*cx + ycc*sx + xcen - xshift
        yy = -ycc*cy - xcc*sy + ycen - yshift

        self.xy_coords = np.array([yy,xx])


    def map_coordinates(self, image, order=None, mode=None, cval=None):
        """
          Front end to scipy.ndimage.map_cordinates function
    
          Input
          -----
              Image: ndarray with image data. 
              order:    Spline interpolator order. If 1 it use a linear
                        interpolation method, zero is the 'nearest' method.
                        Anything greater than 1 will give the specified order
                        of spline interpolation.
              mode:     string. Default is 'constant'. Points outside the 
                        boundaries of the input are filled according to the 
                        given mode ('constant', 'nearest', 'reflect' or 'wrap').
              cval:     scalar. Value used for points outside the boundaries of
                        the input if ``mode='constant'``. Default is 0.0

              If order, mode and/or cval are different from None, they 
              replace the default values.
        """
           
        if order == None:
            order = self.order
        if mode == None:
            mode = self.mode
        if cval == None:
            cval = self.cval
            
        if self.notransform:
            return image

        if not hasattr(self,'xy_coords'):
            self.map_coords_init(image.shape)

        prefilter = order > 1 
        # The xy_ccords member is set in map_coords_init().
        image = nd.map_coordinates(image, self.xy_coords,
                                   prefilter=prefilter,
                                   mode=mode, order=order, cval=cval)

        del self.xy_coords
        return image 

    
    def transform(self,data):
        """
          High level method to drive an already set transformation 
          function. The default one is 'affine'
        """

        if self.affine:    # Use affine_transform
            output = self.affine_transform(data)
        elif self.map_coords:  # Use map_coordinates
            output = self.map_coordinates(data)
        elif self.dq_data:     # For DQ data use map_coordinates
            output = self.transform_dq(data)
        else:
            raise ValueError("Transform function not defined.")

        return output

    def set_affine(self):
        """Set attribute affine to True"""
        self.affine = True
        self.map_coords = False
        self.dq_data = False

    def set_map_coords(self):
        """Set attribute map_coords to True or False"""
        self.map_coords = True
        self.affine = False
        self.dq_data = False

    def set_dq_data(self):
        """Set attribute dq_data to True or False"""
        self.dq_data = True
        self.affine = False
        self.map_coords = False

    def set_interpolator(self,tfunction='linear',spline_order=2):
        """
           Changing the interpolation method to use when
           correcting the blocks for rotation, shifting and
           magnification. The methods and order are defined
           in the global variable 'trans_functions' at the
           bottom of this module. 

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

        # check for allowed values:
        if tfunction not in ('linear', 'nearest', 'spline'):
            raise ValueError("Interpolator:'"+tfunction+\
                "' is not in ('linear','nearest','spline')")

        # Set the order:
        if tfunction == 'linear': 
            order = 1
        elif tfunction == 'nearest': 
            order = 0
        else: 
            order = min(5, max(spline_order,2))  # Spline. No more than 5

        self.order=order

    def transform_dq(self, data):
        """
            Transform the data quality plane. Must be handled a little
            differently. DQ flags are set bit-wise, such that each pixel is the
            sum of any of the following values: 0=good pixel,
            1=bad pixel (from bad pixel mask), 2=nonlinear, 4=saturated, etc.
            To transform the DQ plane without losing flag information, it is
            unpacked into separate masks, each of which is transformed in the same
            way as the science data. A pixel is flagged if it had greater than
            1% influence from a bad pixel. The transformed masks are then added
            back together to generate the transformed DQ plane.

            DQ flags are set bit-wise

            bit 1: bad pixel (1)
            bit 2: nonlinear (2)
            bit 3: saturated (4)
            bit 4: 
            bit 5: nodata (5)

            A pixel can be 0 (good, no flags), or the sum of
            any of the above flags
            (or any others I don't know about)

            (Note: This code was taken from resample.py)

            Parameters
            ----------
            :param data: Ndarray to transform
            :param data,rot,shift, mag, jfactor: See self.affine_tranform help

            Output
            ------
            :param outdata: The tranformed input data. Input remains unchanged.

        """
        if self.notransform:
            return data

        # unpack the DQ data into separate masks
        # NOTE: this method only works for 8-bit masks!
        unp = data.shape + (8,)

        unpack_data = np.unpackbits(np.uint8(data)).reshape(unp)

        # transform each mask
        outdata = np.zeros(data.shape)
        do_nans = False
        gnan = None

        for j in range(0,8):

            # skip the transformation if there are no flags set
            # (but always do the bad pixel mask because it is
            # needed to mask the part of the array that was
            # padded out to match the reference image)
            if not unpack_data[:,:,j].any() and j!=7:
                # first bit is j=7 because unpack is backwards
                continue
            mask = np.float32(unpack_data[:,:,j])

            # if bad pix bit, pad with 1. Otherwise, pad with 0
            if j==7:
                cval = 1
            else:
                # Instead of padding with zeros, we mark the
                # no-data values with Nans
                cval = np.nan
                do_nans = True
            trans_mask = self.map_coordinates(mask,cval=cval)

            del mask; mask = None
            # Get the nans indices:
            if do_nans:
                gnan = np.where(np.isnan(trans_mask))
                # We mark only once
                do_nans = False

            # flag any pixels with >1% influence from bad pixel
            trans_mask = np.where(np.abs(trans_mask)>0.01, 2**(7-j),0)
            # add the flags into the overall mask
            outdata += trans_mask
            del trans_mask; trans_mask = None

            # QUESTION: Do we need to put outdata[gnan] for each plane != 7?
        # put nan's back
        if gnan != None:
            outdata[gnan] = np.nan
        return outdata


