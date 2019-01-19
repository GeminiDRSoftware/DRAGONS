# astromodels.py
#
# This module contains new Model classes to aid with image transformations.

import numpy as np
import math
from astropy.modeling import models, FittableModel, Parameter

class Pix2Sky(FittableModel):
    """
    Wrapper to make an astropy.WCS object act like an astropy.modeling.Model
    object, including having an inverse.
    """
    def __init__(self, wcs, x_offset=0.0, y_offset=0.0, factor=1.0,
                 angle=0.0, origin=1, **kwargs):
        self._wcs = wcs.deepcopy()
        self._direction = 1  # pix->sky direction
        self._origin = origin
        super(Pix2Sky, self).__init__(x_offset, y_offset, factor, angle,
                                      **kwargs)

    inputs = ('x','y')
    outputs = ('x','y')
    x_offset = Parameter()
    y_offset = Parameter()
    factor = Parameter()
    angle = Parameter()

    def evaluate(self, x, y, x_offset, y_offset, factor, angle):
        # x_offset and y_offset are actually arrays in the Model
        #temp_wcs = self.wcs(x_offset[0], y_offset[0], factor, angle)
        temp_wcs = self.wcs
        return temp_wcs.all_pix2world(x, y, self._origin) if self._direction>0 \
            else temp_wcs.all_world2pix(x, y, self._origin)

    @property
    def inverse(self):
        inv = self.copy()
        inv._direction = -self._direction
        return inv

    @property
    def wcs(self):
        """Return the WCS modified by the translation/scaling/rotation"""
        wcs = self._wcs.deepcopy()
        x_offset = self.x_offset.value
        y_offset = self.y_offset.value
        angle = self.angle.value
        factor = self.factor.value
        wcs.wcs.crpix += np.array([x_offset, y_offset])
        if factor != 1:
            wcs.wcs.cd *= factor
        if angle != 0.0:
            m = models.Rotation2D(angle)
            wcs.wcs.cd = m(*wcs.wcs.cd)
        return wcs


class Shift2D(FittableModel):
    """2D translation"""
    inputs = ('x', 'y')
    outputs = ('x', 'y')
    x_offset = Parameter(default=0.0)
    y_offset = Parameter(default=0.0)

    @property
    def inverse(self):
        inv = self.copy()
        inv.x_offset = -self.x_offset
        inv.y_offset = -self.y_offset
        return inv

    @staticmethod
    def evaluate(x, y, x_offset, y_offset):
        return x+x_offset, y+y_offset

class Scale2D(FittableModel):
    """2D scaling"""
    def __init__(self, factor=1.0, **kwargs):
        super(Scale2D, self).__init__(factor, **kwargs)

    inputs = ('x', 'y')
    outputs = ('x', 'y')
    factor = Parameter(default=1.0)

    @property
    def inverse(self):
        inv = self.copy()
        inv.factor = 1.0/self.factor
        return inv

    @staticmethod
    def evaluate(x, y, factor):
        return x*factor, y*factor

class Rotate2D(FittableModel):
    """Rotation; Rotation2D isn't fittable"""
    def __init__(self, angle=0.0, **kwargs):
        super(Rotate2D, self).__init__(angle, **kwargs)

    inputs = ('x', 'y')
    outputs = ('x', 'y')
    angle = Parameter(default=0.0, getter=np.rad2deg, setter=np.deg2rad)

    @property
    def inverse(self):
        inv = self.copy()
        inv.angle = -self.angle
        return inv

    @staticmethod
    def evaluate(x, y, angle):
        if x.shape != y.shape:
            raise ValueError("Expected input arrays to have the same shape")
        orig_shape = x.shape or (1,)
        inarr = np.array([x.flatten(), y.flatten()])
        s, c = math.sin(angle), math.cos(angle)
        x, y = np.dot(np.array([[c, -s], [s, c]], dtype=np.float64), inarr)
        x.shape = y.shape = orig_shape
        return x, y
