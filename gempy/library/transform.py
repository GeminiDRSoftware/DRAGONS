# Copyright(c) 2018-2020 Association of Universities for Research in Astronomy, Inc.
#
"""
transform.py

This module contains classes related to the geometric transformation of
arrays and AstroData objects.

Classes:
    Block: a container for array-like objects that are adjacent to each other
           and so much be transformed together (e.g., each CCD in GMOS is a
           Block)
    Transform: a collection of chained astropy Models that together describe
               a transformation from one set of coordinates to another
    GeoMap: a callable object that accepts coordinates and returns
            geometrically-transformed coordinates
    DataGroup: a collection of array-like objects and transforms that will be
               combined into a single output (more precisely, a single output
               per attribute)

Functions:
    find_reference_extension: returns the index of the slice that is the
                              reference extension of a multi-extension AstroData
                              object (i.e., the one whose header will be used
                              to construct the header of the mosaicked AD)
    add_mosaic_wcs: attaches gWCS objects to all extensions with a "mosaic"
                    frame that defines how the AD will be mosaicked into a
                    single extension
    add_longslit_wcs: attaches gWCS objects to all extensions with 3 output
                      axes: RA, DEC, and wavelength
    resample_from_wcs: creates a new (possibly mosaicked) AstroData object
                       based on the pixel->pixel transform encoded within the
                       gWCS object
    get_output_corners: returns the transformed coordinates of a rectangular
                        region
"""
import numpy as np
from copy import deepcopy
from functools import reduce
import warnings

from astropy.modeling import models, Model
from astropy.modeling.core import _model_oper, fix_inputs
from astropy import table, units as u

from gwcs import coordinate_frames as cf
from gwcs.wcs import WCS as gWCS
from scipy import ndimage

from gempy.library import astromodels as am, astrotools as at
from gempy.gemini import gemini_tools as gt

try:
    from gempy.library.cython_utils import polyinterp
except ImportError:  # pragma: no cover
    raise ImportError("Run 'cythonize -i cython_utils.pyx' in gempy/library")

import multiprocessing as multi
from geminidr.gemini.lookups import DQ_definitions as DQ

import astrodata
from astrodata import wcs as adwcs

from .astromodels import Rotate2D, Shift2D, Scale2D
from ..utils.decorators import insert_descriptor_values
from ..utils import logutils

log= logutils.get_logger(__name__)

# Table attribute names that should be modified to represent the
# coordinates in the Block, not their individual arrays.
# NB. Standard python ordering!
catalog_coordinate_columns = {'OBJCAT': (['Y_IMAGE'], ['X_IMAGE'])}

class Block:
    """
    A Block is a container for multiple AD slices/NDData/ndarray objects
    to organize them prior to being accessed for a transformation.

    In the grand scheme of things, Blocks can be transformed in parallel
    because they have gaps between them and so don't overlap in the final
    transformed image.

    Blocks must have the same dimensionality as their component elements.
    So if you want to layer 2D images into a 3D stack, you need to add a
    third dimension to the data. This requirement could be eliminated.
    """
    def __init__(self, elements, shape=None, xlast=False):
        """
        elements: list of AD slices/NDData/ndarray
        shape: tuple giving shape in which the elements should be arranged
               (if None, assume they're lined up along the x-axis)
        xlast: normally arrays are placed in the standard python order,
               along the x-axis first. This reverses that order.
        """
        super().__init__()
        self._elements = list(elements)
        shapes = [el.shape for el in elements]

        # Check all elements have the same dimensionality
        if len({len(s) for s in shapes}) > 1:
            raise ValueError("Not all elements have same dimensionality")

        # Check the shape for tiling elements matches number of elements
        if shape is None:
            shape = (len(elements),)
        shape = (1,) * (len(shapes[0])-len(shape)) + shape
        if np.multiply.reduce(shape) != len(elements):
            raise ValueError("Incompatible element list and shape")

        # Check that shapes are compatible
        all_axis_lengths = np.rollaxis(np.array([s for s in shapes]).
                                       reshape(shape + (len(shape),)), -1)
        # There's almost certainly a smarter way to do this but it's
        # a struggle for me to visualize it. Basically, we want to go along
        # each axis and keep the lengths of the elements along that axis,
        # while ensuring the lengths along each of the other axes are the
        # same as each other (the np.std()==0 check)
        lengths = []
        for i, axis_lengths in enumerate(all_axis_lengths):
            collapse_axis = 0
            for axis in range(len(shape)):
                if axis != i:
                    if np.std(axis_lengths, axis=collapse_axis).sum() > 0:
                        raise ValueError("Incompatible shapes along axis "
                                         "{}".format(axis))
                    axis_lengths = np.mean(axis_lengths, axis=collapse_axis)
                else:
                    collapse_axis = 1
            lengths.append(axis_lengths.astype(int))
        self._total_shape = tuple(int(l.sum()) for l in lengths)

        corner_grid = np.array(np.meshgrid(*(np.cumsum([0]+list(l))[:-1]
                      for l in lengths), indexing='ij')).reshape(len(shape), len(elements))
        self.corners = (list(zip(*corner_grid[::-1])) if xlast
                        else list(zip(*corner_grid)))

    def __getattr__(self, name):
        """
        Returns the named attribute as if the Block were really a single
        object. The "data" attribute refers to the arrays themselves if
        the elements are simple ndarrays. Tables and arrays are handled
        trivially, other attributes are generally handled by returning a
        list of the individual elements' attributes.

        However, if this list is composed solely of Nones, a single None
        is returned. This is intended to handle .mask and .variance attributes
        if they're not defined on the individual elements.
        """
        attributes = [getattr(el, name, None) for el in self._elements]
        # If we're looking for the .data attributes, this may just be the
        # element, if they're ndarrays. We should handle a mix of ndarrays
        # and NDData-like objects.
        if name == "data":
            attributes = [el if not isinstance(attr, np.ndarray) else attr
                          for el, attr in zip(self._elements, attributes)]

        if any(isinstance(attr, np.ndarray) for attr in attributes):
            try:
                return self._return_array(attributes)
            except AttributeError as e:
                if name == "data":
                    return self._return_array(self._elements)
                else:
                    raise e
        # Handle Tables
        if any(isinstance(attr, table.Table) for attr in attributes):
            return self._return_table(name)

        # Otherwise return a list of the attributes, if they exist
        attributes = [getattr(el, name) for el in self._elements]
        # If all the attributes are None (e.g., .mask), return a single None
        if all(attr is None for attr in attributes):
            return None
        return attributes

    def __getitem__(self, index):
        """Return an element of the Block"""
        return self._elements[index]

    def __len__(self):
        return len(self._elements)

    @property
    def ndim(self):
        """Number of dimensions"""
        return len(self.shape)

    @property
    def shape(self):
        """Overall shape of Block"""
        return self._total_shape

    @property
    def wcs(self):
        """WCS object"""
        if not all (c == 0 for c in self.corners[0]):
            raise ValueError("First element does not start at origin")
        return self._elements[0].wcs

    def _return_array(self, elements):
        """
        Returns a new ndarray composed of the composite elements, which is
        required for the scipy.ndimage transformations. It attempts to handle
        an element list that includes Nones (e.g., OBJMASKs or masks) by
        filling those regions with zeros.
        """
        if len(elements) != len(self._elements):
            raise ValueError("All elements of the Block must be used in the "
                             "output array")
        if len(elements) == 1:
            return elements[0]

        # Cope with an element list that may include Nones
        output = None
        for corner, arr in zip(self.corners, elements):
            if arr is None:
                continue
            if output is None:
                output = np.zeros(self.shape, dtype=arr.dtype)
            slice_ = tuple(slice(c, c+a) for c, a in zip(corner, arr.shape))
            output[slice_] = arr
        return output

    def pixel(self, *coords, attribute="data"):
        """
        Returns the value of a pixel in a Block.

        Parameters
        ----------
        coords: tuple
            the coordinates in the Block whose pixel value we want
        attribute: str
            name of attribute to extract

        Returns
        -------
            float: the pixel value at this coordinate
        """
        if len(coords) != len(self.shape):
            raise ValueError("Incorrect number of coordinates")

        for corner, arr in zip(self.corners, self._elements):
            offset_coords = tuple(coord - corner_coord
                             for coord, corner_coord in zip(coords, corner))
            if all(c >=0 and c < s for c, s in zip(offset_coords, arr.shape)):
                if attribute == "data" and isinstance(arr, np.ndarray):
                    return arr[offset_coords]
                else:
                    attr = getattr(arr, attribute)
                    if attr is None:
                        return None
                    else:
                        return attr[offset_coords]
        raise IndexError("Coordinates not in block")

    def _return_table(self, name):
        """
        Returns a single Table, obtained by concatenating all the Tables of
        the given name. If the name is in the catalog_coordinate_columns dict,
        then the appropriate columns are edited to indicate the coordinates in
        the Block, rather than the individual elements.
        """
        if len(self._elements) == 1:
            return getattr(self._elements[0], name)
        tables = []
        try:
            col_name_list = catalog_coordinate_columns[name]
        except KeyError:
            col_name_list = [[] * self.ndim]
        for el, corner in zip(self._elements, self.corners):
            # ValueError is raised if the Table exists but is empty; trap this
            try:
                t = getattr(el, name).copy()
            except (AttributeError, ValueError):
                continue
            for col_names, coord in zip(col_name_list, corner):
                for col_name in col_names:
                    t[col_name] += coord
            tables.append(t)
        return table.vstack(tables, metadata_conflicts='silent')

#----------------------------------------------------------------------------------

class Transform:
    """
    A chained set of astropy Models with the ability to select a subchain.
    Since many transformations will be concatenated pairs of Models (one
    for each ordinate), this also creates a better structure.

    A Transform can be accessed and iterated over like a list, but also
    like a dict if the individual models have names. So if the Transform
    object "t" has models called "m0", "m1", "m2", "m3", etc., then
    t["m1":"m3"] will create a Transform consisting of "m1" and "m2".
    t["m3":"m1":-1] will create a Transform of "m3.inverse" and "m2.inverse"
    """
    def __init__(self, model=None, copy=True):
        """Initialize with a single model or list. We
        implement a copy option since we want to be able to set attributes
        of temporarily-sliced objects."""
        self._models = []
        self._affine = True
        self._ndim = None
        if model:
            self.append(model, copy)

    def __call__(self, *args, **kwargs):
        if self.ndim is not None and len(args) != self.ndim:
            raise ValueError("Incompatible number of inputs for Transform "
                             "(dimensionality {})".format(self.ndim))
        try:
            inverse = kwargs.pop('inverse')
        except KeyError:
            inverse = False
        return self.asModel(inverse=inverse)(*args, **kwargs)

    def __copy__(self):
        transform = self.__class__(self._models, copy=False)
        transform._affine = self._affine
        return transform

    def __deepcopy__(self, memo):
        transform = self.__class__(self._models, copy=True)
        transform._affine = self._affine
        return transform

    def copy(self):
        return deepcopy(self)

    def __getattr__(self, key):
        """
        The is_affine property and asModel() method should be used to extract
        information, so attributes are intended to refer to the component
        models, and only make sense if this attribute is unique among the models.
        """
        values = []
        for m in self._models:
            try:
                values.append(getattr(m, key))
            except AttributeError:
                pass
        if len(values) == 1:
            return values.pop()
        raise AttributeError("Attribute '{}' found on {} models".
                             format(key, len(values)))

    def __setattr__(self, key, value):
        """
        It is assumed that we're setting attributes of the component models.
        We are allowed to set an attribute on multiple models (e.g., name),
        if that attribute exists on all models, or only a single model.
        """
        if key not in ("_models", "_affine", "_ndim"):
            models = [m for m in self._models if hasattr(m, key)]
            if len(models) in (1, len(self)):
                for m in models:
                    setattr(m, key, value)
                return
            raise AttributeError("Cannot set attribute '{}'".format(key))
        return object.__setattr__(self, key, value)

    def __getitem__(self, key):
        """
        Return a Transform based on index or name, slice, or tuple. If the
        models are in descending index order, then the inverse will be used.
        """
        if isinstance(key, slice):
            # Turn into a slice of integers
            slice_ = slice(self.index(key.start),
                           self.index(key.stop), key.step)
            if key.step is None or key.step > 0:
                models = self._models[slice_]
            else:  # Return chain of inverses
                models = [m.inverse for m in self._models[slice_]]
        elif isinstance(key, tuple):
            indices = [self.index(k) for k in key]
            if all(np.diff(indices) < 0):
                models = [self[i].inverse for i in indices]
            else:
                models = [self[i] for i in indices]
        else:
            try:
                models = self._models[key]
            except TypeError:
                models = [self._models[i] for i in self._indices(key)]
        return self.__class__(models, copy=False)

    def __setitem__(self, key, value):
        """Replace a model in the Transform. Requires that the replacement
        Model have the same number of inputs and outputs as the current one"""
        if not isinstance(value, Model):
            raise ValueError("'Transform' items must be Models")
        if not (value.n_inputs == self._models[key].n_inputs and
                value.n_outputs == self._models[key].n_outputs):
            raise ValueError("Model does not have the same number of inputs "
                             "and outputs as the model it replaces")
        self._models[key] = value
        self._affine = self.__is_affine()

    def __delitem__(self, key):
        """Delete a model by index or name"""
        try:
            del self._models[key]
        except TypeError:
            for i in self._indices(key)[::-1]:
                del self._models[i]
        self._affine = self.__is_affine()

    def __iter__(self):
        yield from self._models

    def __len__(self):
        """Length (number of models)"""
        return len(self._models)

    @property
    def inverse(self):
        """The inverse transform"""
        return self.__class__([m.inverse for m in self._models[::-1]])

    @property
    def ndim(self):
        """Dimensionality (number of inputs)"""
        return self._ndim

    @property
    def fittable(self):
        """Although an Identity Model is fittable, it's not really, and
        we're implementing this by our own function (below)"""
        return len(self) > 0 and self.asModel().fittable

    @staticmethod
    def identity_model(*args):
        return models.Identity(len(args))(*args)

    def asModel(self, inverse=False):
        """
        Return a Model instance of this Transform, by chaining together
        the individual Models.

        Parameters
        ----------
        inverse: bool
            Return the inverse model. t.asModel(inverse=True) is the
            same as t.inverse.asModel() and t[::-1].asModel()
        """
        if len(self) == 0:
            return self.identity_model
        model_list = self.inverse._models if inverse else self._models
        return reduce(Model.__or__, model_list)

    def index(self, key):
        """Works like the list.index() method to return the index of a model
        with a specific name. Traps being sent None (returns None) or an
        integer (which is assumed to be a real index), so don't give models
        names which are integer objects!"""
        if key is None:
            return
        try:
            return [m.name for m in self._models].index(key)
        except ValueError as e:
            if isinstance(key, (int, np.integer)):
                return key
            raise e

    def _indices(self, key):
        """
        Like the index method, but returns the indices of *all* models
        with the name of the key. This is implemented so that a group of
        related models can be given a single name and extracted together.
        """
        indices = [i for i, m in enumerate(self._models) if m.name == key]
        if indices:
            return indices
        raise NameError("Transform has no models named {}".format(key))

    def append(self, model, copy=True):
        """Append Model(s) to the end of the current Transform"""
        if isinstance(model, Model):
            self.insert(len(self), model, copy)
        else:
            if isinstance(model, self.__class__):
                model = model._models
            for m in model:
                self.insert(len(self), m, copy)

    def prepend(self, model, copy=True):
        """Prepend Model(s) to the start of the current Transform"""
        if isinstance(model, Model):
            self.insert(0, model, copy)
        else:
            if isinstance(model, self.__class__):
                model = model._models
            for m in model[::-1]:
                self.insert(0, m, copy)

    def insert(self, index, model, copy=True):
        """
        General method to insert a Model, using the same syntax as list.insert
        This checks that the Model being inserted has the correct number of
        inputs and outputs compared to the adjacent Models.

        Parameters
        ----------
        index: int
            index of item *before* which this model is to be inserted
        model: Model
            Model instance to be inserted
        copy: bool
            make a copy of the model before inserting?
        """
        try:
            required_outputs = self._models[index].n_inputs
        except IndexError:
            pass
        else:
            if len(model.outputs) != required_outputs:
                raise ValueError("Number of outputs ({}) does not match number"
                                 "of inputs of subsequent model ({})".
                                 format(len(model.outputs), required_outputs))
        try:
            required_inputs = self._models[index-1].n_outputs
        except IndexError:
            required_inputs = model.n_inputs
        else:
            if len(model.inputs) != required_inputs:
                raise ValueError("Number of inputs ({}) does not match number"
                                 "of outputs of previous model ({})".
                                 format(len(model.inputs), required_inputs))
        if index == 0:
            self._ndim = model.n_inputs

        # Ugly stuff to break down a CompoundModel. We need a general except
        # here because there are multuple reasons this might fail -- it's not
        # a CompoundModel, or it is, but the mappings are i
        try:
            sequence = self.split_compound_model(model._tree, required_inputs)
        except AttributeError:
            self._models.insert(index, model.copy() if copy else model)
        else:
            i = self.index(index)
            # Avoids corrupting a Model (it will lose its name)
            if len(sequence) == 1:
                self._models.insert(index, deepcopy(model) if copy else model)
            else:
                self._models[i:i] = sequence
        # Update affinity based on new model (leave existing stuff alone)
        self._affine &= adwcs.model_is_affine(model)

    @staticmethod
    def split_compound_model(tree, ndim):
        """
        Break a CompoundModel into a sequence of chained Model instances.
        It's necessary to specify the initial dimensionality of the Model
        to determine whether a chain operator (|) is linking full
        transformations or just components within a submodel. It may prove
        difficult/impossible to split a Model if the number of inputs is
        not preserved. We'll cross that bridge if/when we come to it.

        This has not been tested in extreme circumstances, but works for
        "normal" transformations.

        Parameters
        ----------
        tree: astropy.modeling.utils.ExpressionTree
            The tree to be processed
        ndim: int
            The number of inputs

        Returns
        -------
        list: a list of Models to be inserted into the Transform
        """
        stack = []
        # Expand the tree in a Reverse Polish Notation-type manner
        for node in tree.traverse_postorder():
            if node.isleaf:
                stack.append(node.value)
            else:
                operand = node.value
                # If this is the chain (|) operator we need to see if what
                # we have on the stack takes the right number of inputs.
                # If so, leave it there and updated the required number of
                # inputs; otherwise, combine the models and continue.

                # TODO: Allowing the number of inputs/outputs to change is
                # causing problems with complicated compound models.
                # For now, we will only consider as discrete models those
                # that preserve the number of inputs/outputs
                if operand == "|" and stack[-1].n_inputs == stack[-1].n_outputs == ndim:
                    #ndim = stack[1].n_outputs
                    continue
                right = stack.pop()
                left = stack.pop()
                stack.append(_model_oper(operand)(left, right))
        return stack

    def replace(self, model):
        """
        Gut-and-replace a Transform instance with an instance of a CompoundModel
        that does the same thing. The purpose of this is to update a Transform
        after fitting it, while retaining the names of individual components.

        Parameters
        ----------
        model: the CompoundModel instance
        """
        try:
            sequence = self.split_compound_model(model._tree, self.ndim)
        except AttributeError:
            sequence = [model]
        # Check that the new model is basically the same as the old one, by
        # ensuring the sequence is the same length and the submodels are the
        # same (so no need to re-check the affinity)
        if len(sequence) == len(self):
            for new, old in zip(sequence, self._models):
                if new.__class__ != old.__class__:
                    break
                try:
                    newsubs, oldsubs = new._submodels, old._submodels
                except AttributeError:
                    continue
                for newsub, oldsub in zip(newsubs, oldsubs):
                    if newsub.__class__ != oldsub.__class__:
                        break
            for i, m in enumerate(sequence):
                self._models[i] = m.rename(self[i].name)
            return
        raise ValueError("Replacement Model differs from existing Transform")

    def __is_affine(self):
        """
        Test for affinity.
        """
        return np.logical_and.reduce([adwcs.model_is_affine(m)
                                      for m in self._models])

    @property
    def is_affine(self):
        """Return affinity of model. I have deliberately written this as a
        settable attribute to override the __is_affine() method, in case
        that is unable to handle certain Models."""
        return self._affine

    def affine_matrices(self, shape=None):
        """
        Compute the matrix and offset necessary to turn a Transform into an
        affine transformation. This is done by computing the linear matrix
        along all axes extending from the centre of the region, and then
        calculating the offset such that the transformation is accurate at
        the centre of the region.

        Parameters
        ----------
        shape: sequence
            shape to use for fiducial points

        Returns
        -------
        AffineMatrices(array, array): affine matrix and offset, in
            standard python order
        """
        if shape is None:
            try:
                shape = (1000,) * self.ndim
            except TypeError:  # self.ndim is None
                raise TypeError("Cannot compute affine matrices without a "
                                "dimensionality")
        return adwcs.calculate_affine_matrices(self, shape)

    def add_bounds(self, param, range):
        """Add bounds to a parameter"""
        value = getattr(self, param).value
        getattr(self, param).bounds = (value - range, value + range)

    def info(self):
        """Print out something vaguely intelligible for debugging purposes"""
        new_line_indent = "\n    "
        msg = ['Model information: (dimensionality {})'.format(self.ndim)]
        for model in self._models:
            msg.append(repr(model))
        return new_line_indent.join(msg)

    def apply(self, input_array, output_shape, interpolant="linear",
              cval=0, inverse=False):
        """
        Apply the transform to the pixels of the input array. Recall that the
        scipy.ndimage functions need to know where in the input_array to find
        pixels to put in the output, and therefore they need the *inverse*
        transform. To allow the situation where only the inverse transform has
        been calculated (and which might have no analytic inverse), this can be
        called with inverse=True, so t.inverse.apply() and t.apply(inverse=True)
        are the same, but the latter does not require t.inverse to exist.

        Parameters
        ----------
        input_array: array
            the input "image"
        output_shape: sequence
            shape of the output array
        interpolant: str
            type of interpolant
        cval: float-like
            value to use where there's no pixel data
        inverse: bool
            do the inverse of what you'd expect

        Returns
        -------
        array: of shape output_shape
        """
        if self.is_affine:
            if inverse:
                mapping = self.affine_matrices(input_array.shape)
            else:
                mapping = self.inverse.affine_matrices(input_array.shape)
        else:
            mapping = GeoMap(self, output_shape, inverse=inverse)

        return self.transform_data(input_array, mapping, output_shape,
                                   interpolant=interpolant, cval=cval)

    @staticmethod
    def transform_data(input_array, mapping, output_shape=None,
                       interpolant="linear", cval=0):
        """
        Perform the transformation (a static method so it can be called
        elsewhere). This is the single point where the resampling function
        is called.

        Parameters
        ----------
        input_array
            the input "image"
        mapping: AffineMatrices or GeoMap instance
            describing the mapping from each output pixel to input coordinates
        output_shape: tuple/None
            output array shape (not needed for GeoMap)
        interpolant: str
            type of interpolant
        cval: float
            value for out-of-bounds pixels

        Returns
        -------
        ndarray: output array "image"
        """
        if interpolant == "nearest":
            interpolant = "spline0"
        elif interpolant == "linear":
            interpolant = "spline1"

        if not (interpolant.startswith("spline") or
                interpolant.startswith("poly")):
            raise ValueError(f"Invalid interpolant string '{interpolant}'")
        try:
            order = int(interpolant[-1])
        except ValueError:
            raise ValueError(f"Invalid interpolant string '{interpolant}'")

        if interpolant.startswith("spline"):
            if isinstance(mapping, GeoMap):
                return ndimage.map_coordinates(input_array, mapping.coords,
                                               cval=cval, order=order)
            return ndimage.affine_transform(
                input_array, mapping.matrix, mapping.offset, output_shape,
                cval=cval, order=order)

        # Polynomial interpolation
        if isinstance(mapping, GeoMap):
            affinity = False
            output_shape = mapping.coords[0].shape
            geomap = np.asarray(mapping.coords).ravel().astype(np.float32,
                                                               copy=False)
        else:
            affinity = True
            geomap = np.concatenate([mapping.matrix,
                                     mapping.offset[:, np.newaxis]],
                                    axis=1).astype(np.float32).ravel()
        out_array = np.empty(output_shape, dtype=np.float32)
        polyinterp(input_array.astype(np.float32, copy=False).ravel(),
                   np.asarray(input_array.shape, dtype=np.int32),
                   len(input_array.shape),
                   out_array.astype(np.float32, copy=False).ravel(),
                   np.asarray(out_array.shape, dtype=np.int32),
                   geomap, affinity, order, cval)

        return out_array

    @classmethod
    def create2d(cls, translation=(0, 0), rotation=None, magnification=None,
                 shape=None, center=None):
        """
        Creates a Transform object with any/all of shift/rotate/scale,
        with appropriately named models for easy modification.

        Parameters
        ----------
        translation: 2-tuple/None
            translation IN X, Y ORDER (if None, no translation)
        rotation: float/None
            rotation angle in degrees (if None, no rotation)
        magnification: float/None
            magnification factor (if None, no magnification)
        shape: 2-tuple/None
            shape IN PYTHON ORDER: if provided, the model does a centering
            shift before the other models, and an uncentering shift at the end
        center: 2-tuple/None
            alternatively, simply provide the center
        """
        transform = cls()
        need_to_center = rotation is not None or magnification is not None
        if need_to_center:
            if center is None and shape is not None:
                center = [0.5 * (s - 1) for s in shape]
                shape = None
            if center is not None:
                if shape is not None:
                    raise ValueError("Cannot create Transform with both center and shape")
                # We use regular Shift instead of Shift2D so only the
                # translation model has "x_offset" and "y_offset"
                # The longwinded approach is required because of an astropy bug
                shift0 = models.Shift(center[0])
                shift0.offset.fixed = True
                shift1 = models.Shift(center[1])
                shift1.offset.fixed = True
                shift = shift1 & shift0
                transform.append(shift)
        if translation is not None:
            transform.append(Shift2D(*translation).rename("Translate"))
        if rotation is not None:
            transform.append(Rotate2D(rotation).rename("Rotate"))
        if magnification is not None:
            transform.append(Scale2D(magnification).rename("Magnify"))
        if need_to_center and center is not None:
            transform.append(transform[0].inverse)
        return transform
#----------------------------------------------------------------------------------

class GeoMap:
    """
    Class to store ndim mapping arrays (one for each axis) indicating the
    coordinates in the input frame that each coordinate in the output frame
    transforms back to. The "coords" attribute can be sent directly to
    scipy.ndimage.map_coordinates(). This is created as a class to provide
    the affinity() function describing how good the affine approximation is
    (although this is not yet used in the codebase).

    Parameters
    ----------
    transform: Transform/Model
        the transformation
    shape: tuple
        shape of output array
    inverse: bool
        if True, then the transform is already the output->input transform,
        and doesn't need to be inverted
    """
    def __init__(self, transform, shape, inverse=False):
        # X then Y (for Transform)
        grids = np.meshgrid(*(np.arange(length) for length in shape[::-1]))
        self._transform = transform if inverse else transform.inverse
        self._shape = shape
        transformed = (self._transform(*grids)[::-1] if len(shape) > 1
                       else self._transform(grids))
        #self.coords = [coord.astype(np.float32) for coord in transformed]
        self.coords = transformed  # Y then X

    def affinity(self):
        """
        Calculate deviations from reality when using an affine transform

        Returns
        -------
        tuple: rms and maximum deviation in input-frame pixels
        """
        mapping = self._transform.affine_matrices(shape=self._shape)
        # Y then X (for scipy.ndimage)
        grids = np.meshgrid(*(np.arange(length) for length in self._shape), indexing='ij')
        offset_array = np.array(mapping.offset).reshape(len(grids), 1)
        affine_coords = (np.dot(mapping.matrix, np.array([grid.flatten() for grid in grids]))
                         + offset_array).astype(np.float32, copy=False).reshape(len(grids), *self._shape)
        offsets = np.sum(np.square(self.coords - affine_coords), axis=0)
        return np.sqrt(np.mean(offsets)), np.sqrt(np.max(offsets))

#----------------------------------------------------------------------------------

class DataGroup:
    """
    A DataGroup is a collection of an equal number array-like objects and
    Transforms, with the intention of transforming the arrays into a single
    output array.
    """
    UnequalError = ValueError("Number of arrays and transforms must be equal")

    def __init__(self, arrays=None, transforms=None, loglevel="stdinfo"):
        if transforms is None:
            self._arrays = arrays or []
            self._transforms = [Transform()] * len(self._arrays)
        else:
            try:
                if len(arrays) != len(transforms):
                    raise self.UnequalError
            except TypeError:
                raise self.UnequalError
            # "Freeze" the transforms
            self._transforms = deepcopy(transforms)
            self._arrays = arrays
        self.no_data = {}
        self.output_shape = None
        self.origin = None
        self.log = logutils.get_logger(__name__)
        self.logit = getattr(self.log, loglevel)

    def __getitem__(self, index):
        return (self._arrays[index], self._transforms[index])

    def __iter__(self):
        # Deliberately raise error if list lengths aren't equal
        for i in range(len(self)):
            yield (self._arrays[i], self._transforms[i])

    def __len__(self):
        """Return number of array/transform pairs and checks they're equal"""
        len_ = len(self._arrays)
        if len(self._transforms) == len_:
            return len_
        raise self.UnequalError

    @property
    def arrays(self):
        return self._arrays

    @property
    def transforms(self):
        return self._transforms

    def append(self, array, transform):
        """Add a single array/transform pair"""
        self._arrays.append(array)
        self._transforms.append(deepcopy(transform))

    def calculate_output_shape(self, additional_array_shapes=None,
                               additional_transforms=None):
        """
        This method sets the output_shape and origin attributes of the
        DataGroup. output_shape is the shape that fully encompasses all of
        the transformed arrays, while origin is the location in this region
        where the origin (0,0) lies (since the transformed coordinates may
        be negative).

        Additional arrays and transforms can be provided to assist in the
        case where several DataGroup outputs will be stacked.

        Parameters
        ----------
        additional_array_shapes: list of tuples
            shapes of any other arrays that will be transformed later, to
            ensure a uniform pixel output grid
        additional_transforms: list
            additional transforms, one for each of additional_array_shapes
        """
        array_shapes = [arr.shape for arr in self.arrays]
        transforms = self.transforms
        if additional_array_shapes:
            array_shapes += additional_array_shapes
            transforms += additional_transforms
        if len(array_shapes) != len(transforms):
            raise self.UnequalError
        all_corners = []
        # The [::-1] are because we need to convert from python order to
        # astropy Model order (x-first) and then back again
        for array_shape, transform in zip(array_shapes, transforms):
            corners = np.array(at.get_corners(array_shape)).T[::-1]
            trans_corners = transform(*corners)
            if len(array_shape) == 1:
                trans_corners = (trans_corners,)
            all_corners.extend(corner[::-1] for corner in zip(*trans_corners))
        limits = [(int(np.ceil(min(c))), int(np.floor(max(c))) + 1)
                  for c in zip(*all_corners)]
        self.output_shape = tuple(max_ - min_ for min_, max_ in limits)
        self.origin = tuple(min_ for min_, max_ in limits)

    def shift_origin(self, *args):
        """
        This method shifts the origin of the output frame. Positive values
        will result in the bottom and/or left of the mosaic being eliminated
        from the output image.

        Parameters
        ----------
        args: numbers
            shifts to apply to the origin (standard python order)
        """
        if self.origin is None:
            raise ValueError("Origin has not been set")
        if len(args) != len(self.origin):
            raise ValueError("Number of shifts has wrong dimensionality")
        self.origin = tuple(orig + shift for orig, shift in zip(self.origin, args))

    def reset_origin(self):
        """
        This method appends Shifts to the transforms so that the origin
        becomes (0,0). The origin is then set to None.
        """
        if self.origin is None:
            raise ValueError("Origin has not been set")
        for t in self.transforms:
            t.append(reduce(Model.__and__,
                            [models.Shift(-offset) for offset in self.origin[::-1]]))
        self.origin = None

    def transform(self, attributes=['data'], interpolant="linear", subsample=1,
                  threshold=0.01, conserve=False, parallel=False):
        """
        This method transforms and combines the arrays into a single output
        array. The inputs, after transforming, shouldn't interfere with each
        other (i.e., no output pixel should have signal from more than one
        input object). Bit-masks (identified as being arrays of unsigned
        integer type) are transformed bit-by-bit. If an attribute's name is a
        key in the no_data dict, then that value is used to represent empty
        regions in the output array of this attribute.

        Parameters
        ----------
        attributes: sequence
            attributes of the input objects to transform
        interpolant: str
            type of interpolant
        subsample: int
            subsampling of input/output arrays
        threshold: float
            for bitmask arrays, the fraction that needs to be exceeded
            for the bit to be set in the output
        conserve: bool
            conserve flux by applying Jacobian?
        parallel: bool
            perform operations in parallel using multiprocessing?

        Returns
        -------
        dict: {key: array} of arrays containing the transformed attributes
        """
        self.output_dict = {}
        if parallel:
            processes = []
            process_keys = []
            manager = multi.Manager()
            self.output_arrays = manager.dict()
        else:
            self.output_arrays = {}

        if self.output_shape is None:
            self.calculate_output_shape()

        self.corners = []
        self.jfactors = []

        for input_array, transform in zip(self._arrays, self._transforms):
            # Since this may be modified, deepcopy to preserve the one if
            # the DataGroup's _transforms list
            transform = deepcopy(transform)
            if self.origin is not None and any(x != 0 for x in self.origin):
                transform.append(reduce(Model.__and__,
                                 [models.Shift(-offset) for offset in self.origin[::-1]]))
            output_corners = self._prepare_for_output(input_array, transform)
            output_array_shape = tuple(max_ - min_ for min_, max_ in output_corners)

            # This can happen if the origin and/or output_shape are modified
            if not all(length > 0 for length in output_array_shape):
                self.log.stdinfo("Array falls outside output region")
                continue

            # Create a mapping from output pixel to input pixels
            mapping = transform.inverse.affine_matrices(shape=output_array_shape)
            jfactor = abs(np.linalg.det(mapping.matrix)) if conserve else 1.0
            self.jfactors.append(jfactor)

            integer_shift = (transform.is_affine
                and np.array_equal(mapping.matrix, np.eye(mapping.matrix.ndim)) and
                                   np.array_equal(mapping.offset,
                                                  mapping.offset.astype(int)))

            if not integer_shift:
                ndim = transform.ndim
                # Apply scale and shift for subsampling. Recall that (0,0) is the middle of
                # the pixel, not the corner, so a shift is required as well.
                if subsample > 1:
                    rescale = reduce(Model.__and__, [models.Scale(subsample)] * ndim)
                    rescale_shift = reduce(Model.__and__, [models.Shift(0.5 * (subsample - 1))] * ndim)
                    transform.append([rescale, rescale_shift])

                trans_output_shape = tuple(length * subsample for length in output_array_shape)
                if transform.inverse.is_affine:
                    mapping = transform.inverse.affine_matrices(shape=trans_output_shape)
                else:
                    # If we're conserving the flux, we need to compute the
                    # Jacobian at every input point. This is done by numerical
                    # derivatives so expand the output pixel grid.
                    if conserve:
                        jacobian_shape = tuple(length + 2 for length in trans_output_shape)
                        transform.append(reduce(Model.__and__, [models.Shift(1)] * ndim))

                        # These are the coordinates in the input frame corresponding to each
                        # (subsampled) output pixel in a frame with an additional 1-pixel boundary
                        jacobian_mapping = GeoMap(transform, jacobian_shape)
                        det_matrices = np.empty((ndim, ndim, np.multiply.reduce(trans_output_shape)))
                        for num_axis in range(ndim):
                            coords = jacobian_mapping.coords[num_axis]
                            for denom_axis in range(ndim):
                                # We're numerically estimating the partial
                                # derivatives 2*dx/dx', 2*dx/dy', etc., multiplied
                                # by 1/subsample
                                diff_coords = coords - np.roll(coords, 2, axis=denom_axis)
                                slice_ = [slice(1, -1)] * ndim
                                slice_[denom_axis] = slice(2, None)
                                # Account for the fact that we are measuring
                                # differences in the subsampled plane
                                det_matrices[num_axis, denom_axis] = (0.5 * subsample *
                                    diff_coords[tuple(slice_)].flatten())
                        jfactor = abs(np.linalg.det(np.moveaxis(det_matrices, -1, 0))).reshape(trans_output_shape)
                        # Delete the extra Shift(1) and put a better jfactor in the list
                        del transform[-1]
                        self.jfactors[-1] = np.mean(jfactor)
                    mapping = GeoMap(transform, trans_output_shape)

            for attr in attributes:
                if isinstance(input_array, np.ndarray) and attr == "data":
                    arr = input_array
                else:  # let this raise an AttributeError
                    arr = getattr(input_array, attr)

                # Create an output array if we haven't seen this attribute yet.
                # We only do this now so that we know the dtype.
                cval = self.no_data.get(attr, 0)
                if attr not in self.output_dict:
                    self.output_dict[attr] = np.full(self.output_shape, cval, dtype=arr.dtype)

                # Integer shifts mean the output will be unchanged by the
                # transform, so we can put it straight in the output, since
                # only this array will map into the region.
                #
                # The origin and output_shape may have been set in order to
                # only transform some of the input image into the final output
                # array, so we need to account for that (with slice_2)
                if integer_shift:
                    self.log.debug("Placing {} array in [".format(attr) +
                                   ",".join(["{}:{}".format(limits[0] + 1, limits[1])
                                             for limits in output_corners[::-1]]) + "]")
                    slice_ = tuple(slice(min_, max_) for min_, max_ in output_corners)
                    slice_2 = tuple(slice(int(offset), int(offset) + max_ - min_)
                                    for offset, (min_, max_) in zip(mapping.offset, output_corners))
                    self.output_dict[attr][slice_] = arr[slice_2]
                    continue

                # Set up the functions to call to transform this attribute
                jobs = []
                if np.issubdtype(arr.dtype, np.unsignedinteger):
                    for j in range(0, 16):
                        bit = 2 ** j
                        if bit == cval or np.sum(arr & bit) > 0:
                            key = ((attr,bit), output_corners)
                            # Convert bit plane to float32 so the output
                            # will be float32 and we can threshold
                            jobs.append((key, (arr & bit).astype(np.float32),
                                         {'cval': bit & cval,
                                          'threshold': threshold * bit}))
                else:
                    key = (attr, output_corners)
                    jobs.append((key, arr, {}))

                # Perform the jobs (in parallel, if we can)
                for (key, arr, kwargs) in jobs:
                    args = (arr, mapping, key, output_array_shape)
                    kwargs.update({'dtype': self.output_dict[attr].dtype,
                                   'interpolant': interpolant,
                                   'subsample': subsample,
                                   'jfactor': jfactor})
                    if parallel:
                        p = multi.Process(target=self._apply_geometric_transform,
                                          args=args, kwargs=kwargs)
                        processes.append(p)
                        process_keys.append(key)
                        p.start()
                    else:
                        self._apply_geometric_transform(*args, **kwargs)
                        self._add_to_output(key)

        # If we're in parallel, we need to place the outputs into the final
        # arrays as they finish. This should avoid hogging memory. Note that if
        # we've applied integer shifts, the processes list will be empty so this
        # will pass through quickly.
        if parallel:
            finished = False
            while not finished:
                finished = True
                for p, key in zip(processes, process_keys):
                    if not p.is_alive():
                        if key in self.output_arrays:
                            self._add_to_output(key)
                    else:
                        finished = False

        del self.output_arrays
        return self.output_dict

    def _prepare_for_output(self, input_array, transform):
        """
        Determine the shape of the output array that this input will be
        transformed into, accounting for the overall output array. If
        necessarily, modify the transform to shift it to the "bottom-left"
        so we don't have blank space.

        Parameters
        ----------
        input_array: array-like
            the array being transformed
        output_shape: tuple
            the shape of the overall output array
        transform: Transform/Model callable
            the forward transformation

        Returns
        -------
        sequence of doubletons: limits of the output region this transforms into
        """
        # Find extent of this array in the output, after transformation
        # Invert from standard python order to (x, y[, z]) order
        corners = np.array(at.get_corners(input_array.shape)).T[::-1]
        trans_corners = transform(*corners)
        if len(input_array.shape) == 1:
            trans_corners = (trans_corners,)
        self.corners.append(trans_corners[::-1])  # standard python order
        min_coords = [int(np.ceil(min(coords))) for coords in trans_corners]
        max_coords = [int(np.floor(max(coords)))+1 for coords in trans_corners]
        self.logit("Array maps to ["+",".join(
            [f"{min_+1}:{max_}" for min_, max_ in zip(min_coords, max_coords)])+"]")
        # If this maps to a region not starting in the bottom-left of the
        # output, shift the whole thing so we can efficiently transform it
        # into an array. Coords are still in reverse python order.
        new_min_coords = [max(c, 0) for c in min_coords]
        new_max_coords = [min(c, s) for c, s in zip(max_coords, self.output_shape[::-1])]
        shift = reduce(Model.__and__, [models.Shift(-c) for c in new_min_coords])
        transform.append(shift.rename("Region offset"))

        output_corners = tuple((min_, max_) for min_, max_ in
                               zip(new_min_coords, new_max_coords))[::-1]
        return output_corners  # in standard python order

    def _add_to_output(self, key):
        """
        Adds output_arrays[key] to the final image and deletes it from the
        output_arrays dict

        Parameters
        ----------
        key: tuple
            identifier for this particular array
        """
        attr, output_corners = key
        # 0-indexed but (x, y) order
        self.log.debug("Placing {} array in [".format(attr)+
                       ",".join(["{}:{}".format(limits[0]+1 , limits[1])
                                 for limits in output_corners[::-1]])+"]")
        arr = self.output_arrays[key]
        slice_ = tuple(slice(min_, max_) for min_, max_ in output_corners)
        if isinstance(attr, tuple):  # then attr will be, e.g., ('mask', 2)
            dtype = self.output_dict[attr[0]].dtype
            output_region = self.output_dict[attr[0]][slice_]
            # This is ugly! The DQ.no_data bit should only be set if all arrays
            # have the bit set (and-like), but the other bits combine or-like
            cval = self.no_data.get(attr[0], 0)
            if attr[1] != cval:
                output_region |= (arr * attr[1]).astype(dtype, copy=False)
            else:
                self.output_dict[attr[0]][slice_] = ((output_region & (65535 ^ cval)) |
                                                (output_region & (arr * attr[1]))).astype(dtype, copy=False)
        else:
            self.output_dict[attr][slice_] += arr
        del self.output_arrays[key]

    def _apply_geometric_transform(self, input_array, mapping, output_key,
                                   output_shape, cval=0., dtype=np.float32,
                                   threshold=None, subsample=1,
                                   interpolant="linear", jfactor=1):
        """
        None-returning function to apply geometric transform, so it can be used
        by multiprocessing

        Parameters
        ----------
        input_array: ndarray
            array to be transformed
        mapping: callable
            provides transformation from output -> input coordinates
        output_key;
            key in dict/DictProxy to use when storing this output array
        output_shape: tuple
            shape of this output array
        cval: number
            value for "empty" pixels in output array
        dtype: datatype
            datatype of output array
        threshold: None/float
            limit for deciding what minimum value makes a pixel "bad"
            (if set, a bool array is returned, irrespective of dtype)
        subsample: int
            subsampling in output array for transformation
        interpolant: str
            type of interpolant
        jfactor: float/array
            Jacobian of transformation (basically the increase in pixel area)
        """
        trans_output_shape = tuple(length * subsample for length in output_shape)

        # Any NaN or Inf values in the array prior to resampling can cause
        # havoc. There shouldn't be any, but let's check and spit out a
        # warning if there are, so we at least produce a sensible output array.
        isnan = np.isnan(input_array)
        isinf = np.isinf(input_array)
        if isnan.any() or isinf.any():
            log.warning(f"There are {isnan.sum()} NaN and {isinf.sum()} inf "
                        f"values in the {output_key[0]} array. Setting to zero.")
            input_array[isnan | isinf] = 0

        # We want to transform any DQ bit arrays into floats so we can sample
        # the "ringing" from the interpolation and flag appropriately
        out_dtype = np.float32 if np.issubdtype(
            input_array.dtype, np.unsignedinteger) else input_array.dtype
        out_array = Transform.transform_data(
            input_array, mapping, trans_output_shape, interpolant=interpolant,
            cval=cval)

        # We average to undo the subsampling. This retains the "threshold" and
        # conserves flux according to the Jacobian of input/output arrays.
        # Doing it here saves memory because we only store the 1-sampled arrays
        if subsample > 1:
            intermediate_shape = tuple(x for length in output_shape
                                       for x in (length, subsample))
            out_array = (jfactor * out_array).reshape(intermediate_shape).\
                mean(tuple(range(len(output_shape)*2-1, 0, -2))).\
                reshape(output_shape)
            jfactor = 1  # Since we've done it

        if threshold is None:
            self.output_arrays[output_key] = (out_array * jfactor).astype(dtype, copy=False)
        else:
            self.output_arrays[output_key] = np.where(abs(out_array) > threshold,
                                                      True, False)

#-----------------------------------------------------------------------------
def find_reference_extension(ad):
    """
    This function determines the reference extension of an AstroData object,
    i.e., the extension which is most central, with preference to being to
    the left or down of the centre.

    Parameters
    ----------
    ad: input AstroData object

    Returns
    -------
    int: index of the reference extension
    """
    if len(ad) == 1:
        return 0
    det_corners = np.array([(sec.y1, sec.x1) for sec in ad.detector_section()])
    centre = np.median(det_corners, axis=0)
    distances = list(det_corners - centre)
    ref_index = np.argmax([d.sum() if np.all(d <= 0) else -np.inf for d in distances])
    return ref_index


def add_mosaic_wcs(ad, geotable):
    """
    We assume that the reference extension only experiences a translation,
    not a rotation (otherwise why is it the reference?).

    IMPORTANT: the gWCS object this creates must result in the "mosaic"
    frame coordinates being the same for the same native pixels, regardless
    of the ROI used.

    Parameters
    ----------
    ad: AstroData
        the astrodata instance

    Returns
    -------
    AstroData: the modified input AD, with WCS attributes
    """
    if any('mosaic' in getattr(ext.wcs, 'available_frames', '') for ext in ad):
        raise ValueError("A 'mosaic' frame is already present in one or "
                         f"more extensions of {ad.filename}")

    array_info = gt.array_information(ad)
    offsets = [ad[exts[0]].array_section()
               for exts in array_info.extensions]

    detname = ad.detector_name()
    xbin, ybin = ad.detector_x_bin(), ad.detector_y_bin()
    geometry = geotable.geometry[detname]
    default_shape = geometry.get('default_shape')

    ref_index = find_reference_extension(ad)
    ref_wcs = ad[ref_index].wcs

    for indices, origin, offset in zip(array_info.extensions, array_info.origins, offsets):
        # Origins are in (x, y) order in LUT
        block_geom = geometry[origin[::-1]]
        nx, ny = block_geom.get('shape', default_shape)
        nx /= xbin
        ny /= ybin
        shift = block_geom.get('shift', (0, 0))
        rot = block_geom.get('rotation', 0.)
        mag = block_geom.get('magnification', (1, 1))

        model_list = []

        # This is now being done by the ext_shift (below)
        # Shift the Block's coordinates based on its location within
        # the full array, to ensure any rotation takes place around
        # the true centre.
        # if offset.x1 != 0 or offset.y1 != 0:
        #    model_list.append(models.Shift(offset.x1 / xbin) &
        #                      models.Shift(offset.y1 / ybin))

        if rot != 0 or mag != (1, 1):
            # Shift to centre, do whatever, and then shift back
            model_list.append(models.Shift(-0.5 * (nx - 1)) &
                              models.Shift(-0.5 * (ny - 1)))
            if rot != 0:
                # Cope with non-square pixels by scaling in one
                # direction to make them square before applying the
                # rotation, and then reversing that.
                if xbin != ybin:
                    model_list.append(models.Identity(1) & models.Scale(ybin / xbin))
                model_list.append(models.Rotation2D(rot))
                if xbin != ybin:
                    model_list.append(models.Identity(1) & models.Scale(xbin / ybin))
            if mag != (1, 1):
                model_list.append(models.Scale(mag[0]) &
                                  models.Scale(mag[1]))
            model_list.append(models.Shift(0.5 * (nx - 1)) &
                              models.Shift(0.5 * (ny - 1)))
        model_list.append(models.Shift(shift[0] / xbin) &
                          models.Shift(shift[1] / ybin))
        mosaic_model = reduce(Model.__or__, model_list)

        in_frame = cf.Frame2D(name="pixels")
        tiled_frame = cf.Frame2D(name="tile")
        mos_frame = cf.Frame2D(name="mosaic")
        for i in indices:
            arrsec = ad[i].array_section()
            datsec = ad[i].data_section()
            if len(indices) > 1:
                ext_shift = (models.Shift((arrsec.x1 // xbin - datsec.x1)) &
                             models.Shift((arrsec.y1 // ybin - datsec.y1)))
                pipeline = [(in_frame, ext_shift),
                            (tiled_frame, mosaic_model),
                            (mos_frame, None)]
            else:
                pipeline = [(in_frame, mosaic_model),
                            (mos_frame, None)]
            ad[i].wcs = gWCS(pipeline)

    # We want to put the origin shift before the mosaic frame of each
    # extension in order to keep the post-mosaic model tidy. This requires
    # more code here.
    if ref_wcs is not None:
        new_xorigin, new_yorigin = ad[ref_index].wcs(0, 0)
        # len(ad) > 1 added for applyWCSAdjustment, to preserve the origin
        # as CCD2, and not the origin of this single slice
        if (new_xorigin or new_yorigin) and len(ad) > 1:
            origin_shift = models.Shift(-new_xorigin) & models.Shift(-new_yorigin)
        else:
            origin_shift = None

        for ext in ad:
            ext.wcs.insert_frame(ext.wcs.output_frame, ref_wcs.forward_transform,
                                 ref_wcs.output_frame)
            if origin_shift:
                ext.wcs.insert_transform(mos_frame, origin_shift, after=False)

    return ad


@insert_descriptor_values()
def add_longslit_wcs(ad, central_wavelength=None, pointing=None):
    """
    Attach a gWCS object to all extensions of an AstroData objects,
    representing the approximate spectroscopic WCS, as returned by
    the descriptors, instead of the standard Gemini imaging WCS.

    Parameters
    ----------
    ad : AstroData
        the AstroData instance requiring a WCS
    central_wavelength : float / None
        central wavelength in nm (None => use descriptor)
    pointing: 2-tuple / None
        RA and Dec of pointing center to reproject WCS

    Returns
    -------
    AstroData: the modified input AD, with WCS attributes on each NDAstroData
    """
    if 'SPECT' not in ad.tags:
        raise ValueError(f"Image {ad.filename} is not of type SPECT")

    # TODO: This appears to be true for GMOS. Revisit for other multi-extension
    # spectrographs once they arrive and GMOS tests are written
    if pointing is None:
        crval1 = set(ad.hdr['CRVAL1'])
        crval2 = set(ad.hdr['CRVAL2'])
        if len(crval1) * len(crval2) != 1:
            raise ValueError(f"Not all CRVAL1/CRVAL2 keywords are the same in {ad.filename}")
        pointing = (crval1.pop(), crval2.pop())

    if ad.dispersion() is None:
        raise ValueError(f"Unknown dispersion for {ad.filename}")

    for ext, dispaxis, dw in zip(ad, ad.dispersion_axis(), ad.dispersion(asNanometers=True)):
        if not isinstance(ext.wcs.output_frame, cf.CelestialFrame):
            raise TypeError(f"Output frame of {ad.filename} extension {ext.id}"
                            " is not a CelestialFrame instance")

        # Need to change axes_order in CelestialFrame
        kwargs = {kw: getattr(ext.wcs.output_frame, kw, None)
                  for kw in ('reference_frame', 'unit', 'axes_names',
                             'axis_physical_types')}
        sky_frame = cf.CelestialFrame(axes_order=(1,2), name='sky', **kwargs)
        spectral_frame = cf.SpectralFrame(name='Wavelength in air', unit=u.nm,
                                          axes_names='AWAV')
        output_frame = cf.CompositeFrame([spectral_frame, sky_frame], name='world')

        transform = adwcs.create_new_image_projection(ext.wcs.forward_transform, pointing)
        crpix = transform.inverse(*pointing)
        # Add back the 'crpixN' names, which would otherwise be lost at this
        # point (they're used in the various test_adjust_wcs_with_correlation
        # tests).
        names = ['crpix1', 'crpix2']

        transform.name = None  # so we can reuse "SKY"
        # Pop one of the crpix names (based on the dispersion axis) for the sky
        # model...
        if dispaxis == 1:
            sky_model = (models.Mapping((0, 0)) |
                         (models.Const1D(0) &
                          models.Shift(-crpix[1], name=names.pop(1))))
        else:
            sky_model = (models.Mapping((0, 0)) |
                         (models.Shift(-crpix[0], name=names.pop(0)) &
                          models.Const1D(0)))
        sky_model |= transform[2:]
        sky_model[-1].lon, sky_model[-1].lat = pointing
        sky_model.name = 'SKY'
        # ...then we can use the remaining crpix name for the wave model.
        wave_model = (models.Shift(-crpix[dispaxis-1], name=names[0]) |
                      models.Scale(dw) | models.Shift(central_wavelength))
        wave_model.name = 'WAVE'

        if dispaxis == 1:
            sky_model.inverse = (transform.inverse | models.Mapping((1,)))
            transform = wave_model & sky_model
        else:
            sky_model.inverse = (transform.inverse |
                                 models.Mapping((0,), n_inputs=2))
            transform = models.Mapping((1, 0)) | wave_model & sky_model

        new_wcs = gWCS([(ext.wcs.input_frame, transform),
                        (output_frame, None)])
        ext.wcs = new_wcs

    return ad


def resample_from_wcs(ad, frame_name, attributes=None, interpolant="linear",
                      subsample=1, threshold=0.001, conserve=False, parallel=False,
                      process_objcat=False, output_shape=None, origin=None):
    """
    This takes a single AstroData object with WCS objects attached to the
    extensions and applies some part of the WCS to resample into a
    different pixel coordinate frame.

    This effectively replaces AstroDataGroup since now the transforms are
    part of the AstroData object.

    Parameters
    ----------
    ad : AstroData
        the input image that is going to be resampled/mosaicked
    frame_name : str
        name of the frame to resample to
    attributes : list/None
        list of attributes to resample (None => all standard ones that exist)
    interpolant : str
        type of interpolant
    subsample : int
        if >1, will transform onto finer pixel grid and block-average down
    threshold : float
        for transforming the DQ plane, output pixels larger than this value
        will be flagged as "bad"
    conserve : bool
        conserve flux rather than interpolate?
    parallel : bool
        use parallel processing to speed up operation?
    process_objcat : bool
        merge input OBJCATs into output AD instance?
    output_shape : None/iterable
        final shape of output
    origin : None/iterable
        location of origin

    Returns
    -------
    AstroData: single-extension AD with suitable WCS
    """
    array_attributes = ['data', 'mask', 'variance']
    for k, v in ad.nddata[0].meta['other'].items():
        if isinstance(v, np.ndarray) and v.shape == ad[0].data.shape:
            array_attributes.append(k)
    is_single = ad.is_single

    # It's not clear how much checking we should do here but at a minimum
    # we should probably confirm that each extension is purely data. It's
    # up to a primitive to catch this, call trim_to_data_section(), and try again
    if is_single:
        addatsec = (0, ad.shape[0]) if len(ad.shape) == 1 else ad.data_section()
        shapes_ok = np.array_equal(np.ravel([(0, length) for length in ad.shape[::-1]]),
                                   list(addatsec))
    else:
        shapes_ok = all(np.array_equal(np.ravel([(0, length) for length in shape[::-1]]),
                                   list(datsec)) for datsec, shape in zip(ad.data_section(), ad.shape))
    if not shapes_ok:
        raise ValueError("Not all data sections agree with extension shapes")

    # Create the blocks (individual physical detectors)
    if is_single:
        blocks = [Block(ad)]
    elif len(ad) == 1:
        blocks = [Block(ad[0])]
    else:
        array_info = gt.array_information(ad)
        blocks = [Block(ad[arrays], shape=shape) for arrays, shape in
                  zip(array_info.extensions, array_info.array_shapes)]

    dg = DataGroup()
    dg.no_data['mask'] = DQ.no_data
    dg.origin = origin
    dg.output_shape = output_shape

    for block in blocks:
        wcs = block.wcs
        # Do some checks that, e.g., this is a pixel->pixel mapping
        try:  # Create more informative exceptions
            frame_index = wcs.available_frames.index(frame_name)
        except AttributeError:
            raise TypeError("WCS attribute is not a WCS object on {}"
                            "".format(ad.filename))
        except ValueError:
            raise ValueError("Frame {} is not in WCS for {}"
                             "".format(frame_name, ad.filename))

        frame = getattr(wcs, frame_name)
        if not all(au == u.pix for au in frame.unit):
            raise ValueError("Requested output frame is not a pixel frame")

        transform = Transform(wcs.get_transform(wcs.input_frame, frame))
        dg.append(block, transform)

    if attributes is None:
        attributes = [attr for attr in array_attributes
                      if all(getattr(ext, attr, None) is not None for ext in ad)]
    if 'data' not in attributes:
        log.warning("The 'data' attribute is not specified. Adding to list.")
        attributes += ['data']
    log.fullinfo("Processing the following array attributes: "
                 "{}".format(', '.join(attributes)))
    dg.transform(attributes=attributes, interpolant=interpolant, subsample=subsample,
                 threshold=threshold, conserve=conserve, parallel=parallel)

    ad_out = astrodata.create(ad.phu)
    ad_out.orig_filename = ad.orig_filename

    ref_index = find_reference_extension(ad)
    ref_ext = ad if is_single else ad[ref_index]

    ad_out.append(dg.output_dict['data'], header=ref_ext.hdr.copy())
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        for key, value in dg.output_dict.items():
            if key != 'data':  # already done this
                setattr(ad_out[0], key, value)

    # Store this information so the calling primitive can access it
    ad_out[0].nddata.meta['transform'] = {'origin': dg.origin,
                                          'corners': dg.corners,
                                          'jfactors': dg.jfactors,
                                          'block_corners': [b.corners for b in dg.arrays]}

    # Create a new gWCS object describing the remaining transformation.
    # Not all gWCS objects have to have the same steps, so we need to
    # redetermine the frame_index in the reference extensions's WCS.
    ref_wcs = ref_ext.wcs
    frame_index = ref_wcs.available_frames.index(frame_name)
    new_pipeline = deepcopy(ref_wcs.pipeline[frame_index:])
    new_pipeline[0].frame.name = ref_wcs.input_frame.name
    # Remember, dg.origin is (y, x)
    new_origin = tuple(s for s in dg.origin[::-1])

    origin_model = None
    if len(new_pipeline) == 1:
        new_wcs = None
    else:
        new_wcs = gWCS(new_pipeline)
        if set(new_origin) != {0}:
            origin_model = reduce(Model.__and__, [models.Shift(s) for s in new_origin])
            # For if we tile the OBJCATs
            for transform in dg.transforms:
                transform.append(origin_model.inverse)
            new_wcs.insert_transform(new_wcs.input_frame, origin_model,
                                     after=True)
    ad_out[0].wcs = new_wcs

    # Update and delete keywords from extension (_update_headers)
    ndim = len(ref_ext.shape)
    header = ad_out[0].hdr
    keywords = {sec: ad._keyword_for('{}_section'.format(sec))
                for sec in ('array', 'data', 'detector')}

    # Data section probably has meaning even if ndim!=2
    ad_out.hdr[keywords['data']] = \
        '['+','.join('1:{}'.format(length)
                     for length in ad_out[0].shape[::-1])+']'

    # These descriptor returns are unclear in non-2D data
    if ndim == 2:

        # If detector_section returned something, set an appropriate value
        all_detsec = np.array([ext.detector_section() for ext in ad]).T
        ad_out.hdr[keywords['detector']] = \
            '['+','.join('{}:{}'.format(min(c1)+1, max(c2))
                         for c1, c2 in zip(all_detsec[::2], all_detsec[1::2]))+']'

        # array_section only has meaning now if the inputs were from a
        # single physical array
        if len(blocks) == 1:
            all_arrsec = np.array([ext.array_section() for ext in ad]).T
            ad_out.hdr[keywords['array']] = \
                '[' + ','.join('{}:{}'.format(min(c1) + 1, max(c2))
                               for c1, c2 in zip(all_arrsec[::2], all_arrsec[1::2])) + ']'
        else:
            del ad_out.hdr[keywords['array']]

    # Try to assign an array name for this based on commonality
    if not is_single:
        kw = ad._keyword_for('array_name')
        try:
            array_names = set([name.split(',')[0] for name in ad.array_name()])
        except AttributeError:
            array_name = ''
        else:
            array_name = array_names.pop()
            if array_names:
                array_name = ''
        if array_name:
            ad_out.hdr[kw] = array_name
        elif kw in header:
            del ad_out.hdr[kw]

    if 'CCDNAME' in ad_out[0].hdr:
        ad_out.hdr['CCDNAME'] = ad.detector_name()

    # Finally, delete any keywords that no longer make sense
    for kw in ('FRMNAME', 'FRAMEID', 'CCDSIZE', 'BIASSEC',
               'DATATYP', 'OVERSEC', 'TRIMSEC', 'OVERSCAN', 'OVERRMS'):
        if kw in header:
            del ad_out.hdr[kw]

    # Now let's worry about the tables. Only transfer top-level ones, since
    # we may be combining extensions so it's not clear generally how to merge.
    # If the calling code needs to do some propagation it has to handle that
    # itself. We have to use the private attribute here since the public one
    # doesn't distinguish between top-level and extension-level tables if the
    # AD object is a single slice.
    for table_name in ad._tables:
        setattr(ad_out, table_name, getattr(ad, table_name).copy())
        log.fullinfo("Copying {}".format(table_name))

    # And now the OBJCATs. This code is partially set-up for other types of
    # tables, but it's likely each will need some specific tweaks.
    if 'IMAGE' in ad.tags and process_objcat:
        for table_name, coord_columns in catalog_coordinate_columns.items():
            tables = []
            for block, transform in zip(blocks, dg.transforms):
                # This returns a single Table
                cat_table = getattr(block, table_name, None)
                if cat_table is None:
                    continue
                if 'header' in cat_table.meta:
                    del cat_table.meta['header']
                for ycolumn, xcolumn in zip(*coord_columns):
                    # OBJCAT coordinates are 1-indexed
                    newx, newy = transform(cat_table[xcolumn]-1, cat_table[ycolumn]-1)
                    cat_table[xcolumn] = newx + 1
                    cat_table[ycolumn] = newy + 1
                if new_wcs is not None:
                    ra, dec = new_wcs(cat_table['X_IMAGE'].value-1,
                                      cat_table['Y_IMAGE'].value-1)
                cat_table["X_WORLD"][:] = ra
                cat_table["Y_WORLD"][:] = dec
                tables.append(cat_table)

            if tables:
                log.stdinfo("Processing {}s".format(table_name))
                objcat = table.vstack(tables, metadata_conflicts='silent')
                objcat['NUMBER'] = np.arange(len(objcat)) + 1
                setattr(ad_out[0], table_name, objcat)

    # We may need to remake the gWCS object. The issue here is with 2D spectra,
    # where the resetting of the dispersion direction is done before the
    # complete model and so isn't within the "WAVE" submodel. This is done by
    # splitting up origin shift model and putting the shifts before each of the
    # submodels corresponding to each input axis.
    if origin_model is not None and ad_out[0].wcs is not None:
        try:
            m_wave = am.get_named_submodel(ad_out[0].wcs.forward_transform, "WAVE")
        except IndexError:
            pass
        else:
            ad_out[0].wcs.pipeline[0].transform = add_shifts_to_submodel(
                ad_out[0].wcs.pipeline[0].transform.right, new_origin)
    return ad_out


def add_shifts_to_submodel(m, shifts):
    """
    This function applies specified shifts to each input of a Model but,
    instead of inserting them as a combination of Shifts at the start, it
    adds each shift to the start of the appropriate submodel. So if you have
    a model m(X) & m(Y) and want to add shifts dX and dY it will produce
                (Shift(dX) | m(X)) & (Shift(dY) & m(Y))
    instead of
                (Shift(dX) & Shift(dY)) | (m(X) & m(Y))

    Similarly, if you have m(X) & m(Y,Z) it will give you
           (Shift(dX) | m(X)) & ((Shift(dY) & Shift(dZ)) | m(Y,Z))

    If any of the "m" model instances had a name, this name is now given to
    the compound model that includes the shift.

    Parameters
    ----------
    m : Model instance
        model that needs to be modified
    shifts : iterable
        the shifts that need to be applied to each input of the Model

    Returns
    -------
    new Model instance
    """
    if m.n_inputs != len(shifts):
        raise ValueError(f"Mismatched {m.n_inputs} inputs and {len(shifts)} "
                         "shifts")
    if hasattr(m, "op") and m.op == "&":
        return (add_shifts_to_submodel(m.left, shifts[:m.left.n_inputs]) &
                add_shifts_to_submodel(m.right, shifts[m.left.n_inputs:]))
    else:
        new_model_name = m.name
        m.name = None
        new_model = reduce(Model.__and__, [models.Shift(s) for s in shifts]) | m
        new_model.name = new_model_name
        return new_model


def get_output_corners(transform, input_shape=None, origin=None):
    """
    Determine the locations in output coordinate space of the transformed
    corners of an input array. These coordinates are returned in the
    normal python order, in line with astrotools.get_corners() and DataGroup.

    Parameters
    ----------
    transform: Model
        the transform being applied to the input coordinates
    input_shape: tuple
        shape of the array to be transformed
    origin: tuple
        location in input coordinate space of the bottom-left corner

    Returns
    -------
    array: (ndim, nvertices) with the transformed corner locations
           IN THE PYTHONIC ORDER
    """
    if origin is not None:
        transform = reduce(Model.__and__, [models.Shift(coord) for coord
                                           in origin[::-1]]) | transform
    dg = DataGroup()
    dg.calculate_output_shape(additional_array_shapes=[input_shape],
                              additional_transforms=[transform])
    return (np.array(at.get_corners(dg.output_shape)) + dg.origin).T
