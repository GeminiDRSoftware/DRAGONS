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
    AstroDataGroup: a subclass of DataGroup for AstroData objects
"""
import numpy as np
import copy
from functools import reduce
from collections import namedtuple

from astropy.modeling import models, Model
from astropy.modeling.core import _model_oper
from astropy import table
from astropy.wcs import WCS
from scipy import ndimage

from gempy.library import astrotools as at

import multiprocessing as multi
from geminidr.gemini.lookups import DQ_definitions as DQ

import astrodata, gemini_instruments

from ..utils import logutils

from datetime import datetime

AffineMatrices = namedtuple("AffineMatrices", "matrix offset")

# Table attribute names that should be modified to represent the
# coordinates in the Block, not their individual arrays.
# NB. Standard python ordering!
catalog_coordinate_columns = {'OBJCAT': (['Y_IMAGE'], ['X_IMAGE'])}

class Block(object):
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
        if len(set(shapes)) > 1:
            raise ValueError("Not all elements have the same shape")
        self._arrayshape = shapes[0]

        if shape is None:
            shape = (len(elements),)
        shape = (1,) * (len(self._arrayshape)-len(shape)) + shape
        if np.multiply.reduce(shape) != len(elements):
            print(elements, shape)
            raise ValueError("Incompatible element list and shape")

        self._shape = shape
        corner_grid = np.mgrid[tuple(slice(0, s*a, a)
                                     for s, a in zip(shape, shapes[0]))].\
            reshape(len(shape), len(elements))
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
            attributes = [el if attr is None else attr
                          for el, attr in zip(self._elements, attributes)]

        if any(isinstance(attr, np.ndarray) for attr in attributes):
            try:
                return self._return_array(attributes)
            except AttributeError as e:
                if name == "data":
                    return self._return_array(self._elements)
                else:
                    raise AttributeError(e)
        # Handle Tables
        if any(isinstance(attr, table.Table) for attr in attributes):
            return self._return_table(name)

        # Otherwise return a list of the attributes, if they exist
        # or raise an AttributeError if they don't
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
        return len(self._shape)

    @property
    def shape(self):
        """Overall shape of Block"""
        return tuple(a*s for a, s in zip(self._arrayshape, self._shape))

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
            slice_ = tuple(slice(c, c+a) for c, a in zip(corner, self._arrayshape))
            output[slice_] = arr
        return output

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

class Transform(object):
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
    def __init__(self, model=None, ndim=None, copy=True):
        """Initialize with a single model, list, or dimensionality. We
        implement a copy option since we want to be able to set attributes
        of temporarily-sliced objects."""
        self._models = []
        self._affine = True
        if model:
            self.append(model, copy)
        elif ndim:
            self._ndim = ndim
        else:
            raise ValueError("A Model or dimensionality must be specified")

    def __call__(self, *args, **kwargs):
        if len(args) != self.ndim:
            raise ValueError("Incompatible number of inputs for Transform "
                             "(dimensionality {})".format(self.ndim))
        try:
            inverse = kwargs.pop('inverse')
        except KeyError:
            inverse = False
        return self.asModel(inverse=inverse)(*args, **kwargs)

    def __copy__(self):
        transform = self.__class__(self._models, ndim=self.ndim, copy=False)
        transform._affine = self._affine
        return transform

    def __deepcopy__(self, memo):
        transform = self.__class__(self._models, ndim=self.ndim, copy=True)
        transform._affine = self._affine
        return transform

    def __getattr__(self, key):
        """
        The is_affine property and asModel() method should be used to extract
        information, so attributes are intended to refer to the component
        models, and only make sense if the Transform has a single model.
        """
        if len(self) == 1:
            return getattr(self._models[0], key)
        raise AttributeError(key)

    def __setattr__(self, key, value):
        """
        It is assumed that we're setting attributes of the component models.
        We are allowed to set an attribute on multiple models (e.g., name),
        if that attribute exists on all models. Otherwise, it's set on the
        Transform object.
        """
        if key not in ("_models", "_affine"):
            if all(hasattr(m, key) for m in self._models):
                for m in self._models:
                    setattr(m, key, value)
                return
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
        for model in self._models:
            yield model

    def __len__(self):
        """Length (number of models)"""
        return len(self._models)

    @property
    def inverse(self):
        """The inverse transform"""
        # ndim is only used by __init__() if the model is empty
        return self.__class__([model.inverse for model in self._models[::-1]],
                              ndim=self.ndim)

    @property
    def ndim(self):
        """Dimensionality (number of inputs)"""
        return self._ndim

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
            return models.Identity(self.ndim)
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
            raise ValueError(e)

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

        # Ugly stuff to break down a CompoundModel.
        try:
            sequence = self.split_compound_model(model._tree, required_inputs)
        except AttributeError:
            self._models.insert(index, model.copy() if copy else model)
        else:
            i = self.index(index)
            # Avoids corrupting a Model (it will lose its name)
            if len(sequence) == 1:
                self._models.insert(index, model.copy() if copy else model)
            else:
                self._models[i:i] = sequence
        # Update affinity based on new model (leave existing stuff alone)
        self._affine &= self.__model_is_affine(model)

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
                if operand == "|" and stack[-1].n_inputs == ndim:
                    ndim = stack[1].n_outputs
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
            for new, old in zip(model, self.asModel()):
                if new.__class__ != old.__class__:
                    break
            for i, m in enumerate(sequence):
                self._models[i] = m.rename(self[i].name)
            return
        raise ValueError("Replacement Model differs from existing Transform")

    @staticmethod
    def __model_is_affine(model):
        """Test a single Model for affinity, using its name (or the name
        of its submodels)"""
        try:
            models = model._submodels
        except AttributeError:
            return model.__class__.__name__[:5] in ('Rotat', 'Scale',
                                                    'Shift', 'Ident')
        return np.logical_and.reduce([Transform.__model_is_affine(m)
                                      for m in models])

    def __is_affine(self):
        """Test for affinity, using Model names.
        TODO: Is this the right thing to do? We could compute the affine
        matrices *assuming* affinity, and then check that a number of random
        points behave as expected. Is that better?"""
        return np.logical_and.reduce([self.__model_is_affine(m)
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
            AffineMatrices(array, array): affine matrix and offset
        """
        ndim = self.ndim
        if shape is None:
            shape = (1000,) * ndim
        halfsize = [0.5*length for length in shape]
        points = np.array([halfsize] * (2*ndim+1)).T
        points[:,1:ndim+1] += np.eye(ndim) * points[:,0]
        points[:,ndim+1:] -= np.eye(ndim) * points[:,0]
        transformed = np.array(list(zip(*self.__call__(*points))))
        matrix = np.array([[0.5 * (transformed[j+1,i] - transformed[ndim+j+1,i]) / halfsize[j]
                            for j in range(ndim)] for i in range(ndim)])
        offset = transformed[0] - np.dot(matrix, halfsize)
        return AffineMatrices(matrix.T, offset[::-1])

    def info(self):
        """Print out something vaguely intelligible for debugging purposes"""
        new_line_indent = "\n    "
        msg = ['Model information: (dimensionality {})'.format(self.ndim)]
        for model in self._models:
            msg.append(repr(model))
        return new_line_indent.join(msg)

    def apply(self, input_array, output_shape, cval=0, inverse=False):
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
                matrix, offset = self.affine_matrices(input_array.shape)
            else:
                matrix, offset = self.inverse.affine_matrices(input_array.shape)
            output_array = ndimage.affine_transform(input_array, matrix, offset,
                                                    output_shape, cval=cval)
        else:
            mapping = GeoMap(self, output_shape, inverse=inverse)
            output_array = ndimage.geometric_transform(input_array, mapping,
                                                       output_shape, cval=cval)
        return output_array

#----------------------------------------------------------------------------------

class GeoMap(object):
    """
    Needed to create a callable object for the geometric_transform. Stores ndim
    mapping arrays (one for each axis) and provides a callable that returns
    the value of the array at the location of the tuple passed to that function.
    The coordinates are stored in x-first order.

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
        grids = np.meshgrid(*(np.arange(length) for length in shape[::-1]))
        self._transform = transform if inverse else transform.inverse
        self._shape = shape
        transformed = self._transform(*grids)
        transformed = [np.where(coord > length-1, -1, coord).astype(np.float32)
                       for coord, length in zip(transformed, shape[::-1])]
        self._map = transformed

    def __call__(self, coords):
        # Return in standard python order
        return tuple(map_[coords] for map_ in self._map[::-1])

    def affinity(self):
        """
        Calculate deviations from reality when using an affine transform

        Returns
        -------
        tuple: rms and maximum deviation in input-frame pixels
        """
        mapping = self._transform.affine_matrices(shape=self._shape)
        grids = np.meshgrid(*(np.arange(length) for length in self._shape[::-1]))[::-1]
        offset_array = np.array(mapping.offset).reshape(len(grids), 1)
        affine_coords = (np.dot(mapping.matrix, np.array([grid.flatten() for grid in grids]))
                         + offset_array).astype(np.float32).reshape(len(grids), *self._shape)
        for i, length in enumerate(self._shape):
            affine_coords[i][affine_coords[i] > length-1] = -1
        offsets = np.sum(np.square(np.array(self.__call__(grids)) - affine_coords), axis=0)
        return np.sqrt(np.mean(offsets)), np.sqrt(np.max(offsets))

#----------------------------------------------------------------------------------

class DataGroup(object):
    """
    A DataGroup is a collection of an equal number array-like objects and
    Transforms, with the intention of transforming the arrays into a single
    output array.
    """
    UnequalError = ValueError("Number of arrays and transforms must be equal")

    def __init__(self, arrays=[], transforms=[]):
        if len(arrays) != len(transforms):
            raise self.UnequalError
        self._arrays = arrays
        # "Freeze" the transforms
        if transforms:
            self._transforms = copy.deepcopy(transforms)
        else:
            self._transforms = [Transform(ndim=len(arr.shape)) for arr in arrays]
        self.no_data = {}
        self.output_dict = {}
        self.output_shape = None
        self.origin = None
        self.log = logutils.get_logger(__name__)

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
        self._transforms.append(copy.deepcopy(transform))

    def calculate_output_shape(self, additional_arrays=[],
                               additional_transforms=[]):
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
        additional_arrays: list
            additional arrays to supplement this DataGroup's list when
            calculating the output shape
        additional_transforms: list
            additional transforms, one for each of additional_arrays
        """
        arrays = list(self._arrays) + additional_arrays
        transforms = self._transforms + additional_transforms
        if len(arrays) != len(transforms):
            raise self.UnequalError
        all_corners = []
        # The [::-1] are because we need to convert from python order to
        # astropy Model order (x-first) and then back again
        for array, transform in zip(arrays, transforms):
            corners = np.array(at.get_corners(array.shape)).T[::-1]
            trans_corners = transform(*corners)
            all_corners.extend(corner[::-1] for corner in zip(*trans_corners))
        limits = [(int(np.ceil(min(c))), int(np.floor(max(c))) + 1)
                  for c in zip(*all_corners)]
        self.output_shape = tuple(max_ - min_ for min_, max_ in limits)
        self.origin = tuple(min_ for min_, max_ in limits)

    def transform(self, attributes=['data'],
                  order=3, subsample=1, threshold=0.01, parallel=True):
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
        order: int (0-5)
            order of spline interpolation
        subsample: int
            subsampling of input/output arrays
        threshold: float
            for bitmask arrays, the fraction that needs to be exceeded
            for the bit to be set in the output
        parallel: bool
            perform operations in parallel using multiprocessing?

        Returns
        -------
        dict: {key: array} of arrays containing the transformed attributes
        """
        start = datetime.now()
        jfactor = 1.0

        if parallel:
            processes = []
            process_keys = []
            manager = multi.Manager()
            self.output_arrays = manager.dict()
        else:
            self.output_arrays = {}

        if self.output_shape is None:
            self.calculate_output_shape()

        print(datetime.now()-start, "Completed setup")
        for input_array, transform in zip(self._arrays, self._transforms):
            # Since this may be modified, deepcopy to preserve the one if
            # the DataGroup's _transforms list
            transform = copy.deepcopy(transform)
            if self.origin:
                transform.append(reduce(Model.__and__,
                                 [models.Shift(-offset) for offset in self.origin[::-1]]))
            output_corners = self._prepare_for_output(input_array,
                                                      transform, subsample)
            output_array_shape = tuple(max_ - min_ for min_, max_ in output_corners)
            print(datetime.now() - start, "output_array_shape {}".format(output_array_shape))

            # Create a mapping from output pixel to input pixels
            integer_shift = False
            mapping = transform.inverse.affine_matrices(shape=output_array_shape)
            jfactor = np.linalg.det(mapping.matrix)
            print(datetime.now() - start, "jfactor {}".format(jfactor))
            if transform.is_affine:
                integer_shift = (np.array_equal(mapping.matrix, np.eye(mapping.matrix.ndim)) and
                                 np.array_equal(mapping.offset, mapping.offset.astype(int)))
                print(datetime.now() - start, "affine mapping done; integer shift {}".format(integer_shift))
            else:
                mapping = GeoMap(transform, output_array_shape)
                print(datetime.now() - start, "GeoMap done")

            for attr in attributes:
                try:
                    arr = getattr(input_array, attr)
                except AttributeError as e:
                    # If input_array is just an ndarray
                    if attr == "data":
                        arr = input_array
                    else:
                        raise AttributeError(e)

                # Create an output array if we haven't seen this attribute yet.
                # We only do this now so that we know the dtype.
                cval = self.no_data.get(attr, 0)
                if attr not in self.output_dict:
                    print("New attribute", attr)
                    self.output_dict[attr] = np.full(self.output_shape, cval, dtype=arr.dtype)

                print(datetime.now() - start, attr, "job setup")
                # Integer shifts mean the output will be unchanged by the
                # transform, so we can put it straight in the output, since
                # only this array will map into the region.
                if integer_shift:
                    self.log.stdinfo("Placing {} array in [".format(attr) +
                                     ",".join(["{}:{}".format(limits[0] + 1, limits[1])
                                               for limits in output_corners[::-1]]) + "]")
                    slice_ = tuple(slice(min_, max_) for min_, max_ in output_corners)
                    self.output_dict[attr][slice_] = arr
                    continue

                # Set up the functions to call to transform this attribute
                jobs = []
                if np.issubdtype(arr.dtype, np.unsignedinteger):
                    for j in range(0, 16):
                        bit = 2**j
                        if bit == cval or np.sum(arr & bit) > 0:
                            key = ((attr,bit), output_corners)
                            jobs.append((key, arr & bit, {'cval': bit & cval,
                                                          'threshold': threshold*bit}))
                else:
                    key = (attr, output_corners)
                    jobs.append((key, arr, {}))

                print(datetime.now() - start, attr, "job execution")
                # Perform the jobs (in parallel, if we can)
                for (key, arr, kwargs) in jobs:
                    args = (arr, mapping, key, output_array_shape)
                    kwargs.update({'dtype': self.output_dict[attr].dtype,
                                   'order': order,
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
                        self._add_to_output(key, jfactor)
                        print(datetime.now() - start, attr, "one job completed")

        # If we're in parallel, we need to place the outputs into the final
        # arrays as they finish. This should avoid hogging memory. Note that if
        # we've applied integer shifts, the processes list will be empty so this
        # will pass through quickly.
        if parallel:
            print(datetime.now() - start, "waiting for jobs to finish")
            finished = False
            while not finished:
                finished = True
                for p, key in zip(processes, process_keys):
                    if not p.is_alive():
                        if key in self.output_arrays:
                            self._add_to_output(key)
                    else:
                        finished = False

        print(datetime.now() - start, "leaving")
        del self.output_arrays
        return self.output_dict

    def _prepare_for_output(self, input_array, transform, subsample):
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
        min_coords = [int(np.floor(min(coords))) for coords in trans_corners]
        max_coords = [int(np.ceil(max(coords)))+1 for coords in trans_corners]
        self.log.stdinfo("Array maps to ["+",".join(["{}:{}".format(min_+1, max_)
                                for min_, max_ in zip(min_coords, max_coords)])+"]")
        # If this maps to a region not starting in the bottom-left of the
        # output, shift the whole thing so we can efficiently transform it
        # into an array. Coords are still in reverse python order.
        new_min_coords = [max(c, 0) for c in min_coords]
        new_max_coords = [min(c, s) for c, s in zip(max_coords, self.output_shape[::-1])]
        shift = reduce(Model.__and__, [models.Shift(-c) for c in new_min_coords])
        transform.append(shift.rename("Region offset"))

        # Apply scale and shift for subsampling. Recall that (0,0) is the middle of
        # the pixel, not the corner, so a shift is required as well.
        if subsample > 1:
            rescale = reduce(Model.__and__, [models.Scale(subsample)] * transform.ndim)
            rescale_shift = reduce(Model.__and__, [models.Shift(0.5*(subsample-1))] * transform.ndim)
            transform.append([rescale, rescale_shift])

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
        jfactor: float
            multiplier to conserve flux
        """
        attr, output_corners = key
        # 0-indexed but (x, y) order
        self.log.stdinfo("Placing {} array in [".format(attr)+
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
                output_region |= (arr * attr[1]).astype(dtype)
            else:
                self.output_dict[attr[0]][slice_] = ((output_region & (65535 ^ cval)) |
                                                (output_region & (arr * attr[1]))).astype(dtype)
        else:
            self.output_dict[attr][slice_] += arr
        del self.output_arrays[key]

    def _apply_geometric_transform(self, input_array, mapping, output_key,
                                   output_shape, cval=0., dtype=np.float32,
                                   threshold=None, subsample=1, order=3,
                                   jfactor=1):
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
        order: int (0-5)
            order of spline interpolation
        jfactor: float
            Jacobian of transformation (basically the increase in pixel area)
        """
        trans_output_shape = tuple(length * subsample for length in output_shape)
        if isinstance(mapping, GeoMap):
            out_array = ndimage.geometric_transform(input_array, mapping, trans_output_shape,
                                            cval=cval, order=order)
        else:
            out_array = ndimage.affine_transform(input_array, mapping.matrix, mapping.offset,
                                         trans_output_shape, cval=cval, order=order)

        # We average to undo the subsampling. This retains the "threshold" and
        # conserves flux according to the Jacobian of input/output arrays.
        # Doing it here saves memory because we only store the 1-sampled arrays
        if subsample > 1:
            intermediate_shape = tuple(x for length in output_shape
                                       for x in (length, subsample))
            out_array = out_array.reshape(intermediate_shape).\
                mean(tuple(range(len(output_shape)*2-1, 0, -2))).\
                reshape(output_shape)

        if threshold is None:
            self.output_arrays[output_key] = (jfactor * out_array).astype(dtype)
        else:
            self.output_arrays[output_key] = np.where(abs(out_array) > threshold,
                                                      True, False)

#-----------------------------------------------------------------------------
class AstroDataGroup(DataGroup):
    """
    A subclass of DataGroup for transforming AstroData objects. All the arrays
    are Blocks of AD extensions (single slices are made into 1x1 Blocks).

    The output will be a single-extension AD object, taking the PHU and header
    from a reference extension in the inputs. Header keywords are updated or
    deleted as appropriate after the transformation.
    """
    array_attributes = ['data', 'mask', 'variance', 'OBJMASK']

    def __init__(self, arrays=[], transforms=[]):
        super().__init__(arrays=arrays, transforms=transforms)
        # To ensure uniform behaviour, we wish to encase single AD slices
        # as single-element Block objects
        self._arrays = [arr if isinstance(arr, Block) else Block(arr)
                        for arr in self._arrays]
        self.no_data['mask'] = DQ.no_data
        self.ref_array = 0
        self.ref_index = 0

    def append(self, array, transform):
        """Add a single array/transform pair"""
        self._arrays.append(array if isinstance(array, Block) else Block(array))
        self._transforms.append(copy.deepcopy(transform))

    def descriptor(self, desc, index=None, **kwargs):
        """
        Return a list of descriptor returns for all the AD slices that make
        up this object.

        Parameters
        ----------
        desc: str
            name of descriptor
        index: int/None
            only evaluate on this array (None=>return for all)
        kwargs: dict
            kwargs to pass to descriptor

        Returns
        -------
        list: of descriptor return values
        """
        slice_ = slice(None, None) if index is None else slice(index, index+1)
        return [getattr(ext, desc)(**kwargs)
                for arr in self._arrays[slice_] for ext in arr]

    def set_reference(self, array=None, extver=None):
        """
        This method defines the reference extension upon which to base the
        PHU and header of the output single-extension AD object. The WCS of
        this extension will be used as the basis of the output's WCS.

        Parameters
        ----------
        array: int/None
            index of the array to search for this EXTVER (None => search all)
        extver: int/None
            EXTVER value of the reference extension (None => use centre-bottom-left)
        """
        self.ref_array = None
        if extver is None:
            det_corners = np.array([(sec.y1, sec.x1)
                                    for sec in self.descriptor("detector_section", index=array)])
            if len(det_corners) > 1:
                centre = np.median(det_corners, axis=0)
                distances = list(np.sum(det_corners - centre, axis=1))
                self.ref_index = distances.index(max(v for v in distances if v < 0))
            else:
                self.ref_index = 0
            if array is None:
                for i, arr in enumerate(self._arrays):
                    if self.ref_index < len(arr):
                        self.ref_array = i
                        break
                    self.ref_index -= len(arr)
            else:
                self.ref_array = array
        else:
            for index, arr in enumerate(self._arrays):
                if array in (None, index):
                    try:
                        self.ref_index = [hdr['EXTVER'] for hdr in arr.hdr].index(extver)
                    except ValueError:
                        pass
                    else:
                        self.ref_array = index
        if self.ref_array is None:
            raise ValueError("Cannot locate EXTVER {}".format(extver))

    def transform(self, attributes=None, order=3, subsample=1,
                  threshold=0.01, parallel=True, process_objcat=False):
        if attributes is None:
            attributes = [attr for attr in self.array_attributes
                          if all(getattr(ad, attr, None) is not None for ad in self._arrays)]
        self.log.fullinfo("Processing the following array attributes: "
                          "{}".format(', '.join(attributes)))
        super().transform(attributes=attributes, order=order, subsample=subsample,
                          threshold=threshold, parallel=parallel)

        # Create the output AD object
        ref_ext = self._arrays[self.ref_array][self.ref_index]
        adout = astrodata.create(ref_ext.phu)
        adout.append(self.output_dict['data'], header=ref_ext.hdr.copy())
        for key, value in self.output_dict.items():
            setattr(adout[0], key, value)
        self._update_headers(adout)
        self._process_tables(adout, process_objcat=process_objcat)
        return adout

    def _update_headers(self, ad):
        """
        This method updates the headers of the output AstroData object, to
        reflect the work done.
        """
        ndim = len(ad[0].shape)
        header = ad[0].hdr
        keywords = {sec: ad._keyword_for('{}_section'.format(sec))
                                       for sec in ('array', 'data', 'detector')}
        # Data section probably has meaning even if ndim!=2
        ad.hdr[keywords['data']] = '['+','.join('1:{}'.format(length)
                                    for length in ad[0].shape[::-1])+']'
        # For protection against ndim!=2 arrays where descriptor returns
        # are unspecified, wrap this in a big try...except
        if ndim == 2:
            # If detector_section returned something, set an appropriate value
            all_detsec = np.array(self.descriptor('detector_section')).T
            ad.hdr[keywords['detector']] = '['+','.join('{}:{}'.format(min(c1)+1, max(c2))
                                for c1, c2 in zip(all_detsec[::2], all_detsec[1::2]))+']'
            # array_section only has meaning now if the inputs were from a
            # single physical array
            if len(self._arrays) == 1:
                all_arrsec = np.array(self.descriptor('array_section')).T
                ad.hdr[keywords['array']] = '[' + ','.join('{}:{}'.format(min(c1) + 1, max(c2))
                                    for c1, c2 in zip(all_arrsec[::2], all_arrsec[1::2])) + ']'
            else:
                del ad.hdr[keywords['array']]

        # Now sort out the WCS. CRPIXi values have to be added to the coords
        # of the bottom-left of the Block. We want them in x-first order.
        # Also the CRPIXi values are 1-indexed, so handle that.
        transform = self._transforms[self.ref_array]
        wcs = WCS(header)
        ref_coords = tuple(corner+crpix-1 for corner, crpix in
                           zip(self._arrays[self.ref_array].corners[self.ref_index][::-1],
                               wcs.wcs.crpix))
        new_ref_coords = transform(*ref_coords)
        for i, coord in enumerate(new_ref_coords, start=1):
            ad.hdr['CRPIX{}'.format(i)] = coord+1

        affine_matrix = transform.inverse.affine_matrices(self._arrays[self.ref_array].shape).matrix
        try:
            cd_matrix = np.dot(wcs.wcs.cd, affine_matrix[::-1,::-1])
        except AttributeError:  # No CD matrix
            pass
        else:
            for j in range(ndim):
                for i in range(ndim):
                    ad.hdr['CD{}_{}'.format(i+1, j+1)] = cd_matrix[i, j]

        # Finally, delete any keywords that no longer make sense
        for kw in ('AMPNAME', 'FRMNAME', 'FRAMEID', 'CCDSIZE', 'BIASSEC',
                   'DATATYP', 'OVERSEC', 'TRIMSEC', 'OVERSCAN', 'OVERRMS'):
            if kw in header:
                del ad.hdr[kw]

    def _process_tables(self, ad, process_objcat=False):
        """
        This method propagates the REFCAT and/or OBJCAT catalogs with
        appropriate changes based on the transforms.
        """
        # Copy top-level tables. We assume that the inputs only have a single
        # instance of each table between them, so we take it from the reference
        # extension (which gets it from its parent AD object).
        for name in self._arrays[self.ref_array][self.ref_index].tables:
            setattr(ad, name, getattr(self._arrays[self.ref_array][self.ref_index], name))
            self.log.fullinfo("Copying {}".format(table))

        # Join OBJCATs. We transform the pixel coordinates and then update the
        # RA and DEC based on the output image's WCS.
        if process_objcat:
            self.log.stdinfo("Processing OBJCAT")
            wcs = WCS(ad[0].hdr)
            tables = []
            for array, transform in zip(self._arrays, self._transforms):
                try:
                    objcat = array.OBJCAT
                except AttributeError:
                    continue
                del objcat.meta['header']
                for ycolumn, xcolumn in zip(*catalog_coordinate_columns['OBJCAT']):
                    # OBJCAT coordinates are 1-indexed
                    newx, newy = transform(objcat[xcolumn]-1, objcat[ycolumn]-1)
                    objcat[xcolumn] = newx + 1
                    objcat[ycolumn] = newy + 1
                ra, dec = wcs.all_pix2world(objcat['X_IMAGE'], objcat['Y_IMAGE'], 1)
                objcat["X_WORLD"] = ra
                objcat["Y_WORLD"] = dec
                tables.append(objcat)

            if tables:
                objcat = table.vstack(tables, metadata_conflicts='silent')
                objcat['NUMBER'] = np.arange(len(objcat)) + 1
                ad[0].OBJCAT = objcat