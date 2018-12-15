"""
transform.py

This module contains classes and functions related to the geometric
transformation of arrays and AstroData objects.

Classes:
    Block: a container for array-like objects that are adjacent to each other
           and so much be transformed together (e.g., each CCD in GMOS is a
           Block)
    Transform: a collection of chained astropy Models that together describe
               a transformation from one set of coordinates to another
    GeoMap: a callable object that accepts coordinates and returns
            geometrically-transformed coordinates

functions:
"""
import numpy as np
from functools import reduce
from collections import namedtuple

from astropy.modeling import models, Model
from astropy.modeling.core import _model_oper
from scipy import ndimage

from gempy.library import astrotools as at

import multiprocessing as multi
from geminidr.gemini.lookups import DQ_definitions as DQ

from ..utils import logutils

AffineMatrices = namedtuple("AffineMatrices", "matrix offset")

class Block(object):
    """
    A Block is a container for multiple AD slices/NDData/ndarray objects
    to organize them prior to being accessed for a transformation.

    In the grand scheme of things, Blocks can be transformed in parallel
    because they have gaps between them and so don't overlap in the final
    transformed image.
    """
    def __init__(self, elements, shape):
        """
        elements: list of AD slices/NDData/ndarray
        shape: tuple giving shape in which the elements should be arranged
        """
        super(Block, self).__init__()
        self._elements = list(elements)
        try:
            shapes = [el.shape for el in elements]
        except AttributeError:
            shapes = [el.nddata.shape for el in elements]
        if len(set(shapes)) > 1:
            raise ValueError("Not all elements have the same shape")
        self._arrayshape = shapes[0]
        shape = tuple([1] * (len(self._arrayshape)-len(shape))) + shape
        if np.multiply.reduce(shape) != len(elements):
            raise ValueError("Incompatible element list and shape")
        self._shape = shape

    def __getitem__(self, index):
        """Only able to return a single element"""
        arrayindex = tuple([i % l for i, l in zip(index, self._arrayshape)])
        listindex = int(np.sum((i // l)*np.multiply.reduce(self._shape[j+1:])
                           for j, (i, l) in enumerate(zip(index, self._arrayshape))))
        return self._elements[listindex][arrayindex]

    @property
    def shape(self):
        """Overall shape of Block"""
        return tuple([a*s for a, s in zip(self._arrayshape, self._shape)])

    @property
    def ndim(self):
        """Number of dimensions"""
        return len(self._shape)

    def __getattr__(self, name):
        try:
            output = self._return_array([getattr(el, name)
                                         for el in self._elements])
        except AttributeError as e:
            if name == "data":
                output = self._return_array(self._elements)
            else:
                raise AttributeError(e)
        return output

    def _return_array(self, elements):
        """
        Returns a new ndarray, which is required for the scipy.ndimage
        transformations. This obviously requires a copy of the data,
        which is what we were trying to avoid. But oh well.
        """
        if len(elements) != len(self._elements):
            raise ValueError("All elements of the Block must be used in the "
                             "output array")
        output = np.empty(self.shape, dtype=elements[0].dtype)
        corner = [0] * self.ndim
        for arr in elements:
            section = tuple(slice(c, c+s)
                            for c, s in zip(corner, self._arrayshape))
            output[section] = arr
            for i in range(self.ndim-1, -1, -1):
                corner[i] += self._arrayshape[i]
                if corner[i] >= self.shape[i]:
                    corner[i] = 0
                else:
                    break
        return output

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
    def __init__(self, model=None, ndim=None):
        """Initialize with a single model, list, or dimensionality"""
        self._affine = True
        self._models = []
        if model:
            self.append(model)
        elif ndim:
            self._ndim = ndim
        else:
            raise ValueError("A Model or dimensionality must be specified")

    @property
    def ndim(self):
        """Dimensionality (number of inputs)"""
        return self._ndim

    def __len__(self):
        """Length (number of models)"""
        return len(self._models)

    def __iter__(self):
        return self._models.__iter__()

    def __next__(self):
        return self._models.__next__()

    def __getitem__(self, key):
        """Return a Model or Transform based on index or name,
        or slice or tuple.

        If the models are in descending index order, then the inverse
        will be used
        """
        if isinstance(key, slice):
            # Turn into a slice of integers
            slice_ = slice(self.index(key.start),
                           self.index(key.stop), key.step)
            if key.step is None or key.step > 0:
                return self.__class__(self._models[slice_])
            else:  # Return chain of inverses
                return self.__class__([m.inverse for m in self._models[slice_]])
        elif isinstance(key, tuple):
            indices = [self.index(k) for k in key]
            if all(np.diff(indices) < 0):
                return self.__class__([self[i].inverse for i in indices])
            else:
                return self.__class__([self[i] for i in indices])
        else:
            try:
                return self._models[key]
            except TypeError:
                return self._models[self.index(key)]

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
            del self._models[self.index(key)]
        self._affine = self.__is_affine()

    @property
    def inverse(self):
        """The inverse transform"""
        return self.__class__([model.inverse for model in self._models[::-1]])

    def index(self, key):
        """Works like the list.index() method to return the index of a model
        with a specific name. Traps being sent None (returns None) or an
        integer (which is assumed to be a real index)"""
        if key is None:
            return
        try:
            return [m.name for m in self._models].index(key)
        except ValueError as e:
            if isinstance(key, (int, np.integer)):
                return key
            raise ValueError(e)

    def append(self, model):
        """Append Model(s) to the end of the current Transform"""
        if model is not None:
            if isinstance(model, list):
                for m in model:
                    self.insert(len(self), m)
            else:
                self.insert(len(self), model)

    def prepend(self, model):
        """Prepend Model(s) to the start of the current Transform"""
        if model is not None:
            if isinstance(model, list):
                for m in model[::-1]:
                    self.insert(0, m)
            else:
                self.insert(0, model)

    def insert(self, index, model):
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

        # Ugly stuff to break down a CompoundModel
        try:
            sequence = self.split_compound_model(model._tree, required_inputs)
        except AttributeError:
            self._models.insert(index, model)
        else:
            i = self.index(index)
            # Avoids corrupting a Model (it will lose its name)
            if len(sequence) == 1:
                self._models.insert(index, model)
            else:
                self._models[i:i] = sequence
        # Update affinity since we've modified the model
        self._affine = self.__is_affine()

    @staticmethod
    def split_compound_model(tree, ndim):
        """
        Break a CompoundModel into a sequence of chained Model instances.
        It's necessary to specify the initial dimensionality of the Model
        to determine whether a chain operator (|) is linking full
        transformations or just components within a submodel. It may prove
        difficult/impossible to split a Model if the number of inputs is
        not preserved. We'll cross that bridge if/when we come to it.

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
                # we have on the stack takes the right number of inputs
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

    def __is_affine(self):
        """Test for affinity, using Model names.
        TODO: Is this the right thing to do? We could compute the affine
        matrices *assuming* affinity, and then check that a number of random
        points behave as expected. Is that better?"""
        models = []
        for m in self._models:
            try:
                models.extend(m._submodels)
            except AttributeError:
                models.append(m)
        for m in models:
            if m.__class__.__name__[:5] not in ('Rotat', 'Scale', 'Shift'):
                return False
        return True

    @property
    def is_affine(self):
        """Return affinity of model. I have deliberately written this as a
        settable attribute to override the __is_affine() method, in case
        that is unable to handle certain Models."""
        return self._affine

    def affine_matrices(self, shape=None):
        """
        Compute the matrix and offset necessary to turn a Transform into an
        affine transformation.

        Parameters
        ----------
        shape: sequence
            shape to use for fiducial points

        Returns
        -------
            AffineMatrices(array, array): affine matrix and offset
        """
        if not self.is_affine:
            raise ValueError("Transformation is not affine!")
        ndim = self.ndim
        if shape is None:
            shape = tuple([1000] * ndim)
        corners = np.zeros((ndim, ndim+1))
        for i, length in enumerate(shape):
            corners[i,i+1] = length
        transformed = np.array(list(zip(*self.__call__(*corners))))
        offset = transformed[0]
        matrix = np.empty((ndim, ndim))
        for i in range(ndim):
            for j in range(ndim):
                matrix[i,j] = (transformed[j+1,i] - offset[i]) / corners[j,j+1]
        # Convert to python ordering
        return AffineMatrices(matrix.T, offset[::-1])

    def __call__(self, *args, **kwargs):
        if len(args) != self.ndim:
            raise ValueError("Incompatible number of inputs for Transform "
                             "(dimensionality {})".format(self.ndim))
        try:
            inverse = kwargs.pop('inverse')
        except KeyError:
            inverse = False
        return self.asModel(inverse=inverse)(*args, **kwargs)

    def asModel(self, inverse=False):
        """
        Return a Model instance of this Transform, by chaining together
        the individual Models
        """
        if len(self) == 0:
            return models.Identity(self.ndim)
        model_list = self.inverse._models if inverse else self._models
        return reduce(Model.__or__, model_list)

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
        try:
            if inverse:
                matrix, offset = self.affine_matrices(input_array.shape)
            else:
                matrix, offset = self.inverse.affine_matrices(input_array.shape)
        except ValueError:
            grid = np.meshgrid(*(np.arange(length) for length in input_array.shape))
            if inverse:
                out_coords = self.__call__(*grid)
            else:
                out_coords = self.inverse(*grid)
            for i, length in enumerate(output_shape):
                out_coords[i] = np.where(out_coords[i] > length-1, -1,
                                         out_coords[i]).astype(np.float32)
            mapping = GeoMap(*out_coords)
            output_array = ndimage.geometric_transform(input_array, mapping,
                                                       output_shape, cval=cval)
        else:
            output_array = ndimage.affine_transform(input_array, matrix, offset,
                                                    output_shape, cval=cval)
        return output_array

#----------------------------------------------------------------------------------

class GeoMap(object):
    """
    Needed to create a callable object for the geometric_transform. Stores ndim
    mapping arrays (one for each axis) and provides a function that returns
    the value of the array at the location of the tuple passed to that function
    """
    def __init__(self, *args):
        self._map = args

    def __call__(self, coords):
        return tuple(map_[coords] for map_ in self._map[::-1])

#----------------------------------------------------------------------------------

def apply_transform(inputs, transforms, output_shape, attributes=['data'],
                    subsample=1, threshold=0.01, parallel=True):
    """

    Parameters
    ----------
    inputs: sequence
        input objects (AD/NDDdata/ndarrays)
    transforms: sequence
        transforms to apply (one per input object)
    output_shape: sequence
        shape of the output array(s)
    attributes: sequence
        attributes of the input objects to transform
    subsample: int
        subsampling of input/output arrays
    threshold: float
        for bitmask arrays, the fraction that needs to be exceeded
        for the bit to be set in the output
    parallel: bool
        perform operations in parallel using multiprocessing?

    """
    log = logutils.get_logger(__name__)

    output_dict = {}
    area_scale = 1.0

    if parallel:
        processes = []
        process_keys = []
        manager = multi.Manager()
        output_arrays = manager.dict()
    else:
        output_arrays = {}

    for input_array, transform in zip(inputs, transforms):
        output_corners = _prepare_for_output(input_array, output_shape,
                                             transform, log)
        output_array_shape = tuple(max_-min_ for min_, max_ in output_corners)

        # Create a mapping from output pixel to input pixels
        if transform.is_affine:
            mapping = transform.inverse.affine_matrices(output_array_shape)
        else:
            grids = np.meshgrid(*(np.arange(length)
                                  for length in output_array_shape[::-1]))
            transformed = transform.inverse(*grids)
            transformed = [np.where(coord>length-1, -1, coord).astype(np.float32)
                           for coord, length in zip(transformed, output_array_shape[::-1])]
            mapping = GeoMap(*transformed)

        for attr in attributes:
            try:
                arr = getattr(input_array, attr)
            except AttributeError as e:
                # If input_array is just an ndarray
                if attr == "data":
                    arr = input_array
                else:
                    raise AttributeError(e)

            # Create an output array if we haven't seen this attribute yet
            if attr not in output_dict:
                output_dict[attr] = np.zeros(output_shape, arr.dtype)

            # Set up the functions to call to transform this attribute
            jobs = []
            if np.issubdtype(arr.dtype, np.unsignedinteger):
                for j in range(0, 16):
                    bit = 2**j
                    if ((bit == DQ.no_data and attr == "mask") or
                            np.sum(arr & bit) > 0):
                        cval = bit & DQ.no_data
                        key = ((attr,bit), output_corners)
                        jobs.append((key, arr & bit, {'cval': cval, 'threshold': threshold*bit}))
            else:
                key = (attr, output_corners)
                jobs.append((key, arr, {}))

            # Perform the jobs (in parallel, if we can)
            for (key, arr, kwargs) in jobs:
                args = (arr, mapping, output_arrays, key, output_array_shape)
                kwargs['dtype'] = output_dict[attr].dtype
                if parallel:
                    p = multi.Process(target=_apply_geometric_transform,
                                      args=args, kwargs=kwargs)
                    processes.append(p)
                    process_keys.append(key)
                    p.start()
                else:
                    _apply_geometric_transform(*args, **kwargs)
                    _add_to_output(output_dict, key, output_arrays, area_scale, log)

    # If we're in parallel, we need to place the outputs into the final
    # arrays as they finish. This should avoid hogging memory
    if parallel:
        finished = False
        while not finished:
            finished = True
            for p, key in zip(processes, process_keys):
                if not p.is_alive():
                    if key in output_arrays:
                        _add_to_output(output_dict, key, output_arrays, area_scale, log)
                else:
                    finished = False

    return output_dict


#-----------------------------------------------------------------------------

def _prepare_for_output(input_array, output_shape, transform, log):
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
    log: logger

    Returns
    -------
    sequence of doubletons: limits of the output region this transforms into
    """
    # Find extent of this array in the output, after transformation
    try:
        corners = np.array(at.get_corners(input_array.shape)).T
    except AttributeError:  # AD objects have no .shape attribute
        corners = np.array(at.get_corners(input_array.nddata.shape)).T
    # Invert from standard python order to (x, y[, z]) order
    trans_corners = [c[::-1] for c in transform(*corners)]
    min_coords = [int(np.floor(min(coords))) for coords in trans_corners]
    max_coords = [int(np.ceil(max(coords)))+1 for coords in trans_corners]
    log.stdinfo("Array maps to ("+",".join(["{}:{}".format(min_, max_)
                            for min_, max_ in zip(min_coords, max_coords)])+")")
    # If this maps to a region not starting in the bottom-left of the
    # output, shift the whole thing so we can efficiently transform it
    # into an array. Coords are still in reverse python order.
    new_min_coords = [max(c, 0) for c in min_coords]
    new_max_coords = [min(c, s) for c, s in zip(max_coords, output_shape[::-1])]
    shift = reduce(Model.__and__, [models.Shift(c) for c in new_min_coords])
    transform.append(shift.rename("Region offset"))
    output_corners = tuple((min_, max_) for min_, max_ in
                           zip(new_min_coords, new_max_coords))[::-1]
    return output_corners  # in standard python order

def _add_to_output(output_dict, key, output_arrays, area_scale, log):
    """
    Adds output_arrays[key] to full_output and deletes it from output_arrays

    Parameters
    ----------
    output_dict: dict
        container for the various final output arrays (one per attribute)
    key: tuple
        identifier for this particular array
    output_arrays: dict
        container for the results of the geometric transform
    area_scale: float
        multiplier to conserve flux
    log: logger
    """
    attr, output_corners = key
    # 0-indexed but (x, y) order
    log.stdinfo("Placing {} array in (".format(attr)+
                ",".join(["{}:{}".format(*limits)
                          for limits in output_corners[::-1]])+")")
    arr = output_arrays[key]
    print (arr.shape, output_corners)
    slice_ = tuple(slice(min_,max_) for min_, max_ in output_corners)
    if isinstance(attr, tuple):  # e.g., ('mask', 2)
        output_dict[attr[0]][slice_] += (arr * attr[1]).astype(arr.dtype)
    else:
        output_dict[attr][slice_] += arr * area_scale
    del output_arrays[key]

def _apply_geometric_transform(input_array, mapping, output, output_key,
                               output_shape, cval=0., dtype=np.float32,
                               threshold=None):
    """
    None-returning function to apply geometric transform, so it can be used
    by multiprocessing
    Inputs: input_array = input array
            mapping = callable object with mapping from output->input coords
            output = DictProxy to hold the transformed output
            output_key = key in DictProxy for this output
            output_shape = tuple indicating the shape of these outputs
            cval = value for "empty" pixels
            dtype = datatype of outputarray
            threshold = limit for deciding what value counts a pixel as "bad"
                        (if set, always returns a bool array)
    """
    if isinstance(mapping, GeoMap):
        out_array = ndimage.geometric_transform(input_array, mapping, output_shape,
                                        cval=cval)
    else:
        out_array = ndimage.affine_transform(input_array, mapping.matrix, mapping.offset,
                                     output_shape, cval=cval)
    if threshold is None:
        output[output_key] = out_array.astype(dtype)
    else:
        output[output_key] = np.where(out_array>threshold, True, False)