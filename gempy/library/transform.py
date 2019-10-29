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

Functions:
    create_mosaic_transform: construct an AstroDataGroup instance that will
                             mosaic the detectors
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
from gempy.gemini import gemini_tools as gt

import multiprocessing as multi
from geminidr.gemini.lookups import DQ_definitions as DQ

import astrodata, gemini_instruments

from .astromodels import Rotate2D, Shift2D, Scale2D
from ..utils import logutils

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
        super(Block, self).__init__()
        self._elements = list(elements)
        shapes = [el.shape for el in elements]

        # Check all elements have the same dimensionality
        if len(set([len(s) for s in shapes])) > 1:
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
        return len(self._shape)

    @property
    def shape(self):
        """Overall shape of Block"""
        return self._total_shape

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
        return copy.deepcopy(self)

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
        for model in self._models:
            yield model

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
        if shape is None:
            try:
                shape = (1000,) * self.ndim
            except TypeError:  # self.ndim is None
                raise TypeError("Cannot compute affine matrices without a "
                                "dimensionality")
        ndim = len(shape)
        halfsize = [0.5*length for length in shape]
        points = np.array([halfsize] * (2*ndim+1)).T
        points[:,1:ndim+1] += np.eye(ndim) * points[:,0]
        points[:,ndim+1:] -= np.eye(ndim) * points[:,0]
        if ndim > 1:
            transformed = np.array(list(zip(*self.__call__(*points))))
        else:
            transformed = np.array([self.__call__(*points)]).T
        matrix = np.array([[0.5 * (transformed[j+1,i] - transformed[ndim+j+1,i]) / halfsize[j]
                            for j in range(ndim)] for i in range(ndim)])
        offset = transformed[0] - np.dot(matrix, halfsize)
        return AffineMatrices(matrix.T, offset[::-1])

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

    def apply(self, input_array, output_shape, order=1, cval=0, inverse=False):
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
        order: int (0-5)
            order of interpolation
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
                                        output_shape, order=order, cval=cval)
        else:
            mapping = GeoMap(self, output_shape, inverse=inverse)
            output_array = ndimage.map_coordinates(input_array, mapping.coords,
                                        output_shape, order=order, cval=cval)
        return output_array

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

class GeoMap(object):
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
        self.coords = transformed

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
                         + offset_array).astype(np.float32).reshape(len(grids), *self._shape)
        offsets = np.sum(np.square(self.coords - affine_coords), axis=0)
        return np.sqrt(np.mean(offsets)), np.sqrt(np.max(offsets))

#----------------------------------------------------------------------------------

class DataGroup(object):
    """
    A DataGroup is a collection of an equal number array-like objects and
    Transforms, with the intention of transforming the arrays into a single
    output array.
    """
    UnequalError = ValueError("Number of arrays and transforms must be equal")

    def __init__(self, arrays=[], transforms=None):
        if transforms:
            if len(arrays) != len(transforms):
                raise self.UnequalError
            # "Freeze" the transforms
            self._transforms = copy.deepcopy(transforms)
        else:
            self._transforms = [Transform()] * len(arrays)
        self._arrays = arrays
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

    def calculate_output_shape(self, additional_array_shapes=[],
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
        additional_array_shapes: list of tuples
            shapes of any other arrays that will be transformed later, to
            ensure a uniform pixel output grid
        additional_transforms: list
            additional transforms, one for each of additional_array_shapes
        """
        array_shapes = [arr.shape for arr in self.arrays] + additional_array_shapes
        transforms = self.transforms + additional_transforms
        if len(array_shapes) != len(transforms):
            raise self.UnequalError
        all_corners = []
        # The [::-1] are because we need to convert from python order to
        # astropy Model order (x-first) and then back again
        for array_shape, transform in zip(array_shapes, transforms):
            corners = np.array(at.get_corners(array_shape)).T[::-1]
            trans_corners = transform(*corners)
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

    def transform(self, attributes=['data'], order=1, subsample=1,
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
        order: int (0-5)
            order of spline interpolation
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
            transform = copy.deepcopy(transform)
            if self.origin:
                transform.append(reduce(Model.__and__,
                                 [models.Shift(-offset) for offset in self.origin[::-1]]))
            output_corners = self._prepare_for_output(input_array,
                                                      transform, subsample)
            output_array_shape = tuple(max_ - min_ for min_, max_ in output_corners)

            # This can happen if the origin and/or output_shape are modified
            if not all(length > 0 for length in output_array_shape):
                self.log.stdinfo("Array falls outside output region")
                continue

            # Create a mapping from output pixel to input pixels
            mapping = transform.inverse.affine_matrices(shape=output_array_shape)
            jfactor = abs(np.linalg.det(mapping.matrix))
            if not conserve:
                jfactor = 1
            self.jfactors.append(jfactor)

            integer_shift = (transform.is_affine
                and np.array_equal(mapping.matrix, np.eye(mapping.matrix.ndim)) and
                                 np.array_equal(mapping.offset, mapping.offset.astype(int)))

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
                        jacobian_mapping = GeoMap(transform, jacobian_shape)
                        det_matrices = np.empty((ndim, ndim, np.multiply.reduce(trans_output_shape)))
                        for num_axis in range(ndim):
                            coords = jacobian_mapping.coords[num_axis]
                            for denom_axis in range(ndim):
                                diff_coords = coords - np.roll(coords, 2, axis=denom_axis)
                                slice_ = [slice(1, -1)] * ndim
                                slice_[denom_axis] = slice(2, None)
                                # Account for the fact that we are measuring
                                # differences in the subsampled plane
                                det_matrices[num_axis, denom_axis] = \
                                    diff_coords[slice_].flatten() / (2*subsample)
                        jfactor = 1. / abs(np.linalg.det(np.moveaxis(det_matrices, -1, 0))).reshape(trans_output_shape)
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
                        bit = 2**j
                        if bit == cval or np.sum(arr & bit) > 0:
                            key = ((attr,bit), output_corners)
                            jobs.append((key, arr & bit, {'cval': bit & cval,
                                                          'threshold': threshold*bit}))
                else:
                    key = (attr, output_corners)
                    jobs.append((key, arr, {}))

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
        if len(input_array.shape) == 1:
            trans_corners = (trans_corners,)
        self.corners.append(trans_corners)
        min_coords = [int(np.ceil(min(coords))) for coords in trans_corners]
        max_coords = [int(np.floor(max(coords)))+1 for coords in trans_corners]
        self.log.stdinfo("Array maps to ["+",".join(["{}:{}".format(min_+1, max_)
                                for min_, max_ in zip(min_coords, max_coords)])+"]")
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
                output_region |= (arr * attr[1]).astype(dtype)
            else:
                self.output_dict[attr[0]][slice_] = ((output_region & (65535 ^ cval)) |
                                                (output_region & (arr * attr[1]))).astype(dtype)
        else:
            self.output_dict[attr][slice_] += arr
        del self.output_arrays[key]

    def _apply_geometric_transform(self, input_array, mapping, output_key,
                                   output_shape, cval=0., dtype=np.float32,
                                   threshold=None, subsample=1, order=1,
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
        jfactor: float/array
            Jacobian of transformation (basically the increase in pixel area)
        """
        trans_output_shape = tuple(length * subsample for length in output_shape)
        if isinstance(mapping, GeoMap):
            out_array = ndimage.map_coordinates(input_array, mapping.coords,
                                                cval=cval, order=order)
        else:
            out_array = ndimage.affine_transform(input_array, mapping.matrix,
                                                 mapping.offset, trans_output_shape,
                                                 cval=cval, order=order)

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
            self.output_arrays[output_key] = (out_array * jfactor).astype(dtype)
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

    def __init__(self, arrays=[], transforms=None):
        super(AstroDataGroup, self).__init__(arrays=arrays, transforms=transforms)
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
                distances = list(det_corners - centre)
                self.ref_index = np.argmax([d.sum() if np.all(d <= 0) else -np.inf for d in distances])
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

    def transform(self, attributes=None, order=1, subsample=1, threshold=0.01,
                  conserve=False, parallel=False, process_objcat=False):
        """
        This method

        Parameters
        ----------
        attributes: list-like
            attributes to be transformed (None => all)
        order: int
            order of interpolation
        subsample: int
            if >1, will transform onto finer pixel grid and block-average down
        threshold: float
            for transforming the DQ plane, output pixels larger than this value
            will be flagged as "bad"
        conserve: bool
            conserve flux rather than interpolate?
        parallel: bool
            use parallel processing to speed up operation?
        process_objcat: bool
            merge input OBJCATs into output AD instance?

        Returns
        -------

        """
        if attributes is None:
            attributes = [attr for attr in self.array_attributes
                          if all(getattr(ad, attr, None) is not None for ad in self._arrays)]
        if 'data' not in attributes:
            self.log.warning("The 'data' attribute is not specified. Adding to list.")
            attributes += ['data']

        # Create the output AD object
        ref_ext = self._arrays[self.ref_array][self.ref_index]
        adout = astrodata.create(ref_ext.phu)
        adout.orig_filename = ref_ext.orig_filename

        self.log.fullinfo("Processing the following array attributes: "
                          "{}".format(', '.join(attributes)))
        super(AstroDataGroup, self).transform(attributes=attributes, order=order,
                                              subsample=subsample, threshold=threshold,
                                              conserve=conserve, parallel=parallel)

        adout.append(self.output_dict['data'], header=ref_ext.hdr.copy())
        for key, value in self.output_dict.items():
            if key != 'data':  # already done this
                setattr(adout[0], key, value)
        self._update_headers(adout)
        self._process_tables(adout, process_objcat=process_objcat)

        # Ensure the ADG object doesn't hog memory
        self.output_dict = {}
        return adout

    def _update_headers(self, ad):
        """
        This method updates the headers of the output AstroData object, to
        reflect the work done.
        """
        ndim = len(ad[0].shape)
        if ndim != 2:
            self.log.warning("The updating of header keywords has only been "
                             "fully tested for 2D data.")
        header = ad[0].hdr
        keywords = {sec: ad._keyword_for('{}_section'.format(sec))
                                       for sec in ('array', 'data', 'detector')}
        # Data section probably has meaning even if ndim!=2
        ad.hdr[keywords['data']] = '['+','.join('1:{}'.format(length)
                                    for length in ad[0].shape[::-1])+']'
        # These descriptor returns are unclear in non-2D data
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
        if 'CCDNAME' in ad[0].hdr:
            ad.hdr['CCDNAME'] = ad.detector_name()

        # Now sort out the WCS. CRPIXi values have to be added to the coords
        # of the bottom-left of the Block. We want them in x-first order.
        # Also the CRPIXi values are 1-indexed, so handle that.
        transform = self._transforms[self.ref_array]
        wcs = WCS(header)
        ref_coords = tuple(corner+crpix-1 for corner, crpix in
                           zip(self._arrays[self.ref_array].corners[self.ref_index][::-1],
                               wcs.wcs.crpix))
        new_ref_coords = transform(*ref_coords)
        # The origin shift wasn't appended to the Transform, so apply it here
        if self.origin:
            new_ref_coords = reduce(Model.__and__, [models.Shift(-offset)
                        for offset in self.origin[::-1]])(*new_ref_coords)
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
            self.log.fullinfo("Copying {}".format(name))

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


    def inverse_transform(self, admos, attributes=None, order=1, subsample=1,
                          threshold=0.01, conserve=False):
        """
        The method performs the inverse transformation, which includes breaking
        the input file into multiple extensions.

        Parameters
        ----------
        admos: AstroData
            an AD object compatible with the output of self.transform()
        attributes: list-like
            attributes to be transformed (None => all)
        order: int
            order of interpolation
        subsample: int
            if >1, will transform onto finer pixel grid and block-average down
        threshold: float
            for transforming the DQ plane, output pixels larger than this value
            will be flagged as "bad"
        conserve: bool
            conserve flux rather than interpolate?

        Returns
        -------
        AstroData: a multi-extension AD instance from unmosaicking the input
        """
        if len(admos) != 1:
            raise ValueError("AstroData instance must have only one extension")
        if admos[0].shape != self.output_shape:
            raise ValueError("AstroData shape incompatible with transform")

        adout = astrodata.create(admos.phu)
        adout.orig_filename = admos.orig_filename

        for arr, t in self:
            # Since the origin shift is the last thing applied before
            # transforming, it must be the first thing in the inverse
            t_inverse = t.inverse
            t_inverse.prepend(reduce(Model.__and__,
                                     [models.Shift(o) for o in self.origin[::-1]]))
            adg = self.__class__(admos, [t_inverse])
            # Transformations are interpolations on the input pixel grid. So
            # pixels are typically lost around the edges and the footprint is
            # smaller than the input image's footprint. So we force the shape
            # of the inverse-transformed image to be that of the input array
            # and only take the region starting from (0,0) in the output frame
            adg.origin = (0, 0)
            adg.output_shape = arr.shape
            block = adg.transform(attributes=attributes, order=order,
                                  subsample=subsample, threshold=threshold,
                                  conserve=conserve)

            # Now we split the block into its constituent extensions
            for ext, corner in zip(arr, arr.corners):
                slice_ = tuple(slice(start, start+length)
                               for start, length in zip(corner, ext.shape))
                # We need to deepcopy here to protect the header, because of
                # the way _append_nddata behaves
                ndd = copy.deepcopy(block[0].nddata[slice_])
                adout.append(ndd, header=ext.hdr, reset_ver=False)

        return adout

#-----------------------------------------------------------------------------
def create_mosaic_transform(ad, geotable):
    """
    Constructs an AstroDataGroup object that will perform the mosaicking
    operation on an AstroData instance.

    Parameters
    ----------
    ad: AstroData
        the AD object that will be mosaicked
    geotable: module
        the geometry_conf module with the required information

    Returns
    -------
    AstroDataGroup: the ADG object that can be transformed to perform
                    the mosaicking
    """


    # Create the blocks (individual physical detectors)
    array_info = gt.array_information(ad)
    blocks = [Block(ad[arrays], shape=shape) for arrays, shape in
              zip(array_info.extensions, array_info.array_shapes)]
    offsets = [ad[exts[0]].array_section()
               for exts in array_info.extensions]

    detname = ad.detector_name()
    xbin, ybin = ad.detector_x_bin(), ad.detector_y_bin()
    geometry = geotable.geometry[detname]
    default_shape = geometry.get('default_shape')
    adg = AstroDataGroup()

    for block, origin, offset in zip(blocks, array_info.origins, offsets):
        # Origins are in (x, y) order in LUT
        block_geom = geometry[origin[::-1]]
        nx, ny = block_geom.get('shape', default_shape)
        nx /= xbin
        ny /= ybin
        shift = block_geom.get('shift', (0, 0))
        rot = block_geom.get('rotation', 0.)
        mag = block_geom.get('magnification', (1, 1))
        transform = Transform()

        # Shift the Block's coordinates based on its location within
        # the full array, to ensure any rotation takes place around
        # the true centre.
        if offset.x1 != 0 or offset.y1 != 0:
            transform.append(models.Shift(float(offset.x1) / xbin) &
                             models.Shift(float(offset.y1) / ybin))

        if rot != 0 or mag != (1, 1):
            # Shift to centre, do whatever, and then shift back
            transform.append(models.Shift(-0.5 * (nx - 1)) &
                             models.Shift(-0.5 * (ny - 1)))
            if rot != 0:
                # Cope with non-square pixels by scaling in one
                # direction to make them square before applying the
                # rotation, and then reversing that.
                if xbin != ybin:
                    transform.append(models.Identity(1) & models.Scale(ybin / xbin))
                transform.append(models.Rotation2D(rot))
                if xbin != ybin:
                    transform.append(models.Identity(1) & models.Scale(xbin / ybin))
            if mag != (1, 1):
                transform.append(models.Scale(mag[0]) &
                                 models.Scale(mag[1]))
            transform.append(models.Shift(0.5 * (nx - 1)) &
                             models.Shift(0.5 * (ny - 1)))
        transform.append(models.Shift(float(shift[0]) / xbin) &
                         models.Shift(float(shift[1]) / ybin))
        adg.append(block, transform)

    adg.set_reference()
    return adg