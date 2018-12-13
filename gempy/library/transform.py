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

functions:
"""
import numpy as np

from astropy.modeling import models, Model
from astropy.modeling.core import _model_oper

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

class Transform(object):
    """
    A chained set of astropy Models with the ability to select a subchain.
    Since many transformations will be concatenated pairs of Models (one
    for each ordinate), this also creates a better structure.

    A Transform can be accessed and iterated over like a list, but also
    like a dict if the individual models have names.
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
        or slice or tuple."""
        if isinstance(key, slice):
            # Turn into a slice of integers
            slice_ = slice(self.index(key.start),
                           self.index(key.stop), key.step)
            return self.__class__(self._models[slice_])
        elif isinstance(key, tuple):
            return self.__class__([self[i] for i in key])
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
            array, array: affine matrix and offset
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
        return matrix.T, offset[::-1]


    def __call__(self, *args, **kwargs):
        if len(args) != self.ndim:
            raise ValueError("Incompatible number of inputs for Transform "
                             "(dimensionality {})".format(self.ndim))
        return self.asModel()(*args, **kwargs)

    def asModel(self):
        """
        Return a Model instance of this Transform
        """
        if len(self) == 0:
            return models.Identity(self.ndim)
        composite_model = self._models[0]
        for model in self._models[1:]:
            composite_model = composite_model | model
        return composite_model

    def info(self):
        """Print out something vaguely intelligible for debugging purposes"""
        new_line_indent = "\n    "
        msg = ['Model information: (dimensionality {})'.format(self.ndim)]
        for model in self._models:
            msg.append(repr(model))
        return new_line_indent.join(msg)
