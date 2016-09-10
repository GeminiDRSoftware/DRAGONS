
.. _edge_class:

Edge Class
==========

The Edge class provides functionality to handle a list of coordinates corresponding to one footprint edge.
::

 USAGE
 -----

    ed = Edge(x_array, y_array)

    Parameters
    ----------

    x_array: Array of x-coordinates of a footprint edge.
    y_array: Array of y-coordinates of the same footprint edge.

Class Edge Attributes
^^^^^^^^^^^^^^^^^^^^^^

- **trace** A tuple containing the arrays (x_array,y_array)
- **function** A string with the fitting function name. The default is *polynomial*.
- **order**   An integer with the fitting function order. The default is 2.
- **xlim** Tuple. (minimum x_array value, maximum x_array value)
- **ylim**  Tuple. (minimum y_array value, maximum y_array value)
- **coefficients**  Array with fitting function coefficients. Such that: y=cc[2]*x[0]**2 + cc[1]*x[0] + cc[0] where *cc* is the coefficients array.
- **evalfunction** Evaluator function. Evaluate the polynomial at x, where x is a single point or an array.
 
.. _edge_setf:

Edge.setfunction(function_name) method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set the *function* attribute to the string *function_name*. The possible values are: *polynomial*, *legendre* or *chebyshev*.

.. _edge_seto:

Edge.setorder(new_order) method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set the *order* attribute to the integer value giving by the parameter *new_order*.

.. _edge_fitf:

Edge.fitfunction() method
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Fit the (x_array,y_array) using the current fitting function and order. The fitting coefficients are stored in the class attribute *coefficients* and the evaluator function in *evalfunction* attribute.

Usage
::

 # Instantiate an Edge object with the coordinates tuple corresponding to one edge.  
 ed = Edge(x_array, y_array)

 # Set the orientation of the edge to be vertical.
 ed.orientation = 90

 # Fit a function.
 ed.fitfunction()

 # print the resulting coefficients
 print ed.coefficients

 # print the difference between one point in the edge and the function value
 # at the point. 
 print ed.evalfunction(y_array[10]) - x_array[10]


