To-Do (Docs)
============

- [x] Write preamble about the structure of the interactive module
- [ ] Write a section about using the interactive module as a programmer
    - [ ] PrimitiveVisualizer class
    - [ ] PrimitiveVisualizer methods
    - [ ] PrimitiveVisualizer properties
    - [x] Adding components to a page
        - [x] sizing
        - [x] positioning
        - [x] styling
        - [x] Composing readable code with bokeh
    - [x] Adding callbacks to components
- [ ] link to main documentation

Table of Contents
=================

- [DRAGONS Interactive Module](#dragons-interactive-module)
  - [Module structure](#module-structure)
  - [Styling components with CSS](#styling-components-with-css)
  - [Adding components to a page](#adding-components-to-a-page)
    - [Sizing components](#sizing-components)
    - [Positioning components](#positioning-components)
    - [Readable Code with Bokeh](#readable-code-with-bokeh)
  - [Adding callbacks to components](#adding-callbacks-to-components)


DRAGONS Interactive Module
==========================

The interactive module is a collection of tools for creating interactive
wisualizations of data using a web browser. The module is built on top of
[Bokeh](http://bokeh.pydata.org/en/latest/), a python library for creating
interactive visualizations in a web browser. The interactive module provides a
set of tools for creating visualizations with Bokeh, and for adding
interactivity to those visualizations.

..
    The link to the main page below is important, but I don't think there's a
    user guide for the interactive mode yet (at least not one easy to find on
    the main page).

    TODO: Need to write a user guide for the interactive module, or link an
    existing one to the main page and update it.
The interactive module is designed to be used by programmers to create
interactive visualizations specifically for the primitives in the DRAGONS
framework. The interactive module is not designed to be used by end users.  If
you are an end user looking for documentation on how to use the interactive
module, please see the
[main documentation](https://dragons.readthedocs.io/en/latest/).

Module structure
----------------

.. |PrimitiveVisualizer| replace:: :class:`~geminidr.interactive.primitives.PrimitiveVisualizer`

The interactive module is managed primarily by the |PrimitiveVisualizer| class.
The |PrimitiveVisualizer| class is an abstract base class that defines the
interface for all visualizers. The |PrimitiveVisualizer| class is not intended
to be used directly, but rather to be subclassed by visualizers for specific
primitives.

..
    This should be a more specific link.
Visualizers for specific tasks can be found in `geminidr.interactive.fit`.
These visualizers are subclasses of |PrimitiveVisualizer| that are designed to
perform generic (e.g., 1-D fitting) or specific (e.g., trace an aperture)
tasks. A reference for all the visualizers can be found in the
[main documentation](https://dragons.readthedocs.io/en/latest/).

Individual visualizers are responsible for creating and managing the
layout and parameters of the visualization. The |PrimitiveVisualizer| class
has some methods for common types of components, such as buttons and sliders,
but the visualizers are free to use any components they wish. The visualizers
are also responsible for creating and managing the callbacks for the
visualization.


Styling components with CSS
---------------------------

.. |DragonsStyleManager| replace:: :class:`~geminidr.interactive.styles.DragonsStyleManager`
.. |dragons_styles| replace:: :func:`~geminidr.interactive.styles.dragons_styles`

The interactive module uses CSS to style the components in the visualizations.
The CSS is made available through the `geminidr.interactive.styles` module. It
contains a single class, |DragonsStyleManager|, which is responsible for
managing the CSS sheet available at any time.

Bokeh restricts models, which includes stylesheets, to a single instance per
document. |DragonsStyleManager| checks to see if the current document has not
changed since a style sheet instance was last created.

`geminidr.interactive.styles` also contains a helper function, |dragons_styles|
that can be used as shorthand for the style sheet format expected by bokeh
models. The two `Div`s below are equivalent:

.. code-block:: python

    from geminidr.interactive.styles import DragonsStyleManager
    from bokeh.models import Div

    first_div = Div(
        text="Hello World",
        style=DragonsStyleManager().stylesheets
    )

    second_div = Div(text="Hello World", style=dragons_styles())

This is provided to make adding a style sheet more succinct.


Adding components to a page
---------------------------

To add a bokeh component to a page, it's important to understand the
attributes expected by the bokeh object. The bokeh documentation is the best
place to find this information, but the interactive module deviates from the
bokeh documentation in a few ways.

Firstly, bokeh's support for global CSS stylesheets is limited, and so all
components are required to have a `stylesheet` attribute if the model allows
it. This is not always consistent throughout models, for example,

.. code-block:: python

    from bokeh.models import Div, TabPanel

    # Happily accepts the stylesheets attribute
    div = Div(text="Hello World", stylesheets=dragons_styles())

    # Does not accept the stylesheets attribute
    tabpanel = TabPanel(title="Hello World", stylesheets=dragons_styles())
    # > Raises AttributeError

This means you will need to check the bokeh documentation for the component
you are adding to see if it accepts a `stylesheet` attribute, or you need to
test it yourself should it not be specified. Generally, err on the side of 
assuming it does accept a stylesheet and debugging as needed.

Bokeh does have a `GlobalImportedStyleSheet` class that does not work with the 
current version of bokeh and DRAGONS. Instead, this method links the shadow
elements within a component to the stylesheet. This means that the stylesheet
is applied to the component, but it is not applied globally. This is a
limitation of bokeh, and *may change in the future*.

Secondly, bokeh includes support for a number of different ways to size and
and position components. Per bokeh's documentation, the `sizing_mode` and
`margin` attributes are the preferred way to size and position components.

Sizing components
~~~~~~~~~~~~~~~~~

The `sizing_mode` attribute is used to specify how a component should be sized
relative to the page. The `sizing_mode` attribute accepts a string or list of
strings that specify how the component should be sized. The strings are


+-----------------+---------------------------------------------------------+
| String          | Description                                             |
+=================+=========================================================+
| "fixed"         | The component should be sized to the size of the        |
|                 | component (provided width/height enforced)              |
+-----------------+---------------------------------------------------------+
| "stretch_both"  | The component should be stretched to fill the page      |
+-----------------+---------------------------------------------------------+
| "stretch_width" | The component should be stretched to fill the width of  |
|                 | the page                                                |
+-----------------+---------------------------------------------------------+
| "stretch_height"| The component should be stretched to fill the height of |
|                 | the page                                                |
+-----------------+---------------------------------------------------------+
| "scale_width"   | The component should be scaled to the width of the page |
+-----------------+---------------------------------------------------------+
| "scale_height"  | The component should be scaled to the height of the     |
|                 | page                                                    |
+-----------------+---------------------------------------------------------+
| "scale_both"    | The component should be scaled to the size of the page  |
+-----------------+---------------------------------------------------------+

Bokeh models also often have `width_policy` and `height_policy` attributes,
but these should be avoided in the interactive module. See 
[bokeh's reference](https://docs.bokeh.org/en/latest/docs/reference/models/layouts.html#bokeh.models.Column)
for more details on what these attributes do and when to use them.


Positioning components
~~~~~~~~~~~~~~~~~~~~~~

In a column or row, components are positioned in the order they are added to
the column or row. Aligning and positioning components can be done using
`Spacer` components, `grid` layouts, or `margin` attributes.

The `margin` attribute is used to specify the margins around a component. The
`margin` attribute accepts a tuple of four integers that specify the margins
in the order (top, right, bottom, left). The margins are specified in pixels.
Margins can be used to change where a component is positioned within its own
'box' on the page. 

If you are creating a widget, it is best practice to use margin to add a
static margin around the widget, and `Spacer` to add a dynamic margin between
widgets as part of the larger layout.

`grid` layouts are used to position components in a grid. The `grid` layout
accepts a list of lists of components, where each list of components is a row
in the grid. If you are aligning values in a component, this is likely the best
option.

Readable Code with Bokeh
~~~~~~~~~~~~~~~~~~~~~~~~

Bokeh's API breaks down creating web interfaces into a number of consistent
and clearly represented components. This makes it easy to create a web
interface, but can make writing readable code less straight forward than
other APIs.

In general, it is good to factor individual compoents into their own
variables. This makes it easier to read the code, and makes it easier to
re-use components of an individual widget. For example, the following code
creates a widget with a title, a slider, and a button:

.. code-block:: python

    from bokeh.models import Div, Slider, Button, Column

    title = Div(text="Hello World", style=dragons_styles())
    slider = Slider(start=0, end=10, value=5, step=1, title="Slider")
    button = Button(label="Button", button_type="success")

    widget = Column(title, slider, button)

This is much easier to read than the following code:

.. code-block:: python

    from bokeh.models import Div, Slider, Button, Column

    widget = Column(
        Div(text="Hello World", style=dragons_styles()),
        Slider(start=0, end=10, value=5, step=1, title="Slider"),
        Button(label="Button", button_type="success")
    )

The second example is more compact, but it is harder to read and harder to
read and accessing the components of the widget is less clear if layout
changes are needed.

Adding callbacks to components
------------------------------

Bokeh provides a number of ways to add callbacks to components. The interactive
module uses the `on_change` method to add callbacks to components. The
`on_change` method accepts a string specifying the attribute to watch for
changes, and a callback function. The callback function is called with three
arguments: the model that triggered the callback, the attribute that changed,
and the old value of the attribute.