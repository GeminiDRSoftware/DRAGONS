# Interactive Tools

#This is in flux while I rewrite it on fit_1D as the backend fitter

This is a guide to the interactive APIs and how to use them
in the primitives in the recipe system.  These tools depend
on bokeh and spin up an embedded webserver to provide the
user interface.

## Quick Start

### Identify Coordinate inputs, Fit inputs

Look over the primitive and see what inputs are used to
generate the initial set of coordinates to be fit.  This 
may be an expensive operation and we want to 
separate out the needed inputs accordingly in the UI.
These inputs will be UI widgets on the far left used
to generate the full set of coordinates/weights.

The logic to generate the coordinates from those
inputs will also be placed in a method so it can be 
used by the existing non-interactive code and by the new
interactive UI.

You'll also want to note the inputs that go into the fit.
These will be dynamically adjustable as the recalculation
should be fairly quick.

## Standard Approach

The normal approach in an interactive primitive is to
provide 2 code paths around where the fit is done.

The first is for an interactive
session.  In this case, it spins up the bokeh UI and the
user can interactively modify the fit or results to their
liking.  Once they hit a submit button, the fits are then
returned to the primitive and it carries on as normal.

The second path is non-interactive and the fit or results
are determined normally.

If the fit is being done with `fit_1D` then there is a
feature rich generic UI that you can use for the interactive
fit.  If you need to do something custom, you can subclass
`PrimitiveVisualizer`.  Both approaches are described in
more detail below.

## fit_1D Interactive Fitting



## Custom Interactive Fitting


## Modules

The top-level module is for the bokeh server and for custom UI elements and helpers.  It
has the base PrimitiveVisualizer class that you should subclass if you need a 
custom UI.

The `fit` module holds the custom `PrimitiveVisualizer` implementations.  This keeps
them from cluttering up the top level as we add more.

The `templates` folder holds HTML and styling information.

The `deprecated` folder holds code that is being retired as a result of a large
refactor.  I have kept it around for now in case it is useful for reference.

### Top Level

#### server

The server has two calls, `interactive_fitter` and `stop_server`.  The `interactive_fitter`
is called with a `PrimitiveVisualizer` instance and will run the UI interaction
in a browser until the UI is stopped.  `stop_server` is generally called by the
user clicking the Submit button or closing the browser tab.

### interactive

`interactive` has the abstract base class, `PrimitiveVisualizer`.  It also has some
helper methods for building slider/text box widgets and some code for displaying and
managing bands (range-selections with plot shading) and apertures (also with plot 
shading).

### controls

`controls` has code that is wired up to key presses in the browser.  This includes
a `Controller` that listens for key presses and passes control to various `Task`
managers.  For instance, one such `Task` is for bands and it allows a user to
use the mouse to add, resize, or delete the band selections.

The easiest way to make use of this logic is to use the helper function to
connect a `GIBandModel` and/or `GIApertureModel` to a bokeh `Figure` and `Div`.
The connected `Figure` will show the selected areas and the `Div` will show
help text for the currently active task (such as `BandTask`).

Connecting the models to a `Figure` is done with 
`connect_figure_extras(fig, aperture_model, band_model)`

To cause a figure to respond to key presses and connect a help `Div` area, 
you build a `Controller` like: `Controller(fig, aperture_model, band_model, helptext)`.
Note that `aperture_model` or `band_model` may be passed as None if it
is not relevant.

## fit

The `fit` module is for the implementations of visualizers and supporting code.
Right now, it includes a 1-D fitter for tracing apertures and an aperture
visualizer for finding apertures.

## templates

The `templates` folder has the HTML for displaying the UI and support for things
like a modal overlay message for long-run operations.

## deprecated

The `deprecated` folder holds some older prototype UI code that we want to
migrate away from.  It has been moved there to keep it clear to avoid it
as well as have it around for easy reference.

# Other

These are random notes on things that it is useful to be aware of.

The slider/textbox combinations in the `build_*` methods infer if they are 
working with `float` or `int` data based on the initial value.  If you
call the helper with a starting value of `1`, it will assume it should only
accept `int` types and will reject entries like `1.1`.

The `PrimitiveVisualizer` creates a `submit` button when it is constructed.
You can place this in a custom subclass UI to provde a button that will
stop the server (and cause a return to the calling primitive).

The `PrimitiveVisualizer` has a `do_later()` helper method.  This can post
a function onto the bokeh event loop to execute in the next pass.  This is
useful primarily if you are currently outside the event loop (such as 
another thread, or responding to the key press URL endpoint).  It is also
useful for long-running operations when you want the modal dialog to pop
up (described next).

The `PrimitiveVisualizer` has a `make_modal()` helper method.  This can
be used to tie a widget's `disabled` state to showing a modal message
over the whole interface.  For instance, if you connect it to a button
then when that button is disabled with `button.disabled=True`, the modal
will automatically show over the UI.  Then when the button is re-enabled
with `button.disabled=False`, the modal overlay will be removed.  For
this to work properly, the long-running work should be queued up via
the `do_later` method mentioned previously.  The pattern works like this:

```
self.make_modal(self.button, "<b>This may take 20 seconds")
# ...
self.button.disabled = True

def fn():
    # do something expensive
    self.button.disabled = False
self.do_later(fn)
```

The `PrimitiveVisualizer` has a `make_widgets_from_config` helper method.
For the passed list of parameter names, this will build a panel of
widgets to update the corresponding config values.  This provides a 
quick and easy way to make a UI for the config inputs to a primitive.
