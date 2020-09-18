import pathlib

from bokeh.server.server import Server
from jinja2 import Template

from geminidr.interactive import controls

__all__ = ["interactive_fitter", "stop_server"]

_bokeh_server = None
_visualizer = None


def _handle_key(doc):
    key = doc.session_context.request.arguments['key'][0].decode('utf-8')
    if controls.controller:
        controls.controller.handle_key(key)


def _bkapp(doc):
    """
    Callback for bokeh on server start.

    When the bokeh service starts, it is given this method as the root level
    handler.  When the user's browser opens, the `doc` for that page will be
    passed down through here.

    Parameters
    ----------
    doc
        Document reference for the user's browser tab to hold the user interface.

    Returns
    -------
    none
    """
    global _visualizer

    with open('%s/templates/index.html' % pathlib.Path(__file__).parent.absolute()) as f:
        t = Template(f.read())
        doc.template = t
    _visualizer.visualize(doc)


def set_visualizer(visualizer):
    """
    Set the visualizer for the UI.

    This sets the visualizer that will control the user interface when we start
    the bokeh server.  This is generally defined in the same file with the
    fit model and helper method.  For instance, see the `chebyshev1d` example.

    Parameters
    ----------
    visualizer : :class:`PrimitiveVisualizer`
    """
    global _visualizer
    _visualizer = visualizer


def start_server():
    """
    Start the bokeh server.

    This will begin an IO loop to handle user interaction via
    bokeh.  Until the server is explicitly stopped, the method
    will block and this call will not return.

    Returns
    -------
    none
    """
    global _bokeh_server

    if not _bokeh_server:
        _bokeh_server = Server({'/': _bkapp, '/handle_key': _handle_key}, num_procs=1)
        _bokeh_server.start()

    _bokeh_server.io_loop.add_callback(_bokeh_server.show, "/")
    _bokeh_server.io_loop.start()

    # The server normally stops when the user hits the Submit button in the
    # visualizer, or when they close the tab.


def stop_server():
    """
    Stop the bokeh server.

    This will end the IO look and unblock the `start_server()` call.
    This normally gets called when the user hits the submit button
    or closes the UI browser tab.
    """
    global _bokeh_server
    _bokeh_server.io_loop.stop()


def interactive_fitter(visualizer):
    """
    Start the interactive fitter with the given visualizer.

    This will spin up the bokeh server using the provided
    visualizer to build a UI.  It returns when the user
    submits the result.

    Parameters
    ----------
    visualizer : `~PrimitiveVisualizer`
        The visualizer UI to display
    """
    set_visualizer(visualizer)
    start_server()
    set_visualizer(None)
    return visualizer.user_satisfied
