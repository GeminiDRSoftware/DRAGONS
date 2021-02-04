import uuid

from bokeh.io import curdoc

from astrodata import version

import pathlib

import tornado
from bokeh.application import Application
from bokeh.application.handlers import Handler
from bokeh.server.server import Server
from jinja2 import Template, Environment, FileSystemLoader

from geminidr.interactive import controls

__all__ = ["interactive_fitter", "stop_server"]

_bokeh_server = None
_visualizer = None


__version__ = version()


class VersionHandler(tornado.web.RequestHandler):
    """
    Handler for geting the DRAGONS version.
    """
    def get(self):
        self.write(__version__)


def _handle_key(doc):
    """
    Web endpoint for telling the bokeh python a key was pressed.

    Parameters
    ----------
    doc : :class:`~bokeh.document.Document`
        Document reference for the user's browser tab to hold the user interface.

    Returns
    -------
    none
    """
    key = doc.session_context.request.arguments['key'][0].decode('utf-8')
    if controls.controller:
        controls.controller.handle_key(key)


# to store the mapping from ID to callable for the registered web callbacks
_callbacks = dict()


def register_callback(fn):
    """
    Register a function with the `callback` web endpoint.

    This will register a function to respond to calls to the `callback`
    web endpoint.  It will assign the function a unique UUID and return it
    to the caller as a string.  This ID should be passed in the web calls
    as the query argument `callback`.

    Parameters
    ----------
    fn : callable
        Function to call when callback is accessed via the URL.  It will be passed the web call arguments.

    Returns
    -------
    str : Unique ID for the callback, to be passed in the URL
    """
    name = str(uuid.uuid1())
    _callbacks[name] = fn
    return name


def _handle_callback(doc):
    """
    Web handler for 'callback' urls.

    This handler is for functions registered as callbacks for the javascript.
    This allows us to write javascript logic that calls back down into the python
    via this endpoint.

    Parameters
    ----------
    doc : :class:`~bokeh.document.Document`
        Document reference for the user's browser tab to hold the user interface.

    Returns
    -------
    none
    """
    cb = doc.session_context.request.arguments['callback'][0].decode('utf-8')
    args = doc.session_context.request.arguments
    callback = _callbacks.get(cb)
    if callback is not None:
        callback(args)


def _bkapp(doc):
    """
    Callback for bokeh on server start.

    When the bokeh service starts, it is given this method as the root level
    handler.  When the user's browser opens, the `doc` for that page will be
    passed down through here.

    Parameters
    ----------
    doc : :class:`~bokeh.document.Document`
        Document reference for the user's browser tab to hold the user interface.

    Returns
    -------
    none
    """
    global _visualizer

    template = "index.html"
    if _visualizer.template:
        template = _visualizer.template
    template_path = '%s/templates/' % pathlib.Path(__file__).parent.absolute()
    with open('%s/templates/%s' % (pathlib.Path(__file__).parent.absolute(), template)) as f:
        # Because Bokeh has broken templating...
        title = _visualizer.title
        if not title:
            title = 'Interactive'
        primitive_name = _visualizer.primitive_name
        template = f.read()
        # template = template.replace('{{ title }}', title.replace(' ', '&nbsp;')) \
        #                    .replace('{{ primitive_name }}', primitive_name.replace(' ', '&nbsp;'))
        t = Environment(loader=FileSystemLoader(template_path)).from_string(template)
        doc.template = t
        doc.template_variables['primitive_title'] = title.replace(' ', '&nbsp;')
        doc.template_variables['primitive_name'] = primitive_name.replace(' ', '&nbsp;')
    _visualizer.visualize(doc)
    doc.title = title


def set_visualizer(visualizer):
    """
    Set the visualizer for the UI.

    This sets the visualizer that will control the user interface when we start
    the bokeh server.  This is generally defined under `fit` in the same file with the
    fit model and helper method.  For instance, see the `~geminidr.interactive.fit.fit1d.Fit1DVisualizer`.

    Parameters
    ----------
    visualizer : :class:`~geminidr.interactive.interactive.PrimitiveVisualizer`
    """
    global _visualizer
    _visualizer = visualizer


class DRAGONSStaticHandler(Handler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._static = '%s/static' % pathlib.Path(__file__).parent.absolute()


def start_server():
    """
    Start the bokeh server.

    This will begin an IO loop to handle user interaction via
    bokeh.  Until the server is explicitly stopped, the method
    will block and this call will not return.
    """
    global _bokeh_server

    if not _bokeh_server:
        static_app = Application(DRAGONSStaticHandler())
        port = 5006
        while port < 5701 and _bokeh_server is None:
            try:
                _bokeh_server = Server({'/': _bkapp, '/dragons': static_app, '/handle_key': _handle_key,
                                        '/handle_callback': _handle_callback,
                                        }, num_procs=1, extra_patterns=[('/version', VersionHandler),], port=port)
            except OSError:
                port = port+1
                if port >= 5701:
                    raise
        _bokeh_server.start()

    _bokeh_server.io_loop.add_callback(_bokeh_server.show, "/")
    _bokeh_server.io_loop.start()

    # The server normally stops when the user hits the Submit button in the
    # visualizer, or when they close the tab.


def stop_server():
    """
    Stop the bokeh server.

    This will end the IO loop and unblock the :meth:`~geminidr.interactive.server.start_server()` call.
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
    visualizer : :class:`~geminidr.interactive.interactive.PrimitiveVisualizer`
        The visualizer UI to display
    """
    set_visualizer(visualizer)
    start_server()
    set_visualizer(None)
    return visualizer.user_satisfied
