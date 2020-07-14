import pathlib

from bokeh.server.server import Server
from jinja2 import Template

from geminidr.interactive import controls


def handle_key(doc):
    key = doc.session_context.request.arguments['key'][0].decode('utf-8')
    if controls.controller:
        controls.controller.handle_key(key)


def bkapp(doc):
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
    global visualizer

    with open('%s/templates/index.html' % pathlib.Path(__file__).parent.absolute()) as f:
        t = Template(f.read())
        doc.template = t
    visualizer.visualize(doc)


bokeh_server = None
visualizer = None


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
    global bokeh_server

    if not bokeh_server:
        bokeh_server = Server({'/': bkapp, '/handle_key': handle_key}, num_procs=1)
        bokeh_server.start()

    bokeh_server.io_loop.add_callback(bokeh_server.show, "/")
    bokeh_server.io_loop.start()

    # The server normally stops when the user hits the Submit button in the
    # visualizer, or when the close the tab.
