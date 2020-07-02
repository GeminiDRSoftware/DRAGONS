import pathlib

from bokeh.server.server import Server
from jinja2 import Template


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

    # Setting num_procs here means we can't touch the IOLoop before now, we must
    # let Server handle that. If you need to explicitly handle IOLoops then you
    # will need to use the lower level BaseServer class.
    if not bokeh_server:
        bokeh_server = Server({'/': bkapp}, num_procs=1)
        bokeh_server.start()

    # Setting num_procs here means we can't touch the IOLoop before now, we must
    # let Server handle that. If you need to explicitly handle IOLoops then you
    # will need to use the lower level BaseServer class.

    bokeh_server.io_loop.add_callback(bokeh_server.show, "/")
    bokeh_server.io_loop.start()
