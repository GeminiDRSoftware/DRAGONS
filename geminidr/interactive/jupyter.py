"""
Support for Jupyter with interactive widgets.

This has some Jupyter specific handling to allow us to use the interactive
tools within Jupyter.  This is work in progress with three modes currently
being investigated:

1) Jupyter using a full embedded bokeh server and browser windows for the normal experience
2) Jupyter using an embedded interface via pulling the Document, stuck in the first place the interface is shown
3) Jupyter using widgets and show() to allow it to be repositioned within each output cell as needed
"""
import yaml
from bokeh.themes import Theme

EMBEDDED_SERVER_MODE = "embedded_server_mode"
DOCUMENT_MODE = "document_mode"
PANEL_MODE = "panel_mode"


__all__ = ["using_jupyter", "start_server_jupyter", "jupyter_doc", "jupyter_mode", "stop_jupyter"]

jupyter_doc = None
jupyter_mode = EMBEDDED_SERVER_MODE
_jup_running = False


# Check if we are running in a Jupyter notebook and set using_jupyter accordingly
try:
    __IPYTHON__
    using_jupyter = True
except NameError:
    using_jupyter = False


def setup_jupyter(doc, fn=None):
    global jupyter_doc
    # if jupyter_doc is None:
    jupyter_doc = doc
    doc.theme = Theme(json=yaml.load("""
        attrs:
            Figure:
                background_fill_color: "#DDDDDD"
                outline_line_color: white
                toolbar_location: above
                height: 500
                width: 800
            Grid:
                grid_line_dash: [6, 4]
                grid_line_color: white
    """, Loader=yaml.FullLoader))
    if fn is not None:
        # trying threaded version
        def threaded_redux():
            import threading
            t = threading.Thread(name='jupyter_dragons_interactive_thread', target=fn)
            t.start()

        if not jupyter_mode == PANEL_MODE:
            doc.add_next_tick_callback(threaded_redux)


def start_server_jupyter():
    """
    Tell the interactive code we are starting interactivity in Jupyter

    This is analagous to :meth:`start_server` but it just blocks while
    polling for the interactive session to end.  This way, it can be
    used as a drop-in replacement for that call when running in a
    notebook.  This works best from a separate thread as seein in
    :meth:`jupyter_reduce`
    """
    global _jup_running
    _jup_running = True

    while _jup_running:
        import time
        time.sleep(2)


def jupyter_reduce(files=[], recipename=None, uparms=None, upload=[], mode=None):
    from recipe_system.reduction.coreReduce import Reduce
    from bokeh.io import output_notebook

    if jupyter_mode == PANEL_MODE:
        output_notebook()

    def stub_redux():
        redux = Reduce()
        redux.files = files
        redux.recipename= recipename
        if uparms:
            redux.uparms = uparms
        redux.upload = upload
        if mode:
            redux.mode = mode
        redux.runr()

    # uncomment output_notebook above and remove this, then uncomment the show if we want to go back to single cell
    def threaded_redux():
        import threading
        t = threading.Thread(name='jupyter_dragons_interactive_thread', target=stub_redux)
        t.start()
    if jupyter_mode == DOCUMENT_MODE:
        jupyter_doc.add_next_tick_callback(threaded_redux)
    else:
        stub_redux()
        # threaded_redux()

    # show(lambda x: setup_jupyter(x, stub_redux))


def stop_jupyter():
    global _jup_running
    _jup_running = False
