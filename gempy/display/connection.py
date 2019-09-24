# gempy.display.connection.py
#
# Creates a class that provides the standard imexam methods, plus the
# additional annotation methods we require. We aim to provide a uniform
# interface for both ds9 and ginga.
#
# There's a fair bit of cut-and-paste from the imexam.Connect().__init__
# because we're providing different parameters but need the underlying
# code and logic.

import imexam

try:
    from .ginga_viewer import gingaViewer
    have_ginga = True
except ImportError:
    have_ginga = False

try:
    import xpa
    have_xpa = True
    from .ds9_viewer import ds9Viewer
except ImportError:
    have_xpa = False


class Connect(object):
    """
    Connect to a display device to look at and examine images.
    The control features below are a basic set that should be available
    in all display tools.
    The class for the display tool should override them and add it's own
    extra features.

    Parameters
    ----------
    viewer: string, optional
        The name of the image viewer you want to use, DS9 is the default
    path : string, optional
        absolute path to the viewers executable
    wait_time: int, optional
        The time to wait for a connection to be eastablished before quitting

    Attributes
    ----------
    window: a pointer to an object
        controls the viewers functions
    imexam: a pointer to an object
        controls the imexamine functions and options
    """
    def __new__(cls, viewer='ds9', path=None, use_existing=False,
                use_dragons=False, wait_time=10, quit_window=True,
                port=None, browser=None):

        _possible_viewers = []
        if have_xpa:
            _possible_viewers.append("ds9")
        if have_ginga:
            _possible_viewers.append('ginga')

        vwr = viewer.lower()
        if vwr not in _possible_viewers or len(_possible_viewers) == 0:
            raise NotImplementedError("Unsupported viewer, check your "
                                      "installed packages")

        # Here's the DRAGONS-specific stuff
        if 'ds9' in vwr:
            target = None
            existing_ds9_viewers = [v[0] for v in imexam.list_active_ds9(False).values()]
            if use_dragons:
                if 'DRAGONS' in existing_ds9_viewers:
                    target = 'DRAGONS'
            elif use_existing and len(existing_ds9_viewers):
                target = existing_ds9_viewers[0]

            instance = ds9Viewer(target=target, path=path,
                                 wait_time=wait_time,
                                 quit_ds9_on_del=quit_window)

        else:  # Must be ginga if we've got this far
            instance = gingaViewer(port=port, close_on_del=quit_window,
                                   browser=browser)

        instance.logfile = 'imexam_log.txt'  # default logfile name
        instance.log = imexam.util.set_logging()  # points to the package logger
        instance._current_slice = None
        instance._current_frame = None

        return instance
