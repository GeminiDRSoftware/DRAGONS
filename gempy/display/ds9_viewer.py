import imexam
import os
import time
import warnings
from subprocess import Popen

from astropy.wcs import WCS

class ds9Viewer(imexam.connect):
    def __init__(self, *args, **kwargs):
        self._viewer = 'ds9'
        self.exam = imexam.imexamine.Imexamine()
        self.window = ds9(*args, **kwargs)
        self._event_driven_exam = False  # use the imexam loop

        self.currentpoint = None
        self.origin = 0
        self.color = "green"
        self.width = 1

    def _offset(self, origin, *args):
        """Convert coordinates to ds9 1-indexed form"""
        offset = 1 - (self.origin if origin is None else origin)
        return list(coord + offset for coord in args)

    # We define the display methods here because this is the object the
    # user creates and interacts with, even though almost everything is
    # sent to self.window

    def display_image(self, ext, attribute='data', title=None):
        """
        Displays an extension of an AD object in the current ds9 frame and
        provides the ds9 window with its WCS information.

        Parameters
        ----------
        ext: single-slice AstroData object
            the slice to display
        attribute: str ("data"/"mask"/"variance"/other)
            which attribute to display
        title: str/None
            string for annotation
        """
        data_to_display = getattr(ext, attribute)
        frame = self.window.frame()
        if frame is None:
            frame = 1
        self.view(data_to_display)
        header = WCS(ext.hdr).to_header()
        if title is None:
            title = "{}:{}[{}] {}".format(ext.filename, ext.hdr['EXTVER'],
                                          attribute, ext.phu['OBJECT'])
        header['OBJECT'] = title
        fname = "temp{}.wcs".format(time.time())
        with open(fname, "w") as f:
            f.write(header.tostring())
        self.window.set("wcs replace {} {}".format(frame, fname))
        os.remove(fname)

    def regions(self, str):
        """
        Simple interface to xpa

        Parameters
        ----------
        str: string
            regions command to execute
        """
        extras = ""
        if self.color != "green":
            extras += " color={}".format(self.color)
        if self.width != 1:
            extras += " width={}".format(self.width)
        if extras:
            str = str + " # " + extras
        self.window.set("regions command {{{}}}".format(str))

    def clear_regions(self):
        self.window.set("regions delete all")

    def line(self, x1=None, y1=None, x2=None, y2=None, origin=None):
        """Draw a line from (x1, y1) to (x2, y2)"""
        coords = self._offset(origin, x1, y1, x2, y2)
        self.regions("line {} {} {} {}".format(*coords))
        self.currentpoint = tuple(coords[-2:])

    def lineto(self, x=None, y=None, origin=None):
        """Draw a line from the currentpoint to (x, y)"""
        try:
            x1, y1 = self.currentpoint
        except TypeError:
            warnings.warn("No currentpoint set. Cannot use lineto")
        x, y = self._offset(origin, x, y)
        self.line(x1=x1, y1=y1, x2=x, y2=y)

    def rlineto(self, dx=None, dy=None):
        """Draw a line from the currentpoint to a point offset by (dx, dy)"""
        try:
            y1, x1 = self.currentpoint
        except TypeError:
            warnings.warn("No currentpoint set. Cannot use rlineto")
        self.line(x1=x1, y1=y1, x2=x1+dx, y2=y1+dy)

    def moveto(self, x=None, y=None, origin=None):
        """Move the currentpoint to (x,y)"""
        self.currentpoint = tuple(self._offset(origin, x, y))

    def polygon(self, points, closed=True, xfirst=False, origin=None):
        """Draw lines between consecutive points, possibly returning to the first one"""
        for i, point in enumerate(points):
            v1, v2 = self._offset(origin, *point)
            kwargs = {'x': v1, 'y': v2} if xfirst else {'x': v2, 'y': v1}
            if i == 0:
                self.moveto(**kwargs)
                initial_kwargs = kwargs
            else:
                self.lineto(**kwargs)
        if closed:
            self.lineto(**initial_kwargs)

class ds9(imexam.ds9_viewer.ds9):
    """
    An extension of the imexam.ds9_viewer class to allow us to name the
    viewer as we wish, and to sleep after creation, as required.
    """
    def __init__(self, target=None, path=None, wait_time=5,
                 quit_ds9_on_del=True):
        super().__init__(target=target, path=path, wait_time=wait_time,
                         quit_ds9_on_del=quit_ds9_on_del)
        # No need to sleep if we're connceting to an existing ds9
        if target is None:
            time.sleep(2)

    def run_inet_ds9(self):
        """start a new ds9 window using an inet socket connection.

        Notes
        -----
        It is given a unique title so it can be identified later. I have
        copied this method so we can give the window a nice DRAGONS-y name.
        """
        env = os.environ

        existing_ds9_viewers = [v[0] for v in imexam.list_active_ds9(False).values()]
        xpaname = 'DRAGONS'
        n = 1
        while xpaname in existing_ds9_viewers:
            n += 1
            xpaname = 'DRAGONS_{}'.format(n)
        try:
            p = Popen([self._ds9_path,
                       "-xpa", "inet",
                       "-title", xpaname],
                      shell=False, env=env)
            self._ds9_process = p
            self._process_list.append(p)
            self._need_to_purge = False
            #time.sleep(2)
            return xpaname

        except Exception as e:  # refine error class
            warnings.warn("Opening ds9 failed")
            print("Exception: {}".format(repr(e)))
            from signal import SIGTERM
            try:
                pidtokill = p.pid
            except NameError:
                # in case p failed at the initialization level
                pidtokill = None
            if pidtokill is not None:
                os.kill(pidtokill, SIGTERM)
            raise e
