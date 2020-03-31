import imexam
import webbrowser


class gingaViewer(imexam.connect):

    def __init__(self, *args, **kwargs):
        self._viewer = 'ginga'
        self.exam = imexam.imexamine.Imexamine()
        kwargs['exam'] = self.exam
        self.window = ginga(*args, **kwargs)
        # the viewer will track imexam with callbacks
        self._event_driven_exam = True

        self.currentpoint = None
        self.origin = 0
        self.color = "green"
        self.width = 1


class ginga(imexam.ginga_viewer.ginga):
    """Extend from imexam.ginga_viewer.ginga class to have a bit
    more control over what we do"""

    def __init__(self, *args, **kwargs):
        # Allow connection to an existing browser

        try:
            self.browser = webbrowser.get(using=kwargs.pop('browser'))
        except (KeyError, webbrowser.Error):
            self.browser = None

        super().__init__(*args, **kwargs)

    def _open_browser(self):
        if self.browser:
            self.browser.open(self.ginga_view.url)
        else:
            webbrowser.open_new_tab(self.ginga_view.url)
