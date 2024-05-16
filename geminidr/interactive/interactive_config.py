# Adding a central location for any configurable
# options we may want.

from recipe_system.config import globalConf


__all__ = ["interactive_conf", "show_add_aperture_button"]


# Hit if we want to show or hide the add aperture button
# It was decided to remove it in favor of keystrokes
# but at some point I may want it back for Jupyter
show_add_aperture_button = False


# This is a singleton, Guido van Rossum’s implementation
class InteractiveConfig:
    """
    Singleton configuration data for interactive code.
    """
    def __new__(cls, *args, **kwds):
        it = cls.__dict__.get("__it__")
        if it is not None:
            return it
        cls.__it__ = it = object.__new__(cls)
        it.init(*args, **kwds)
        return it

    def init(self, *args, **kwargs):
        self.bokeh_theme = None
        self.bokeh_data_color = 'black'
        self.bokeh_line_color = 'crimson'
        self.bokeh_template_css = 'template_default.css'
        self.browser = None

        # Default port number for Bokeh server
        self.port_number = 5006

        try:
            cfg = globalConf['interactive']

            try:
                themes = [
                    'caliber',
                    'dark_minimal',
                    'light_minimal',
                    'night_sky',
                    'contrast'
                ]

                if cfg["theme"] in themes:
                    self.bokeh_theme = cfg["theme"]

                    if self.bokeh_theme == 'dark_minimal':
                        self.bokeh_template_css = 'template_dark_minimal.css'
                        self.bokeh_data_color = 'white'

            except KeyError:
                pass

            try:
                self.bokeh_data_color = cfg["line_color"]

            except KeyError:
                pass

            try:
                self.browser = cfg["browser"]

            except KeyError:
                pass

            try:
                self.port_number = int(cfg["port_number"])

            except KeyError:
                pass

        except KeyError:
            # ok, no config section for us, use defaults
            pass


def interactive_conf():
    """
    Get the interactive configuration singleton.

    Returns
    -------
    :class:`~geminidr.interactive.interactive_config.InteractiveConfig`
        Singleton interactive configuration
    """
    return InteractiveConfig()
