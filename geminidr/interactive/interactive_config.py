# Adding a central location for any configurable
# options we may want.

from recipe_system.config import globalConf

# Hit if we want to show or hide the add aperture button
# It was decided to remove it in favor of keystrokes
# but at some point I may want it back for Jupyter
show_add_aperture_button = False


class InteractiveConfig:
    def __init__(self, theme=None):
        self.bokeh_theme = 'caliber'
        self.bokeh_data_color = 'black'
        self.bokeh_line_color = 'crimson'
        self.bokeh_template_css = 'template_default.css'
        self.browser = None

        try:
            cfg = globalConf['interactive']
            try:
                if cfg["theme"] in ['caliber', 'dark_minimal', 'light_minimal', 'night_sky', 'contrast']:
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
        except KeyError:
            # ok, no config section for us, use defaults
            pass


def interactiveConf():
    return InteractiveConfig()
