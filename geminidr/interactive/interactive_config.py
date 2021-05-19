# Adding a central location for any configurable
# options we may want.

from recipe_system.config import globalConf

# Hit if we want to show or hide the add aperture button
# It was decided to remove it in favor of keystrokes
# but at some point I may want it back for Jupyter
show_add_aperture_button = False

# Theme related settings
bokeh_theme = 'caliber'
bokeh_data_color = 'black'
bokeh_line_color = 'crimson'
bokeh_template_css = 'template_default.css'


try:
    cfg = globalConf['interactive']
    try:
        if cfg.theme in ['caliber', 'dark_minimal', 'light_minimal', 'night_sky', 'contrast']:
            bokeh_theme = cfg.theme
            if bokeh_theme == 'dark_minimal':
                bokeh_template_css = 'template_dark_minimal.css'
                bokeh_data_color = 'white'
    except AttributeError:
        pass
    try:
        bokeh_data_color = cfg.line_color
    except AttributeError:
        pass
except KeyError:
    # ok, no config section for us, use defaults
    pass


