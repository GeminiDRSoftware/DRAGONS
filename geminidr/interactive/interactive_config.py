from recipe_system.config import globalConf


bokeh_theme = 'caliber'
bokeh_data_color = 'black'
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
