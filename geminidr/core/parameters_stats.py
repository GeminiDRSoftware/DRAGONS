from gempy.library import config


class statsConfig(config.Config):
    prefix = config.Field('4-character header prefix', str, None, optional=True)

