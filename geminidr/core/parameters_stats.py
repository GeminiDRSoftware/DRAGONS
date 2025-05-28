from gempy.library import config


class statsConfig(config.Config):
    pre = config.Field('4-character header prefix', str, None, optional=True)

