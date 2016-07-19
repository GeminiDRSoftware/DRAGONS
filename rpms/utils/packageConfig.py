from ConfigParser import SafeConfigParser as configparser

class PackageConfig(object):

    def __init__(self):
        self.descriptor_path  = None
        self.lookups_path     = None
        self.parameter_path   = None
        self.parameter_prefix = None
        self.primitive_path   = None
        self.primitive_prefix = None
        self.class_prefix     = None
        self.recipe_path      = None
        self.tag_path         = None

    def configure_pkg(self, configfile):
        """
        parameter configfile: Gemini package config file. 
        :type configfile: <str>, path to a package configuration, default 'pkg.cfg'

        :return: <void>

        """
        conf = configparser()
        conf.read(configfile)

        self.descriptor_path  = conf.get('descriptors', 'descriptor_path')
        self.lookups_path     = conf.get('lookups'    , 'lookups_path')
        self.parameter_path   = conf.get('parameters' , 'parameter_path')
        self.parameter_prefix = conf.get('parameters' , 'parameter_prefix')
        self.primitive_path   = conf.get('primitives' , 'primitive_path')
        self.primitive_prefix = conf.get('primitives' , 'primitive_prefix')
        self.class_prefix     = conf.get('primitives' , 'class_prefix')
        self.recipe_path      = conf.get('recipes'    , 'recipe_path')
        self.tag_path         = conf.get('tags'       , 'tag_path')

        return
