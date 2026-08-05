from geminidr.core import parameters_standardize


class validateDataConfig(parameters_standardize.validateDataConfig):
    def setDefaults(self):
        self.require_wcs = False
