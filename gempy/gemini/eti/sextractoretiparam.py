from gempy.eti_core.etiparam import ETIParam

PARAMETERS = "parameters"
__PARAMETERS_SET_INTERNALLY__ = ["FLAG_IMAGE",
                                 "CHECKIMAGE_NAME",
                                 "CATALOG_NAME"]

class SExtractorETIParam(ETIParam):
    def __init__(self, params):
        super(SExtractorETIParam, self).__init__(params=params)

    # Delete any parameters from the dict that need to be set interally
    def prepare(self):
        if self.params:
            for param in __PARAMETERS_SET_INTERNALLY__:
                try:
                    self.params.pop(param)
                except KeyError:
                    pass