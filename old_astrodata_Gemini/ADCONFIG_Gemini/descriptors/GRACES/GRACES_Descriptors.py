from GEMINI_Descriptors import GEMINI_DescriptorCalc

class GRACES_DescriptorCalc(GEMINI_DescriptorCalc):

    def __init__(self):
        GEMINI_DescriptorCalc.__init__(self)

    def ra(self, dataset, **args):
        return dataset.target_ra()

    def dec(self, dataset, **args):
        return dataset.target_dec()

    def central_wavelength(self, dataset, **args):
        return 0.7

    def disperser(self, dataset, **args):
        return "GRACES"

