from GEMINI_Descriptors import GEMINI_DescriptorCalc

class GRACES_DescriptorCalc(GEMINI_DescriptorCalc):

    def __init__(self):
        GEMINI_DescriptorCalc.__init__(self)

    def ra(self, dataset, **args):
        return dataset.target_ra()

    def dec(self, dataset, **args):
        return dataset.target_dec()


