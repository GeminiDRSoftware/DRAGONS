class OBSERVED_DescriptorCalc:
    def observatory(self, dataset, **args):
        return dataset.get_phu_key_value("OBSERVAT")

    def telescope(self, dataset, **args):
        return dataset.get_phu_key_value("TELESCOP")
