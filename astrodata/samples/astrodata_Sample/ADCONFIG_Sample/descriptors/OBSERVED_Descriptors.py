class OBSERVED_DescriptorCalc:
    def observatory(self, dataset, **args):
        return dataset.phu_get_key_value("OBSERVAT")
