class MOS(DataClassification):
    name="MOS"
    usage = """
        Applies to all Gemini multi object spectroscopy datasets
        """
    parent = "SPECT"
    requirement = OR([  ISCLASS("F2_MOS"),
                        ISCLASS("GMOS_MOS")
                     ])

newtypes.append(MOS())
