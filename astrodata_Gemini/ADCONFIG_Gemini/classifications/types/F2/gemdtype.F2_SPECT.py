class F2_SPECT(DataClassification):
    name="F2_SPECT"
    # this a description of the intent of the classification
    # to what does the classification apply?
    usage = """
        Applies to all spectroscopic datasets from the FLAMINGOS-2 instrument
        """
    parent = "F2"
    requirement = ISCLASS("F2") & OR([  PHU(DCKERPOS="Long_slit"),
                                        PHU(MOSPOS=".?pix-slit"),
                                        PHU(DCKERPOS="mos"),
                                        PHU(MOSPOS="mos.?")  ])

newtypes.append(F2_SPECT())
