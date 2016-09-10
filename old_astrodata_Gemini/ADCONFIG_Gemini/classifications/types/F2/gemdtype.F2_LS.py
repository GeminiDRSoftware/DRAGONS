class F2_LS(DataClassification):
    name="F2_LS"
    usage = """
        Applies to all longslit datasets from the FLAMINGOS-2 instrument
        """
    parent = "F2_SPECT"
    requirement = AND ([  ISCLASS("F2_SPECT"),
                          OR([  PHU(DECKER="Long_slit"),
                                PHU(DCKERPOS="Long_slit"),
                                PHU(MOSPOS=".?pix-slit")  ])  ])

newtypes.append(F2_LS())
 
