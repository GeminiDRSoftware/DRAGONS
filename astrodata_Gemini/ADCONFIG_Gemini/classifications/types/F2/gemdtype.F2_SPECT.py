class F2_SPECT(DataClassification):
    name="F2_SPECT"
    usage = """
        Applies to all spectroscopic datasets from the FLAMINGOS-2 instrument
        """
    parent = "F2"
    requirement = AND([  ISCLASS("F2"),
                         OR([  PHU(GRISMPOS="JH.?"),
                               PHU(GRISMPOS="HK.?"),
                               PHU(GRISMPOS="R3K.?"),
                               PHU(GRISM="JH.?"),
                               PHU(GRISM="HK.?"),
                               PHU(GRISM="R3K.?"),  ]),
                         NOT(ISCLASS("F2_DARK"))  ])

newtypes.append(F2_SPECT())
