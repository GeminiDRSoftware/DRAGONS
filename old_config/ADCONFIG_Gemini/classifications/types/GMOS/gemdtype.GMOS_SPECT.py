class GMOS_SPECT(DataClassification):
    name="GMOS_SPECT"
    # this a description of the intent of the classification
    # to what does the classification apply?
    usage = """
        Applies to all spectroscopic datasets from the GMOS instruments
        """
    parent = "GMOS"
    requirement = AND([  ISCLASS("GMOS"),
                         PHU({"{prohibit}MASKTYP": "0"}),
                         PHU({"{prohibit}MASKNAME": "None"}),
                         PHU({"{prohibit}GRATING": "MIRROR"}),
                         NOT(ISCLASS("GMOS_BIAS"))  ])

newtypes.append(GMOS_SPECT())
