class F2_DARK(DataClassification):
    name="F2_DARK"
    usage = """
        Applies to all dark datasets from the FLAMINGOS-2 instrument
        """
    parent = "F2"
    requirement = ISCLASS("F2") & OR([  PHU(OBSTYPE="DARK"),
                                        PHU(FILTER1="DK.?"),
                                        PHU(FILTER2="DK.?")  ])

newtypes.append(F2_DARK())
