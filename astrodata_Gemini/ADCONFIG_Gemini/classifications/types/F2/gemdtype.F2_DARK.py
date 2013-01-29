class F2_DARK(DataClassification):
    name="F2_DARK"
    usage = """
        Applies to all dark datasets from the FLAMINGOS-2 instrument
        """
    parent = "F2"
    requirement = ISCLASS("F2") & PHU(OBSTYPE="DARK")

newtypes.append(F2_DARK())
