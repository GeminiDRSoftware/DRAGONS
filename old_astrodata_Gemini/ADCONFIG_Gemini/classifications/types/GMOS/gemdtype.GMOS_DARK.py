class GMOS_DARK(DataClassification):
    name="GMOS_DARK"
    usage = """
        Applies to all dark datasets from the GMOS instruments
        """
    parent = "GMOS"
    requirement = ISCLASS("GMOS") & PHU(OBSTYPE="DARK")

newtypes.append(GMOS_DARK())
