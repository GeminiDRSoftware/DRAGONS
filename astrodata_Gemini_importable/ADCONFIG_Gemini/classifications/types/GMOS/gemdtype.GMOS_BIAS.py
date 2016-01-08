class GMOS_BIAS(DataClassification):
    name="GMOS_BIAS"
    usage = """
        Applies to all bias datasets from the GMOS instruments
        """
    parent = "GMOS"
    requirement = ISCLASS("GMOS") & PHU(OBSTYPE="BIAS")

newtypes.append(GMOS_BIAS())
