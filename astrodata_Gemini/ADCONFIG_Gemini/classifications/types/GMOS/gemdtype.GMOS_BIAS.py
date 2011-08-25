class GMOS_BIAS(DataClassification):
    name="GMOS_BIAS"
    usage = """
        Applies to all dark datasets from the GMOS instruments
        """
    parent = "GMOS_IMAGE"
    requirement = PHU(OBSTYPE="BIAS")

newtypes.append(GMOS_BIAS())
