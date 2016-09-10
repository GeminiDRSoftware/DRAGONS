class GNIRS_DARK(DataClassification):
    name = "GNIRS_DARK"
    usage = """
        Applies to all dark datasets from GNIRS.
        """
    parent = "GNIRS"
    requirement = ISCLASS("GNIRS") & PHU(OBSTYPE="DARK")

newtypes.append(GNIRS_DARK())