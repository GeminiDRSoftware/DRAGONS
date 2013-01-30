class GSAOI_DARK(DataClassification):
    name="GSAOI_DARK"
    usage = """
        Applies to all dark datasets from the GSAOI instrument
        """
    parent = "GSAOI"
    requirement = ISCLASS("GSAOI") & PHU(OBSTYPE="DARK")

newtypes.append(GSAOI_DARK())
