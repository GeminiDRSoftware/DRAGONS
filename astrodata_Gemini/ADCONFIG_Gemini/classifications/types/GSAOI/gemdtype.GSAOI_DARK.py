class GSAOI_DARK(DataClassification):
    name="GSAOI_DARK"
    usage = """
        Applies to all dark datasets from the GSAOI instrument
        """
    parent = "GSAOI_IMAGE"
    requirement = AND([  ISCLASS("GSAOI_IMAGE"),
                         PHU(OBSTYPE="DARK")  ])

newtypes.append(GSAOI_DARK())
