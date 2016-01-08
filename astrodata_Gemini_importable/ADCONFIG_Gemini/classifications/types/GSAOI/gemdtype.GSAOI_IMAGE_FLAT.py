class GSAOI_IMAGE_FLAT(DataClassification):
    name="GSAOI_IMAGE_FLAT"
    usage = """
        Applies to all imaging flat datasets from the GSAOI instrument
        """
    parent = "GSAOI_IMAGE"
    requirement = AND([  ISCLASS("GSAOI_IMAGE"),
                         OR([PHU(OBSTYPE="FLAT"),
                           PHU(OBJECT="(([Tt]wilight)+?)"),
                           PHU(OBJECT="(([Dd]omeflat)+?)")  ])  ])

newtypes.append(GSAOI_IMAGE_FLAT())
