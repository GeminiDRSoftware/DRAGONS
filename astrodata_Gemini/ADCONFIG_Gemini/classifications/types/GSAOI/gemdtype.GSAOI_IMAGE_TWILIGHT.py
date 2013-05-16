class GSAOI_IMAGE_TWILIGHT(DataClassification):
    name="GSAOI_IMAGE_TWILIGHT"
    usage = """
        Applies to all imaging twilight flat datasets from the GSAOI
        instrument
        """
    parent = "GSAOI_IMAGE_FLAT"
    requirement = AND([  ISCLASS("GSAOI_IMAGE"),
                         PHU(OBJECT="(([Tt]wilight)+?)")  ])

newtypes.append(GSAOI_IMAGE_TWILIGHT())
