class GSAOI_IMAGE_TWILIGHT(DataClassification):
    name="GSAOI_IMAGE_TWILIGHT"
    usage = """
        Applies to all imaging twilight flat datasets from the GSAOI
        instrument
        """
    parent = "GSAOI_IMAGE_FLAT"
    requirement = AND([  ISCLASS("GSAOI_IMAGE_FLAT"),
                         OR([  PHU(OBJECT="Twilight"),
                               PHU(OBJECT="twilight")  ])  ])

newtypes.append(GSAOI_IMAGE_TWILIGHT())
