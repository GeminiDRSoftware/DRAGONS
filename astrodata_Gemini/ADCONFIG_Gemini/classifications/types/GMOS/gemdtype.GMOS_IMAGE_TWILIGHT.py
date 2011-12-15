class GMOS_IMAGE_TWILIGHT(DataClassification):
    name="GMOS_IMAGE_TWILIGHT"
    usage = """
        Applies to all imaging twilight flat datasets from the GMOS instruments
        """
    parent = "GMOS_IMAGE_FLAT"
    requirement = AND([  ISCLASS("GMOS_IMAGE_FLAT"),
                         OR([PHU(OBJECT="Twilight"),
                             PHU(OBJECT="twilight")]) ])

newtypes.append(GMOS_IMAGE_TWILIGHT())
