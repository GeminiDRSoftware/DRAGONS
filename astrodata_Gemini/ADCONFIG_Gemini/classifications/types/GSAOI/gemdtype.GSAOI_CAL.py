class GSAOI_CAL(DataClassification):
    name="GSAOI_CAL"
    usage = """
        Applies to all calibration datasets from the GSAOI instrument
        """
    parent = "GSAOI"
    requirement = AND([  ISCLASS("GSAOI"),
                         OR([  ISCLASS("GSAOI_IMAGE_FLAT"),
                               ISCLASS("GSAOI_IMAGE_DOMEFLAT"),
                               ISCLASS("GSAOI_IMAGE_TWILIGHT"),
                               ISCLASS("GSAOI_DARK")  ])  ])

newtypes.append(GSAOI_CAL())
