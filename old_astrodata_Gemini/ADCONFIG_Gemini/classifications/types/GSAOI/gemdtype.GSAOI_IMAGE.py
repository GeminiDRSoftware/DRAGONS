class GSAOI_IMAGE(DataClassification):
    name="GSAOI_IMAGE"
    usage = """
        Applies to all imaging datasets from the GSAOI instrument
        """
    parent = "GSAOI"
    requirement = ISCLASS("GSAOI")

newtypes.append(GSAOI_IMAGE())
