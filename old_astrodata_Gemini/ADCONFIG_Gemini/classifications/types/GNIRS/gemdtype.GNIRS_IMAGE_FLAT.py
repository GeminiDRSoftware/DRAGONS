class GNIRS_IMAGE_FLAT(DataClassification):
    name="GNIRS_IMAGE_FLAT"
    usage = """
        Applies to all imaging flat datasets from the GNIRS instrument
        """
    parent = "GNIRS_IMAGE"
    requirement = AND([ ISCLASS("GNIRS_IMAGE"), PHU(OBSTYPE="FLAT") ])

newtypes.append(GNIRS_IMAGE_FLAT())
