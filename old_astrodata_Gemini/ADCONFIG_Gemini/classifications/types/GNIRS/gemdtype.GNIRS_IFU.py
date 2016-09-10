class GNIRS_IFU(DataClassification):
    name="GNIRS_IFU"
    usage = """
        Applies to all integral field unit datasets from the GNIRS instruments
        """
    parent = "GNIRS_SPECT"
    requirement = AND([  ISCLASS("GNIRS_SPECT"),
                         PHU(SLIT="IFU")  ])

newtypes.append(GNIRS_IFU())
