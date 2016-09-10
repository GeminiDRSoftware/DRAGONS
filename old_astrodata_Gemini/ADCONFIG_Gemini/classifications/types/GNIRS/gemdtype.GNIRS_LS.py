class GNIRS_LS(DataClassification):
    name="GNIRS_LS"
    usage = """
        Applies to all longslit datasets from the GNIRS instruments
        """
    parent = "GNIRS_SPECT"
    requirement = AND([  ISCLASS("GNIRS_SPECT"),
                         PHU(SLIT=".*arcsec.*")  ])

newtypes.append(GNIRS_LS())
