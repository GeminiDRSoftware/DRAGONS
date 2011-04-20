class GMOS_LS(DataClassification):
    name="GMOS_LS"
    usage = """
        Applies to all longslit datasets from the GMOS instruments
        """
    parent = "GMOS_SPECT"
    requirement = AND([  ISCLASS("GMOS_SPECT"),
                         PHU(MASKTYP="1"),
                         PHU(MASKNAME=".*arcsec")  ])

newtypes.append(GMOS_LS())
