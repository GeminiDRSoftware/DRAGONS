class GMOS_MOS(DataClassification):
    name="GMOS_MOS"
    usage = """
        Applies to all MOS datasets from the GMOS instruments
        """
    parent = "GMOS_SPECT"
    requirement = AND([  ISCLASS("GMOS_SPECT"),
                         PHU(MASKTYP="1"),
                         PHU({"{prohibit}MASKNAME": ".*arcsec"}),
                         PHU({"{prohibit}MASKNAME": "IFU*"}),
                         PHU({"{prohibit}MASKNAME": "focus*"})  ])

newtypes.append(GMOS_MOS())
