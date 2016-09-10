class GMOS_IFU(DataClassification):
    name="GMOS_IFU"
    usage = """
        Data taken in the IFU instrument mode with either GMOS instrument
        """
    parent = "GMOS_SPECT"
    requirement = AND([  ISCLASS("GMOS_SPECT"),
                         PHU(MASKTYP="-1"),
                         (PHU(MASKNAME="IFU*") | PHU(MASKNAME='g.ifu_slit._mdf'))  ])

newtypes.append(GMOS_IFU())
