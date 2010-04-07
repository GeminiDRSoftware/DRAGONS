class GMOS_LS(DataClassification):
    name="GMOS_LS"
    usage = ""
    parent = "GMOS_SPECT"
    requirement = ISCLASS('GMOS') & PHU(OBSMODE='LONGSLIT',
                                        OBSTYPE='OBJECT')

newtypes.append(GMOS_LS())
