class GMOS_LS(DataClassification):
    name="GMOS_LS"
    usage = ""
    requirement = ISCLASS('GMOS') & PHU(OBSMODE='LONGSLIT',
                                        OBSTYPE='OBJECT')

newtypes.append(GMOS_LS())
