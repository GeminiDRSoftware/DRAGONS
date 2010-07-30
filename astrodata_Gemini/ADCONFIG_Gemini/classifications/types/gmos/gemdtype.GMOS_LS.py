class GMOS_LS(DataClassification):
    name="GMOS_LS"
    usage = ""
    parent = "GMOS_SPECT"
    requirement = AND( ISCLASS('GMOS_SPECT'),
                       PHU(OBSMODE='LONGSLIT',OBSTYPE='OBJECT') )

newtypes.append(GMOS_LS())
