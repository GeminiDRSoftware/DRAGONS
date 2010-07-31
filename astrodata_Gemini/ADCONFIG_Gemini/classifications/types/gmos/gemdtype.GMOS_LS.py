class GMOS_LS(DataClassification):
    name="GMOS_LS"
    usage = ""
    parent = "GMOS_SPECT"
    requirement = OR(AND(ISCLASS('GMOS_SPECT'),
                         PHU(OBSMODE='LONGSLIT',OBSTYPE='OBJECT')),
                     ISCLASS(    'GMOS_LS_FLAT'),
                     ISCLASS('GMOS_LS_TWILIGHT'),
                     ISCLASS(     'GMOS_LS_ARC') )
newtypes.append(GMOS_LS())
