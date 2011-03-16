class GMOS_LS_FLAT(DataClassification):
    name="GMOS_LS_FLAT"
    usage = ""
    parent = "GMOS_LS"
    requirement = AND( ISCLASS(                 'GMOS_SPECT'),
                       PHU(OBSMODE='LONGSLIT',OBSTYPE='FLAT'),
                       NOT(ISCLASS(      'GMOS_LS_TWILIGHT')) )
    
newtypes.append(GMOS_LS_FLAT())
