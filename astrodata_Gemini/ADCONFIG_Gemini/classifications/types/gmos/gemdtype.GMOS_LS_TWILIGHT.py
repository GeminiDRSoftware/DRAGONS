class GMOS_LS_TWILIGHT(DataClassification):
    name="GMOS_LS_TWILIGHT"
    usage = ""
    parent = "GMOS_LS"
    requirement = AND( ISCLASS(                                     'GMOS_SPECT'),
                       PHU( OBSMODE='LONGSLIT',OBSTYPE='FLAT',OBJECT='Twilight' ) )
    
newtypes.append(GMOS_LS_TWILIGHT())