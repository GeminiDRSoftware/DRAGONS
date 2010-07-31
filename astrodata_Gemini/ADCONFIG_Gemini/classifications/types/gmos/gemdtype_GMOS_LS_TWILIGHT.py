class GMOS_LS_TWILIGHT(DataClassification):
    name="GMOS_LS_TWILIGHT"
    usage = ""
    parent = "GMOS_LS"
    requirement = PHU( OBSMODE='LONGSLIT',OBSTYPE='FLAT',OBJECT='Twilight' )
    
newtypes.append(GMOS_LS_TWILIGHT())