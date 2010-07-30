class GMOS_LS_TWFLAT(DataClassification):
    name="GMOS_LS_TWFLAT"
    usage = ""
    parent = "GMOS_LS"
    requirement = PHU( OBSMODE='LONGSLIT',OBSTYPE='FLAT',OBJECT='Twilight' )
    
newtypes.append(GMOS_LS_TWFLAT())