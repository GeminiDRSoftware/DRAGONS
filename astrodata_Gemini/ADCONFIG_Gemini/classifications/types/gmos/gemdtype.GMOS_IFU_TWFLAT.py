class GMOS_IFU_TWFLAT(DataClassification):
    name="GMOS_IFU_TWFLAT"
    usage = ""
    parent = "GMOS_IFU"
    requirement = PHU( OBSMODE='IFU',OBSTYPE='FLAT',OBJECT='Twilight' )
    
newtypes.append(GMOS_IFU_TWFLAT())