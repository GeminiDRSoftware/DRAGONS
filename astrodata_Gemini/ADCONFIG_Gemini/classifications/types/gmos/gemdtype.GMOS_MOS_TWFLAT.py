class GMOS_MOS_TWFLAT(DataClassification):
    name="GMOS_MOS_TWFLAT"
    usage = ""
    parent = "GMOS_MOS"
    requirement = PHU( OBSMODE='MOS',OBSTYPE='FLAT',OBJECT='Twilight' )
    
newtypes.append(GMOS_MOS_TWFLAT())