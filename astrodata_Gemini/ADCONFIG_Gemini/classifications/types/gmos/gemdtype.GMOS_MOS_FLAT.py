class GMOS_MOS_FLAT(DataClassification):
    name="GMOS_MOS_FLAT"
    usage = ""
    parent = "GMOS_MOS"
    requirement = PHU(OBSMODE='MOS',OBSTYPE='FLAT' )
    
newtypes.append(GMOS_MOS_FLAT())
