class GMOS_IFU_FLAT(DataClassification):
    name="GMOS_IFU_FLAT"
    usage = ""
    parent = "GMOS_IFU"
    requirement = PHU(OBSMODE='IFU',OBSTYPE='FLAT' )
    
newtypes.append(GMOS_IFU_FLAT())
