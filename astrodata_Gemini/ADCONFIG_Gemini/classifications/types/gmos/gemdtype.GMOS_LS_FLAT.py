class GMOS_LS_FLAT(DataClassification):
    name="GMOS_LS_FLAT"
    usage = ""
    parent = "GMOS_LS"
    requirement = PHU(OBSMODE='LONGSLIT',OBSTYPE='FLAT' )
    
newtypes.append(GMOS_LS_FLAT())
