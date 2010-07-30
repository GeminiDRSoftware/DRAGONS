class GMOS_IM_FLAT(DataClassification):
    name="GMOS_IM_FLAT"
    usage = ""
    parent = "GMOS_IMAGE"
    requirement = AND( ISCLASS('GMOS_IMAGE'),
                       PHU(OBSTYPE=  'FLAT') ) 

newtypes.append(GMOS_IM_FLAT())