class GMOS_IM_TWFLAT(DataClassification):
    name="GMOS_IM_TWFLAT"
    usage = ""
    parent = "GMOS_IMAGE"
    requirement = AND( ISCLASS( 'GMOS_IMAGE'),
                       PHU(OBJECT='Twilight') ) 

newtypes.append(GMOS_IM_TWFLAT())