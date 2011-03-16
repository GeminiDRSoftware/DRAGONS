class GMOS_IMAGE_TWILIGHT(DataClassification):
    name="GMOS_IMAGE_TWILIGHT"
    usage = ""
    parent = "GMOS_IMAGE"
    requirement = AND( ISCLASS( 'GMOS_IMAGE'),
                       PHU(OBJECT='Twilight') ) 

newtypes.append(GMOS_IMAGE_TWILIGHT())