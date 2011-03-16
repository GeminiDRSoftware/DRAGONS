class GMOS_IFU_TWILIGHT(DataClassification):
    name="GMOS_IFU_TWILIGHT"
    usage = ""
    parent = "GMOS_IFU"
    requirement = AND( ISCLASS(                                'GMOS_SPECT'),
                       PHU( OBSMODE='IFU',OBSTYPE='FLAT',OBJECT='Twilight' ) )
    
newtypes.append(GMOS_IFU_TWILIGHT())