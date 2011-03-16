class GMOS_IFU_FLAT(DataClassification):
    name="GMOS_IFU_FLAT"
    usage = ""
    parent = "GMOS_IFU"
    requirement = AND( ISCLASS(             'GMOS_SPECT'),
                       PHU(OBSMODE='IFU',OBSTYPE='FLAT' ),
                       NOT(ISCLASS( 'GMOS_IFU_TWILIGHT')) )
    
newtypes.append(GMOS_IFU_FLAT())
