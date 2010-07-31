class GMOS_MOS_FLAT(DataClassification):
    name="GMOS_MOS_FLAT"
    usage = ""
    parent = "GMOS_MOS"
    requirement = AND( ISCLASS(             'GMOS_SPECT'),
                       PHU(OBSMODE='MOS',OBSTYPE='FLAT' ),
                       NOT(ISCLASS( 'GMOS_MOS_TWILIGHT')) )
    
newtypes.append(GMOS_MOS_FLAT())
