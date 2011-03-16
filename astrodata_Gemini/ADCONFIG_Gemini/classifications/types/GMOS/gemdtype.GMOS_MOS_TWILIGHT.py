class GMOS_MOS_TWILIGHT(DataClassification):
    name="GMOS_MOS_TWILIGHT"
    usage = ""
    parent = "GMOS_MOS"
    requirement = AND( ISCLASS(                                'GMOS_SPECT'),
                       PHU( OBSMODE='MOS',OBSTYPE='FLAT',OBJECT='Twilight' ) )
    
newtypes.append(GMOS_MOS_TWILIGHT())