class GMOS_IFU_ARC(DataClassification):
    name="GMOS_IFU_ARC"
    usage = ""
    parent = "GMOS_IFU"
    requirement = AND( ISCLASS(           'GMOS_SPECT'),
                       PHU(OBSMODE='IFU',OBSTYPE='ARC') )

newtypes.append(GMOS_IFU_ARC())