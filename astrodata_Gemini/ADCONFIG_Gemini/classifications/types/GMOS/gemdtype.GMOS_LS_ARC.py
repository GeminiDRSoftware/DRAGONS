class GMOS_LS_ARC(DataClassification):
    name="GMOS_LS_ARC"
    usage = ""
    parent = "GMOS_LS"
    requirement = AND( ISCLASS(                'GMOS_SPECT'),
                       PHU(OBSMODE='LONGSLIT',OBSTYPE='ARC') )

newtypes.append(GMOS_LS_ARC())