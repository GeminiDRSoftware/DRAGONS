class GMOS_MOS_ARC(DataClassification):
    name="GMOS_MOS_ARC"
    usage = ""
    parent = "GMOS_MOS"
    requirement = PHU(OBSMODE='MOS',OBSTYPE='ARC')

newtypes.append(GMOS_MOS_ARC())