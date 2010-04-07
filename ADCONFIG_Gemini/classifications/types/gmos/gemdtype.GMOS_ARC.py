
class GMOS_ARC(DataClassification):
    name="GMOS_ARC"
    usage = ""
    parent = "GMOS_CAL"
    requirement = ISCLASS('GMOS') & PHU(OBSTYPE='ARC')

newtypes.append(GMOS_ARC())
