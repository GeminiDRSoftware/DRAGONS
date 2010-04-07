class GMOS_FLAT(DataClassification):
    name="GMOS_FLAT"
    usage = ""
    parent = "GMOS_CAL"
    requirement = ISCLASS('GMOS') & PHU(OBSTYPE='FLAT')

newtypes.append(GMOS_FLAT())
