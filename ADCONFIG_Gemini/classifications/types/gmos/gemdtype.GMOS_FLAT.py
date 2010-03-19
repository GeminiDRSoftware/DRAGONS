
class GMOS_FLAT(DataClassification):
    name="GMOS_FLAT"
    usage = ""
    requirement = ISCLASS('GMOS') & PHU(OBSTYPE='FLAT')

newtypes.append(GMOS_FLAT())
