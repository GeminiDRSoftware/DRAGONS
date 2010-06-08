
class GMOS_TWILIGHT(DataClassification):
    name="GMOS_TWILIGHT"
    usage = ""
    parent = "GMOS_FLAT"
    requirement = ISCLASS('GMOS') & PHU(OBJECT='Twilight')

newtypes.append(GMOS_TWILIGHT())
