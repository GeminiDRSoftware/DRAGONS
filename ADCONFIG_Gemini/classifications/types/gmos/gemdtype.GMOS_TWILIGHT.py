
class GMOS_TWILIGHT(DataClassification):
    name="GMOS_TWILIGHT"
    usage = ""
    parent = "GMOS_CAL"
    requirement = ISCLASS('GMOS') & PHU(OBJECT='Twilight')

newtypes.append(GMOS_TWILIGHT())
