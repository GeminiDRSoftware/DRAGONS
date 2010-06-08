
class GMOS_DARK(DataClassification):
    name="GMOS_DARK"
    usage = ""
    parent = "GMOS_CAL"
    requirement = ISCLASS('GMOS') & PHU( OBSTYPE = 'DARK')

newtypes.append(GMOS_DARK())
