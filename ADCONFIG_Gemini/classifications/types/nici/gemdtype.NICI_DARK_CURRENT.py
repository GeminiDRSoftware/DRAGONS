
class NICI_DARK_CURRENT(DataClassification):
    name="NICI_DARK_CURRENT"
    usage = ""
    requirement = ISCLASS('NICI') & PHU(OBSTYPE='DARK')

newtypes.append(NICI_DARK_CURRENT())
