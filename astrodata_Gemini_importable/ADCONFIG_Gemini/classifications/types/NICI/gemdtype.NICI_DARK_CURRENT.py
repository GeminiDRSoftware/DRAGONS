
class NICI_DARK_CURRENT(DataClassification):
    name="NICI_DARK_CURRENT"
    usage = "Applies to current dark current calibrations for the NICI instrument."
    parent = "NICI_DARK"
    requirement = ISCLASS('NICI') & PHU(OBSTYPE='DARK')

newtypes.append(NICI_DARK_CURRENT())
