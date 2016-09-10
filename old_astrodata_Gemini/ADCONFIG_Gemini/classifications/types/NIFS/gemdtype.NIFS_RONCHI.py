
class NIFS_RONCHI(DataClassification):
    name="NIFS_RONCHI"
    usage = "Applies to NIFS Ronchi mask calibration observations"
    parent = "NIFS"
    requirement = AND(ISCLASS('NIFS'), PHU(OBSTYPE='FLAT'), PHU(APERTURE='Ronchi_Screen_G5615'))

newtypes.append(NIFS_RONCHI())
