
class GNIRS_PINHOLE(DataClassification):
    name="GNIRS_PINHOLE"
    usage = "Applies to GNIRS Pinhole mask calibration observations"
    parent = "GNIRS"
    requirement = AND(ISCLASS('GNIRS'), PHU(OBSTYPE='FLAT'), OR( PHU(SLIT='LgPinholes_G5530'), PHU(SLIT='SmPinholes_G5530') ))

newtypes.append(GNIRS_PINHOLE())
