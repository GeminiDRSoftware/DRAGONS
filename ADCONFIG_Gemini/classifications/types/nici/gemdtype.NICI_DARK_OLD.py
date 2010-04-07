
class NICI_DARK_OLD(DataClassification):
    name="NICI_DARK_OLD"
    usage = "Applies to OLD NICI dark current calibration datasets."
    parent = "NICI_DARK"
    requirement = ISCLASS('NICI') & PHU(OBSTYPE='FLAT',
                                        GCALSHUT='CLOSED')

newtypes.append(NICI_DARK_OLD())
