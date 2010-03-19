
class NICI_DARK_OLD(DataClassification):
    name="NICI_DARK_OLD"
    usage = ""
    requirement = ISCLASS('NICI') & PHU(OBSTYPE='FLAT',
                                        GCALSHUT='CLOSED')

newtypes.append(NICI_DARK_OLD())
