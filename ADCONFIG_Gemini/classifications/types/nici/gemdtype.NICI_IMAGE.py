
class NICI_IMAGE(DataClassification):
    name="NICI_IMAGE"
    usage = ""
    requirement = ISCLASS('NICI') & PHU(INSTRUME='NICI')

newtypes.append(NICI_IMAGE())
