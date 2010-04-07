
class NICI_IMAGE(DataClassification):
    name="NICI_IMAGE"
    usage = "Applies to imaging datasts from the NICI instrument."
    parent = "NICI"
    requirement = ISCLASS('NICI') & PHU(INSTRUME='NICI')

newtypes.append(NICI_IMAGE())
