
class GRACES(DataClassification):
    name="GRACES"
    usage = "Applies to all datasets from the GRACES instrument."
    parent = "GEMINI"
    requirement = PHU(INSTRUME='GRACES') | PHU(INSTRUME='graces')

newtypes.append(GRACES())
