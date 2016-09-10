
class TEXES(DataClassification):
    name="TEXES"
    usage = "Applies to datasets from the TEXES instrument"
    parent = "GEMINI"
    requirement = PHU(INSTRUME='TEXES')

newtypes.append(TEXES())
