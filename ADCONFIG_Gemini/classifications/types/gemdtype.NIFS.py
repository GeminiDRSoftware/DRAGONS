
class NIFS(DataClassification):
    name="NIFS"
    usage = "Applies to datasets from NIFS instrument"
    parent = "GEMINI"
    requirement = PHU(INSTRUME='NIFS')

newtypes.append(NIFS())
