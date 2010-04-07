
class MICHELLE(DataClassification):
    name="MICHELLE"
    usage = "Applies to datasets from the MICHELLE instrument"
    parent = "GEMINI"
    requirement = PHU(INSTRUME= 'michelle')

newtypes.append(MICHELLE())
