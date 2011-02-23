
class OSCIR(DataClassification):
    name="OSCIR"
    usage = "Applies to datasets from the OSCIR instrument"
    parent = "GEMINI"
    requirement = PHU(INSTRUME='oscir')

newtypes.append(OSCIR())
