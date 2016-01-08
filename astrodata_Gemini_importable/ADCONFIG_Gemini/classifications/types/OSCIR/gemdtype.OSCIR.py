
class OSCIR(DataClassification):
    name="OSCIR"
    usage = "Applies to datasets from the OSCIR instrument"
    parent = "GEMINI"
    requirement = OR(PHU(INSTRUME='oscir'), PHU(INSTRUME='OSCIR'))

newtypes.append(OSCIR())
