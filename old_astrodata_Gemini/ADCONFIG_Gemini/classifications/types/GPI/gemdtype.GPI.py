
class GPI(DataClassification):
    name="GPI"
    usage = "Applies to datasets from the GPI instrument"
    parent = "GEMINI"
    requirement = PHU(INSTRUME='GPI')

newtypes.append(GPI())
