
class FLAMINGOS(DataClassification):
    name="FLAMINGOS"
    usage = "Applies to all datasets from the FLAMINGOS visitor instrument. Note, this is NOT F2."
    parent = "GEMINI"
    requirement = PHU(INSTRUME='FLAMINGOS')

newtypes.append(FLAMINGOS())
