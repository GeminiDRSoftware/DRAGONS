
class CIRPASS(DataClassification):
    name="CIRPASS"
    usage = "Applies to datasets from the CIRPASS instrument"
    parent = "GEMINI"
    requirement = PHU(INSTRUME='CIRPASS')

newtypes.append(CIRPASS())
