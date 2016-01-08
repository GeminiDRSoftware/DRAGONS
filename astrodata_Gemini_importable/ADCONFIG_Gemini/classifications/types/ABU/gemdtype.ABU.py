
class ABU(DataClassification):
    name="ABU"
    usage = "Applies to datasets from the ABU instrument"
    parent = "GEMINI"
    requirement = PHU(INSTRUME='ABU')

newtypes.append(ABU())
